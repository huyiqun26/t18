
import asyncio
import atexit
import logging
import math
from collections import defaultdict
from logging.handlers import RotatingFileHandler
import os
import re
from pathlib import Path
from functools import partial
import signal
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


# ========================== 基础配置 ==========================
APP_TITLE = "铁路运输配载服务"
API_HOST = os.getenv("RAILWAY_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("RAILWAY_API_PORT", "2376"))
HEALTH_PATH = "/health"
LOG_FILE_NAME = "railway_service.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
MAX_ALGO_WORKERS = 4
REQUEST_TIMEOUT_SECONDS = 45.0


def get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_DIR = get_app_dir()
LOG_PATH = APP_DIR / LOG_FILE_NAME


def ensure_log_dir() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


ensure_log_dir()

logger = logging.getLogger("railway_service_linux")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

_file_handler = RotatingFileHandler(
    LOG_PATH,
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(threadName)s - %(message)s"))
logger.addHandler(_file_handler)


class StreamToLogger:
    def __init__(self, log_obj: logging.Logger, level: int):
        self.log_obj = log_obj
        self.level = level
        self._buffer = ""

    def write(self, message: str) -> None:
        if not message:
            return
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.log_obj.log(self.level, line)

    def flush(self) -> None:
        line = self._buffer.strip()
        if line:
            self.log_obj.log(self.level, line)
        self._buffer = ""

    def isatty(self) -> bool:
        return False


sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)


def handle_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        return
    logger.exception("未捕获异常", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_uncaught_exception


def _threading_excepthook(args) -> None:
    logger.exception("线程未捕获异常", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


if hasattr(threading, "excepthook"):
    threading.excepthook = _threading_excepthook


def configure_process_signals() -> None:
    if hasattr(signal, "SIGHUP"):
        try:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            logger.info("已忽略 SIGHUP，降低 SSH 断开导致进程退出的概率")
        except Exception:
            logger.exception("配置 SIGHUP 忽略失败")


# ========================== 数据模型 ==========================
class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class Organization(FlexibleModel):
    organizationID: str = ""
    organizationName: str = ""
    personCount: int = 0
    leiXing: str = ""
    yingjiName: str = ""
    componentList: List[Dict[str, Any]] = Field(default_factory=list)
    goodsList: List[Dict[str, Any]] = Field(default_factory=list)


class OptimizationRequest(FlexibleModel):
    systemSettings: Dict[str, Any]
    data: List[Organization]


def model_to_payload(model):
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        return dumper()

    dumper = getattr(model, "dict", None)
    if callable(dumper):
        return dumper()

    raise TypeError(f"不支持的请求模型类型: {type(model)}")


# ========================== 算法核心（公司成组拼车 + 同SC内物资混装） ==========================
class SubContainer:
    def __init__(self, box_type, length_unit, weight_empty, max_capacity, capacity_type='count', category=None,
                 zzsbid='', zhuang_zai=''):
        self.box_type = box_type
        self.length_unit = float(length_unit)
        self.weight = float(weight_empty)
        self.max_capacity = max_capacity
        self.capacity_type = capacity_type
        self.current_load = 0.0
        self.contents: List[Dict[str, Any]] = []
        self.owners = set()
        self.equip_category = category
        self.zzsbid = str(zzsbid or '').strip()
        self.zhuang_zai = str(zhuang_zai or '').strip()

        # 新增：用于物资（goodsList）按体积和载重进行装箱
        self.max_volume = 0.0
        self.max_payload = 0.0
        self.current_volume = 0.0
        self.current_payload = 0.0

        # 物资混装规则：
        # - 同类物资识别键固定为 name + zzsbid，不再优先使用 ID；
        # - 优先让同一类物资按 zzsbidNumber 装满一个小箱；
        # - 某一类在本箱达到 zzsbidNumber 后，整箱关闭，不允许其他物资继续进入；
        # - 只有某类物资剩余数量不足一箱时，才允许与其他同样不足一箱的物资按体积/载重/zjdh规则拼入同一箱。
        self.goods_item_counts = defaultdict(int)
        self.goods_item_limits = {}
        self.goods_closed = False

    def add_item(self, company_id, item_info, item_weight, item_load_value, item_volume=0.0):
        # 物资/装备特有的装箱逻辑：
        # - Small物资：校验体积、载重、同类件数上限；
        # - Large装备：不校验sbrl/sbzz，但必须校验装载占用比例，sum(1/zzsbidNumber)<=1。
        # zjdh/尾数拼箱资格在选择 best_box 前统一校验，避免 add_item 依赖外部矩阵参数。
        if self.capacity_type == 'component_pack':
            item_key = item_info.get('_component_item_key') or component_item_key(item_info)
            item_limit = safe_int(item_info.get('_component_item_limit', item_info.get('zzsbidNumber', 1)), 1)
            if item_limit <= 0:
                item_limit = 1

            item_fraction = safe_float(
                item_info.get('_component_item_fraction', item_info.get('occupancy', item_load_value)),
                0.0
            )
            if item_fraction <= 0:
                item_fraction = 1.0 / item_limit

            if self.goods_closed:
                return False
            if self.goods_item_counts[item_key] + 1 > item_limit:
                return False
            if self.current_load + item_fraction <= self.max_capacity + 1e-6:
                self.current_load += item_fraction
                # Large不以体积/载重作为箱内拼装限制，但保留统计值，便于调试与输出扩展。
                self.current_volume += float(item_volume)
                self.current_payload += float(item_weight)
                self.weight += float(item_weight)
                self.contents.append(item_info)
                self.owners.add(company_id)
                self.goods_item_counts[item_key] += 1
                self.goods_item_limits[item_key] = item_limit
                if self.goods_item_counts[item_key] >= item_limit:
                    # 某一类装备达到自身zzsbidNumber后，视为单类满箱，本箱关闭。
                    self.goods_closed = True
                return True
            return False

        if self.capacity_type == 'goods_pack':
            item_key = item_info.get('_goods_item_key') or goods_item_key(item_info)
            item_limit = safe_int(item_info.get('_goods_item_limit', item_info.get('zzsbidNumber', 1)), 1)
            if item_limit <= 0:
                item_limit = 1

            if self.goods_closed:
                return False
            if self.goods_item_counts[item_key] + 1 > item_limit:
                return False
            if self.current_volume + item_volume <= self.max_volume + 1e-6 and \
                    self.current_payload + item_weight <= self.max_payload + 1e-6:
                self.current_volume += item_volume
                self.current_payload += item_weight
                self.weight += float(item_weight)
                self.contents.append(item_info)
                self.owners.add(company_id)
                self.goods_item_counts[item_key] += 1
                self.goods_item_limits[item_key] = item_limit
                if self.goods_item_counts[item_key] >= item_limit:
                    # 恢复原始闭箱逻辑：某一类物资达到该箱上限后，本箱视为已满，其他类型不能再进入。
                    self.goods_closed = True
                return True
            return False
        # 人员的装箱逻辑：按件数或载物比例
        else:
            if self.current_load + item_load_value <= self.max_capacity + 1e-6:
                self.current_load += item_load_value
                self.weight += float(item_weight)
                self.contents.append(item_info)
                self.owners.add(company_id)
                return True
            return False

    @property
    def is_mixed(self):
        return len(self.owners) > 1


class AlgorithmError(Exception):
    pass


ALGO_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_ALGO_WORKERS, thread_name_prefix="algo-worker")
ALGO_GATE = threading.BoundedSemaphore(MAX_ALGO_WORKERS)


def safe_int(value, default=0):
    try:
        if value is None or value == '':
            return default
        return int(value)
    except Exception:
        return default


def normalize_yingji_name(value):
    if value is None:
        return ''
    return str(value).strip()


def is_effective_yingji_name(value):
    return normalize_yingji_name(value) != ''


def get_company_yingji_name(comp):
    if 'yingjiName' in comp:
        return normalize_yingji_name(comp.get('yingjiName'))
    u_class = safe_int(comp.get('Unitclass'), 0)
    if u_class in (1, 2, 3):
        return str(u_class)
    return ''


def safe_float(value, default=0.0):
    try:
        if value is None or value == '':
            return default
        return float(value)
    except Exception:
        return default


def normalize_lei_xing(value):
    s = str(value or '').strip().lower()
    return s if s in {'j', 's', 'l', 't'} else ''


def get_public_box_type(box_type):
    if str(box_type).startswith('Person_Box') or box_type == 'Person':
        return 'Person'
    if box_type in ('Equip_Box_Large', 'Large'):
        return 'Large'
    if box_type in ('Equip_Box_Small', 'Small'):
        return 'Small'
    return str(box_type or '')


def parse_tl_zzsb_specs(box_specs):
    spec_list = box_specs.get('tlZzsbVOList')
    if not isinstance(spec_list, list):
        raise AlgorithmError('缺少 systemSettings.Box_Specs.tlZzsbVOList 配置')

    by_id = {}
    by_name = defaultdict(list)
    for spec in spec_list:
        if not isinstance(spec, dict):
            continue
        sid = str(spec.get('id', '')).strip()
        sbmc = str(spec.get('sbmc', '')).strip()
        parsed = {
            'id': sid,
            'sbmc': sbmc,
            'sbhc': safe_float(spec.get('sbhc'), 0.0),
            'sbzl': safe_float(spec.get('sbzl'), 0.0),
            'sbryrl': safe_int(spec.get('sbryrl'), 0),
            'sbzz': safe_float(spec.get('sbzz'), 0.0),  # 最大载重
            'sbrl': safe_float(spec.get('sbrl'), 0.0),  # 最大体积
            'raw': spec,
        }
        if sid:
            by_id[sid] = parsed
        if sbmc:
            by_name[sbmc].append(parsed)

    if not by_id and not by_name:
        raise AlgorithmError('tlZzsbVOList 中没有有效的装载车辆配置')
    return by_id, by_name


def choose_person_spec(specs_by_name, sbmc):
    candidates = specs_by_name.get(sbmc, [])
    if not candidates:
        raise AlgorithmError(f'缺少人员装载车辆配置：sbmc={sbmc}')
    with_capacity = [s for s in candidates if s.get('sbryrl', 0) > 0]
    spec = with_capacity[0] if with_capacity else candidates[0]
    if spec.get('sbryrl', 0) <= 0:
        raise AlgorithmError(f'人员装载车辆 sbmc={sbmc} 缺少有效 sbryrl 容量')
    return spec


def choose_loading_spec_by_id(specs_by_id, zzsbid, item_name=''):
    sid = str(zzsbid or '').strip()
    if not sid:
        raise AlgorithmError(f'{item_name} 缺少 zzsbid，无法匹配装载车辆')
    spec = specs_by_id.get(sid)
    if spec is None:
        raise AlgorithmError(f'{item_name} 的 zzsbid={sid} 未在 Box_Specs.tlZzsbVOList.id 中找到')
    return spec


def item_fraction_capacity(zzsbid_number, item_name=''):
    num = safe_float(zzsbid_number, 0.0)
    if num <= 0:
        raise AlgorithmError(f'{item_name} 的 zzsbidNumber 必须大于 0')
    return 1.0 / num



def goods_item_key(item):
    """
    同类物资识别键。
    甲方的 ID 可能是每件物资的唯一 ID，不能用 ID 判断“同类”。
    现在固定使用 name + zzsbid：同名且使用同一装载车辆配置，才视为同类物资。
    """
    return '|'.join([
        str(item.get('name', '')).strip(),
        str(item.get('zzsbid', '')).strip(),
    ])




def prepare_goods_items_for_tailmix(items):
    """
    为“先同类装满、尾数再拼箱”准备物资列表。
    - 同类识别：name + zzsbid；
    - 每类先按 zzsbidNumber 切出可装满整箱的部分；
    - 只有最后不足 zzsbidNumber 的尾数物资标记为 _goods_tail_candidate=True，允许与其他尾数物资拼箱。
    """
    grouped = defaultdict(list)
    for raw in items:
        item = dict(raw)
        key = goods_item_key(item)
        item['_goods_item_key'] = key
        grouped[key].append(item)

    prepared = []
    for key, arr in grouped.items():
        if not arr:
            continue
        # 同一类物资应使用相同 zzsbidNumber；若输入有差异，取最小正数作为保守上限。
        limits = [safe_int(x.get('_goods_item_limit', x.get('zzsbidNumber', 1)), 1) for x in arr]
        limits = [x for x in limits if x > 0]
        item_limit = min(limits) if limits else 1
        tail_count = len(arr) % item_limit
        full_count = len(arr) - tail_count
        for pos, item in enumerate(arr):
            item['_goods_item_key'] = key
            item['_goods_item_limit'] = item_limit
            item['_goods_tail_candidate'] = bool(tail_count > 0 and pos >= full_count)
            item['_goods_group_count'] = len(arr)
            item['_goods_tail_count'] = tail_count
            prepared.append(item)

    # 先处理非尾数部分，使每类物资优先装满本类箱；再处理尾数部分用于拼箱。
    prepared.sort(
        key=lambda x: (
            1 if x.get('_goods_tail_candidate') else 0,
            str(x.get('zzsbid', '')).strip(),
            -item_fraction_capacity(x.get('_goods_item_limit', x.get('zzsbidNumber', 1)), x.get('name', '物资')),
            -safe_float(x.get('tj'), 0.0),
            -safe_float(x.get('weight'), 0.0),
            str(x.get('name', '')).strip(),
        )
    )
    return prepared


def component_item_key(item):
    """
    同类装备识别键。
    与物资规则保持一致，不使用 componentID 判断同类，固定使用 componentname + zzsbid。
    """
    return '|'.join([
        str(item.get('componentname', '')).strip(),
        str(item.get('zzsbid', '')).strip(),
    ])


def component_item_volume(item):
    """装备体积字段兼容。优先 tj，其次 componentvolume/componenttj/volume；没有则按 0 处理。"""
    for key in ('tj', 'componentvolume', 'componentVolume', 'componenttj', 'volume'):
        if key in item and item.get(key) not in (None, ''):
            return safe_float(item.get(key), 0.0)
    return 0.0


def component_item_fraction(item):
    """
    Large装备箱内占用比例。
    规则：单件装备占用比例 = 1 / zzsbidNumber。
    例如 zzsbidNumber=2 表示单件占 0.5；两个这样的装备已占满，不能再拼第三件。
    """
    explicit = safe_float(item.get('_component_item_fraction', item.get('occupancy', 0.0)), 0.0)
    if explicit > 0:
        return explicit

    item_limit = safe_int(item.get('_component_item_limit', item.get('zzsbidNumber', 1)), 1)
    if item_limit <= 0:
        item_limit = 1
    return 1.0 / item_limit


def prepare_component_items_for_tailmix(items):
    """
    为 Large 装备执行“先同类装满、尾数再拼箱”的准备。
    - 同类识别：componentname + zzsbid；
    - 每类先按 zzsbidNumber 切出可装满整箱的部分；
    - 只有最后不足 zzsbidNumber 的尾数装备标记为 _component_tail_candidate=True，允许与其他尾数装备拼箱。
    """
    grouped = defaultdict(list)
    for raw in items:
        item = dict(raw)
        key = component_item_key(item)
        item['_component_item_key'] = key
        grouped[key].append(item)

    prepared = []
    for key, arr in grouped.items():
        if not arr:
            continue
        limits = [safe_int(x.get('_component_item_limit', x.get('zzsbidNumber', 1)), 1) for x in arr]
        limits = [x for x in limits if x > 0]
        item_limit = min(limits) if limits else 1
        tail_count = len(arr) % item_limit
        full_count = len(arr) - tail_count
        for pos, item in enumerate(arr):
            item['_component_item_key'] = key
            item['_component_item_limit'] = item_limit
            item['_component_tail_candidate'] = bool(tail_count > 0 and pos >= full_count)
            item['_component_group_count'] = len(arr)
            item['_component_tail_count'] = tail_count
            prepared.append(item)

    # 非尾数先处理，尾数后处理；同一 zzsbid 内再按体积/重量大的优先。
    prepared.sort(
        key=lambda x: (
            1 if x.get('_component_tail_candidate') else 0,
            str(x.get('zzsbid', '')).strip(),
            -item_fraction_capacity(x.get('_component_item_limit', x.get('zzsbidNumber', 1)), x.get('componentname', '装备')),
            -component_item_volume(x),
            -safe_float(x.get('componentweight'), 0.0),
            str(x.get('componentname', '')).strip(),
        )
    )
    return prepared


# ========================== zjdh 字段映射与默认禁配矩阵 ==========================
# 用户给定的 27 行矩阵行号与 zjdh 字段值的对应关系。
ZJDH_ROW_LABELS = {
    1: ("1组1级",),
    2: ("1组2级",),
    3: ("1组3级",),
    4: ("1组4级",),
    5: ("1组5级",),
    6: ("1组6级", "1组7级"),
    7: ("1组8级",),
    8: ("1组9级",),
    9: ("1组10级",),
    10: ("1组11级",),
    11: ("1组12级",),
    12: ("1组13级",),
    13: ("2组21级",),
    14: ("2组22级",),
    15: ("2组23级",),
    16: ("3组32级", "3组33级"),
    17: ("3组34级", "3组35级"),
    18: ("4组41级",),
    19: ("4组43级",),
    20: ("4组44级",),
    21: ("5组51级",),
    22: ("6组61级",),
    23: ("6组62级",),
    24: ("7组72级",),
    25: ("7组71级", "7组73级", "7组74级", "7组75级", "7组76级", "7组77级"),
    26: ("8组81级",),
    27: ("8组82级",),
}


def _normalize_zjdh_label_text(value):
    return re.sub(r"\s+", "", str(value or "").strip())


ZJDH_LABEL_TO_ROW = {}
for _row_idx, _labels in ZJDH_ROW_LABELS.items():
    for _label in _labels:
        ZJDH_LABEL_TO_ROW[_normalize_zjdh_label_text(_label)] = _row_idx

# 由用户提供的 27x27 数字禁配矩阵内置而来。
# 值含义：1=不能混装，0=可以混装。
DEFAULT_ZJDH_FORBID_MATRIX = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
 [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]]

def normalize_zjdh_value(value):
    """
    把物资 zjdh 字段转换为禁配矩阵的行号 1..27。

    这里只按业务字段值识别，不把 yingjiName 当作 zjdh，也不把纯数字字符串
    例如 "25" 当作第25行，避免测试样例或外部输入产生歧义。
    例如：
    - "1组6级" 和 "1组7级" 均映射到第6行；
    - "7组71级/73级/74级/75级/76级/77级" 均映射到第25行。
    """
    if value is None or value == '':
        return None

    s = _normalize_zjdh_label_text(value)
    if not s:
        return None

    return ZJDH_LABEL_TO_ROW.get(s)


def _parse_matrix_from_string(text):
    rows = []
    for line in str(text).splitlines():
        nums = re.findall(r'[01]', line)
        if nums:
            rows.append([int(x) for x in nums])
    return rows


def normalize_zjdh_forbid_matrix(matrix):
    """
    规范化 zjdh 禁配矩阵。矩阵值含义：1=不能放一起，0=能放一起。
    兼容 27x27、27x28、28x27、28x28；只取前27行、前27列。
    """
    if matrix is None or matrix == '':
        return None
    if isinstance(matrix, str):
        matrix = _parse_matrix_from_string(matrix)
    if not isinstance(matrix, list):
        return None

    rows = []
    for row in matrix[:27]:
        if isinstance(row, str):
            vals = re.findall(r'[01]', row)
        elif isinstance(row, (list, tuple)):
            vals = row
        else:
            continue
        clean = []
        for v in vals[:27]:
            clean.append(1 if str(v).strip() in {'1', '1.0', 'true', 'True', '是'} else 0)
        if len(clean) < 27:
            return None
        rows.append(clean[:27])

    if len(rows) < 27:
        return None
    return rows[:27]


def load_zjdh_forbid_matrix(sys_settings=None):
    """
    返回代码内置的 zjdh 禁配矩阵。

    按当前接口要求，zjdhMatrix 不再放在 input/systemSettings 里传入，
    而是固定写在 DEFAULT_ZJDH_FORBID_MATRIX 中。
    input 只需要在每类物资中提供 zjdh 字段值。
    """
    return DEFAULT_ZJDH_FORBID_MATRIX


def can_mix_zjdh(existing_item, new_item, zjdh_forbid_matrix):
    """
    按 zjdh 禁配矩阵判断两件物资是否允许混装。

    当前业务口径：
    - 只有两件物资都提供了有效 zjdh，且都能映射到内置的“1组1级”等 27 行范围时，才查 0/1 矩阵；
    - 只要任意一件物资未提供 zjdh 字段、zjdh 为空，或 zjdh 值不在映射范围内，就不触发禁配矩阵，直接视为 zjdh 层面允许混装；
    - zjdh 层面允许后，仍继续执行 name+zzsbid、尾数拼箱、体积/载重、yingjiName 等其他规则。
    """
    old_idx = normalize_zjdh_value(existing_item.get('zjdh'))
    new_idx = normalize_zjdh_value(new_item.get('zjdh'))

    # 新规则：缺失、空值、无法识别时，不再保守禁止，而是视为 zjdh 层面允许混装。
    if old_idx is None or new_idx is None:
        return True

    mat = zjdh_forbid_matrix if zjdh_forbid_matrix is not None else DEFAULT_ZJDH_FORBID_MATRIX

    # 矩阵可能不是完全对称，安全起见两边任一方向为1都视为不能混装。
    a = mat[old_idx - 1][new_idx - 1]
    b = mat[new_idx - 1][old_idx - 1]
    return a == 0 and b == 0


def can_pack_goods_item(box, new_item, company_yingji_name, zjdh_forbid_matrix):
    """统一校验一个物资能否进入某个 Small 箱。

    恢复“先同类装满、尾数再拼箱”逻辑：
    - 若箱子已经因某一类达到 zzsbidNumber 而关闭，则任何物资都不能再进入；
    - 同类物资可继续进入，直到达到自身 zzsbidNumber；
    - 不同类物资只有在双方都属于“尾数不足一箱”时才允许拼箱；若双方 zjdh 均有效，则须满足禁配矩阵，同时仍须满足公司/yingjiName限制。
    """
    if getattr(box, 'goods_closed', False):
        return False

    cid = new_item.get('company_id', '')
    if not can_mix_goods_owner(box, cid, company_yingji_name):
        return False

    new_key = new_item.get('_goods_item_key') or goods_item_key(new_item)
    new_is_tail = bool(new_item.get('_goods_tail_candidate', False))
    for old_item in getattr(box, 'contents', []):
        if old_item.get('type') != 'goods':
            continue
        old_key = old_item.get('_goods_item_key') or goods_item_key(old_item)
        # 先执行 zjdh 禁配判断；若任一方 zjdh 缺失/空/无法识别，则该层面直接放行，再继续判断同类/尾数拼箱规则。
        if not can_mix_zjdh(old_item, new_item, zjdh_forbid_matrix):
            return False
        if old_key != new_key:
            old_is_tail = bool(old_item.get('_goods_tail_candidate', False))
            if not (old_is_tail and new_is_tail):
                return False
    return True


def _has_nonempty_zjdh(item):
    return str(item.get('zjdh', '') or '').strip() != ''


def can_mix_component_zjdh_if_present(existing_item, new_item, zjdh_forbid_matrix):
    """Large装备的zjdh兼容判断。

    当前componentList历史上不一定提供zjdh，因此Large装备默认不强制zjdh；
    但如果任一装备提供了zjdh，则要求两者zjdh都可识别且矩阵允许，避免部分填字段后误混。
    """
    if not (_has_nonempty_zjdh(existing_item) or _has_nonempty_zjdh(new_item)):
        return True
    return can_mix_zjdh(existing_item, new_item, zjdh_forbid_matrix)


def can_pack_component_item(box, new_item, company_yingji_name, zjdh_forbid_matrix):
    """统一校验一个装备能否进入某个 Large 箱。

    Large规则：
    - 同一装载车辆zzsbid；
    - 同类识别为 componentname + zzsbid；
    - 单类达到 zzsbidNumber 后闭箱；
    - 不同类装备只有双方都是尾数不足一箱时才允许拼箱；
    - 不检查sbrl/sbzz；
    - 必须检查箱内占用比例，sum(1/zzsbidNumber)<=1；
    - 跨公司混装时该箱涉及的yingjiName种类不能超过2。
    """
    if getattr(box, 'goods_closed', False):
        return False

    if str(getattr(box, 'zzsbid', '')).strip() != str(new_item.get('zzsbid', '')).strip():
        return False

    cid = new_item.get('company_id', '')
    if not can_mix_goods_owner(box, cid, company_yingji_name):
        return False

    new_fraction = component_item_fraction(new_item)
    if getattr(box, 'current_load', 0.0) + new_fraction > getattr(box, 'max_capacity', 1.0) + 1e-6:
        return False

    new_key = new_item.get('_component_item_key') or component_item_key(new_item)
    new_is_tail = bool(new_item.get('_component_tail_candidate', False))
    for old_item in getattr(box, 'contents', []):
        if old_item.get('type') != 'component':
            continue
        old_key = old_item.get('_component_item_key') or component_item_key(old_item)
        # Large装备不走zjdh矩阵；不同component只按 componentname+zzsbid、尾数规则、占用比例和yingjiName判断。
        if old_key != new_key:
            old_is_tail = bool(old_item.get('_component_tail_candidate', False))
            if not (old_is_tail and new_is_tail):
                return False
    return True


def effective_yingji_names_for_owners(owners, company_yingji_name):
    return {
        company_yingji_name.get(cid, '')
        for cid in owners
        if is_effective_yingji_name(company_yingji_name.get(cid, ''))
    }


def can_mix_goods_owner(box, new_owner_id, company_yingji_name):
    """物资箱允许跨公司混装，但不能让该箱涉及的 yingjiName 种类超过 2。"""
    owners = set(getattr(box, 'owners', set()))
    owners.add(new_owner_id)
    return len(effective_yingji_names_for_owners(owners, company_yingji_name)) <= 2


def normalize_is_chaoxian(value):
    if value is None:
        return ''
    s = str(value).strip()
    if s in ('是', '否', ''):
        return s
    yes_values = {'Y', 'YES', 'Yes', 'yes', 'true', 'True', 'TRUE', '1', '超限'}
    no_values = {'N', 'NO', 'No', 'no', 'false', 'False', 'FALSE', '0', '不超限'}
    if s in yes_values:
        return '是'
    if s in no_values:
        return '否'
    return ''


def box_has_chaoxian_equipment(box):
    if get_public_box_type(getattr(box, 'box_type', '')) != 'Large':
        return False
    for item in getattr(box, 'contents', []):
        if item.get('type') == 'component' and normalize_is_chaoxian(item.get('is_chaoXian', '')) == '是':
            return True
    return False


def box_chaoxian_owners(box):
    owners = set()
    if get_public_box_type(getattr(box, 'box_type', '')) != 'Large':
        return owners
    for item in getattr(box, 'contents', []):
        if item.get('type') == 'component' and normalize_is_chaoxian(item.get('is_chaoXian', '')) == '是':
            owners.add(item.get('company_id', ''))
    owners.discard('')
    return owners


def dominant_ratio(weight, length, max_weight, max_length):
    return max(weight / max_weight if max_weight else 0.0,
               length / max_length if max_length else 0.0)


class VehicleState:
    def __init__(self):
        self.weight = 0.0
        self.length = 0.0
        self.units = []
        self.companies = set()
        self.yingji_companies = defaultdict(set)
        self.chaoXian_companies = set()

    def clone(self):
        other = VehicleState()
        other.weight = self.weight
        other.length = self.length
        other.units = list(self.units)
        other.companies = set(self.companies)
        other.yingji_companies = defaultdict(set, {g: set(cids) for g, cids in self.yingji_companies.items()})
        other.chaoXian_companies = set(self.chaoXian_companies)
        return other

    def can_place(self, unit, max_weight, max_length, company_yingji_name):
        if self.weight + unit['weight'] > max_weight + 1e-6:
            return False
        if self.length + unit['length'] > max_length + 1e-6:
            return False
        current_yingji_names = {
            y for y, cids in self.yingji_companies.items()
            if is_effective_yingji_name(y) and len(cids) > 0
        }
        unit_yingji_names = {
            company_yingji_name.get(cid, '') for cid in unit['owners']
            if is_effective_yingji_name(company_yingji_name.get(cid, ''))
        }
        if len(current_yingji_names | unit_yingji_names) > 2:
            return False
        return True

    def place(self, unit, company_yingji_name):
        self.weight += unit['weight']
        self.length += unit['length']
        self.units.append(unit)
        for cid in unit['owners']:
            self.companies.add(cid)
            yingji_name = company_yingji_name.get(cid, '')
            if is_effective_yingji_name(yingji_name):
                self.yingji_companies[yingji_name].add(cid)
        if unit.get('has_chaoXian_equipment'):
            self.chaoXian_companies.update(unit.get('chaoXian_owners', set()))

    def remove(self, unit, company_yingji_name):
        self.weight -= unit['weight']
        self.length -= unit['length']
        self.units.remove(unit)
        self.companies = set()
        self.yingji_companies = defaultdict(set)
        self.chaoXian_companies = set()
        for u in self.units:
            for cid in u['owners']:
                self.companies.add(cid)
                yingji_name = company_yingji_name.get(cid, '')
                if is_effective_yingji_name(yingji_name):
                    self.yingji_companies[yingji_name].add(cid)
            if u.get('has_chaoXian_equipment'):
                self.chaoXian_companies.update(u.get('chaoXian_owners', set()))


def run_engine(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sys_settings = raw_data.get('systemSettings', {})
        sc_limit = sys_settings.get('SC_Constraint', {'maxWeightLimit': 60000, 'maxLengthLimit': 800.0})
        person_weight = sys_settings.get('Person_Weight', {'weight_per_person': 75.0})['weight_per_person']
        box_specs = sys_settings.get('Box_Specs', {})
        zjdh_forbid_matrix = load_zjdh_forbid_matrix(sys_settings)

        max_weight_per_sc = float(sc_limit['maxWeightLimit'])
        max_length_per_sc = float(sc_limit['maxLengthLimit'])

        all_sub_containers = []
        open_person_boxes = defaultdict(list)  # key=(owner_id, sbmc/id)
        open_large_boxes = defaultdict(list)  # key=(owner_id, zzsbid)
        open_small_boxes = defaultdict(list)  # key=(owner_id, zzsbid)；预处理阶段只在公司内部混装，跨公司物资混装只在同一SC确定后执行

        specs_by_id, specs_by_name = parse_tl_zzsb_specs(box_specs)
        person_specs = {
            '软卧': choose_person_spec(specs_by_name, '软卧'),
            '硬卧': choose_person_spec(specs_by_name, '硬卧'),
            '硬座': choose_person_spec(specs_by_name, '硬座'),
        }

        companies = raw_data.get('data', [])
        company_yingji_name = {}
        company_name = {}
        for comp in companies:
            cid = comp.get('organizationID', '')
            if not cid:
                raise AlgorithmError('存在 organizationID 为空的单位数据')
            company_yingji_name[cid] = get_company_yingji_name(comp)
            company_name[cid] = comp.get('organizationName', '')

        missing_yingji_name = [cid for cid, y in company_yingji_name.items() if not is_effective_yingji_name(y)]
        if missing_yingji_name:
            print('提示：以下公司未提供非空 yingjiName，装车时不参与yingjiName种类数限制：')
            print(', '.join(missing_yingji_name[:20]) + ('...' if len(missing_yingji_name) > 20 else ''))

        def add_people_to_boxes(sbmc, num_people, owner_id):
            spec = person_specs.get(sbmc)
            if not spec or num_people <= 0:
                return 0
            cap = int(spec['sbryrl'])
            remaining = int(num_people)
            added_total = 0
            zzsbid = spec.get('id', '')
            zhuang_zai = spec.get('sbmc', sbmc)
            box_key = (owner_id, zzsbid or zhuang_zai)

            def create_person_info(count):
                return {
                    "type": "person",
                    "company_id": owner_id,
                    "box_type": "Person",
                    "count": int(count),
                    "zhuangZai": zhuang_zai,
                    "zzsbid": zzsbid,
                }

            for box in open_person_boxes[box_key]:
                if box.current_load < box.max_capacity:
                    space = int(box.max_capacity - box.current_load)
                    to_add = min(remaining, space)
                    if to_add > 0:
                        box.add_item(owner_id, create_person_info(to_add), to_add * person_weight, to_add)
                        remaining -= to_add
                        added_total += to_add
                    if remaining <= 0:
                        break

            while remaining > 0:
                to_add = min(remaining, cap)
                new_box = SubContainer(
                    'Person',
                    spec['sbhc'],
                    spec['sbzl'],
                    cap,
                    'count',
                    zzsbid=zzsbid,
                    zhuang_zai=zhuang_zai,
                )
                new_box.add_item(owner_id, create_person_info(to_add), to_add * person_weight, to_add)
                all_sub_containers.append(new_box)
                open_person_boxes[box_key].append(new_box)
                remaining -= to_add
                added_total += to_add

            return added_total

        def allocate_people_by_lei_xing(owner_id, person_count, lei_xing):
            remaining = int(person_count)
            lx = normalize_lei_xing(lei_xing)

            if lx == 'j':
                for _ in range(1):
                    if remaining > 0:
                        to_add = min(remaining, person_specs['软卧']['sbryrl'])
                        add_people_to_boxes('软卧', to_add, owner_id)
                        remaining -= to_add
                for _ in range(3):
                    if remaining > 0:
                        to_add = min(remaining, person_specs['硬卧']['sbryrl'])
                        add_people_to_boxes('硬卧', to_add, owner_id)
                        remaining -= to_add
            elif lx in {'s', 'l'}:
                for _ in range(2):
                    if remaining > 0:
                        to_add = min(remaining, person_specs['硬卧']['sbryrl'])
                        add_people_to_boxes('硬卧', to_add, owner_id)
                        remaining -= to_add
            elif lx == 't':
                if remaining > 0:
                    to_add = min(remaining, person_specs['硬卧']['sbryrl'])
                    add_people_to_boxes('硬卧', to_add, owner_id)
                    remaining -= to_add

            if remaining > 0:
                add_people_to_boxes('硬座', remaining, owner_id)

        for comp in companies:
            cid = comp.get('organizationID', '')
            p_count = safe_int(comp.get('personCount'), 0)
            allocate_people_by_lei_xing(cid, p_count, comp.get('leiXing', ''))

            # === 处理装备组件（Large：同zzsbid、componentname+zzsbid识别同类、尾数拼箱、yingjiName≤2；不检查sbrl/sbzz） ===
            comps_list = comp.get('componentList', []) or []
            expanded_components = []
            for c_item in comps_list:
                count = safe_int(c_item.get('count', 1), 1)
                for _ in range(count):
                    expanded_components.append(dict(c_item))

            # Large装备也执行“先同类装满、尾数再拼箱”。
            prepared_components = prepare_component_items_for_tailmix(expanded_components)

            for item in prepared_components:
                name = item.get('componentname', '')
                comp_id = item.get('componentID', '')
                spec = choose_loading_spec_by_id(specs_by_id, item.get('zzsbid', ''), f'装备 {name or comp_id}')
                w = safe_float(item.get('componentweight'), 0.0)
                vol = component_item_volume(item)
                item_limit = safe_int(item.get('zzsbidNumber', 1), 1)
                if item_limit <= 0:
                    raise AlgorithmError(f'装备 {name or comp_id} 的 zzsbidNumber 必须大于 0')
                occupancy = 1.0 / item_limit
                zzsbid = spec.get('id', '')
                zhuang_zai = spec.get('sbmc', '')
                c_key = component_item_key(item)

                item_info = {
                    "type": "component",
                    "company_id": cid,
                    "componentname": name,
                    "componentID": comp_id,
                    "componentweight": w,
                    "componentvolume": vol,
                    "tj": vol,
                    "bddxid": item.get('bddxid', ''),
                    "dxcode": item.get('dxcode', ''),
                    "is_chaoXian": normalize_is_chaoxian(item.get('is_chaoXian', '')),
                    "zzsbid": item.get('zzsbid', ''),
                    "zzsbidNumber": item.get('zzsbidNumber', ''),
                    "zjdh": item.get('zjdh', ''),
                    "zhuangZai": zhuang_zai,
                    "count": 1,
                    "occupancy": occupancy,
                    # 内部字段：仅用于Large执行“同类zzsbidNumber上限”“尾数拼箱”和“箱内占用比例”规则，输出时不会保留下划线字段。
                    "_component_item_key": c_key,
                    "_component_item_limit": item_limit,
                    "_component_item_fraction": occupancy,
                    "_component_tail_candidate": bool(item.get('_component_tail_candidate', False)),
                    "_component_group_count": item.get('_component_group_count', ''),
                    "_component_tail_count": item.get('_component_tail_count', ''),
                }

                # 初装阶段仍按“同公司 + 同zzsbid”开放Large箱；跨公司Large混装放到同一SC确定后的二次重装阶段执行。
                key = (cid, zzsbid)
                placed = False
                best_box = None
                best_score = None
                for box in open_large_boxes[key]:
                    if box.capacity_type != 'component_pack':
                        continue
                    if getattr(box, 'goods_closed', False):
                        continue
                    if not can_pack_component_item(box, item_info, company_yingji_name, zjdh_forbid_matrix):
                        continue
                    if box.goods_item_counts[c_key] + 1 > item_limit:
                        continue
                    # Large不按sbrl/sbzz判断；只按同zzsbid、尾数拼箱、zzsbidNumber、占用比例和yingjiName规则。
                    new_load_ratio = (box.current_load + occupancy) / box.max_capacity if box.max_capacity else 1.0
                    count_ratio = (box.goods_item_counts[c_key] + 1) / item_limit if item_limit else 1.0
                    owner_bonus = 0.15 if cid in box.owners else 0.0
                    chao_bonus = 0.10 if normalize_is_chaoxian(item_info.get('is_chaoXian', '')) == '是' else 0.0
                    tail_bonus = 0.08 if item_info.get('_component_tail_candidate') else 0.0
                    score = 0.75 * new_load_ratio + 0.15 * count_ratio + owner_bonus + chao_bonus + tail_bonus
                    if best_score is None or score > best_score:
                        best_score = score
                        best_box = box

                if best_box is not None:
                    if not best_box.add_item(cid, item_info, w, occupancy, item_volume=vol):
                        raise AlgorithmError(f'装备 {name or comp_id} 匹配到旧Large箱但装入失败，请检查体积/载重/zzsbidNumber')
                    placed = True

                if not placed:
                    new_box = SubContainer(
                        'Large',
                        spec['sbhc'],
                        spec['sbzl'],
                        1.0,
                        'component_pack',
                        zzsbid=zzsbid,
                        zhuang_zai=zhuang_zai,
                    )
                    # Large不使用sbrl/sbzz作为拼箱限制；这里设为极大值，仅保留SC总重/总换长校验。
                    new_box.max_payload = 999999999.0
                    new_box.max_volume = 999999999.0
                    if not new_box.add_item(cid, item_info, w, occupancy, item_volume=vol):
                        raise AlgorithmError(
                            f'装备 {name or comp_id} Large装入失败：zzsbid={zzsbid}, '
                            f'请检查zzsbidNumber或尾数拼箱规则'
                        )
                    all_sub_containers.append(new_box)
                    open_large_boxes[key].append(new_box)

            # === 处理物资（按文档新规则：不再按category分装，按zzsbidNumber、体积、载重混装） ===
            goods_list = comp.get('goodsList', []) or []
            expanded_goods = []
            for g in goods_list:
                count = safe_int(g.get('count', 1), 1)
                for _ in range(count):
                    expanded_goods.append(g.copy())

            # 恢复“先同类装满、尾数再拼箱”：非尾数部分先处理，尾数部分后处理并允许按规则拼箱。
            flat_goods = prepare_goods_items_for_tailmix(expanded_goods)

            for item in flat_goods:
                name = item.get('name', '')
                gid = item.get('ID', '')
                spec = choose_loading_spec_by_id(specs_by_id, item.get('zzsbid', ''), f'物资 {name or gid}')
                w = safe_float(item.get('weight'), 0.0)
                tj = safe_float(item.get('tj'), 0.0)
                item_limit = safe_int(item.get('zzsbidNumber', 1), 1)
                if item_limit <= 0:
                    raise AlgorithmError(f'物资 {name or gid} 的 zzsbidNumber 必须大于 0')
                cat = item.get('category', '未分类')
                zzsbid = spec.get('id', '')
                zhuang_zai = spec.get('sbmc', '')
                g_key = goods_item_key(item)

                item_info = {
                    "type": "goods",
                    "company_id": cid,
                    "name": name,
                    "ID": gid,
                    "bddxid": item.get('bddxid', ''),
                    "category": cat,
                    "dxcode": item.get('dxcode', ''),
                    "tj": tj,
                    "weight": w,
                    "zzsbid": item.get('zzsbid', ''),
                    "zzsbidNumber": item.get('zzsbidNumber', ''),
                    "zjdh": item.get('zjdh', ''),
                    "zhuangZai": zhuang_zai,
                    "count": 1,
                    # 内部字段：仅用于执行“每类物资zzsbidNumber件数上限”和“尾数拼箱”规则，输出时不会保留。
                    "_goods_item_key": g_key,
                    "_goods_item_limit": item_limit,
                    "_goods_tail_candidate": bool(item.get('_goods_tail_candidate', False)),
                    "_goods_group_count": item.get('_goods_group_count', ''),
                    "_goods_tail_count": item.get('_goods_tail_count', ''),
                }

                # 文档要求：物资暂不按category分装；但跨公司物资混装必须以这些公司已经同列SC为前提。
                # 因此预处理阶段只按“同公司 + 同装载车辆zzsbid”开放物资箱；
                # 等SC装车方案确定后，再在同一SC内部对Small物资箱进行跨公司重装/混装。
                key = (cid, zzsbid)
                placed = False
                best_box = None
                best_score = None

                for box in open_small_boxes[key]:
                    if box.capacity_type != 'goods_pack':
                        continue
                    if getattr(box, 'goods_closed', False):
                        continue
                    if not can_pack_goods_item(box, item_info, company_yingji_name, zjdh_forbid_matrix):
                        continue
                    if box.goods_item_counts[g_key] + 1 > item_limit:
                        continue
                    if box.current_volume + tj > box.max_volume + 1e-6:
                        continue
                    if box.current_payload + w > box.max_payload + 1e-6:
                        continue

                    new_volume = box.current_volume + tj
                    new_payload = box.current_payload + w
                    vol_ratio = new_volume / box.max_volume if box.max_volume else 0.0
                    wt_ratio = new_payload / box.max_payload if box.max_payload else 0.0
                    count_ratio = (box.goods_item_counts[g_key] + 1) / item_limit if item_limit else 1.0
                    owner_bonus = 0.15 if cid in box.owners else 0.0
                    # 优先选装后更满的箱；同公司略优先，但不禁止跨公司混装。
                    score = 0.55 * max(vol_ratio, wt_ratio) + 0.25 * min(vol_ratio, wt_ratio) + 0.20 * count_ratio + owner_bonus
                    if best_score is None or score > best_score:
                        best_score = score
                        best_box = box

                if best_box is not None:
                    if not best_box.add_item(cid, item_info, w, 0.0, item_volume=tj):
                        raise AlgorithmError(f'物资 {name or gid} 匹配到旧箱但装入失败，请检查体积/载重/zzsbidNumber')
                    placed = True

                if not placed:
                    new_box = SubContainer(
                        'Small',
                        spec['sbhc'],
                        spec['sbzl'],
                        1.0,
                        'goods_pack',
                        category=None,
                        zzsbid=zzsbid,
                        zhuang_zai=zhuang_zai,
                    )
                    # 文档使用 sbzz 表示最大载重、sbrl 表示最大体积；输入暂未提供时沿用大容量兜底，保证兼容旧样例。
                    new_box.max_payload = spec.get('sbzz') if spec.get('sbzz', 0) > 0 else 999999999.0
                    new_box.max_volume = spec.get('sbrl') if spec.get('sbrl', 0) > 0 else 999999999.0

                    if not new_box.add_item(cid, item_info, w, 0.0, item_volume=tj):
                        raise AlgorithmError(
                            f'物资 {name or gid} 自身超过装载车辆限制：weight={w:.1f}, tj={tj:.2f}, '
                            f'zzsbid={zzsbid}, sbzz={new_box.max_payload:.1f}, sbrl={new_box.max_volume:.2f}'
                        )
                    all_sub_containers.append(new_box)
                    open_small_boxes[key].append(new_box)

        logger.info("预处理完成，生成小车/小箱总数=%s", len(all_sub_containers))
        original_boxes = all_sub_containers.copy()

        for i, box in enumerate(original_boxes):
            if box.weight > max_weight_per_sc + 1e-6 or box.length_unit > max_length_per_sc + 1e-6:
                raise AlgorithmError(
                    f"Box_{i + 1:04d} 自身超过单车限制：weight={box.weight:.1f}, length={box.length_unit:.2f}"
                )

        def person_count_in_box(box):
            if get_public_box_type(getattr(box, 'box_type', '')) != 'Person':
                return 0
            return sum(safe_int(item.get('count'), 0) for item in getattr(box, 'contents', []) if item.get('type') == 'person')

        def clone_person_box_with_count(src_box, count):
            """按给定人数拆出一个新的人员箱。

            为满足“同一公司有人又有物时，每个装车单元必须人-物同车”的硬规则，
            必要时允许把原本一个满载人员箱拆成多个同型号人员箱，并把人员数量分摊进去。
            拆分会增加人员箱自重和换长，后续仍按超重/超换长硬约束校验。
            """
            count = int(count)
            if count <= 0:
                raise AlgorithmError('人员箱拆分失败：拆分人数必须大于0')
            total_count = person_count_in_box(src_box)
            empty_weight = max(0.0, float(src_box.weight) - total_count * float(person_weight))
            first_item = None
            for item in getattr(src_box, 'contents', []):
                if item.get('type') == 'person':
                    first_item = item
                    break
            if first_item is None:
                raise AlgorithmError('人员箱拆分失败：原人员箱缺少人员明细')

            new_box = SubContainer(
                src_box.box_type,
                src_box.length_unit,
                empty_weight,
                src_box.max_capacity,
                src_box.capacity_type,
                zzsbid=getattr(src_box, 'zzsbid', ''),
                zhuang_zai=getattr(src_box, 'zhuang_zai', ''),
            )
            item_info = dict(first_item)
            item_info['count'] = count
            if not new_box.add_item(item_info.get('company_id', ''), item_info, count * float(person_weight), count):
                raise AlgorithmError('人员箱拆分失败：拆分后的人员箱超过自身容量')
            return new_box

        def split_person_boxes_for_hard_balance():
            """为人-物同车硬规则准备足够细粒度的人员箱。

            如果某公司同时存在人员和物资/装备，而其总量至少需要多辆车，
            但人员箱数量少于理论最少装车单元数，则把已有人员箱按人数拆成更多同型号人员箱。
            这样后续每个装车单元才有机会至少配到一个人员箱。
            """
            nonlocal original_boxes

            def company_presence():
                presence = defaultdict(lambda: {'person': [], 'non_person': []})
                for idx, box in enumerate(original_boxes):
                    public_type = get_public_box_type(getattr(box, 'box_type', ''))
                    for cid0 in getattr(box, 'owners', set()):
                        if public_type == 'Person':
                            presence[cid0]['person'].append(idx)
                        else:
                            presence[cid0]['non_person'].append(idx)
                return presence

            changed = True
            guard = 0
            while changed:
                changed = False
                guard += 1
                if guard > 10000:
                    raise AlgorithmError('人员箱拆分次数异常，请检查输入数据')

                presence = company_presence()
                for cid0, parts in presence.items():
                    person_idxs = parts['person']
                    non_person_idxs = parts['non_person']
                    if not person_idxs or not non_person_idxs:
                        continue

                    all_idxs = person_idxs + non_person_idxs
                    total_w = sum(original_boxes[i].weight for i in all_idxs)
                    total_l = sum(original_boxes[i].length_unit for i in all_idxs)
                    required_by_weight = int(math.ceil(total_w / max_weight_per_sc)) if max_weight_per_sc > 0 else 1
                    required_by_length = int(math.ceil(total_l / max_length_per_sc)) if max_length_per_sc > 0 else 1
                    required_units = max(1, required_by_weight, required_by_length)

                    if len(person_idxs) >= required_units:
                        continue

                    total_people = sum(person_count_in_box(original_boxes[i]) for i in person_idxs)
                    if total_people < required_units:
                        raise AlgorithmError(
                            f'公司 {company_name.get(cid0, cid0)}({cid0}) 同时存在人员和物资/装备，'
                            f'至少需要{required_units}个装车单元，但人员总数只有{total_people}，'
                            f'无法保证每个装车单元都至少有人'
                        )

                    # 选择人数最多且可继续拆分的人员箱，一分为二。
                    splittable = [i for i in person_idxs if person_count_in_box(original_boxes[i]) >= 2]
                    if not splittable:
                        raise AlgorithmError(
                            f'公司 {company_name.get(cid0, cid0)}({cid0}) 人员箱数量不足且无法继续拆分，'
                            f'无法满足每个装车单元人-物同车硬规则'
                        )
                    split_idx = max(splittable, key=lambda i: person_count_in_box(original_boxes[i]))
                    src_box = original_boxes[split_idx]
                    cnt = person_count_in_box(src_box)
                    left = cnt // 2
                    right = cnt - left
                    original_boxes[split_idx] = clone_person_box_with_count(src_box, left)
                    original_boxes.append(clone_person_box_with_count(src_box, right))
                    changed = True
                    break

        split_person_boxes_for_hard_balance()

        for i, box in enumerate(original_boxes):
            if box.weight > max_weight_per_sc + 1e-6 or box.length_unit > max_length_per_sc + 1e-6:
                raise AlgorithmError(
                    f"Box_{i + 1:04d} 拆分后自身超过单车限制：weight={box.weight:.1f}, length={box.length_unit:.2f}"
                )

        def create_merged_box(box_list):
            first = box_list[0]
            if get_public_box_type(first.box_type) == 'Person':
                merged = SubContainer(first.box_type, 0.0, 0.0, first.max_capacity, first.capacity_type,
                                      zzsbid=getattr(first, 'zzsbid', ''), zhuang_zai=getattr(first, 'zhuang_zai', ''))
            elif get_public_box_type(first.box_type) == 'Small':
                merged = SubContainer(first.box_type, 0.0, 0.0, first.max_capacity, first.capacity_type,
                                      category=first.equip_category, zzsbid=getattr(first, 'zzsbid', ''),
                                      zhuang_zai=getattr(first, 'zhuang_zai', ''))
                # 同步继承物资装箱信息
                merged.max_volume = getattr(first, 'max_volume', 0.0)
                merged.max_payload = getattr(first, 'max_payload', 0.0)
                merged.current_volume = sum(getattr(b, 'current_volume', 0.0) for b in box_list)
                merged.current_payload = sum(getattr(b, 'current_payload', 0.0) for b in box_list)
            else:
                raise ValueError(f"不支持合并的箱子类型: {first.box_type}")

            merged.length_unit = sum(b.length_unit for b in box_list)
            merged.weight = sum(b.weight for b in box_list)
            merged.owners = set()
            merged.contents = []
            for b in box_list:
                merged.contents.extend(b.contents)
                merged.owners.update(b.owners)
                if getattr(b, 'capacity_type', '') == 'goods_pack':
                    for k, v in getattr(b, 'goods_item_counts', {}).items():
                        merged.goods_item_counts[k] += v
                    merged.goods_item_limits.update(getattr(b, 'goods_item_limits', {}))
                    merged.goods_closed = merged.goods_closed or getattr(b, 'goods_closed', False)
            return merged

        def can_merge(box_list):
            total_len = sum(b.length_unit for b in box_list)
            total_weight = sum(b.weight for b in box_list)
            return (total_len <= max_length_per_sc + 1e-6 and total_weight <= max_weight_per_sc + 1e-6)

        company_box_type_presence = defaultdict(lambda: {'person': False, 'non_person': False})
        for box in original_boxes:
            public_type = get_public_box_type(getattr(box, 'box_type', ''))
            for cid0 in getattr(box, 'owners', set()):
                if public_type == 'Person':
                    company_box_type_presence[cid0]['person'] = True
                else:
                    company_box_type_presence[cid0]['non_person'] = True
        companies_need_hard_person_nonperson_balance = {
            cid0 for cid0, flags in company_box_type_presence.items()
            if flags.get('person') and flags.get('non_person')
        }

        group_dict = defaultdict(list)
        for idx, box in enumerate(original_boxes):
            if box.is_mixed:
                group_dict[('mixed', idx)].append(idx)
            else:
                cid = list(box.owners)[0]
                public_type = get_public_box_type(box.box_type)
                if public_type == 'Large':
                    group_dict[('large', idx)].append(idx)
                elif public_type == 'Person':
                    # 人员箱不再预先合并成接近一整列的“大块”，避免大人数公司出现人物分离。
                    # 后续 split_company_into_chunks 会优先把人员箱分摊到该公司已有物品/装备的装车单元中。
                    group_dict[('person', idx)].append(idx)
                elif public_type == 'Small':
                    if cid in companies_need_hard_person_nonperson_balance:
                        # 对同时存在人员和物资/装备的公司，Small箱先保持细粒度，
                        # 避免提前合并成大块后无法给每个装车单元搭配人员。
                        group_dict[('small_balance', idx)].append(idx)
                    else:
                        # 物资不再按category分组；同一公司、同一装载车辆类型的小车可作为合并装车单元。
                        group_dict[(cid, public_type, getattr(box, 'zzsbid', ''))].append(idx)
                else:
                    group_dict[(cid, public_type, getattr(box, 'zzsbid', ''))].append(idx)

        merged_boxes = []
        merge_map = []

        for key, indices in group_dict.items():
            if key[0] in ('mixed', 'large', 'person', 'small_balance'):
                for idx in indices:
                    merged_boxes.append(original_boxes[idx])
                    merge_map.append([idx])
                continue

            current_group = []
            current_group_indices = []
            for idx in indices:
                current_box = original_boxes[idx]
                if current_group and can_merge(current_group + [current_box]):
                    current_group.append(current_box)
                    current_group_indices.append(idx)
                else:
                    if current_group:
                        merged_box = create_merged_box(current_group)
                        merged_boxes.append(merged_box)
                        merge_map.append(list(current_group_indices))
                    current_group = [current_box]
                    current_group_indices = [idx]
            if current_group:
                merged_box = create_merged_box(current_group)
                merged_boxes.append(merged_box)
                merge_map.append(list(current_group_indices))

        print(f"合并后箱子总数: {len(merged_boxes)} (原始: {len(original_boxes)})")
        all_sub_containers = merged_boxes

        indices_by_company = defaultdict(list)
        mixed_indices = []
        for idx, box in enumerate(all_sub_containers):
            if len(box.owners) == 1:
                cid = next(iter(box.owners))
                indices_by_company[cid].append(idx)
            else:
                mixed_indices.append(idx)

        units = []

        def make_unit(box_indices, forced_owners=None):
            owners = set(forced_owners or [])
            total_w = 0.0
            total_l = 0.0
            chao_owners = set()
            for i in box_indices:
                b = all_sub_containers[i]
                owners.update(b.owners)
                total_w += b.weight
                total_l += b.length_unit
                chao_owners.update(box_chaoxian_owners(b))
            return {
                'box_indices': list(box_indices),
                'owners': owners,
                'weight': total_w,
                'length': total_l,
                'dominant': dominant_ratio(total_w, total_l, max_weight_per_sc, max_length_per_sc),
                'has_chaoXian_equipment': len(chao_owners) > 0,
                'chaoXian_owners': chao_owners
            }

        def split_company_into_chunks(cid, box_indices):
            """
            将同一公司拆成若干装车单元。

            新增均衡规则：
            - 若公司同时存在人员箱和物资/装备箱，不再先装完物资、最后再装人员；
            - 先按总重量/总换长估算该公司至少需要的车辆数；
            - 在这些目标车辆块之间分别均衡分摊“非人员箱”和“人员箱”；
            - 硬性保证：只要该公司同时存在人员箱和物资/装备箱，则该公司形成的每个装车单元都必须同时含有人和物资/装备；
            - 不允许出现该公司的纯人员装车单元或纯物资/装备装车单元；
            - 若受单箱尺寸、超重、超换长等约束影响无法做到人-物同车，则直接报错，不输出违反规则的方案。
            """
            def can_add_to_chunk(chunk, box_idx):
                b = all_sub_containers[box_idx]
                return (chunk['weight'] + b.weight <= max_weight_per_sc + 1e-6 and
                        chunk['length'] + b.length_unit <= max_length_per_sc + 1e-6)

            def add_box_to_chunk(chunk, box_idx):
                b = all_sub_containers[box_idx]
                chunk['box_indices'].append(box_idx)
                chunk['weight'] += b.weight
                chunk['length'] += b.length_unit
                chunk['dominant'] = dominant_ratio(chunk['weight'], chunk['length'], max_weight_per_sc,
                                                   max_length_per_sc)
                chunk['chaoXian_owners'].update(box_chaoxian_owners(b))
                chunk['has_chaoXian_equipment'] = len(chunk['chaoXian_owners']) > 0

            def chunk_person_nonperson_load(chunk):
                person_w = person_l = non_person_w = non_person_l = 0.0
                person_count = non_person_count = 0
                for bi in chunk.get('box_indices', []):
                    bb = all_sub_containers[bi]
                    if get_public_box_type(bb.box_type) == 'Person':
                        person_w += bb.weight
                        person_l += bb.length_unit
                        person_count += 1
                    else:
                        non_person_w += bb.weight
                        non_person_l += bb.length_unit
                        non_person_count += 1
                return {
                    'person_w': person_w,
                    'person_l': person_l,
                    'non_person_w': non_person_w,
                    'non_person_l': non_person_l,
                    'person_count': person_count,
                    'non_person_count': non_person_count,
                }

            def best_chunk_for_box(chunks, box_idx, prefer_chao_chunk=False, spread_person=False):
                b = all_sub_containers[box_idx]
                best_k = None
                best_score = None
                for k, chunk in enumerate(chunks):
                    if not can_add_to_chunk(chunk, box_idx):
                        continue
                    new_w = chunk['weight'] + b.weight
                    new_l = chunk['length'] + b.length_unit
                    fill_w = new_w / max_weight_per_sc
                    fill_l = new_l / max_length_per_sc
                    score = 0.7 * max(fill_w, fill_l) + 0.3 * min(fill_w, fill_l) - 0.08 * abs(fill_w - fill_l)
                    if prefer_chao_chunk and chunk.get('has_chaoXian_equipment'):
                        score += 0.45
                    if spread_person:
                        load = chunk_person_nonperson_load(chunk)
                        # 人员过多或人员较少时都优先把人员分摊到已经有物资/装备、且人员负载较小的块里。
                        if load['non_person_count'] > 0:
                            score += 0.80
                        score -= 0.80 * (load['person_l'] / max_length_per_sc if max_length_per_sc else 0.0)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_k = k
                return best_k

            def pack_indices(seed_chunks, indices, prefer_chao_chunk=False):
                chunks = seed_chunks
                ordered = sorted(
                    indices,
                    key=lambda i: (
                        dominant_ratio(all_sub_containers[i].weight, all_sub_containers[i].length_unit,
                                       max_weight_per_sc, max_length_per_sc),
                        all_sub_containers[i].length_unit,
                        all_sub_containers[i].weight
                    ),
                    reverse=True
                )
                for idx in ordered:
                    best_k = best_chunk_for_box(chunks, idx, prefer_chao_chunk=prefer_chao_chunk)
                    if best_k is None:
                        chunks.append(make_unit([idx], forced_owners={cid}))
                    else:
                        add_box_to_chunk(chunks[best_k], idx)
                return chunks

            def make_empty_chunk():
                return make_unit([], forced_owners={cid})

            def lower_bound_vehicle_count(indices):
                total_w = sum(all_sub_containers[i].weight for i in indices)
                total_l = sum(all_sub_containers[i].length_unit for i in indices)
                by_weight = int(math.ceil(total_w / max_weight_per_sc)) if max_weight_per_sc > 0 else 1
                by_length = int(math.ceil(total_l / max_length_per_sc)) if max_length_per_sc > 0 else 1
                return max(1, by_weight, by_length)

            def rebalance_single_type_chunks(chunks):
                """尝试修复只含人员或只含物资/装备的块。

                现在“不能出现纯人员块/纯物资装备块”是硬规则：
                - 能通过搬移箱子修复，则返回修复后的块；
                - 不能修复，则后续 assert_no_single_type_chunks 会报错，禁止输出违规方案。
                """
                changed = True
                while changed:
                    changed = False
                    person_only = []
                    non_person_only = []
                    for k, chunk in enumerate(chunks):
                        load = chunk_person_nonperson_load(chunk)
                        if load['person_count'] > 0 and load['non_person_count'] == 0:
                            person_only.append(k)
                        elif load['non_person_count'] > 0 and load['person_count'] == 0:
                            non_person_only.append(k)

                    if not person_only or not non_person_only:
                        break

                    # 先尝试从人员块中移动一个人员箱到物资块。
                    moved = False
                    for pk in list(person_only):
                        if pk >= len(chunks):
                            continue
                        p_boxes = [bi for bi in list(chunks[pk]['box_indices'])
                                   if get_public_box_type(all_sub_containers[bi].box_type) == 'Person']
                        p_boxes.sort(key=lambda bi: (all_sub_containers[bi].length_unit, all_sub_containers[bi].weight))
                        for bi in p_boxes:
                            targets = sorted(non_person_only, key=lambda kk: chunk_person_nonperson_load(chunks[kk])['person_l'])
                            for nk in targets:
                                if nk >= len(chunks) or not can_add_to_chunk(chunks[nk], bi):
                                    continue
                                chunks[pk]['box_indices'].remove(bi)
                                chunks[pk] = make_unit(chunks[pk]['box_indices'], forced_owners={cid})
                                add_box_to_chunk(chunks[nk], bi)
                                changed = moved = True
                                break
                            if moved:
                                break
                        if moved:
                            break

                    if moved:
                        chunks = [c for c in chunks if c.get('box_indices')]
                        continue

                    # 如果人员箱搬不过去，则尝试把一个较小的物资/装备箱搬到人员块。
                    for nk in list(non_person_only):
                        if nk >= len(chunks):
                            continue
                        np_boxes = [bi for bi in list(chunks[nk]['box_indices'])
                                    if get_public_box_type(all_sub_containers[bi].box_type) != 'Person']
                        np_boxes.sort(key=lambda bi: (all_sub_containers[bi].length_unit, all_sub_containers[bi].weight))
                        for bi in np_boxes:
                            targets = sorted(person_only, key=lambda kk: chunk_person_nonperson_load(chunks[kk])['non_person_l'])
                            for pk in targets:
                                if pk >= len(chunks) or not can_add_to_chunk(chunks[pk], bi):
                                    continue
                                chunks[nk]['box_indices'].remove(bi)
                                chunks[nk] = make_unit(chunks[nk]['box_indices'], forced_owners={cid})
                                add_box_to_chunk(chunks[pk], bi)
                                changed = moved = True
                                break
                            if moved:
                                break
                        if moved:
                            break

                    if moved:
                        chunks = [c for c in chunks if c.get('box_indices')]
                return [c for c in chunks if c.get('box_indices')]

            def assert_no_single_type_chunks(chunks):
                """硬校验：同一公司同时有人和物资/装备时，不允许产生纯人员或纯物资/装备装车单元。"""
                violations = []
                for k, chunk in enumerate(chunks):
                    load = chunk_person_nonperson_load(chunk)
                    has_person = load['person_count'] > 0
                    has_non_person = load['non_person_count'] > 0
                    if has_person and not has_non_person:
                        violations.append((k + 1, '纯人员'))
                    elif has_non_person and not has_person:
                        violations.append((k + 1, '纯物资/装备'))
                if violations:
                    detail = '；'.join([f'第{idx}个装车单元为{kind}单元' for idx, kind in violations[:10]])
                    if len(violations) > 10:
                        detail += f'；另有{len(violations) - 10}个违规单元'
                    raise AlgorithmError(
                        f'公司 {company_name.get(cid, cid)}({cid}) 同时存在人员和物资/装备，'
                        f'但无法在不超重、不超换长的前提下实现每个装车单元人-物同车：{detail}'
                    )

            def pack_company_balanced(person_indices, non_person_indices):
                """公司内人员-物资均衡打包：适用于同时有人员和物资/装备的公司。

                这里把“不能产生纯人员/纯物资装备装车单元”作为硬规则：
                - 每个候选装车单元先至少放入一个非人员箱和一个人员箱；
                - 后续剩余箱子只允许继续加入已有混合单元，不再新建纯类型单元；
                - 如果现有箱子粒度与容量约束导致无法满足，则直接报错。
                """
                all_indices = list(person_indices) + list(non_person_indices)
                lower_count = lower_bound_vehicle_count(all_indices)
                max_mixed_count = min(len(person_indices), len(non_person_indices))
                if lower_count > max_mixed_count:
                    raise AlgorithmError(
                        f'公司 {company_name.get(cid, cid)}({cid}) 同时存在人员和物资/装备，'
                        f'但至少需要{lower_count}个装车单元，而当前人员箱数={len(person_indices)}、'
                        f'物资/装备箱数={len(non_person_indices)}，无法保证每个装车单元都人-物同车'
                    )

                chao_indices = [i for i in non_person_indices if box_has_chaoxian_equipment(all_sub_containers[i])]
                normal_non_person_indices = [i for i in non_person_indices if i not in set(chao_indices)]
                base_ordered_non_person = sorted(
                    chao_indices + normal_non_person_indices,
                    key=lambda i: (
                        box_has_chaoxian_equipment(all_sub_containers[i]),
                        dominant_ratio(all_sub_containers[i].weight, all_sub_containers[i].length_unit,
                                       max_weight_per_sc, max_length_per_sc),
                        all_sub_containers[i].length_unit,
                        all_sub_containers[i].weight,
                    ),
                    reverse=True,
                )
                base_ordered_person = sorted(
                    person_indices,
                    key=lambda i: (all_sub_containers[i].length_unit, all_sub_containers[i].weight),
                    reverse=True,
                )

                last_error = None

                def build_with_target_count(target_count):
                    """
                    构造指定数量的人-物混合装车单元。

                    本轮修正重点：
                    - 先把非人员箱分摊好，再按非人员负载比例分配人员箱；
                    - 避免某个单元只有一个物资/装备箱，却被分到大量人员；
                    - 仍保持硬规则：最终每个单元必须同时有人和物资/装备。
                    """
                    chunks = [make_empty_chunk() for _ in range(target_count)]

                    remaining_non_person = list(base_ordered_non_person)
                    remaining_person = list(base_ordered_person)

                    total_person_l = sum(all_sub_containers[i].length_unit for i in person_indices)
                    total_person_w = sum(all_sub_containers[i].weight for i in person_indices)
                    total_non_person_l = sum(all_sub_containers[i].length_unit for i in non_person_indices)
                    total_non_person_w = sum(all_sub_containers[i].weight for i in non_person_indices)
                    avg_non_person_l = total_non_person_l / target_count if target_count else total_non_person_l

                    def non_person_share_for_chunk(chunk):
                        """按非人员箱的换长/重量综合估计该块应分到的人员比例。"""
                        load = chunk_person_nonperson_load(chunk)
                        if total_non_person_l <= 1e-9 and total_non_person_w <= 1e-9:
                            return 1.0 / target_count if target_count else 1.0
                        shares = []
                        if total_non_person_l > 1e-9:
                            shares.append(load['non_person_l'] / total_non_person_l)
                        if total_non_person_w > 1e-9:
                            shares.append(load['non_person_w'] / total_non_person_w)
                        if not shares:
                            return 1.0 / target_count if target_count else 1.0
                        # 换长更能反映“占用列车空间”，权重略高；重量作为辅助。
                        if len(shares) == 2:
                            return 0.65 * shares[0] + 0.35 * shares[1]
                        return shares[0]

                    def target_person_l_for_chunk(chunk):
                        return total_person_l * non_person_share_for_chunk(chunk)

                    def target_person_w_for_chunk(chunk):
                        return total_person_w * non_person_share_for_chunk(chunk)

                    def can_fit_any_person(chunk, person_list):
                        for pi in person_list:
                            if can_add_to_chunk(chunk, pi):
                                return True
                        return False

                    def can_add_non_person_and_still_fit_person(chunk, np_idx, person_list):
                        """非人员箱加入后，仍要至少能塞进一个人员箱，否则会制造纯物资单元。"""
                        if not can_add_to_chunk(chunk, np_idx):
                            return False
                        tmp_indices = list(chunk.get('box_indices', [])) + [np_idx]
                        tmp_chunk = make_unit(tmp_indices, forced_owners={cid})
                        return can_fit_any_person(tmp_chunk, person_list)

                    # 第一步：每个目标块先放至少一个物资/装备箱，杜绝纯人员块。
                    # 此时就要预留至少一个人员箱的容量，避免后续出现“物资块已经满了、人员进不去”。
                    for k in range(target_count):
                        if not remaining_non_person:
                            return None, '物资/装备箱数量不足，无法给每个装车单元配置物资/装备'

                        best_pos = None
                        best_score = None
                        for pos, idx in enumerate(remaining_non_person):
                            if not can_add_non_person_and_still_fit_person(chunks[k], idx, remaining_person):
                                continue
                            b = all_sub_containers[idx]
                            # 种子阶段优先放较大的非人员箱，使大箱先获得人员容量保障。
                            score = dominant_ratio(b.weight, b.length_unit, max_weight_per_sc, max_length_per_sc)
                            score += 0.05 * b.length_unit + 0.000001 * b.weight
                            if box_has_chaoxian_equipment(b):
                                score += 0.20
                            if best_score is None or score > best_score:
                                best_score = score
                                best_pos = pos
                        if best_pos is None:
                            return None, f'第{k + 1}个装车单元无法找到既能放入又能预留人员容量的物资/装备箱'
                        idx = remaining_non_person.pop(best_pos)
                        add_box_to_chunk(chunks[k], idx)

                    # 第二步：先分摊剩余物资/装备箱。
                    # 与上一版不同，这一步放在配人员之前，避免人员先占满某个小物资块的容量，导致后续物资进不去。
                    for idx in remaining_non_person:
                        best_k = None
                        best_score = None
                        b = all_sub_containers[idx]
                        for k, chunk in enumerate(chunks):
                            if not can_add_non_person_and_still_fit_person(chunk, idx, remaining_person):
                                continue
                            load = chunk_person_nonperson_load(chunk)
                            new_non_l = load['non_person_l'] + b.length_unit
                            new_non_w = load['non_person_w'] + b.weight
                            new_total_l = chunk['length'] + b.length_unit
                            new_total_w = chunk['weight'] + b.weight
                            fill_l = new_total_l / max_length_per_sc if max_length_per_sc else 0.0
                            fill_w = new_total_w / max_weight_per_sc if max_weight_per_sc else 0.0
                            l_gap = abs(new_non_l - avg_non_person_l) / max(avg_non_person_l, 1e-6)
                            w_share_gap = 0.0
                            if total_non_person_w > 1e-9:
                                w_share_gap = abs((new_non_w / total_non_person_w) - (1.0 / target_count))
                            # 优先让非人员负载均匀；不让某个块只保留一个很小物资箱。
                            score = -1.45 * l_gap - 0.55 * w_share_gap - 0.18 * max(fill_l, fill_w) - 0.08 * abs(fill_l - fill_w)
                            if box_has_chaoxian_equipment(b) and chunk.get('has_chaoXian_equipment'):
                                score += 0.20
                            if best_score is None or score > best_score:
                                best_score = score
                                best_k = k
                        if best_k is None:
                            return None, f'剩余物资/装备箱 BoxIndex={idx} 无法加入任何可预留人员容量的装车单元'
                        add_box_to_chunk(chunks[best_k], idx)

                    # 第三步：每个目标块放至少一个人员箱，杜绝纯物资/装备块。
                    # 人员目标不再按“每车平均人数”，而是按该块非人员负载比例分配。
                    chunk_order = sorted(
                        range(target_count),
                        key=lambda kk: target_person_l_for_chunk(chunks[kk]),
                        reverse=True,
                    )
                    for k in chunk_order:
                        best_pos = None
                        best_score = None
                        target_l = target_person_l_for_chunk(chunks[k])
                        target_w = target_person_w_for_chunk(chunks[k])
                        for pos, idx in enumerate(remaining_person):
                            if not can_add_to_chunk(chunks[k], idx):
                                continue
                            b = all_sub_containers[idx]
                            load = chunk_person_nonperson_load(chunks[k])
                            new_person_l = load['person_l'] + b.length_unit
                            new_person_w = load['person_w'] + b.weight
                            new_total_l = chunks[k]['length'] + b.length_unit
                            new_total_w = chunks[k]['weight'] + b.weight
                            fill_l = new_total_l / max_length_per_sc if max_length_per_sc else 0.0
                            fill_w = new_total_w / max_weight_per_sc if max_weight_per_sc else 0.0
                            l_gap = abs(new_person_l - target_l) / max(target_l, 1e-6)
                            w_gap = abs(new_person_w - target_w) / max(target_w, 1e-6) if target_w > 1e-9 else 0.0
                            over_l = max(0.0, new_person_l - target_l) / max(target_l, 1e-6)
                            # 小物资块的目标人员少，优先给它更小的人员箱，避免“一个小物资箱配一堆人”。
                            score = -1.55 * l_gap - 0.45 * w_gap - 0.85 * over_l - 0.16 * max(fill_l, fill_w) - 0.08 * abs(fill_l - fill_w)
                            if best_score is None or score > best_score:
                                best_score = score
                                best_pos = pos
                        if best_pos is None:
                            return None, f'第{k + 1}个装车单元已有物资/装备，但没有任何人员箱能在不超重、不超换长的前提下放入'
                        idx = remaining_person.pop(best_pos)
                        add_box_to_chunk(chunks[k], idx)

                    # 第四步：剩余人员继续按“非人员负载比例”分摊到已有混合块，不能新建纯人员块。
                    for idx in remaining_person:
                        best_k = None
                        best_score = None
                        b = all_sub_containers[idx]
                        for k, chunk in enumerate(chunks):
                            if not can_add_to_chunk(chunk, idx):
                                continue
                            load = chunk_person_nonperson_load(chunk)
                            target_l = target_person_l_for_chunk(chunk)
                            target_w = target_person_w_for_chunk(chunk)
                            new_person_l = load['person_l'] + b.length_unit
                            new_person_w = load['person_w'] + b.weight
                            new_total_l = chunk['length'] + b.length_unit
                            new_total_w = chunk['weight'] + b.weight
                            fill_l = new_total_l / max_length_per_sc if max_length_per_sc else 0.0
                            fill_w = new_total_w / max_weight_per_sc if max_weight_per_sc else 0.0
                            l_gap_after = abs(new_person_l - target_l) / max(target_l, 1e-6)
                            l_gap_before = abs(load['person_l'] - target_l) / max(target_l, 1e-6)
                            w_gap_after = abs(new_person_w - target_w) / max(target_w, 1e-6) if target_w > 1e-9 else 0.0
                            over_l = max(0.0, new_person_l - target_l) / max(target_l, 1e-6)
                            # 只把人员继续放到“按物资比例还缺人”的块里；如果已经超目标，强惩罚。
                            improvement = l_gap_before - l_gap_after
                            score = 1.20 * improvement - 1.35 * l_gap_after - 0.45 * w_gap_after - 1.10 * over_l - 0.16 * max(fill_l, fill_w) - 0.08 * abs(fill_l - fill_w)
                            if best_score is None or score > best_score:
                                best_score = score
                                best_k = k
                        if best_k is None:
                            return None, f'剩余人员箱 BoxIndex={idx} 无法加入任何已有人-物混合装车单元'
                        add_box_to_chunk(chunks[best_k], idx)

                    chunks = [c for c in chunks if c.get('box_indices')]
                    chunks = rebalance_single_type_chunks(chunks)
                    try:
                        assert_no_single_type_chunks(chunks)
                    except AlgorithmError as exc:
                        return None, str(exc)
                    return chunks, ''

                # 从理论下界开始尝试；如果为了满足人-物同车需要增加车辆数，则逐步增加。
                for target_count in range(lower_count, max_mixed_count + 1):
                    chunks, err = build_with_target_count(target_count)
                    if chunks is not None:
                        return chunks
                    last_error = err

                raise AlgorithmError(
                    f'公司 {company_name.get(cid, cid)}({cid}) 同时存在人员和物资/装备，'
                    f'但无法在不超重、不超换长的前提下实现每个装车单元人-物同车；'
                    f'最后一次失败原因：{last_error or "未知"}'
                )

            person_indices = [i for i in box_indices if get_public_box_type(all_sub_containers[i].box_type) == 'Person']
            non_person_indices = [i for i in box_indices if i not in set(person_indices)]

            # 只要同一公司同时存在人员和物资/装备，就启用均衡打包，而不是只在人员接近一整车时才启用。
            if person_indices and non_person_indices:
                return pack_company_balanced(person_indices, non_person_indices)

            # 只有单一类型时，保留原有压缩装车逻辑。
            chao_indices = [i for i in non_person_indices if box_has_chaoxian_equipment(all_sub_containers[i])]
            normal_non_person_indices = [i for i in non_person_indices if i not in set(chao_indices)]
            chunks = []
            chunks = pack_indices(chunks, chao_indices, prefer_chao_chunk=True)
            chunks = pack_indices(chunks, normal_non_person_indices + person_indices, prefer_chao_chunk=False)
            return chunks

        for cid, box_indices in indices_by_company.items():
            total_w = sum(all_sub_containers[i].weight for i in box_indices)
            total_l = sum(all_sub_containers[i].length_unit for i in box_indices)
            if total_w <= max_weight_per_sc + 1e-6 and total_l <= max_length_per_sc + 1e-6:
                units.append(make_unit(box_indices, forced_owners={cid}))
            else:
                units.extend(split_company_into_chunks(cid, box_indices))

        for idx in mixed_indices:
            units.append(make_unit([idx]))

        print(f"装车单元数: {len(units)} (优先按公司整体成组；超限公司自动拆分)")

        def vehicle_score_after_place(vehicle, unit):
            fill_w = (vehicle.weight + unit['weight']) / max_weight_per_sc
            fill_l = (vehicle.length + unit['length']) / max_length_per_sc
            return 0.75 * max(fill_w, fill_l) + 0.25 * min(fill_w, fill_l) - 0.10 * abs(fill_w - fill_l)

        def find_best_vehicle(vehicles, unit, exclude_index=None):
            best_v = None
            best_score = None
            for v, vehicle in enumerate(vehicles):
                if exclude_index is not None and v == exclude_index:
                    continue
                if not vehicle.can_place(unit, max_weight_per_sc, max_length_per_sc, company_yingji_name):
                    continue
                score = vehicle_score_after_place(vehicle, unit)
                if unit['owners'] & vehicle.companies:
                    score += 0.15
                if unit.get('has_chaoXian_equipment'):
                    if vehicle.chaoXian_companies:
                        score += 0.45
                    if unit.get('chaoXian_owners', set()) & vehicle.chaoXian_companies:
                        score += 0.10
                if best_score is None or score > best_score:
                    best_score = score
                    best_v = v
            return best_v

        units.sort(key=lambda u: (u.get('has_chaoXian_equipment', False), u['dominant'], u['length'], u['weight']),
                   reverse=True)
        vehicles = []

        for unit in units:
            v = find_best_vehicle(vehicles, unit)
            if v is None:
                new_vehicle = VehicleState()
                if not new_vehicle.can_place(unit, max_weight_per_sc, max_length_per_sc, company_yingji_name):
                    owners = ','.join(sorted(unit['owners']))
                    raise AlgorithmError(
                        f"装车单元无法单独放入车辆，owners={owners}, weight={unit['weight']:.1f}, length={unit['length']:.2f}"
                    )
                new_vehicle.place(unit, company_yingji_name)
                vehicles.append(new_vehicle)
            else:
                vehicles[v].place(unit, company_yingji_name)

        def compact_vehicles(vehicles):
            changed = True
            while changed:
                changed = False
                order = sorted(
                    range(len(vehicles)),
                    key=lambda i: (dominant_ratio(vehicles[i].weight, vehicles[i].length,
                                                  max_weight_per_sc, max_length_per_sc), len(vehicles[i].units))
                )
                for source_idx in order:
                    if source_idx >= len(vehicles):
                        continue
                    source_units = sorted(
                        list(vehicles[source_idx].units),
                        key=lambda u: (u['dominant'], u['length'], u['weight']),
                        reverse=True
                    )
                    snapshot = [v.clone() for v in vehicles]
                    success = True
                    for unit in source_units:
                        vehicles[source_idx].remove(unit, company_yingji_name)
                        target_idx = find_best_vehicle(vehicles, unit, exclude_index=source_idx)
                        if target_idx is None:
                            success = False
                            break
                        vehicles[target_idx].place(unit, company_yingji_name)

                    if success and not vehicles[source_idx].units:
                        del vehicles[source_idx]
                        changed = True
                        break
                    else:
                        vehicles = snapshot
            return vehicles

        vehicles = compact_vehicles(vehicles)
        total_sc_used = len(vehicles)
        print(f"启发式装车完成，使用 SC 总数: {total_sc_used}")

        heuristic_assign = [-1] * len(all_sub_containers)
        for v, vehicle in enumerate(vehicles):
            for unit in vehicle.units:
                for idx in unit['box_indices']:
                    if heuristic_assign[idx] != -1:
                        raise AlgorithmError(f"合并箱 {idx} 被重复分配")
                    heuristic_assign[idx] = v

        def validate_assignment():
            if any(v == -1 for v in heuristic_assign):
                missing = [i for i, v in enumerate(heuristic_assign) if v == -1]
                raise AlgorithmError(f"存在未分配合并箱: {missing[:10]}")

            for v, vehicle in enumerate(vehicles):
                if vehicle.weight > max_weight_per_sc + 1e-6:
                    raise AlgorithmError(f"SC_{v + 1:03d} 超重: {vehicle.weight:.1f} > {max_weight_per_sc:.1f}")
                if vehicle.length > max_length_per_sc + 1e-6:
                    raise AlgorithmError(f"SC_{v + 1:03d} 超换长: {vehicle.length:.2f} > {max_length_per_sc:.2f}")
                used_yingji_names = sorted(
                    y for y, cids in vehicle.yingji_companies.items()
                    if is_effective_yingji_name(y) and len(cids) > 0
                )
                if len(used_yingji_names) > 2:
                    raise AlgorithmError(f"SC_{v + 1:03d} yingjiName种类数超限: used_yingjiNames={used_yingji_names}")

            # 硬规则终检：若某公司在全局同时存在人员和物资/装备，则该公司出现的任何SC都不能只有人员或只有物资/装备。
            # 这一步防止后续车辆压缩或跨公司混放导致局部出现“人车/物资车”分离。
            company_type_presence = defaultdict(lambda: {'person': False, 'non_person': False})
            for box in all_sub_containers:
                public_type = get_public_box_type(getattr(box, 'box_type', ''))
                for cid0 in getattr(box, 'owners', set()):
                    if public_type == 'Person':
                        company_type_presence[cid0]['person'] = True
                    else:
                        company_type_presence[cid0]['non_person'] = True
            companies_need_mixed = {
                cid0 for cid0, flags in company_type_presence.items()
                if flags.get('person') and flags.get('non_person')
            }

            for v, vehicle in enumerate(vehicles):
                per_company_vehicle_presence = defaultdict(lambda: {'person': False, 'non_person': False})
                for unit in vehicle.units:
                    for idx0 in unit['box_indices']:
                        box = all_sub_containers[idx0]
                        public_type = get_public_box_type(getattr(box, 'box_type', ''))
                        for cid0 in getattr(box, 'owners', set()):
                            if public_type == 'Person':
                                per_company_vehicle_presence[cid0]['person'] = True
                            else:
                                per_company_vehicle_presence[cid0]['non_person'] = True
                for cid0, flags in per_company_vehicle_presence.items():
                    if cid0 not in companies_need_mixed:
                        continue
                    if flags.get('person') != flags.get('non_person'):
                        only_type = '仅人员' if flags.get('person') else '仅物资/装备'
                        raise AlgorithmError(
                            f"SC_{v + 1:03d} 违反人-物同车硬规则：公司 {company_name.get(cid0, cid0)}({cid0}) "
                            f"在该车中为{only_type}，但该公司全局同时存在人员和物资/装备"
                        )

            spread = defaultdict(set)
            for idx, assign in enumerate(heuristic_assign):
                for cid in all_sub_containers[idx].owners:
                    spread[cid].add(assign)
            split_companies = {cid: sorted(vs) for cid, vs in spread.items() if len(vs) > 1}
            if split_companies:
                print("提示：以下公司因容量/装载组合原因被分到多辆车（软约束，已尽量压缩）：")
                for cid, vs in list(split_companies.items())[:20]:
                    print(f"  {cid}: {len(vs)} 辆 -> {[f'SC_{v + 1:03d}' for v in vs]}")
                if len(split_companies) > 20:
                    print(f"  ... 共 {len(split_companies)} 个公司")

        validate_assignment()

        if vehicles:
            avg_w = sum(v.weight / max_weight_per_sc for v in vehicles) / len(vehicles)
            avg_l = sum(v.length / max_length_per_sc for v in vehicles) / len(vehicles)
            print(f"平均重量利用率: {avg_w:.2%}，平均换长利用率: {avg_l:.2%}")

        def repack_large_and_small_boxes_within_sc(sc_boxes):
            """
            在SC车辆组合已经确定后，对同一SC内的Large装备箱和Small物资箱分别二次重装。
            - Person固定不动；
            - Large装备：同一zzsbid内可跨公司尾数拼箱，不检查sbrl/sbzz，只检查zzsbidNumber尾数规则和yingjiName≤2；
            - Small物资：保持原有同一zzsbid内跨公司尾数拼箱、zjdh矩阵、体积/载重、yingjiName≤2规则。
            """
            fixed_boxes = []
            component_items = []
            goods_items = []
            for orig_idx, box in sc_boxes:
                public_type = get_public_box_type(box.box_type)
                if public_type == 'Large':
                    for item in getattr(box, 'contents', []):
                        if item.get('type') == 'component':
                            component_items.append(dict(item))
                elif public_type == 'Small':
                    for item in getattr(box, 'contents', []):
                        if item.get('type') == 'goods':
                            goods_items.append(dict(item))
                else:
                    fixed_boxes.append((orig_idx, box))

            virtual_boxes = []

            # === Large装备二次重装：同一SC内同zzsbid允许跨公司尾数拼箱 ===
            component_items = prepare_component_items_for_tailmix(component_items)
            open_large_repacked = defaultdict(list)
            for item in component_items:
                cid = item.get('company_id', '')
                name = item.get('componentname', '')
                comp_id = item.get('componentID', '')
                spec = choose_loading_spec_by_id(specs_by_id, item.get('zzsbid', ''), f'装备 {name or comp_id}')
                zzsbid = spec.get('id', '')
                zhuang_zai = spec.get('sbmc', '')
                w = safe_float(item.get('componentweight'), 0.0)
                vol = component_item_volume(item)
                item_limit = safe_int(item.get('zzsbidNumber', 1), 1)
                if item_limit <= 0:
                    item_limit = 1
                occupancy = component_item_fraction(item)
                c_key = item.get('_component_item_key') or component_item_key(item)
                item['_component_item_key'] = c_key
                item['_component_item_limit'] = item_limit
                item['_component_item_fraction'] = occupancy
                item['occupancy'] = occupancy
                item['_component_tail_candidate'] = bool(item.get('_component_tail_candidate', False))
                item['componentvolume'] = vol
                item['tj'] = vol
                item['zhuangZai'] = item.get('zhuangZai', zhuang_zai) or zhuang_zai

                best_box = None
                best_score = None
                for box in open_large_repacked[zzsbid]:
                    if box.capacity_type != 'component_pack':
                        continue
                    if getattr(box, 'goods_closed', False):
                        continue
                    if not can_pack_component_item(box, item, company_yingji_name, zjdh_forbid_matrix):
                        continue
                    if box.goods_item_counts[c_key] + 1 > item_limit:
                        continue
                    # Large不按sbrl/sbzz判断；只按同zzsbid、尾数拼箱、zzsbidNumber、占用比例和yingjiName规则。
                    new_load_ratio = (box.current_load + occupancy) / box.max_capacity if box.max_capacity else 1.0
                    count_ratio = (box.goods_item_counts[c_key] + 1) / item_limit if item_limit else 1.0
                    owner_bonus = 0.08 if cid in box.owners else 0.0
                    chao_bonus = 0.10 if normalize_is_chaoxian(item.get('is_chaoXian', '')) == '是' else 0.0
                    tail_bonus = 0.08 if item.get('_component_tail_candidate') else 0.0
                    score = 0.80 * new_load_ratio + 0.12 * count_ratio + owner_bonus + chao_bonus + tail_bonus
                    if best_score is None or score > best_score:
                        best_score = score
                        best_box = box

                if best_box is None:
                    new_box = SubContainer(
                        'Large',
                        spec['sbhc'],
                        spec['sbzl'],
                        1.0,
                        'component_pack',
                        zzsbid=zzsbid,
                        zhuang_zai=zhuang_zai,
                    )
                    # Large不使用sbrl/sbzz作为拼箱限制；这里设为极大值，仅保留SC总重/总换长校验。
                    new_box.max_payload = 999999999.0
                    new_box.max_volume = 999999999.0
                    if not new_box.add_item(cid, item, w, occupancy, item_volume=vol):
                        raise AlgorithmError(
                            f'装备 {name or comp_id} 在SC内Large重装失败：zzsbid={zzsbid}, 请检查zzsbidNumber或尾数拼箱规则'
                        )
                    open_large_repacked[zzsbid].append(new_box)
                    virtual_boxes.append((None, new_box))
                else:
                    if not best_box.add_item(cid, item, w, occupancy, item_volume=vol):
                        raise AlgorithmError(f'装备 {name or comp_id} 在SC内Large混装失败')

            # === Small物资二次重装：保持原规则 ===
            goods_items = prepare_goods_items_for_tailmix(goods_items)
            open_repacked = defaultdict(list)

            for item in goods_items:
                cid = item.get('company_id', '')
                name = item.get('name', '')
                gid = item.get('ID', '')
                spec = choose_loading_spec_by_id(specs_by_id, item.get('zzsbid', ''), f'物资 {name or gid}')
                zzsbid = spec.get('id', '')
                zhuang_zai = spec.get('sbmc', '')
                w = safe_float(item.get('weight'), 0.0)
                tj = safe_float(item.get('tj'), 0.0)
                item_limit = safe_int(item.get('zzsbidNumber', 1), 1)
                if item_limit <= 0:
                    item_limit = 1
                g_key = item.get('_goods_item_key') or goods_item_key(item)
                item['_goods_item_key'] = g_key
                item['_goods_item_limit'] = item_limit
                item['_goods_tail_candidate'] = bool(item.get('_goods_tail_candidate', False))
                item['zhuangZai'] = item.get('zhuangZai', zhuang_zai) or zhuang_zai

                best_box = None
                best_score = None
                for box in open_repacked[zzsbid]:
                    if box.capacity_type != 'goods_pack':
                        continue
                    if getattr(box, 'goods_closed', False):
                        continue
                    if not can_pack_goods_item(box, item, company_yingji_name, zjdh_forbid_matrix):
                        continue
                    if box.goods_item_counts[g_key] + 1 > item_limit:
                        continue
                    if box.current_volume + tj > box.max_volume + 1e-6:
                        continue
                    if box.current_payload + w > box.max_payload + 1e-6:
                        continue

                    new_volume = box.current_volume + tj
                    new_payload = box.current_payload + w
                    vol_ratio = new_volume / box.max_volume if box.max_volume else 0.0
                    wt_ratio = new_payload / box.max_payload if box.max_payload else 0.0
                    owner_bonus = 0.08 if cid in box.owners else 0.0
                    # 已经同SC后，允许跨公司混装；优先选装后更满的小车。
                    score = 0.65 * max(vol_ratio, wt_ratio) + 0.25 * min(vol_ratio, wt_ratio) + owner_bonus
                    if best_score is None or score > best_score:
                        best_score = score
                        best_box = box

                if best_box is None:
                    new_box = SubContainer(
                        'Small',
                        spec['sbhc'],
                        spec['sbzl'],
                        1.0,
                        'goods_pack',
                        category=None,
                        zzsbid=zzsbid,
                        zhuang_zai=zhuang_zai,
                    )
                    new_box.max_payload = spec.get('sbzz') if spec.get('sbzz', 0) > 0 else 999999999.0
                    new_box.max_volume = spec.get('sbrl') if spec.get('sbrl', 0) > 0 else 999999999.0
                    if not new_box.add_item(cid, item, w, 0.0, item_volume=tj):
                        raise AlgorithmError(
                            f'物资 {name or gid} 在SC内重装失败：weight={w:.1f}, tj={tj:.2f}, zzsbid={zzsbid}'
                        )
                    open_repacked[zzsbid].append(new_box)
                    virtual_boxes.append((None, new_box))
                else:
                    if not best_box.add_item(cid, item, w, 0.0, item_volume=tj):
                        raise AlgorithmError(f'物资 {name or gid} 在SC内混装失败')

            return fixed_boxes + virtual_boxes

        res_data = {
            "code": 0,
            "msg": "success",
            "data": {
                "total_SC_used": total_sc_used,
                "SC_list": []
            }
        }

        for v in range(total_sc_used):
            sc_info = {
                "SC_ID": f"SC_{v + 1:03d}",
                "summary": {},
                "box_list": []
            }

            owners_set = set()
            curr_w = 0.0
            curr_l = 0.0
            has_mixed = False

            merged_indices = [i for i, assign in enumerate(heuristic_assign) if assign == v]
            sc_source_boxes = []
            for i in merged_indices:
                for orig_idx in merge_map[i]:
                    sc_source_boxes.append((orig_idx, original_boxes[orig_idx]))

            # 只有在同一SC内部，才允许不同公司的Large装备/Small物资按各自规则二次重装；Person保持原箱。
            output_boxes = repack_large_and_small_boxes_within_sc(sc_source_boxes)

            virtual_counter = 1
            for orig_idx, orig_box in output_boxes:
                owners_set.update(orig_box.owners)
                curr_w += orig_box.weight
                curr_l += orig_box.length_unit
                if orig_box.is_mixed:
                    has_mixed = True

                entity_desc = build_entities(orig_box, company_yingji_name)

                box_yingji_names = sorted({
                    company_yingji_name.get(cid, '') for cid in orig_box.owners
                    if is_effective_yingji_name(company_yingji_name.get(cid, ''))
                })
                if orig_idx is None:
                    box_id = f"Box_{v + 1:03d}_M{virtual_counter:03d}"
                    virtual_counter += 1
                else:
                    box_id = f"Box_{orig_idx + 1:04d}"
                box_dict = {
                    "box_id": box_id,
                    "box_type": get_public_box_type(orig_box.box_type),
                    "is_mixed": orig_box.is_mixed,
                    "owners": list(orig_box.owners),
                    "yingjiName": ';'.join(box_yingji_names),
                    "is_chaoXian": "是" if box_has_chaoxian_equipment(orig_box) else (
                        "否" if get_public_box_type(orig_box.box_type) == 'Large' else ""),
                    "content_desc": entity_desc,
                    "weight": round(orig_box.weight, 1),
                    "length_unit": round(orig_box.length_unit, 2)
                }

                if get_public_box_type(orig_box.box_type) == 'Small':
                    categories = sorted({
                        str(item.get('category', '')).strip()
                        for item in getattr(orig_box, 'contents', [])
                        if str(item.get('category', '')).strip()
                    })
                    box_dict["category"] = ';'.join(categories) if categories else '未分类'

                sc_info["box_list"].append(box_dict)

            yingji_names_in_sc = sorted({
                company_yingji_name.get(cid, '') for cid in owners_set
                if is_effective_yingji_name(company_yingji_name.get(cid, ''))
            })
            yingji_company_distribution = {}
            for y in yingji_names_in_sc:
                yingji_company_distribution[y] = len(
                    [cid for cid in owners_set if company_yingji_name.get(cid, '') == y])

            chao_companies_in_sc = sorted({
                                              entity.get('company_id', '')
                                              for box in sc_info['box_list']
                                              for entity in box.get('content_desc', [])
                                              if
                                              entity.get('type') == 'component' and entity.get('is_chaoXian', '') == '是'
                                          } - {''})

            sc_info["summary"] = {
                "companies_included": list(owners_set),
                "yingjiName_list": yingji_names_in_sc,
                "yingjiName_count": len(yingji_names_in_sc),
                "yingjiName_company_distribution": yingji_company_distribution,
                "has_chaoXian_equipment": len(chao_companies_in_sc) > 0,
                "chaoXian_companies": chao_companies_in_sc,
                "total_weight": round(curr_w, 1),
                "total_length_unit": round(curr_l, 2),
                "has_mixed_box": has_mixed,
                "description": f"包含 {len(owners_set)} 个公司: {','.join(list(owners_set)[:3])}... 共 {len(sc_info['box_list'])} 个小箱"
            }

            res_data["data"]["SC_list"].append(sc_info)

        validate_output_result(res_data, company_yingji_name, max_weight_per_sc, max_length_per_sc, zjdh_forbid_matrix)
        return res_data
    except AlgorithmError as exc:
        logger.warning("算法输入错误: %s", exc)
        return {"code": 1, "msg": str(exc)}
    except Exception:
        logger.exception("算法执行失败")
        return {"code": 1, "msg": "算法执行失败，请查看日志"}


def validate_output_result(res_data, company_yingji_name, max_weight_per_sc, max_length_per_sc, zjdh_forbid_matrix=None):
    for sc in res_data.get('data', {}).get('SC_list', []):
        sid = sc.get('SC_ID', '')
        summary = sc.get('summary', {})
        box_list = sc.get('box_list', []) or []

        # 最终出参以 box_list 重新汇总校验，不能只相信 summary。
        total_w = sum(safe_float(box.get('weight'), 0.0) for box in box_list)
        total_l = sum(safe_float(box.get('length_unit'), 0.0) for box in box_list)
        summary_w = safe_float(summary.get('total_weight', total_w), total_w)
        summary_l = safe_float(summary.get('total_length_unit', total_l), total_l)

        if total_w > max_weight_per_sc + 1e-6:
            raise AlgorithmError(f"{sid} 出参校验超重: {total_w:.1f} > {max_weight_per_sc:.1f}")
        if total_l > max_length_per_sc + 1e-6:
            raise AlgorithmError(f"{sid} 出参校验超换长: {total_l:.2f} > {max_length_per_sc:.2f}")
        if summary_w > max_weight_per_sc + 1e-6:
            raise AlgorithmError(f"{sid} summary超重: {summary_w:.1f} > {max_weight_per_sc:.1f}")
        if summary_l > max_length_per_sc + 1e-6:
            raise AlgorithmError(f"{sid} summary超换长: {summary_l:.2f} > {max_length_per_sc:.2f}")
        if abs(summary_w - total_w) > 0.2 or abs(summary_l - total_l) > 0.02:
            raise AlgorithmError(
                f"{sid} summary与box_list汇总不一致: "
                f"summary_weight={summary_w:.1f}, box_weight={total_w:.1f}, "
                f"summary_length={summary_l:.2f}, box_length={total_l:.2f}"
            )

        used_yingji_names = set()
        owners_from_boxes = set()
        for box in box_list:
            if safe_float(box.get('weight'), 0.0) > max_weight_per_sc + 1e-6:
                raise AlgorithmError(f"{sid} 存在单箱超重: box_id={box.get('box_id')}")
            if safe_float(box.get('length_unit'), 0.0) > max_length_per_sc + 1e-6:
                raise AlgorithmError(f"{sid} 存在单箱超换长: box_id={box.get('box_id')}")
            for cid in box.get('owners', []) or []:
                owners_from_boxes.add(cid)

            # Small箱内再次校验 zjdh 混装规则，确保不会输出矩阵禁止的混装组合。
            goods_entities = [
                e for e in (box.get('content_desc', []) or [])
                if isinstance(e, dict) and e.get('type') == 'goods'
            ]
            if len(goods_entities) > 1:
                for entity in goods_entities:
                    item_limit = safe_int(entity.get('zzsbidNumber', 1), 1)
                    entity_count = safe_int(entity.get('count', 0), 0)
                    # 一个Small箱若混装了不同类物资，则每一类都必须是“尾数不足一箱”的部分；
                    # 如果某类已经达到自身 zzsbidNumber，还和其他类混装，说明违反“装满即闭箱”。
                    if item_limit > 0 and entity_count >= item_limit:
                        raise AlgorithmError(
                            f"{sid} 尾数拼箱校验失败: box_id={box.get('box_id')}, "
                            f"{entity.get('name', '')} 已达到 zzsbidNumber={item_limit}，不应再与其他物资混装"
                        )

            for i in range(len(goods_entities)):
                for j in range(i + 1, len(goods_entities)):
                    if not can_mix_zjdh(goods_entities[i], goods_entities[j], zjdh_forbid_matrix):
                        raise AlgorithmError(
                            f"{sid} 物资混装规则校验失败: box_id={box.get('box_id')}, "
                            f"{goods_entities[i].get('name', '')}/{goods_entities[i].get('zjdh', '')} 与 "
                            f"{goods_entities[j].get('name', '')}/{goods_entities[j].get('zjdh', '')} 不允许混装"
                        )

            # Large箱内再次校验同zzsbid、componentname+zzsbid尾数拼箱规则。
            # Large不检查sbrl/sbzz，也不走zjdh矩阵。
            component_entities = [
                e for e in (box.get('content_desc', []) or [])
                if isinstance(e, dict) and e.get('type') == 'component'
            ]
            if len(component_entities) > 1:
                zzsbids = {str(e.get('zzsbid', '')).strip() for e in component_entities if str(e.get('zzsbid', '')).strip()}
                if len(zzsbids) > 1:
                    raise AlgorithmError(f"{sid} Large装备混装规则校验失败: box_id={box.get('box_id')} 存在不同zzsbid={sorted(zzsbids)}")

                comp_class_counts = defaultdict(lambda: {'count': 0, 'limit': 1, 'name': '', 'zzsbid': ''})
                for entity in component_entities:
                    c_key = (str(entity.get('componentname', '')).strip(), str(entity.get('zzsbid', '')).strip())
                    cnt = safe_int(entity.get('count', 1), 1)
                    lim = safe_int(entity.get('zzsbidNumber', 1), 1)
                    if lim <= 0:
                        lim = 1
                    comp_class_counts[c_key]['count'] += cnt
                    comp_class_counts[c_key]['limit'] = lim
                    comp_class_counts[c_key]['name'] = c_key[0]
                    comp_class_counts[c_key]['zzsbid'] = c_key[1]

                component_occupancy_sum = 0.0
                for info in comp_class_counts.values():
                    lim = info['limit'] if info['limit'] > 0 else 1
                    if info['count'] > lim:
                        raise AlgorithmError(
                            f"{sid} Large装备数量校验失败: box_id={box.get('box_id')}, "
                            f"{info['name']} 数量={info['count']} 超过 zzsbidNumber={lim}"
                        )
                    component_occupancy_sum += info['count'] / lim

                if component_occupancy_sum > 1.0 + 1e-6:
                    raise AlgorithmError(
                        f"{sid} Large装备占用比例校验失败: box_id={box.get('box_id')}, "
                        f"sum(count/zzsbidNumber)={component_occupancy_sum:.3f} > 1.000"
                    )

                if len(comp_class_counts) > 1:
                    for info in comp_class_counts.values():
                        if info['count'] >= info['limit']:
                            raise AlgorithmError(
                                f"{sid} Large尾数拼箱校验失败: box_id={box.get('box_id')}, "
                                f"{info['name']} 已达到 zzsbidNumber={info['limit']}，不应再与其他装备混装"
                            )


        for cid in owners_from_boxes or set(summary.get('companies_included', []) or []):
            yingji_name = company_yingji_name.get(cid, '')
            if is_effective_yingji_name(yingji_name):
                used_yingji_names.add(yingji_name)
        if len(used_yingji_names) > 2:
            raise AlgorithmError(f"{sid} 出参校验yingjiName种类数超限: used_yingjiNames={sorted(used_yingji_names)}")


def build_entities(box, company_yingji_name=None):
    if not box.contents:
        return []

    entities = []
    public_type = get_public_box_type(box.box_type)
    if public_type == 'Person':
        merged = {}
        for item in box.contents:
            key = (item['company_id'], item.get('zhuangZai', getattr(box, 'zhuang_zai', '')),
                   item.get('zzsbid', getattr(box, 'zzsbid', '')))
            if key not in merged:
                merged[key] = {
                    "type": "person",
                    "company_id": item['company_id'],
                    "yingjiName": (company_yingji_name or {}).get(item['company_id'], ''),
                    "box_type": "Person",
                    "count": 0,
                    "zhuangZai": item.get('zhuangZai', getattr(box, 'zhuang_zai', '')),
                }
            merged[key]['count'] += item['count']
        entities = list(merged.values())

    elif public_type == 'Large':
        merged = {}
        for item in box.contents:
            key = (
                item.get('company_id', ''),
                item.get('componentname', ''),
                item.get('componentID', ''),
                normalize_is_chaoxian(item.get('is_chaoXian', '')),
                item.get('bddxid', ''),
                item.get('dxcode', ''),
                item.get('zzsbid', getattr(box, 'zzsbid', '')),
                item.get('zjdh', ''),
                item.get('zhuangZai', getattr(box, 'zhuang_zai', '')),
            )
            if key not in merged:
                merged[key] = {
                    "type": "component",
                    "company_id": item['company_id'],
                    "yingjiName": (company_yingji_name or {}).get(item['company_id'], ''),
                    "componentname": item.get('componentname', ''),
                    "componentID": item.get('componentID', ''),
                    "componentweight": item.get('componentweight', 0),
                    "componentvolume": item.get('componentvolume', item.get('tj', 0)),
                    "tj": item.get('tj', item.get('componentvolume', 0)),
                    "is_chaoXian": normalize_is_chaoxian(item.get('is_chaoXian', '')),
                    "bddxid": item.get('bddxid', ''),
                    "dxcode": item.get('dxcode', ''),
                    "zzsbid": item.get('zzsbid', getattr(box, 'zzsbid', '')),
                    "zzsbidNumber": item.get('zzsbidNumber', ''),
                    "occupancy": item.get('occupancy', item.get('_component_item_fraction', component_item_fraction(item))),
                    "zjdh": item.get('zjdh', ''),
                    "zhuangZai": item.get('zhuangZai', getattr(box, 'zhuang_zai', '')),
                    "tail_candidate": bool(item.get('_component_tail_candidate', False)),
                    "group_count": item.get('_component_group_count', ''),
                    "tail_count": item.get('_component_tail_count', ''),
                    "count": 0,
                }
            merged[key]['count'] += item.get('count', 1)
        entities = list(merged.values())

    elif public_type == 'Small':
        merged = {}
        for item in box.contents:
            key = (
                item['company_id'],
                item.get('name', ''),
                item.get('ID', ''),
                item.get('category', ''),
                item.get('zzsbid', getattr(box, 'zzsbid', '')),
                item.get('zjdh', ''),
                item.get('zhuangZai', getattr(box, 'zhuang_zai', '')),
            )
            if key not in merged:
                merged[key] = {
                    "type": "goods",
                    "company_id": item['company_id'],
                    "yingjiName": (company_yingji_name or {}).get(item['company_id'], ''),
                    "name": item.get('name', ''),
                    "ID": item.get('ID', ''),
                    "bddxid": item.get('bddxid', ''),
                    "category": item.get('category', ''),
                    "dxcode": item.get('dxcode', ''),
                    "tj": item.get('tj', 0),
                    "weight": item.get('weight', 0),
                    "zzsbid": item.get('zzsbid', getattr(box, 'zzsbid', '')),
                    "zzsbidNumber": item.get('zzsbidNumber', ''),
                    "zjdh": item.get('zjdh', ''),
                    "zhuangZai": item.get('zhuangZai', getattr(box, 'zhuang_zai', '')),
                    "tail_candidate": bool(item.get('_goods_tail_candidate', False)),
                    "group_count": item.get('_goods_group_count', ''),
                    "tail_count": item.get('_goods_tail_count', ''),
                    "count": 0,
                }
            merged[key]['count'] += item.get('count', 1)
        entities = list(merged.values())

    return entities


atexit.register(ALGO_EXECUTOR.shutdown, wait=False, cancel_futures=False)

def build_cors_origins() -> List[str]:
    raw = os.getenv("RAILWAY_CORS_ALLOW_ORIGINS", "*").strip()
    if raw == "*" or not raw:
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=build_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(HEALTH_PATH)
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "app": APP_TITLE,
        "host": API_HOST,
        "port": API_PORT,
    }


@app.post("/api/v1/optimize")
async def optimize(req: OptimizationRequest, request: Request):
    request_id = uuid4().hex[:8]
    client_host = request.client.host if request.client else "unknown"
    logger.info("[%s] 收到计算请求，来源=%s", request_id, client_host)

    try:
        payload = model_to_payload(req)
    except Exception as exc:
        logger.warning("[%s] 请求格式转换失败: %s", request_id, exc)
        raise HTTPException(status_code=400, detail=f"请求格式错误: {exc}")

    if not ALGO_GATE.acquire(blocking=False):
        logger.warning("[%s] worker 全部繁忙，拒绝请求", request_id)
        raise HTTPException(status_code=503, detail="服务繁忙，当前已有计算任务正在执行，请稍后重试")

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(ALGO_EXECUTOR, partial(run_engine, payload))

    gate_released = False

    def _release_gate(_):
        nonlocal gate_released
        if not gate_released:
            gate_released = True
            ALGO_GATE.release()
            logger.info("[%s] worker 名额已释放", request_id)

    future.add_done_callback(_release_gate)

    try:
        result = await asyncio.wait_for(asyncio.shield(future), timeout=REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error("[%s] 接口层超时兜底触发，启发式任务仍可能继续执行", request_id)
        raise HTTPException(status_code=504, detail="计算超时，请检查输入数据规模或联系技术人员")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[%s] 接口调用失败", request_id)
        raise HTTPException(status_code=500, detail=f"接口处理失败: {exc}")

    if result.get("code") != 0:
        logger.warning("[%s] 算法返回失败: %s", request_id, result.get("msg"))
        raise HTTPException(status_code=400, detail=result.get("msg", "算法执行失败"))

    logger.info("[%s] 请求完成，返回结果", request_id)
    return result


# ========================== 网络/启动辅助 ==========================
def can_bind_port(host: str, port: int) -> bool:
    test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        test_sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        test_sock.close()


def detect_local_ipv4_addresses() -> List[str]:
    candidates: List[str] = []

    probe_targets = [("10.255.255.255", 1), ("192.168.255.255", 1), ("172.16.255.255", 1)]
    for target, port in probe_targets:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((target, port))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127."):
                candidates.append(ip)
        except Exception:
            pass
        finally:
            sock.close()

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                candidates.append(ip)
    except Exception:
        pass

    deduped: List[str] = []
    seen = set()
    for ip in candidates:
        if ip not in seen:
            seen.add(ip)
            deduped.append(ip)

    return deduped if deduped else ["127.0.0.1"]


def log_startup_banner() -> None:
    ip_list = detect_local_ipv4_addresses()
    logger.info("程序启动，目录=%s", APP_DIR)
    logger.info("日志文件=%s", LOG_PATH)
    logger.info("服务监听=%s:%s", API_HOST, API_PORT)
    for ip in ip_list:
        logger.info("接口地址=http://%s:%s/api/v1/optimize", ip, API_PORT)


def bootstrap() -> None:
    configure_process_signals()

    if not can_bind_port(API_HOST, API_PORT):
        logger.error("端口 %s 已被占用，程序退出", API_PORT)
        raise SystemExit(1)

    log_startup_banner()
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_config=None,
        access_log=False,
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    bootstrap()
