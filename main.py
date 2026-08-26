import asyncio
import atexit
import http.client
import logging
import math
from logging.handlers import RotatingFileHandler
import os
import re
from pathlib import Path
import socket
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import Any, Dict, List, Optional
from functools import partial

import uvicorn
from fastapi import FastAPI, HTTPException, Request
import uuid
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

# ========================== 基础配置 ==========================
APP_TITLE = "铁路运输配载服务"
API_HOST = "0.0.0.0"
API_PORT = 2376
INSTANCE_LOCK_PORT = 23761
HEALTH_PATH = "/health"
LOG_FILE_NAME = "railway_service.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
SERVER_START_TIMEOUT = 20.0
SERVER_STOP_TIMEOUT = 10.0
STATUS_POLL_MS = 1000
SERVER_MONITOR_INTERVAL = 5.0
MAX_ALGO_WORKERS = 4
REQUEST_TIMEOUT_SECONDS = None  # 兼容保留：接口已取消固定超时，不参与运行
BUILD_VERSION = "2026-08-25-linux-v11-seat-demand-balance"

# 车辆换长均衡参数：只作为启发式目标，不替代任何硬约束。
# 目标：尽可能少用车，并让已使用车辆的换长尽量贴近最大换长，避免出现明显低换长尾车。
BALANCE_MIN_LENGTH_RATIO = 0.88
BALANCE_TARGET_LENGTH_RATIO = 0.96
BALANCE_MAX_GAP_RATIO = 0.10
BALANCE_MAX_ITERATIONS = 1400

# 公司分散度软惩罚：避免为了补换长把同一公司拆散到过多车辆。
# 该项只参与候选方案评分，不改变人-物同车、超重、超换长、yingjiName等硬约束。
COMPANY_SPREAD_WEIGHT = 0.35

# 公司人员分布均衡参数：只作为软目标。
# 若某公司被分到多辆车，尽量让各车上的该公司人数接近；
# 但不为追求人数均衡破坏人-物同车、超重、超换长、yingjiName等硬约束。
PERSON_BALANCE_MAX_RATIO = 0.25
PERSON_BALANCE_MAX_ABS_GAP = 6
PERSON_BALANCE_WEIGHT = 0.35


def get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_DIR = get_app_dir()
LOG_PATH = APP_DIR / LOG_FILE_NAME


def ensure_log_dir() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


ensure_log_dir()

logger = logging.getLogger("railway_service")
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

    def isatty(self):
        return False


sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)


def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        return
    logger.exception("未捕获异常", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_uncaught_exception

if hasattr(threading, "excepthook"):
    def _threading_excepthook(args):
        logger.exception("线程未捕获异常", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


    threading.excepthook = _threading_excepthook


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
        # 等价性能缓存：箱内类别/尾数/zjdh摘要，避免每个候选都重新遍历全部contents。
        self._goods_key_all_tail = {}
        self._component_key_all_tail = {}
        self._goods_zjdh_indices = set()
        self._pack_cache_size = 0

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
                old_tail = self._component_key_all_tail.get(item_key, True)
                self._component_key_all_tail[item_key] = old_tail and bool(item_info.get('_component_tail_candidate', False))
                self._pack_cache_size = len(self.contents)
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
                old_tail = self._goods_key_all_tail.get(item_key, True)
                self._goods_key_all_tail[item_key] = old_tail and bool(item_info.get('_goods_tail_candidate', False))
                zjdh_idx = _cached_zjdh_index(item_info)
                if zjdh_idx is not None:
                    self._goods_zjdh_indices.add(zjdh_idx)
                self._pack_cache_size = len(self.contents)
                self.goods_item_counts[item_key] += 1
                self.goods_item_limits[item_key] = item_limit
                if self.goods_item_counts[item_key] >= item_limit:
                    # 恢复原始闭箱逻辑：某一类物资达到该箱上限后，本箱视为已满，其他类型不能再进入。
                    self.goods_closed = True
                return True
            return False
        # 人员的装箱逻辑：按件数或载物比例
        else:
            # 人员箱严禁跨公司混装。公司唯一性始终以 organizationID/company_id 判断，
            # 不能用 organizationName 或 yingjiName 代替。
            if get_public_box_type(self.box_type) == 'Person' and self.owners and company_id not in self.owners:
                return False
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


class UnitDict(dict):
    """内部装车单元/块：保持dict全部行为，同时兼容历史分支的属性式读取。"""
    __slots__ = ()

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


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


def _cached_zjdh_index(item):
    """缓存算法内部物资的zjdh映射；出参校验字典不写入任何内部字段。"""
    raw = item.get('zjdh')
    # 只有装箱阶段的内部item带有_goods_item_key；最终content_desc不得被校验过程改写。
    if '_goods_item_key' not in item:
        return normalize_zjdh_value(raw)
    marker = item.get('_zjdh_cache_raw', object())
    if marker == raw and '_zjdh_cache_index' in item:
        return item.get('_zjdh_cache_index')
    idx = normalize_zjdh_value(raw)
    item['_zjdh_cache_raw'] = raw
    item['_zjdh_cache_index'] = idx
    return idx


def _ensure_box_pack_cache(box):
    """兼容contents被合并代码直接扩展的情况：仅在摘要失效时重建一次。"""
    contents = getattr(box, 'contents', [])
    if getattr(box, '_pack_cache_size', -1) == len(contents):
        return
    goods_key_all_tail = {}
    component_key_all_tail = {}
    goods_zjdh_indices = set()
    for item in contents:
        item_type = item.get('type')
        if item_type == 'goods':
            key = item.get('_goods_item_key') or goods_item_key(item)
            goods_key_all_tail[key] = goods_key_all_tail.get(key, True) and bool(item.get('_goods_tail_candidate', False))
            idx = _cached_zjdh_index(item)
            if idx is not None:
                goods_zjdh_indices.add(idx)
        elif item_type == 'component':
            key = item.get('_component_item_key') or component_item_key(item)
            component_key_all_tail[key] = component_key_all_tail.get(key, True) and bool(item.get('_component_tail_candidate', False))
    box._goods_key_all_tail = goods_key_all_tail
    box._component_key_all_tail = component_key_all_tail
    box._goods_zjdh_indices = goods_zjdh_indices
    box._pack_cache_size = len(contents)


def can_mix_zjdh(existing_item, new_item, zjdh_forbid_matrix):
    """
    按 zjdh 禁配矩阵判断两件物资是否允许混装。

    当前业务口径：
    - 只有两件物资都提供了有效 zjdh，且都能映射到内置的“1组1级”等 27 行范围时，才查 0/1 矩阵；
    - 只要任意一件物资未提供 zjdh 字段、zjdh 为空，或 zjdh 值不在映射范围内，就不触发禁配矩阵，直接视为 zjdh 层面允许混装；
    - zjdh 层面允许后，仍继续执行 name+zzsbid、尾数拼箱、体积/载重、yingjiName 等其他规则。
    """
    old_idx = _cached_zjdh_index(existing_item)
    new_idx = _cached_zjdh_index(new_item)

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

    _ensure_box_pack_cache(box)
    new_key = new_item.get('_goods_item_key') or goods_item_key(new_item)
    new_is_tail = bool(new_item.get('_goods_tail_candidate', False))

    # 等价执行原逐件zjdh检查：未知/空值不进入集合；有效行号与新物资逐一按双向矩阵判断。
    new_idx = _cached_zjdh_index(new_item)
    if new_idx is not None:
        mat = zjdh_forbid_matrix if zjdh_forbid_matrix is not None else DEFAULT_ZJDH_FORBID_MATRIX
        for old_idx in box._goods_zjdh_indices:
            if mat[old_idx - 1][new_idx - 1] != 0 or mat[new_idx - 1][old_idx - 1] != 0:
                return False

    # 等价执行原尾数规则：只要存在不同类别，该类别的既有物资与新物资都必须是尾数。
    for old_key, old_all_tail in box._goods_key_all_tail.items():
        if old_key != new_key and not (old_all_tail and new_is_tail):
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

    _ensure_box_pack_cache(box)
    new_key = new_item.get('_component_item_key') or component_item_key(new_item)
    new_is_tail = bool(new_item.get('_component_tail_candidate', False))
    # Large装备不走zjdh矩阵；等价汇总原逐件 componentname+zzsbid / 尾数判断。
    for old_key, old_all_tail in box._component_key_all_tail.items():
        if old_key != new_key and not (old_all_tail and new_is_tail):
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
        # 等价性能缓存：只记录同一公司/超限公司当前出现在多少个装车单元中。
        # 不改变任何装车规则、评分或输出，仅避免 remove 时反复扫描整辆车的全部单元。
        self._company_unit_counts = defaultdict(int)
        self._chaoxian_unit_counts = defaultdict(int)
        # 等价性能缓存：车辆内各公司人员数量。只复用已有unit统计，不改变人员分配。
        self._person_counts = defaultdict(int)

    def clone(self):
        other = VehicleState()
        other.weight = self.weight
        other.length = self.length
        other.units = list(self.units)
        other.companies = set(self.companies)
        other.yingji_companies = defaultdict(set, {g: set(cids) for g, cids in self.yingji_companies.items()})
        other.chaoXian_companies = set(self.chaoXian_companies)
        other._company_unit_counts = defaultdict(int, self._company_unit_counts)
        other._chaoxian_unit_counts = defaultdict(int, self._chaoxian_unit_counts)
        other._person_counts = defaultdict(int, self._person_counts)
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
            self._company_unit_counts[cid] += 1
            if self._company_unit_counts[cid] == 1:
                self.companies.add(cid)
                yingji_name = company_yingji_name.get(cid, '')
                if is_effective_yingji_name(yingji_name):
                    self.yingji_companies[yingji_name].add(cid)
        if unit.get('has_chaoXian_equipment'):
            for cid in unit.get('chaoXian_owners', set()):
                self._chaoxian_unit_counts[cid] += 1
                if self._chaoxian_unit_counts[cid] == 1:
                    self.chaoXian_companies.add(cid)
        for cid, count in unit.get('_person_counts', {}).items():
            self._person_counts[cid] += count

    def remove(self, unit, company_yingji_name):
        self.weight -= unit['weight']
        self.length -= unit['length']
        self.units.remove(unit)
        for cid in unit['owners']:
            remaining = self._company_unit_counts.get(cid, 0) - 1
            if remaining > 0:
                self._company_unit_counts[cid] = remaining
                continue
            self._company_unit_counts.pop(cid, None)
            self.companies.discard(cid)
            yingji_name = company_yingji_name.get(cid, '')
            if is_effective_yingji_name(yingji_name):
                cids = self.yingji_companies.get(yingji_name)
                if cids is not None:
                    cids.discard(cid)
                    if not cids:
                        self.yingji_companies.pop(yingji_name, None)
        if unit.get('has_chaoXian_equipment'):
            for cid in unit.get('chaoXian_owners', set()):
                remaining = self._chaoxian_unit_counts.get(cid, 0) - 1
                if remaining > 0:
                    self._chaoxian_unit_counts[cid] = remaining
                    continue
                self._chaoxian_unit_counts.pop(cid, None)
                self.chaoXian_companies.discard(cid)
        for cid, count in unit.get('_person_counts', {}).items():
            remaining = self._person_counts.get(cid, 0) - count
            if remaining:
                self._person_counts[cid] = remaining
            else:
                self._person_counts.pop(cid, None)


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
                    # 已闭箱的Large后续必定会被原逻辑跳过，立即移出候选列表只减少无效扫描。
                    if getattr(best_box, 'goods_closed', False):
                        open_large_boxes[key].remove(best_box)
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
                    # 新箱若单件即满，原逻辑后续只会跳过；不放入开放候选列表。
                    if not getattr(new_box, 'goods_closed', False):
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
                    # 已闭箱的Small后续必定会被原逻辑跳过，立即移出候选列表只减少无效扫描。
                    if getattr(best_box, 'goods_closed', False):
                        open_small_boxes[key].remove(best_box)
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
                    # 新箱若单件即满，原逻辑后续只会跳过；不放入开放候选列表。
                    if not getattr(new_box, 'goods_closed', False):
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

            为满足“最终SC有某公司物资/装备时必须有该公司人员”的硬规则，
            必要时允许把原本一个满载人员箱拆成多个同公司、同型号人员箱，并分摊人数。
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
            """按预计SC需求为同一公司重分配人员箱。

            保留既有人员车型规则：leiXing先决定软卧/硬卧/硬座人数，绝不跨公司混箱，
            也不在不同箱型之间转移人数。软卧、硬卧仍按原顺序依次装满，只有最终确实
            缺少同公司人员箱时才拆分；硬座人员可按预计SC需求在硬座箱之间均衡分配。

            这里仍只生成候选人员箱；最终硬规则在SC层校验：某SC若有某公司的
            物资/装备，该SC必须同时有该公司的人员。
            """
            nonlocal original_boxes

            def estimated_vehicle_count(indices):
                """用二维Best-Fit-Decreasing估计这些箱子实际需要的SC数。

                总重量/总换长的ceil只是理论下界，边界数据受单箱不可再拆影响时可能偏小；
                增加这一等价可行装箱估计，可提前准备足够的同公司人员箱。
                """
                bins = []
                ordered = sorted(
                    indices,
                    key=lambda i: dominant_ratio(
                        original_boxes[i].weight,
                        original_boxes[i].length_unit,
                        max_weight_per_sc,
                        max_length_per_sc,
                    ),
                    reverse=True,
                )
                for idx0 in ordered:
                    box0 = original_boxes[idx0]
                    best_pos = None
                    best_score = None
                    for pos, (used_w, used_l) in enumerate(bins):
                        new_w = used_w + box0.weight
                        new_l = used_l + box0.length_unit
                        if new_w > max_weight_per_sc + 1e-6 or new_l > max_length_per_sc + 1e-6:
                            continue
                        # 放入后越满越优先，减少理论下界低估造成的人员箱数量不足。
                        score = dominant_ratio(new_w, new_l, max_weight_per_sc, max_length_per_sc)
                        if best_score is None or score > best_score:
                            best_score = score
                            best_pos = pos
                    if best_pos is None:
                        bins.append((box0.weight, box0.length_unit))
                    else:
                        used_w, used_l = bins[best_pos]
                        bins[best_pos] = (used_w + box0.weight, used_l + box0.length_unit)
                return max(1, len(bins))

            def estimated_material_vehicle_count(person_indices0, non_person_indices0):
                """估计承载物资/装备的SC数，并为每辆预留一个同公司人员箱。

                与单纯按总换长取ceil不同，这能识别“两个物资箱可以同车，但再放人员箱
                就超限”的边界情形，避免人员箱准备数量仍然偏少。
                """
                reserve_idx = min(
                    person_indices0,
                    key=lambda i: dominant_ratio(
                        original_boxes[i].weight,
                        original_boxes[i].length_unit,
                        max_weight_per_sc,
                        max_length_per_sc,
                    ),
                )
                reserve_box = original_boxes[reserve_idx]
                usable_weight = max_weight_per_sc - reserve_box.weight
                usable_length = max_length_per_sc - reserve_box.length_unit
                if usable_weight < -1e-6 or usable_length < -1e-6:
                    return len(non_person_indices0) + 1

                bins = []
                ordered = sorted(
                    non_person_indices0,
                    key=lambda i: dominant_ratio(
                        original_boxes[i].weight,
                        original_boxes[i].length_unit,
                        max(usable_weight, 1e-9),
                        max(usable_length, 1e-9),
                    ),
                    reverse=True,
                )
                for idx0 in ordered:
                    box0 = original_boxes[idx0]
                    if box0.weight > usable_weight + 1e-6 or box0.length_unit > usable_length + 1e-6:
                        # 该物资/装备箱连同当前最小占用人员箱也无法同车。
                        return len(non_person_indices0) + 1
                    best_pos = None
                    best_score = None
                    for pos, (used_w, used_l) in enumerate(bins):
                        new_w = used_w + box0.weight
                        new_l = used_l + box0.length_unit
                        if new_w > usable_weight + 1e-6 or new_l > usable_length + 1e-6:
                            continue
                        score = dominant_ratio(
                            new_w,
                            new_l,
                            max(usable_weight, 1e-9),
                            max(usable_length, 1e-9),
                        )
                        if best_score is None or score > best_score:
                            best_score = score
                            best_pos = pos
                    if best_pos is None:
                        bins.append((box0.weight, box0.length_unit))
                    else:
                        used_w, used_l = bins[best_pos]
                        bins[best_pos] = (used_w + box0.weight, used_l + box0.length_unit)
                return max(1, len(bins))

            def person_spec_key(box):
                count0 = person_count_in_box(box)
                empty_weight0 = max(0.0, float(box.weight) - count0 * float(person_weight))
                return (
                    str(box.box_type),
                    str(getattr(box, 'zzsbid', '')),
                    str(getattr(box, 'zhuang_zai', '')),
                    round(float(box.length_unit), 9),
                    round(empty_weight0, 6),
                    int(box.max_capacity),
                    str(box.capacity_type),
                )

            def balanced_counts(total_count, box_count):
                base, extra = divmod(int(total_count), int(box_count))
                return [base + (1 if i < extra else 0) for i in range(box_count)]

            def rebuild_company_person_boxes(cid0, person_idxs, target_count):
                """按业务优先级重建人员箱，同时保持各箱型总人数不变。

                - 硬座：允许按需求均分，并优先承担新增人员箱需求；
                - 硬卧/软卧：正常情况下保持原来的依次装满结果；只有人员箱数量不足时，
                  才从该箱型人数最多的箱开始二分，不主动参与均衡；
                - 任意情况下均不跨公司、跨人员箱型转移人员。
                """
                grouped = defaultdict(list)
                for idx0 in person_idxs:
                    grouped[person_spec_key(original_boxes[idx0])].append(idx0)

                group_infos = []
                for key0, idxs0 in grouped.items():
                    idxs0 = sorted(idxs0)
                    total0 = sum(person_count_in_box(original_boxes[i]) for i in idxs0)
                    group_infos.append({
                        'key': key0,
                        'indices': idxs0,
                        'template': original_boxes[idxs0[0]],
                        'zhuang_zai': str(getattr(original_boxes[idxs0[0]], 'zhuang_zai', '')).strip(),
                        'total': total0,
                        'boxes': len(idxs0),
                    })
                group_infos.sort(key=lambda info: info['indices'][0])

                allocated = [info['boxes'] for info in group_infos]
                while sum(allocated) < target_count:
                    # 新增人员箱优先使用硬座；硬座确实无法继续拆时，才依次拆硬卧、软卧。
                    candidates = []
                    for zhuang_zai0 in ('硬座', '硬卧', '软卧'):
                        candidates = [
                            pos for pos, info in enumerate(group_infos)
                            if info['zhuang_zai'] == zhuang_zai0 and info['total'] > allocated[pos]
                        ]
                        if candidates:
                            break
                    if not candidates:
                        candidates = [
                            pos for pos, info in enumerate(group_infos)
                            if info['total'] > allocated[pos]
                        ]
                    if not candidates:
                        raise AlgorithmError(
                            f'公司 {company_name.get(cid0, cid0)}({cid0}) 人员不足，'
                            f'无法生成{target_count}个非空且不跨公司的人员箱'
                        )
                    # 同一优先级内选择当前平均人数最多的箱型。
                    chosen = max(
                        candidates,
                        key=lambda pos: group_infos[pos]['total'] / max(1, allocated[pos]),
                    )
                    allocated[chosen] += 1

                rebuilt = []
                desired_signature = []
                current_signature = []
                for pos, info in enumerate(group_infos):
                    current = [person_count_in_box(original_boxes[i]) for i in info['indices']]
                    if info['zhuang_zai'] == '硬座':
                        # 只有硬座主动按最终需求均分。
                        desired = balanced_counts(info['total'], allocated[pos])
                    else:
                        # 软卧/硬卧保持依次装满；只有箱数不足时才拆人数最多的现有箱。
                        desired = list(current)
                        while len(desired) < allocated[pos]:
                            split_pos = max(range(len(desired)), key=lambda i: desired[i])
                            split_count = desired[split_pos]
                            if split_count < 2:
                                raise AlgorithmError(
                                    f'公司 {company_name.get(cid0, cid0)}({cid0}) 的{info["zhuang_zai"] or "人员"}箱'
                                    f'无法继续拆分为非空人员箱'
                                )
                            left = split_count // 2
                            right = split_count - left
                            desired[split_pos] = left
                            desired.append(right)
                    desired_signature.append((info['key'], tuple(desired)))
                    current_signature.append((info['key'], tuple(current)))
                    for count0 in desired:
                        rebuilt.append(clone_person_box_with_count(info['template'], count0))

                if desired_signature == current_signature:
                    return False

                remove_set = set(person_idxs)
                insert_at = min(person_idxs)
                new_original = []
                for idx0, box0 in enumerate(original_boxes):
                    if idx0 == insert_at:
                        new_original.extend(rebuilt)
                    if idx0 not in remove_set:
                        new_original.append(box0)
                original_boxes[:] = new_original
                return True

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
                    if non_person_idxs and not person_idxs:
                        raise AlgorithmError(
                            f'公司 {company_name.get(cid0, cid0)}({cid0}) 存在物资/装备但人员数为0，'
                            f'无法满足“含该公司物资/装备的SC必须有该公司人员”规则'
                        )
                    if not non_person_idxs:
                        continue

                    all_idxs = person_idxs + non_person_idxs
                    total_w = sum(original_boxes[i].weight for i in all_idxs)
                    total_l = sum(original_boxes[i].length_unit for i in all_idxs)
                    required_by_weight = int(math.ceil(total_w / max_weight_per_sc)) if max_weight_per_sc > 0 else 1
                    required_by_length = int(math.ceil(total_l / max_length_per_sc)) if max_length_per_sc > 0 else 1
                    required_by_packing = estimated_vehicle_count(all_idxs)
                    required_by_material = estimated_material_vehicle_count(person_idxs, non_person_idxs)
                    if required_by_material > len(non_person_idxs):
                        raise AlgorithmError(
                            f'公司 {company_name.get(cid0, cid0)}({cid0}) 存在单个物资/装备箱无法与任何'
                            f'该公司人员箱在不超重、不超换长条件下同车的情况'
                        )
                    required_units = max(
                        1,
                        required_by_weight,
                        required_by_length,
                        required_by_packing,
                        required_by_material,
                    )

                    total_people = sum(person_count_in_box(original_boxes[i]) for i in person_idxs)
                    if total_people < required_units:
                        raise AlgorithmError(
                            f'公司 {company_name.get(cid0, cid0)}({cid0}) 同时存在人员和物资/装备，'
                            f'预计至少分布到{required_units}辆SC，但人员总数只有{total_people}，'
                            f'无法保证每辆含该公司物资/装备的SC都至少有1名该公司人员'
                        )
                    target_person_boxes = max(len(person_idxs), required_units)
                    if rebuild_company_person_boxes(cid0, person_idxs, target_person_boxes):
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

        # 等价性能缓存：合并箱在后续车辆搜索阶段内容不再变化，箱型、人员数和超限归属只计算一次。
        for box in all_sub_containers:
            public_type = get_public_box_type(getattr(box, 'box_type', ''))
            person_counts = defaultdict(int)
            if public_type == 'Person':
                for item in getattr(box, 'contents', []):
                    if item.get('type') == 'person':
                        cid0 = item.get('company_id', '')
                        person_counts[cid0] += safe_int(item.get('count'), 0)
            box._cached_public_type = public_type
            box._cached_person_counts = dict(person_counts)
            box._cached_chao_owners = box_chaoxian_owners(box)

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
            person_owners = set()
            non_person_owners = set()
            person_counts = defaultdict(int)
            has_person_box = False
            has_non_person_box = False
            person_w = person_l = non_person_w = non_person_l = 0.0
            person_box_count = non_person_box_count = 0
            for i in box_indices:
                b = all_sub_containers[i]
                owners.update(b.owners)
                total_w += b.weight
                total_l += b.length_unit
                chao_owners.update(getattr(b, '_cached_chao_owners', set()))
                if getattr(b, '_cached_public_type', get_public_box_type(b.box_type)) == 'Person':
                    has_person_box = True
                    person_box_count += 1
                    person_w += b.weight
                    person_l += b.length_unit
                    person_owners.update(b.owners)
                    for cid0, count0 in getattr(b, '_cached_person_counts', {}).items():
                        person_counts[cid0] += count0
                else:
                    has_non_person_box = True
                    non_person_box_count += 1
                    non_person_w += b.weight
                    non_person_l += b.length_unit
                    non_person_owners.update(b.owners)
            return UnitDict({
                'box_indices': list(box_indices),
                'owners': owners,
                'weight': total_w,
                'length': total_l,
                'dominant': dominant_ratio(total_w, total_l, max_weight_per_sc, max_length_per_sc),
                'has_chaoXian_equipment': len(chao_owners) > 0,
                'chaoXian_owners': chao_owners,
                # 以下均为内部缓存，不参与任何业务规则或最终出参。
                '_person_owners': person_owners,
                '_non_person_owners': non_person_owners,
                '_person_counts': dict(person_counts),
                '_has_person_box': has_person_box,
                '_has_non_person_box': has_non_person_box,
                '_person_w': person_w,
                '_person_l': person_l,
                '_non_person_w': non_person_w,
                '_non_person_l': non_person_l,
                '_person_box_count': person_box_count,
                '_non_person_box_count': non_person_box_count,
            })

        _readonly_unit_cache = {}

        def make_unit_readonly_cached(box_indices, forced_owners=None):
            """只用于车辆搜索中的只读候选单元；业务字段与make_unit完全一致。"""
            indices_key = tuple(box_indices)
            owners_key = tuple(sorted(forced_owners)) if forced_owners else ()
            key = (indices_key, owners_key)
            unit = _readonly_unit_cache.get(key)
            if unit is None:
                unit = make_unit(list(indices_key), forced_owners=set(owners_key) if owners_key else None)
                _readonly_unit_cache[key] = unit
            return unit

        def split_company_into_chunks(cid, box_indices):
            """
            将同一公司拆成若干装车单元。

            新增均衡规则：
            - 若公司同时存在人员箱和物资/装备箱，不再先装完物资、最后再装人员；
            - 先按总重量/总换长估算该公司至少需要的车辆数；
            - 在这些目标车辆块之间分别均衡分摊“非人员箱”和“人员箱”；
            - 硬性保证：任何承载该公司物资/装备的最终SC都必须同时含该公司人员；
            - 允许纯人员单元/纯人员SC，但不允许纯物资/装备SC；
            - 若受单箱尺寸、超重、超换长等约束影响无法做到“有物必有人”，则直接报错。
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
                chunk['chaoXian_owners'].update(getattr(b, '_cached_chao_owners', set()))
                chunk['has_chaoXian_equipment'] = len(chunk['chaoXian_owners']) > 0
                # 同步维护 make_unit 中的内部性能缓存；业务字段、评分和候选顺序均不变。
                if getattr(b, '_cached_public_type', get_public_box_type(b.box_type)) == 'Person':
                    chunk['_has_person_box'] = True
                    chunk['_person_owners'].update(b.owners)
                    chunk['_person_w'] += b.weight
                    chunk['_person_l'] += b.length_unit
                    chunk['_person_box_count'] += 1
                    person_counts = chunk['_person_counts']
                    for cid0, count0 in getattr(b, '_cached_person_counts', {}).items():
                        person_counts[cid0] = person_counts.get(cid0, 0) + count0
                else:
                    chunk['_has_non_person_box'] = True
                    chunk['_non_person_owners'].update(b.owners)
                    chunk['_non_person_w'] += b.weight
                    chunk['_non_person_l'] += b.length_unit
                    chunk['_non_person_box_count'] += 1

            def chunk_person_nonperson_load(chunk):
                # make_unit/add_box_to_chunk已同步维护，避免边界换长下对箱明细反复全扫描。
                return {
                    'person_w': chunk.get('_person_w', 0.0),
                    'person_l': chunk.get('_person_l', 0.0),
                    'non_person_w': chunk.get('_non_person_w', 0.0),
                    'non_person_l': chunk.get('_non_person_l', 0.0),
                    'person_count': chunk.get('_person_box_count', 0),
                    'non_person_count': chunk.get('_non_person_box_count', 0),
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

                尝试修复单一类型块：
                - 能通过搬移箱子修复，则返回修复后的块；
                - 纯人员块允许保留；纯物资/装备块不能修复时由后续硬校验报错。
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
                """硬校验：只禁止纯物资/装备单元；纯人员单元允许存在。

                业务规则是单向蕴含：某SC有该公司物资/装备 => 同SC有该公司人员。
                人员单独成单元或最终形成纯人员SC不违反该规则。
                """
                violations = []
                for k, chunk in enumerate(chunks):
                    load = chunk_person_nonperson_load(chunk)
                    has_person = load['person_count'] > 0
                    has_non_person = load['non_person_count'] > 0
                    if has_non_person and not has_person:
                        violations.append((k + 1, '纯物资/装备'))
                if violations:
                    detail = '；'.join([f'第{idx}个装车单元为{kind}单元' for idx, kind in violations[:10]])
                    if len(violations) > 10:
                        detail += f'；另有{len(violations) - 10}个违规单元'
                    raise AlgorithmError(
                        f'公司 {company_name.get(cid, cid)}({cid}) 同时存在人员和物资/装备，'
                        f'但无法在不超重、不超换长的前提下保证含物资/装备的单元都有该公司人员：{detail}'
                    )

            def pack_company_balanced(person_indices, non_person_indices):
                """公司内人员-物资均衡打包：适用于同时有人员和物资/装备的公司。

                快速路径优先构造人-物混合单元；如果剩余人员不能全部放入混合单元，
                后续精确兜底允许形成纯人员单元，但始终禁止纯物资/装备单元。
                """
                all_indices = list(person_indices) + list(non_person_indices)
                lower_count = lower_bound_vehicle_count(all_indices)
                # 每个最终块至少需要一个本公司人员箱；允许其中部分块为纯人员块。
                # 因此最大可用块数由人员箱数决定，而不是由物资/装备箱数决定。
                max_mixed_count = len(person_indices)
                if lower_count > max_mixed_count:
                    raise AlgorithmError(
                        f'公司 {company_name.get(cid, cid)}({cid}) 同时存在人员和物资/装备，'
                        f'预计至少需要{lower_count}辆SC，但当前只有{len(person_indices)}个同公司人员箱，'
                        f'无法保证每辆含该公司物资/装备的SC都有该公司人员'
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
                # 下列总量与target_count/排序策略无关，只计算一次。
                total_person_l = sum(all_sub_containers[i].length_unit for i in person_indices)
                total_person_w = sum(all_sub_containers[i].weight for i in person_indices)
                total_non_person_l = sum(all_sub_containers[i].length_unit for i in non_person_indices)
                total_non_person_w = sum(all_sub_containers[i].weight for i in non_person_indices)

                def build_with_target_count(target_count, non_person_order=None, person_order=None):
                    """
                    构造指定数量的人-物混合装车单元。

                    本轮修正重点：
                    - 先把非人员箱分摊好，再按非人员负载比例分配人员箱；
                    - 避免某个单元只有一个物资/装备箱，却被分到大量人员；
                    - 快速路径仍优先让每个目标块同时有人和物；精确兜底可保留纯人员块。
                    """
                    chunks = [make_empty_chunk() for _ in range(target_count)]

                    remaining_non_person = list(non_person_order if non_person_order is not None else base_ordered_non_person)
                    remaining_person = list(person_order if person_order is not None else base_ordered_person)

                    avg_non_person_l = total_non_person_l / target_count if target_count else total_non_person_l
                    # 当前构造轮次内，人员箱列表及大量chunk状态会被重复查询。
                    person_options_cache = {}
                    reservation_cache = {}

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

                    def _person_options(chunk, person_list):
                        # box_indices和person_list顺序完整进入key，缓存不会改变原匹配顺序。
                        # 人员箱能否放入只取决于当前块的重量、换长和人员箱列表；
                        # 不同箱组合只要总重量/换长相同，其可选人员箱严格相同。
                        key = (chunk.get('weight', 0.0), chunk.get('length', 0.0), tuple(person_list))
                        cached = person_options_cache.get(key)
                        if cached is None:
                            cached = tuple(pi for pi in person_list if can_add_to_chunk(chunk, pi))
                            person_options_cache[key] = cached
                        return cached

                    def can_fit_any_person(chunk, person_list):
                        """单块快速预检：至少存在一个可放入的人员箱。"""
                        return bool(_person_options(chunk, person_list))

                    def reserve_distinct_persons(candidate_chunks, person_list):
                        """为各个未配人员的非人员块分配一个互不重复的人员箱。

                        返回 ``{chunk_index: person_box_index}``；无可行匹配时返回 ``None``。
                        """
                        cache_key = (
                            tuple((
                                c.get('weight', 0.0),
                                c.get('length', 0.0),
                                c.get('_person_box_count', 0),
                                c.get('_non_person_box_count', 0),
                            ) for c in candidate_chunks),
                            tuple(person_list),
                        )
                        if cache_key in reservation_cache:
                            cached = reservation_cache[cache_key]
                            return None if cached is None else dict(cached)

                        need_chunks = []
                        for k, chunk0 in enumerate(candidate_chunks):
                            load0 = chunk_person_nonperson_load(chunk0)
                            if load0['non_person_count'] > 0 and load0['person_count'] == 0:
                                need_chunks.append(k)
                        if len(need_chunks) > len(person_list):
                            reservation_cache[cache_key] = None
                            return None

                        # 优先处理可选人员箱更少的块，降低匹配贪心误判概率。
                        # 使用稠密下标和代际标记代替dict+set；DFS顺序与原实现完全一致。
                        person_pos = {pi: pos for pos, pi in enumerate(person_list)}
                        choices = {}
                        for k in need_chunks:
                            opts = _person_options(candidate_chunks[k], person_list)
                            if not opts:
                                reservation_cache[cache_key] = None
                                return None
                            choices[k] = tuple(person_pos[pi] for pi in opts)
                        ordered_chunks = sorted(need_chunks, key=lambda k: len(choices[k]))
                        person_to_chunk = [-1] * len(person_list)
                        seen_marks = [0] * len(person_list)
                        epoch = 0

                        def augment(k):
                            for pos in choices[k]:
                                if seen_marks[pos] == epoch:
                                    continue
                                seen_marks[pos] = epoch
                                old_k = person_to_chunk[pos]
                                if old_k < 0 or augment(old_k):
                                    person_to_chunk[pos] = k
                                    return True
                            return False

                        matched = True
                        for k in ordered_chunks:
                            epoch += 1
                            if not augment(k):
                                matched = False
                                break
                        if not matched:
                            reservation_cache[cache_key] = None
                            return None
                        result = {k: person_list[pos] for pos, k in enumerate(person_to_chunk) if k >= 0}
                        reservation_cache[cache_key] = tuple(result.items())
                        return result

                    def can_reserve_distinct_persons(candidate_chunks, person_list):
                        """全局预检：每个待配人员块都能预留一个不同的人员箱。"""
                        return reserve_distinct_persons(candidate_chunks, person_list) is not None

                    def _prepare_matching_context(base_chunks, person_list):
                        matching = reserve_distinct_persons(base_chunks, person_list)
                        if matching is None:
                            return None
                        person_pos = {pi: pos for pos, pi in enumerate(person_list)}
                        owner_by_pos = [-1] * len(person_list)
                        for k, pi in matching.items():
                            owner_by_pos[person_pos[pi]] = k
                        choice_positions = {
                            k: tuple(person_pos[pi] for pi in _person_options(base_chunks[k], person_list))
                            for k in matching
                        }
                        return matching, person_pos, owner_by_pos, choice_positions

                    def _can_repair_matching_after_one_chunk_change(
                            base_chunks, changed_pos, changed_chunk, person_list, matching_context):
                        """复用当前匹配，只对一个发生变化的块执行增量修复。"""
                        if matching_context is None:
                            # 加入一个非人员箱只会收紧容量或增加待配块，不可能把原不可行变可行。
                            return False

                        base_matching, person_pos, base_owner_by_pos, base_choice_positions = matching_context
                        changed_load = chunk_person_nonperson_load(changed_chunk)
                        changed_needs_person = (
                            changed_load['non_person_count'] > 0 and changed_load['person_count'] == 0
                        )
                        if not changed_needs_person:
                            return True

                        assigned = base_matching.get(changed_pos)
                        if assigned is not None and can_add_to_chunk(changed_chunk, assigned):
                            return True

                        changed_choices = tuple(
                            person_pos[pi] for pi in _person_options(changed_chunk, person_list)
                        )
                        if not changed_choices:
                            return False

                        owner_by_pos = list(base_owner_by_pos)
                        if assigned is not None:
                            owner_by_pos[person_pos[assigned]] = -1
                        seen = [False] * len(person_list)

                        def choices_for(k):
                            return changed_choices if k == changed_pos else base_choice_positions[k]

                        def augment(k):
                            for pos in choices_for(k):
                                if seen[pos]:
                                    continue
                                seen[pos] = True
                                old_k = owner_by_pos[pos]
                                if old_k < 0 or augment(old_k):
                                    owner_by_pos[pos] = k
                                    return True
                            return False

                        return augment(changed_pos)

                    def _matching_candidate_after_non_person(chunk, box_idx):
                        """只构造人员预留判断所需字段，数值与完整make_unit候选完全一致。"""
                        b = all_sub_containers[box_idx]
                        return UnitDict({
                            'weight': chunk['weight'] + b.weight,
                            'length': chunk['length'] + b.length_unit,
                            '_person_box_count': chunk.get('_person_box_count', 0),
                            '_non_person_box_count': chunk.get('_non_person_box_count', 0) + 1,
                        })

                    def can_add_non_person_and_still_fit_person(
                            chunk, np_idx, person_list, chunk_pos, matching_context):
                        """非人员箱加入后，须为各块全局预留互不重复的人员箱。"""
                        if not can_add_to_chunk(chunk, np_idx):
                            return False
                        tmp_chunk = _matching_candidate_after_non_person(chunk, np_idx)
                        if not can_fit_any_person(tmp_chunk, person_list):
                            return False
                        return _can_repair_matching_after_one_chunk_change(
                            chunks, chunk_pos, tmp_chunk, person_list, matching_context
                        )

                    def try_relocate_non_person_for_reservation(chunks, pending_idx, person_list):
                        """一跳搬移修复非人员箱的贪心死路。

                        当待放箱子无法直接加入任何块时，尝试把一个已放入的非人员箱
                        搬到别的块，给待放箱子腾出位置；每个候选方案都重新做“不同人员箱”
                        的全局预留校验。这样 55 换长下不会因为早期均衡策略略有偏差就直接报无解。
                        """
                        best_plan = None
                        best_score = None
                        for src_k, src in enumerate(chunks):
                            movable = [
                                bi for bi in src.get('box_indices', [])
                                if get_public_box_type(all_sub_containers[bi].box_type) != 'Person'
                            ]
                            # 优先搬换长大的箱，通常更容易为 pending_idx 腾出空间。
                            movable.sort(
                                key=lambda bi: (all_sub_containers[bi].length_unit, all_sub_containers[bi].weight),
                                reverse=True,
                            )
                            for move_idx in movable:
                                src_after = [bi for bi in src['box_indices'] if bi != move_idx] + [pending_idx]
                                new_src = make_unit(src_after, forced_owners={cid})
                                if (new_src['weight'] > max_weight_per_sc + 1e-6 or
                                        new_src['length'] > max_length_per_sc + 1e-6):
                                    continue
                                for dst_k, dst in enumerate(chunks):
                                    if dst_k == src_k:
                                        continue
                                    dst_after = list(dst['box_indices']) + [move_idx]
                                    new_dst = make_unit(dst_after, forced_owners={cid})
                                    if (new_dst['weight'] > max_weight_per_sc + 1e-6 or
                                            new_dst['length'] > max_length_per_sc + 1e-6):
                                        continue
                                    candidate_chunks = list(chunks)
                                    candidate_chunks[src_k] = new_src
                                    candidate_chunks[dst_k] = new_dst
                                    if not can_reserve_distinct_persons(candidate_chunks, person_list):
                                        continue
                                    # 留白越均衡越好，避免刚修复一个箱又堵死后续箱。
                                    fills = [
                                        dominant_ratio(c['weight'], c['length'], max_weight_per_sc, max_length_per_sc)
                                        for c in candidate_chunks
                                    ]
                                    score = max(fills) + 0.15 * (max(fills) - min(fills))
                                    if best_score is None or score < best_score:
                                        best_score = score
                                        best_plan = (src_k, dst_k, new_src, new_dst)
                        if best_plan is None:
                            return False
                        src_k, dst_k, new_src, new_dst = best_plan
                        chunks[src_k] = new_src
                        chunks[dst_k] = new_dst
                        return True

                    # 第一步：每个目标块先放至少一个物资/装备箱，杜绝纯人员块。
                    # 此时就要预留至少一个人员箱的容量，避免后续出现“物资块已经满了、人员进不去”。
                    for k in range(target_count):
                        if not remaining_non_person:
                            return None, '物资/装备箱数量不足，无法给每个装车单元配置物资/装备'

                        best_pos = None
                        best_score = None
                        matching_context = _prepare_matching_context(chunks, remaining_person)
                        for pos, idx in enumerate(remaining_non_person):
                            if not can_add_non_person_and_still_fit_person(
                                    chunks[k], idx, remaining_person, k, matching_context):
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
                        matching_context = _prepare_matching_context(chunks, remaining_person)
                        for k, chunk in enumerate(chunks):
                            if not can_add_non_person_and_still_fit_person(
                                    chunk, idx, remaining_person, k, matching_context):
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
                            # 先尝试一跳搬移，修复由前序贪心均衡造成的局部死路；
                            # 仍失败才说明当前 target_count 下无法继续构造。
                            if try_relocate_non_person_for_reservation(chunks, idx, remaining_person):
                                continue
                            return None, f'剩余物资/装备箱 BoxIndex={idx} 无法加入任何可预留人员容量的装车单元'
                        add_box_to_chunk(chunks[best_k], idx)

                    # 第三步：按第二步已经验证过的全局匹配，给每个块落入一个不同的人员箱。
                    # 不能再按局部评分临时抢人箱，否则会破坏前面验证过的可行匹配。
                    reserved_persons = reserve_distinct_persons(chunks, remaining_person)
                    if reserved_persons is None:
                        return None, '物资/装备分摊完成后，无法为每个装车单元匹配不同的人员箱'
                    for k, idx in sorted(
                        reserved_persons.items(),
                        key=lambda pair: target_person_l_for_chunk(chunks[pair[0]]),
                        reverse=True,
                    ):
                        if idx not in remaining_person or not can_add_to_chunk(chunks[k], idx):
                            return None, f'第{k + 1}个装车单元的预留人员箱无法落位'
                        remaining_person.remove(idx)
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

                def order_variants():
                    """给同一target_count提供多种排序重试，避免50这类边界换长下被单一路径卡死。"""
                    variants = []

                    def add_variant(name, np_order, p_order):
                        sig = (tuple(np_order), tuple(p_order))
                        if any(v[0] == sig for v in variants):
                            return
                        variants.append((sig, name, list(np_order), list(p_order)))

                    add_variant('dominant_desc/person_desc', base_ordered_non_person, base_ordered_person)
                    add_variant(
                        'length_desc/person_asc',
                        sorted(base_ordered_non_person, key=lambda i: (box_has_chaoxian_equipment(all_sub_containers[i]), all_sub_containers[i].length_unit, all_sub_containers[i].weight), reverse=True),
                        sorted(base_ordered_person, key=lambda i: (all_sub_containers[i].length_unit, all_sub_containers[i].weight)),
                    )
                    add_variant(
                        'length_asc/person_desc',
                        sorted(base_ordered_non_person, key=lambda i: (box_has_chaoxian_equipment(all_sub_containers[i]), all_sub_containers[i].length_unit, all_sub_containers[i].weight)),
                        base_ordered_person,
                    )
                    add_variant(
                        'weight_desc/person_asc',
                        sorted(base_ordered_non_person, key=lambda i: (box_has_chaoxian_equipment(all_sub_containers[i]), all_sub_containers[i].weight, all_sub_containers[i].length_unit), reverse=True),
                        sorted(base_ordered_person, key=lambda i: (all_sub_containers[i].weight, all_sub_containers[i].length_unit)),
                    )
                    return [(name, np_order, p_order) for _sig, name, np_order, p_order in variants]

                def chunks_balance_score(chunks):
                    """公司内部候选块评分：优先可行，其次换长更均衡；必要时允许多拆一块。"""
                    if not chunks:
                        return float('inf')
                    lengths = [c['length'] for c in chunks]
                    weights = [c['weight'] for c in chunks]
                    target_l = max_length_per_sc * BALANCE_TARGET_LENGTH_RATIO
                    min_target_l = max_length_per_sc * BALANCE_MIN_LENGTH_RATIO
                    under_penalty = sum(max(0.0, min_target_l - l) ** 2 for l in lengths)
                    target_gap = sum((l - target_l) ** 2 for l in lengths) / max(1, len(lengths))
                    range_penalty = (max(lengths) - min(lengths)) ** 2 if len(lengths) > 1 else 0.0
                    weight_range = (max(weights) - min(weights)) ** 2 / max(max_weight_per_sc, 1.0) if len(weights) > 1 else 0.0
                    extra_count_penalty = 0.35 * max(0, len(chunks) - lower_count) * (max_length_per_sc ** 2)
                    return 2.20 * under_penalty + 0.85 * target_gap + 0.55 * range_penalty + 0.05 * weight_range + extra_count_penalty

                def build_exact_with_target_count(target_count):
                    """用 MILP 精确寻找指定装车单元数的可行分配。

                    正常数据仍先走上面的快速启发式；只有所有启发式排序都失败时才调用本函数。
                    与启发式不同，这里同时决定所有箱子的去向，所以不会因为前面某个箱子的
                    局部贪心选择，导致后面的箱子在临界最大换长下被误判为无解。

                    模型只表达现有硬规则，不改变装车规则或接口：
                    1. 每个箱子恰好进入一个装车单元；
                    2. 每个单元不超重、不超换长；
                    3. 每个单元至少有一个本公司人员箱；允许纯人员单元；
                    4. 任何含物资/装备的单元因此必然同时含本公司人员。
                    """
                    try:
                        import numpy as np
                        from scipy.optimize import Bounds, LinearConstraint, milp
                        from scipy.sparse import coo_matrix
                    except Exception as exc:
                        raise AlgorithmError(
                            '精确可行性兜底需要 scipy（含 scipy.optimize.milp）；'
                            '请在运行/打包环境安装 scipy>=1.9'
                        ) from exc

                    exact_indices = list(person_indices) + list(non_person_indices)
                    box_count = len(exact_indices)
                    variable_count = box_count * target_count
                    if box_count == 0 or target_count <= 0:
                        return None, '精确求解收到空箱集合或非法装车单元数'

                    weights = np.asarray(
                        [float(all_sub_containers[i].weight) for i in exact_indices], dtype=float
                    )
                    lengths = np.asarray(
                        [float(all_sub_containers[i].length_unit) for i in exact_indices], dtype=float
                    )
                    is_person = np.asarray(
                        [1.0 if get_public_box_type(all_sub_containers[i].box_type) == 'Person' else 0.0
                         for i in exact_indices],
                        dtype=float,
                    )
                    is_non_person = 1.0 - is_person

                    row_indices = []
                    col_indices = []
                    coefficients = []
                    lower_bounds = []
                    upper_bounds = []
                    row = 0

                    def add_row(entries, lower, upper):
                        nonlocal row
                        for col, value in entries:
                            if abs(value) > 1e-12:
                                row_indices.append(row)
                                col_indices.append(col)
                                coefficients.append(float(value))
                        lower_bounds.append(float(lower))
                        upper_bounds.append(float(upper))
                        row += 1

                    # 每个箱子必须且只能分配一次。
                    for i in range(box_count):
                        add_row(((i * target_count + k, 1.0) for k in range(target_count)), 1.0, 1.0)

                    # 每个装车单元的容量和人-物同车约束。
                    for k in range(target_count):
                        add_row(
                            ((i * target_count + k, weights[i]) for i in range(box_count)),
                            -np.inf,
                            max_weight_per_sc,
                        )
                        add_row(
                            ((i * target_count + k, lengths[i]) for i in range(box_count)),
                            -np.inf,
                            max_length_per_sc,
                        )
                        add_row(
                            ((i * target_count + k, is_person[i]) for i in range(box_count)),
                            1.0,
                            np.inf,
                        )
                        # 不要求每个单元必须有物资/装备；纯人员单元符合业务规则。

                    # 对称性消除：按总换长非增序排列装车单元。
                    # 任意可行方案都可以按此顺序重编号，因此不会删掉真实可行解，
                    # 但可显著减少完全相同车辆编号造成的搜索分支。
                    for k in range(target_count - 1):
                        add_row(
                            (
                                (i * target_count + kk, lengths[i] * sign)
                                for i in range(box_count)
                                for kk, sign in ((k, 1.0), (k + 1, -1.0))
                            ),
                            0.0,
                            np.inf,
                        )

                    matrix = coo_matrix(
                        (coefficients, (row_indices, col_indices)),
                        shape=(row, variable_count),
                    ).tocsr()

                    # 目标函数恒为 0：这里只需第一组严格可行解，不额外消耗时间追求最优评分。
                    # 常规方案质量仍由前面的启发式和后续车辆均衡处理负责。
                    result = milp(
                        c=np.zeros(variable_count, dtype=float),
                        integrality=np.ones(variable_count, dtype=np.int8),
                        bounds=Bounds(
                            np.zeros(variable_count, dtype=float),
                            np.ones(variable_count, dtype=float),
                        ),
                        constraints=LinearConstraint(
                            matrix,
                            np.asarray(lower_bounds, dtype=float),
                            np.asarray(upper_bounds, dtype=float),
                        ),
                        options={
                            'presolve': True,
                            'mip_rel_gap': 0.0,
                            'disp': False,
                        },
                    )

                    if result.x is None:
                        if getattr(result, 'status', None) == 2:
                            return None, f'精确求解证明 target_count={target_count} 不可行'
                        return None, (
                            f'精确求解未返回方案：target_count={target_count}, '
                            f'status={getattr(result, "status", "")}, '
                            f'message={getattr(result, "message", "")}'
                        )

                    assignment = np.rint(np.asarray(result.x).reshape(box_count, target_count)).astype(int)
                    if np.any(assignment.sum(axis=1) != 1):
                        return None, f'精确求解结果完整性校验失败：target_count={target_count}'

                    exact_chunks = []
                    for k in range(target_count):
                        selected = [exact_indices[i] for i in range(box_count) if assignment[i, k] == 1]
                        if not selected:
                            return None, f'精确求解产生空装车单元：target_count={target_count}, k={k}'
                        chunk = make_unit(selected, forced_owners={cid})
                        if (chunk['weight'] > max_weight_per_sc + 1e-6 or
                                chunk['length'] > max_length_per_sc + 1e-6):
                            return None, f'精确求解结果复核超限：target_count={target_count}, k={k}'
                        exact_chunks.append(chunk)

                    try:
                        assert_no_single_type_chunks(exact_chunks)
                    except AlgorithmError as exc:
                        return None, f'精确求解结果人-物同车复核失败：{exc}'
                    return exact_chunks, ''

                best_chunks = None
                best_score = None
                best_target_count = None
                best_strategy = ''
                order_variant_list = order_variants()

                # 从理论下界开始尝试；如果为了满足人-物同车或均衡需要增加车辆数，则逐步增加。
                # 不再遇到第一个可行方案就返回，避免45/55可行而50被局部贪心误判，
                # 同时避免公司内部产生40、26这类明显不均衡的块。
                heuristic_target_max = min(max_mixed_count, len(non_person_indices))
                for target_count in range(lower_count, heuristic_target_max + 1):
                    for strategy_name, np_order, p_order in order_variant_list:
                        chunks, err = build_with_target_count(target_count, np_order, p_order)
                        if chunks is not None:
                            score = chunks_balance_score(chunks)
                            if best_score is None or score < best_score:
                                best_score = score
                                best_chunks = chunks
                                best_target_count = target_count
                                best_strategy = strategy_name
                        else:
                            last_error = err

                    # 若已经达到最低车辆数且所有块换长均不低于目标下限，可提前结束；
                    # 否则继续看多一个混合块是否能显著改善均衡。
                    if best_chunks is not None and best_target_count == lower_count:
                        min_l = min(c['length'] for c in best_chunks)
                        max_l = max(c['length'] for c in best_chunks)
                        if (min_l >= max_length_per_sc * BALANCE_MIN_LENGTH_RATIO - 1e-6 and
                                max_l - min_l <= max_length_per_sc * BALANCE_MAX_GAP_RATIO + 1e-6):
                            break

                # 快速启发式全部失败后，再进行精确可行性兜底。
                # 这是修复“改变最大换长偶尔就能运行”的关键：不再把贪心死路当成数学无解。
                if best_chunks is None:
                    exact_errors = []
                    for target_count in range(lower_count, max_mixed_count + 1):
                        exact_chunks, exact_error = build_exact_with_target_count(target_count)
                        if exact_chunks is not None:
                            best_chunks = exact_chunks
                            best_target_count = target_count
                            best_strategy = 'exact-milp-fallback'
                            best_score = chunks_balance_score(exact_chunks)
                            break
                        exact_errors.append(exact_error)
                    if exact_errors:
                        last_error = '；'.join(exact_errors[-3:])

                if best_chunks is not None:
                    print(
                        f"公司 {company_name.get(cid, cid)}({cid}) 人-物同车打包完成："
                        f"target_count={best_target_count}, strategy={best_strategy}, "
                        f"换长={[round(c['length'], 2) for c in best_chunks]}"
                    )
                    return best_chunks

                raise AlgorithmError(
                    f'公司 {company_name.get(cid, cid)}({cid}) 同时存在人员和物资/装备，'
                    f'但无法在不超重、不超换长的前提下保证每辆含该公司物资/装备的SC都有该公司人员；'
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
            """车辆选择评分。

            原逻辑主要追求装得更满，容易在边界换长下形成一辆40左右、另一辆26左右的尾车。
            现在改为：在不增加硬约束的前提下，优先让放置后的换长接近目标利用率，
            同时保留重量利用率和同公司/超限加分。
            """
            fill_w = (vehicle.weight + unit['weight']) / max_weight_per_sc if max_weight_per_sc else 0.0
            fill_l = (vehicle.length + unit['length']) / max_length_per_sc if max_length_per_sc else 0.0
            target_l = BALANCE_TARGET_LENGTH_RATIO
            min_l = BALANCE_MIN_LENGTH_RATIO
            under = max(0.0, min_l - fill_l)
            over = max(0.0, fill_l - 0.98)
            return (
                1.10 * fill_l
                + 0.25 * fill_w
                - 1.35 * abs(fill_l - target_l)
                - 0.85 * under
                - 0.60 * over
                - 0.10 * abs(fill_w - fill_l)
            )

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

        def companies_requiring_person_nonperson_mix():
            presence = defaultdict(lambda: {'person': False, 'non_person': False})
            for box in all_sub_containers:
                public_type = get_public_box_type(getattr(box, 'box_type', ''))
                for cid0 in getattr(box, 'owners', set()):
                    if public_type == 'Person':
                        presence[cid0]['person'] = True
                    else:
                        presence[cid0]['non_person'] = True
            # 业务规则是“有物必有人”，因此所有存在物资/装备的公司都需要校验。
            return {cid0 for cid0, flags in presence.items() if flags.get('non_person')}

        companies_need_mixed_final = companies_requiring_person_nonperson_mix()

        def vehicle_respects_person_nonperson_rule(vehicle):
            """搬移预检：单车有某公司物资/装备时，必须同时有该公司人员。"""
            person_owners = set()
            non_person_owners = set()
            for unit in vehicle.units:
                person_owners.update(unit.get('_person_owners', set()))
                non_person_owners.update(unit.get('_non_person_owners', set()))
            for cid0 in non_person_owners:
                if cid0 in companies_need_mixed_final and cid0 not in person_owners:
                    return False
            return True

        def unit_has_person_box(unit):
            return bool(unit.get('_has_person_box', False))

        def unit_has_non_person_box(unit):
            return bool(unit.get('_has_non_person_box', False))

        def box_person_count_for_company(box, cid0):
            return getattr(box, '_cached_person_counts', {}).get(cid0, 0)

        def unit_person_count_for_company(unit, cid0):
            return unit.get('_person_counts', {}).get(cid0, 0)

        def vehicle_person_count_for_company(vehicle, cid0):
            return vehicle._person_counts.get(cid0, 0)

        def vehicle_has_company(vehicle, cid0):
            return cid0 in vehicle.companies

        def company_people_distribution_penalty(vehicle_list):
            """同一公司人员在其所在车辆上的均衡软惩罚。

            只统计人员数量，不改变硬规则；公司只出现在一辆车时惩罚为0。
            使用相对差距，避免大公司天然因人数多而被过度惩罚。
            """
            company_to_vehicle_counts = defaultdict(list)
            for cid0 in company_name.keys():
                counts = []
                for vehicle in vehicle_list:
                    if vehicle_has_company(vehicle, cid0):
                        counts.append(vehicle_person_count_for_company(vehicle, cid0))
                if len(counts) > 1 and sum(counts) > 0:
                    company_to_vehicle_counts[cid0] = counts

            penalty = 0.0
            for cid0, counts in company_to_vehicle_counts.items():
                avg = sum(counts) / len(counts)
                if avg <= 1e-9:
                    continue
                rel_var = sum(((c - avg) / avg) ** 2 for c in counts) / len(counts)
                allowed_gap = max(PERSON_BALANCE_MAX_ABS_GAP, avg * PERSON_BALANCE_MAX_RATIO)
                range_over = max(0.0, max(counts) - min(counts) - allowed_gap) / max(avg, 1.0)
                penalty += rel_var + 1.50 * (range_over ** 2)
            return penalty

        def company_spread_penalty(vehicle_list):
            """同一公司被分散到过多车辆的软惩罚。

            说明：
            - 只统计公司是否出现在某辆车中，不区分人、物资或装备；
            - 该惩罚不作为硬约束，不会阻止为消除低换长尾车而进行的必要跨公司成组补车；
            - 目标是在换长接近、车辆数不增加的候选方案之间，优先选择公司分散更少的方案。
            """
            penalty = 0.0
            for cid0 in company_name.keys():
                vehicle_count = sum(1 for vehicle in vehicle_list if vehicle_has_company(vehicle, cid0))
                if vehicle_count <= 1:
                    continue
                # 轻惩罚“多出现一辆车”，平方项避免过度分散。
                penalty += (vehicle_count - 1) ** 2
            return penalty

        def overall_balance_objective(vehicle_list):
            return (
                fleet_balance_objective(vehicle_list)
                + PERSON_BALANCE_WEIGHT * (max_length_per_sc ** 2) * company_people_distribution_penalty(vehicle_list)
                + COMPANY_SPREAD_WEIGHT * (max_length_per_sc ** 2) * company_spread_penalty(vehicle_list)
            )

        def fleet_balance_objective(vehicle_list):
            if not vehicle_list:
                return 0.0
            lengths = [v.length for v in vehicle_list if v.units]
            weights = [v.weight for v in vehicle_list if v.units]
            if not lengths:
                return 0.0
            target_l = max_length_per_sc * BALANCE_TARGET_LENGTH_RATIO
            min_target_l = max_length_per_sc * BALANCE_MIN_LENGTH_RATIO
            allowed_gap = max_length_per_sc * BALANCE_MAX_GAP_RATIO
            under_penalty = sum(max(0.0, min_target_l - l) ** 2 for l in lengths)
            target_penalty = sum((l - target_l) ** 2 for l in lengths) / max(1, len(lengths))
            range_penalty = max(0.0, (max(lengths) - min(lengths)) - allowed_gap) ** 2
            full_range_penalty = (max(lengths) - min(lengths)) ** 2 if len(lengths) > 1 else 0.0
            weight_range_penalty = (max(weights) - min(weights)) ** 2 / max(max_weight_per_sc, 1.0) if len(weights) > 1 else 0.0
            return 7.50 * under_penalty + 1.15 * target_penalty + 2.20 * range_penalty + 0.35 * full_range_penalty + 0.04 * weight_range_penalty

        def _people_penalty_for_counts(counts):
            """与 company_people_distribution_penalty 中单公司公式完全一致。"""
            if len(counts) <= 1 or sum(counts) <= 0:
                return 0.0
            avg = sum(counts) / len(counts)
            if avg <= 1e-9:
                return 0.0
            rel_var = sum(((c - avg) / avg) ** 2 for c in counts) / len(counts)
            allowed_gap = max(PERSON_BALANCE_MAX_ABS_GAP, avg * PERSON_BALANCE_MAX_RATIO)
            range_over = max(0.0, max(counts) - min(counts) - allowed_gap) / max(avg, 1.0)
            return rel_var + 1.50 * (range_over ** 2)

        class _ObjectiveIterationCache:
            """单次搜索迭代的等价评分缓存。

            每个候选只改变 donor/receiver 两辆车，因此：
            - 车队换长目标仍按原函数逐车计算；
            - 人员均衡、公司分散度仅重算受影响公司；
            - 最终仍按 company_name 原插入顺序累加，避免改变候选比较顺序。
            """
            __slots__ = (
                'vehicles', 'company_order', 'people_contrib', 'spread_contrib',
                'people_total', 'spread_total'
            )

            def __init__(self, vehicles):
                self.vehicles = vehicles
                self.company_order = tuple(company_name.keys())
                self.people_contrib = {}
                self.spread_contrib = {}
                for cid0 in self.company_order:
                    counts = [v._person_counts.get(cid0, 0) for v in vehicles if cid0 in v.companies]
                    self.people_contrib[cid0] = _people_penalty_for_counts(counts)
                    vehicle_count = sum(1 for v in vehicles if cid0 in v.companies)
                    self.spread_contrib[cid0] = (vehicle_count - 1) ** 2 if vehicle_count > 1 else 0.0
                # 保持原公司顺序累加。
                people_total = 0.0
                spread_total = 0.0
                for cid0 in self.company_order:
                    people_total += self.people_contrib[cid0]
                    spread_total += self.spread_contrib[cid0]
                self.people_total = people_total
                self.spread_total = spread_total

            @staticmethod
            def affected_companies(old_donor, old_receiver, new_donor, new_receiver):
                affected = set(old_donor.companies) | set(old_receiver.companies)
                affected.update(new_donor.companies)
                affected.update(new_receiver.companies)
                affected.update(old_donor._person_counts.keys())
                affected.update(old_receiver._person_counts.keys())
                affected.update(new_donor._person_counts.keys())
                affected.update(new_receiver._person_counts.keys())
                return affected

            def penalties(self, candidate, affected):
                # 按原company_name插入顺序累加，确保评分计算顺序与原函数一致。
                replacement_people = {}
                replacement_spread = {}
                for cid0 in affected:
                    if cid0 not in self.people_contrib:
                        continue
                    counts = [v._person_counts.get(cid0, 0) for v in candidate if cid0 in v.companies]
                    replacement_people[cid0] = _people_penalty_for_counts(counts)
                    vehicle_count = sum(1 for v in candidate if cid0 in v.companies)
                    replacement_spread[cid0] = (vehicle_count - 1) ** 2 if vehicle_count > 1 else 0.0
                people_total = 0.0
                spread_total = 0.0
                for cid0 in self.company_order:
                    people_total += replacement_people.get(cid0, self.people_contrib[cid0])
                    spread_total += replacement_spread.get(cid0, self.spread_contrib[cid0])
                return people_total, spread_total

            def objective(self, candidate, affected):
                people_penalty, spread_penalty = self.penalties(candidate, affected)
                return (
                    fleet_balance_objective(candidate)
                    + PERSON_BALANCE_WEIGHT * (max_length_per_sc ** 2) * people_penalty
                    + COMPANY_SPREAD_WEIGHT * (max_length_per_sc ** 2) * spread_penalty
                )

        def clone_vehicle_list_for_move(vehicle_list, donor_idx, receiver_idx):
            """候选搬移只复制会发生变化的供给车和接收车，其余车辆只读共享。"""
            candidate = list(vehicle_list)
            candidate[donor_idx] = vehicle_list[donor_idx].clone()
            candidate[receiver_idx] = vehicle_list[receiver_idx].clone()
            return candidate

        def balance_vehicles_by_unit_moves(vehicle_list):
            """车辆层换长均衡后处理。

            策略顺序：
            1）优先从同公司其他车辆中搬完整装车单元或可拆小箱，补到换长偏低车辆；
            2）同公司补不动时，再尝试从其他公司车辆中拆出可行候选补车；
            3）跨公司补车若涉及一个同时有人和物的公司，必须搬入完整人-物成组包，不能造成该公司在任一车辆中只有人或只有物；
            4）评分中加入公司分散度惩罚，避免为了补换长把同一公司分散到过多车辆；
            5）每次搬移前后都校验超重、超换长、yingjiName种类数和人-物同车硬规则。
            """
            vehicles_local = [v for v in vehicle_list if v.units]
            if len(vehicles_local) <= 1:
                return vehicles_local

            base_vehicle_validity = []

            def valid_after_move(candidate, donor_idx, receiver_idx):
                # 本候选只改变供给车和接收车；其余车辆复用本轮开始时的相同硬校验结果。
                for vi, vehicle in enumerate(candidate):
                    if not vehicle.units:
                        continue
                    if vi == donor_idx or vi == receiver_idx:
                        if not vehicle_respects_person_nonperson_rule(vehicle):
                            return False
                    elif not base_vehicle_validity[vi]:
                        return False
                return True

            current_score = overall_balance_objective(vehicles_local)
            for _ in range(BALANCE_MAX_ITERATIONS):
                objective_cache = _ObjectiveIterationCache(vehicles_local)
                base_vehicle_validity = [vehicle_respects_person_nonperson_rule(v) for v in vehicles_local]
                best = None
                best_score = current_score
                under_order = sorted(
                    range(len(vehicles_local)),
                    key=lambda i: (vehicles_local[i].length, vehicles_local[i].weight),
                )
                lengths = [v.length for v in vehicles_local]
                length_gap = max(lengths) - min(lengths)
                donor_units_cache = {}
                move_options_cache = {}

                for receiver_idx in under_order:
                    receiver = vehicles_local[receiver_idx]
                    # 低于目标下限，或全局最大/最小换长差距过大时才主动补车。
                    need_fill = (
                        receiver.length < max_length_per_sc * BALANCE_MIN_LENGTH_RATIO - 1e-6
                        or length_gap > max_length_per_sc * BALANCE_MAX_GAP_RATIO + 1e-6
                    )
                    if not need_fill:
                        continue

                    # 两轮候选：先同公司，再其他公司。
                    for prefer_same_company in (True, False):
                        for donor_idx, donor in enumerate(vehicles_local):
                            if donor_idx == receiver_idx or donor.length <= receiver.length + 1e-6:
                                continue
                            donor_units = donor_units_cache.get(donor_idx)
                            if donor_units is None:
                                donor_units = sorted(
                                    list(donor.units),
                                    key=lambda u: (u['length'], u['weight'], u['dominant']),
                                )
                                donor_units_cache[donor_idx] = donor_units
                            for unit in donor_units:
                                # 候选搬移粒度：先尝试完整装车单元；若单元过粗，再尝试搬移其中一个原始小箱。
                                # 小箱级搬移只在最终硬校验仍满足时接受，用于修复40/26这类尾车不均衡。
                                unit_cache_key = id(unit)
                                move_options = move_options_cache.get(unit_cache_key)
                                if move_options is None:
                                    move_options = [('unit', unit, None)]
                                if move_options_cache.get(unit_cache_key) is None and len(unit.get('box_indices', [])) > 1:
                                    single_boxes = sorted(
                                        list(unit.get('box_indices', [])),
                                        key=lambda bi: (
                                            all_sub_containers[bi].length_unit,
                                            all_sub_containers[bi].weight,
                                            get_public_box_type(all_sub_containers[bi].box_type) != 'Person',
                                        ),
                                    )
                                    for bi in single_boxes:
                                        move_options.append(('single_box', make_unit_readonly_cached([bi]), bi))

                                    # 对跨公司补车很关键：如果单独搬一个其他公司的物资/人员箱会破坏
                                    # “该公司在每辆车中必须人-物同车”的硬规则，则尝试搬一个最小成组包。
                                    # 成组包通常由同一公司的一个人员箱 + 一个非人员箱组成；
                                    # 若该公司在该单元中已有多箱，可生成若干候选组合，由后续硬校验筛选。
                                    bundle_seen = set()
                                    owners_in_unit = sorted(set(unit.get('owners', set())))
                                    for owner0 in owners_in_unit:
                                        if owner0 not in companies_need_mixed_final:
                                            continue
                                        person_bis = [
                                            bi for bi in unit.get('box_indices', [])
                                            if owner0 in all_sub_containers[bi].owners
                                            and get_public_box_type(all_sub_containers[bi].box_type) == 'Person'
                                        ]
                                        non_person_bis = [
                                            bi for bi in unit.get('box_indices', [])
                                            if owner0 in all_sub_containers[bi].owners
                                            and get_public_box_type(all_sub_containers[bi].box_type) != 'Person'
                                        ]
                                        person_bis = sorted(person_bis, key=lambda bi: (all_sub_containers[bi].length_unit, all_sub_containers[bi].weight))[:4]
                                        non_person_bis = sorted(non_person_bis, key=lambda bi: (all_sub_containers[bi].length_unit, all_sub_containers[bi].weight))[:4]
                                        for p_bi in person_bis:
                                            for np_bi in non_person_bis:
                                                bundle = tuple(sorted({p_bi, np_bi}))
                                                if len(bundle) < 2 or bundle in bundle_seen:
                                                    continue
                                                bundle_seen.add(bundle)
                                                move_options.append(('mixed_bundle', make_unit_readonly_cached(list(bundle), forced_owners={owner0}), None))
                                if unit_cache_key not in move_options_cache:
                                    move_options_cache[unit_cache_key] = move_options

                                for move_kind, moving_unit, single_box_idx in move_options:
                                    same_company = bool(moving_unit['owners'] & receiver.companies)
                                    if prefer_same_company and not same_company:
                                        continue
                                    if not prefer_same_company and same_company:
                                        continue
                                    # 跨公司补车允许搬入其他公司人员箱，但必须与该公司物资/装备成组满足人-物同车。
                                    # 单独搬入人员箱或单独搬入物资箱若会造成该公司在某车中人物分离，会在 valid_after_move 中被拒绝。
                                    if not receiver.can_place(moving_unit, max_weight_per_sc, max_length_per_sc, company_yingji_name):
                                        continue
                                    # 不把一个本来均衡的供给车拆成新的严重低换长车，除非该车被整体清空。
                                    donor_after_length = donor.length - moving_unit['length']
                                    if donor_after_length > 1e-6 and donor_after_length < max_length_per_sc * (BALANCE_MIN_LENGTH_RATIO - 0.08):
                                        continue

                                    candidate = clone_vehicle_list_for_move(vehicles_local, donor_idx, receiver_idx)
                                    cand_unit = None
                                    for u in candidate[donor_idx].units:
                                        if u is unit or u == unit:
                                            cand_unit = u
                                            break
                                    if cand_unit is None:
                                        continue

                                    candidate[donor_idx].remove(cand_unit, company_yingji_name)
                                    if move_kind == 'unit':
                                        candidate[receiver_idx].place(cand_unit, company_yingji_name)
                                    else:
                                        moving_indices = set(moving_unit.get('box_indices', []))
                                        rest_indices = [bi for bi in cand_unit.get('box_indices', []) if bi not in moving_indices]
                                        if rest_indices:
                                            rest_unit = make_unit_readonly_cached(rest_indices)
                                            if not candidate[donor_idx].can_place(rest_unit, max_weight_per_sc, max_length_per_sc, company_yingji_name):
                                                continue
                                            candidate[donor_idx].place(rest_unit, company_yingji_name)
                                        candidate[receiver_idx].place(moving_unit, company_yingji_name)

                                    if not valid_after_move(candidate, donor_idx, receiver_idx):
                                        continue
                                    affected = objective_cache.affected_companies(
                                        vehicles_local[donor_idx], vehicles_local[receiver_idx],
                                        candidate[donor_idx], candidate[receiver_idx]
                                    )
                                    new_score = objective_cache.objective(candidate, affected)
                                    candidate = [v for v in candidate if v.units]
                                    # 同公司搬移优先；跨公司搬移必须在补换长/少浪费方面带来更明确收益，且评分会惩罚公司过度分散。
                                    improvement_tol = 1e-7 if same_company else 5e-4
                                    if new_score < best_score - improvement_tol:
                                        best_score = new_score
                                        best = candidate
                        if best is not None:
                            break

                if best is None:
                    break
                vehicles_local = best
                current_score = best_score

            return vehicles_local

        def balance_company_people_distribution(vehicle_list):
            """公司人员跨车分布均衡后处理。

            原则：
            - 只搬同一公司的人员箱，不把其他公司人员拿来补车；
            - 只在搬移后仍满足人-物同车、超重、超换长、yingjiName种类数等硬规则时接受；
            - 目标是减少同一公司在其所在车辆上的人数差距，不能保证绝对平均。
            """
            vehicles_local = [v for v in vehicle_list if v.units]
            if len(vehicles_local) <= 1:
                return vehicles_local

            base_vehicle_validity = []

            def valid_after_move(candidate, donor_idx, receiver_idx):
                for vi, vehicle in enumerate(candidate):
                    if not vehicle.units:
                        continue
                    if vi == donor_idx or vi == receiver_idx:
                        if not vehicle_respects_person_nonperson_rule(vehicle):
                            return False
                    elif not base_vehicle_validity[vi]:
                        return False
                return True

            def company_vehicle_counts(vehicle_list0, cid0):
                pairs = []
                for vi, vehicle in enumerate(vehicle_list0):
                    if vehicle_has_company(vehicle, cid0):
                        pairs.append((vi, vehicle_person_count_for_company(vehicle, cid0)))
                return pairs

            def people_gap_needs_fix(counts):
                if len(counts) <= 1:
                    return False
                values = [c for _vi, c in counts]
                if not values or sum(values) <= 0:
                    return False
                avg = sum(values) / len(values)
                allowed_gap = max(PERSON_BALANCE_MAX_ABS_GAP, avg * PERSON_BALANCE_MAX_RATIO)
                return (max(values) - min(values)) > allowed_gap + 1e-6

            current_score = overall_balance_objective(vehicles_local)
            for _ in range(BALANCE_MAX_ITERATIONS):
                objective_cache = _ObjectiveIterationCache(vehicles_local)
                base_vehicle_validity = [vehicle_respects_person_nonperson_rule(v) for v in vehicles_local]
                best = None
                best_score = current_score
                current_people_penalty = objective_cache.people_total

                for cid0 in company_name.keys():
                    counts = company_vehicle_counts(vehicles_local, cid0)
                    if not people_gap_needs_fix(counts):
                        continue
                    avg = sum(c for _vi, c in counts) / len(counts)
                    high_list = sorted([x for x in counts if x[1] > avg + 1e-6], key=lambda x: x[1], reverse=True)
                    low_list = sorted([x for x in counts if x[1] < avg - 1e-6], key=lambda x: x[1])
                    if not high_list or not low_list:
                        continue

                    for donor_idx, donor_count in high_list:
                        donor = vehicles_local[donor_idx]
                        # 只取该公司自己的人员箱作为候选；不移动其他公司人员。
                        donor_units = sorted(
                            list(donor.units),
                            key=lambda u: (unit_person_count_for_company(u, cid0), u['length'], u['weight']),
                            reverse=True,
                        )
                        move_options = []
                        for unit in donor_units:
                            person_boxes = [
                                bi for bi in unit.get('box_indices', [])
                                if box_person_count_for_company(all_sub_containers[bi], cid0) > 0
                            ]
                            # 优先按小箱搬移，避免整单元搬走造成源车人-物同车被破坏。
                            for bi in sorted(
                                person_boxes,
                                key=lambda x: (box_person_count_for_company(all_sub_containers[x], cid0), all_sub_containers[x].length_unit),
                                reverse=True,
                            ):
                                moving_unit = make_unit_readonly_cached([bi], forced_owners={cid0})
                                move_options.append((unit, moving_unit, bi))

                        for receiver_idx, receiver_count in low_list:
                            if receiver_idx == donor_idx:
                                continue
                            receiver = vehicles_local[receiver_idx]
                            for source_unit, moving_unit, single_box_idx in move_options:
                                if not receiver.can_place(moving_unit, max_weight_per_sc, max_length_per_sc, company_yingji_name):
                                    continue
                                # 如果搬移人数明显超过低车缺口，仍允许尝试，但评分会自然惩罚过度搬移。
                                candidate = clone_vehicle_list_for_move(vehicles_local, donor_idx, receiver_idx)
                                cand_source_unit = None
                                for u in candidate[donor_idx].units:
                                    if u == source_unit:
                                        cand_source_unit = u
                                        break
                                if cand_source_unit is None:
                                    continue

                                candidate[donor_idx].remove(cand_source_unit, company_yingji_name)
                                rest_indices = [bi for bi in cand_source_unit.get('box_indices', []) if bi != single_box_idx]
                                if rest_indices:
                                    rest_unit = make_unit_readonly_cached(rest_indices)
                                    if not candidate[donor_idx].can_place(rest_unit, max_weight_per_sc, max_length_per_sc, company_yingji_name):
                                        continue
                                    candidate[donor_idx].place(rest_unit, company_yingji_name)
                                candidate[receiver_idx].place(moving_unit, company_yingji_name)
                                if not valid_after_move(candidate, donor_idx, receiver_idx):
                                    continue
                                affected = objective_cache.affected_companies(
                                    vehicles_local[donor_idx], vehicles_local[receiver_idx],
                                    candidate[donor_idx], candidate[receiver_idx]
                                )
                                new_people_penalty, new_spread_penalty = objective_cache.penalties(candidate, affected)
                                new_score = (
                                    fleet_balance_objective(candidate)
                                    + PERSON_BALANCE_WEIGHT * (max_length_per_sc ** 2) * new_people_penalty
                                    + COMPANY_SPREAD_WEIGHT * (max_length_per_sc ** 2) * new_spread_penalty
                                )
                                candidate = [v for v in candidate if v.units]
                                # 人员分布必须确实改善；整体换长均衡不能明显变差。
                                if new_people_penalty >= current_people_penalty - 1e-9:
                                    continue
                                if new_score < best_score - 1e-7:
                                    best_score = new_score
                                    best = candidate

                if best is None:
                    break
                vehicles_local = best
                current_score = best_score

            return vehicles_local

        # 多轮执行“压缩车辆数 -> 换长均衡 -> 公司人员均衡”。
        # 先尽量少用车，再在不增加车辆数、不破坏硬约束的前提下把尾车换长补起来。
        last_signature = None
        for _balance_round in range(6):
            vehicles = compact_vehicles(vehicles)
            vehicles = balance_vehicles_by_unit_moves(vehicles)
            vehicles = balance_company_people_distribution(vehicles)
            signature = (
                len(vehicles),
                tuple(sorted(round(v.length, 3) for v in vehicles)),
                tuple(sorted(round(v.weight, 3) for v in vehicles)),
            )
            if signature == last_signature:
                break
            last_signature = signature
        # 均衡后再尝试一次压缩；如压缩成功，再做一次均衡，避免新尾车过小。
        before_count = len(vehicles)
        vehicles = compact_vehicles(vehicles)
        if len(vehicles) < before_count:
            vehicles = balance_vehicles_by_unit_moves(vehicles)
            vehicles = balance_company_people_distribution(vehicles)
        total_sc_used = len(vehicles)
        print(f"启发式装车完成，使用 SC 总数: {total_sc_used}")
        if vehicles:
            lengths_after_balance = [round(v.length, 2) for v in vehicles]
            print(f"车辆换长均衡结果: min={min(lengths_after_balance):.2f}, max={max(lengths_after_balance):.2f}, values={lengths_after_balance}")

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
                if flags.get('non_person')
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
                    if flags.get('non_person') and not flags.get('person'):
                        raise AlgorithmError(
                            f"SC_{v + 1:03d} 违反人-物同车硬规则：公司 {company_name.get(cid0, cid0)}({cid0}) "
                            f"在该车中有物资/装备但没有该公司人员"
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
            在SC车辆组合已经确定后，只对同一SC内的Large、Small二次重装。
            - Person人员箱保持原箱，不做跨公司人员拼箱；
            - Large装备：同一zzsbid内可跨公司尾数拼箱，不检查sbrl/sbzz，只检查zzsbidNumber尾数规则和yingjiName≤2；
            - Small物资：保持原有同一zzsbid内跨公司尾数拼箱、zjdh矩阵、体积/载重、yingjiName≤2规则。
            """
            fixed_boxes = []
            component_items = []
            goods_items = []
            for orig_idx, box in sc_boxes:
                public_type = get_public_box_type(box.box_type)
                if public_type == 'Person':
                    # 明确保留原人员箱，不再把不同公司人员重装到同一个Person箱。
                    fixed_boxes.append((orig_idx, box))
                elif public_type == 'Large':
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
                    if not getattr(new_box, 'goods_closed', False):
                        open_large_repacked[zzsbid].append(new_box)
                    virtual_boxes.append((None, new_box))
                else:
                    if not best_box.add_item(cid, item, w, occupancy, item_volume=vol):
                        raise AlgorithmError(f'装备 {name or comp_id} 在SC内Large混装失败')
                    if getattr(best_box, 'goods_closed', False):
                        open_large_repacked[zzsbid].remove(best_box)

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
                    if not getattr(new_box, 'goods_closed', False):
                        open_repacked[zzsbid].append(new_box)
                    virtual_boxes.append((None, new_box))
                else:
                    if not best_box.add_item(cid, item, w, 0.0, item_volume=tj):
                        raise AlgorithmError(f'物资 {name or gid} 在SC内混装失败')
                    if getattr(best_box, 'goods_closed', False):
                        open_repacked[zzsbid].remove(best_box)

            return fixed_boxes + virtual_boxes

        res_data = {
            "code": 0,
            "msg": "success",
            "data": {
                "total_SC_used": total_sc_used,
                "SC_list": []
            }
        }

        assigned_indices_by_vehicle = [[] for _ in range(total_sc_used)]
        for idx0, assign0 in enumerate(heuristic_assign):
            assigned_indices_by_vehicle[assign0].append(idx0)

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

            merged_indices = assigned_indices_by_vehicle[v]
            sc_source_boxes = []
            for i in merged_indices:
                for orig_idx in merge_map[i]:
                    sc_source_boxes.append((orig_idx, original_boxes[orig_idx]))

            # 只有在同一SC内部，才允许不同公司的Large/Small按各自规则二次重装；Person人员箱保持原箱，不跨公司拼箱。
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
        logger.warning("算法执行失败，具体原因: %s", exc, exc_info=True)
        return {"code": 1, "msg": f"算法执行失败：{exc}"}
    except Exception as exc:
        logger.exception("算法执行失败，具体原因: %s", exc)
        return {"code": 1, "msg": f"算法执行失败：{exc}，请查看日志"}


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

            # 人员箱不得跨公司混装；允许同一SC存在多个公司的独立人员箱。
            if get_public_box_type(box.get('box_type', '')) == 'Person':
                person_company_ids = {
                    str(entity.get('company_id', '')).strip()
                    for entity in (box.get('content_desc', []) or [])
                    if isinstance(entity, dict) and entity.get('type') == 'person'
                    and str(entity.get('company_id', '')).strip()
                }
                person_owner_ids = {
                    str(cid).strip() for cid in (box.get('owners', []) or []) if str(cid).strip()
                }
                if len(person_company_ids) > 1 or len(person_owner_ids) > 1:
                    raise AlgorithmError(
                        f"{sid} 人员箱跨公司混装: box_id={box.get('box_id')}, "
                        f"company_ids={sorted(person_company_ids | person_owner_ids)}"
                    )

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


# ========================== 服务状态 ==========================
class ServiceState:
    def __init__(self):
        self._lock = threading.Lock()
        self.status = "初始化中"
        self.detail = "程序启动中"
        self.last_error = ""
        self.started_at: Optional[float] = None

    def set(self, status: str, detail: str = "", last_error: str = "") -> None:
        with self._lock:
            self.status = status
            self.detail = detail
            if last_error:
                self.last_error = last_error
            elif status not in {"启动失败", "运行异常"}:
                self.last_error = ""
            if status == "运行中" and self.started_at is None:
                self.started_at = time.time()
            if status in {"已停止", "启动失败", "运行异常"}:
                self.started_at = None if status != "运行异常" else self.started_at
        logger.info("服务状态更新: %s - %s", status, detail)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "detail": self.detail,
                "last_error": self.last_error,
                "started_at": self.started_at,
            }


SERVICE_STATE = ServiceState()

# ========================== FastAPI 接口 ==========================
app = FastAPI(title="铁路运输配载优化")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get(HEALTH_PATH)
def health_check():
    snapshot = SERVICE_STATE.snapshot()
    return {
        "status": snapshot["status"],
        "detail": snapshot["detail"],
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "build_version": BUILD_VERSION,
    }


@app.post("/api/v1/optimize")
async def optimize(req: OptimizationRequest, request: Request):
    request_id = uuid.uuid4().hex[:8]
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
        # 不再设置45秒接口等待上限：保持等待，直到算法线程正常结束。
        # 仅修改接口等待行为，不改变任何装箱、混装、约束、评分或均衡规则。
        result = await asyncio.shield(future)
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


# ========================== 网络/进程辅助 ==========================
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


def is_api_reachable(timeout: float = 1.5) -> bool:
    conn = None
    try:
        conn = http.client.HTTPConnection("127.0.0.1", API_PORT, timeout=timeout)
        conn.request("GET", HEALTH_PATH)
        resp = conn.getresponse()
        resp.read()
        return 200 <= resp.status < 500
    except Exception:
        return False
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


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

    deduped = []
    seen = set()
    for ip in candidates:
        if ip not in seen:
            seen.add(ip)
            deduped.append(ip)

    if deduped:
        return deduped
    return ["127.0.0.1"]


# ========================== 服务管理 ==========================
class ServerManager:
    def __init__(self):
        self.server: Optional[uvicorn.Server] = None
        self.server_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.server_exception: Optional[BaseException] = None
        self._lock = threading.Lock()

    def _server_worker(self):
        try:
            config = uvicorn.Config(
                app,
                host=API_HOST,
                port=API_PORT,
                log_config=None,
                access_log=False,
                lifespan="off",
            )
            self.server = uvicorn.Server(config)
            self.server.run()
        except BaseException as exc:
            self.server_exception = exc
            logger.exception("服务线程异常退出")
        finally:
            logger.info("服务线程已结束")

    def start(self) -> bool:
        with self._lock:
            if self.server_thread and self.server_thread.is_alive():
                SERVICE_STATE.set("运行中", f"服务已在 {API_PORT} 端口运行")
                return True

            if not can_bind_port("0.0.0.0", API_PORT):
                msg = f"端口 {API_PORT} 已被占用，服务未启动"
                SERVICE_STATE.set("启动失败", msg, msg)
                logger.error(msg)
                return False

            self.stop_event.clear()
            self.server_exception = None
            SERVICE_STATE.set("启动中", f"正在启动 {API_PORT} 端口服务")
            self.server_thread = threading.Thread(target=self._server_worker, name="uvicorn-server", daemon=False)
            self.server_thread.start()

        deadline = time.time() + SERVER_START_TIMEOUT
        while time.time() < deadline:
            if self.server_exception:
                msg = f"服务启动失败: {self.server_exception}"
                SERVICE_STATE.set("启动失败", msg, msg)
                return False
            if self.server and getattr(self.server, "started", False) and is_api_reachable():
                SERVICE_STATE.set("运行中", f"服务已监听 0.0.0.0:{API_PORT}")
                return True
            if self.server_thread and not self.server_thread.is_alive():
                msg = "服务线程已退出，启动失败"
                SERVICE_STATE.set("启动失败", msg, msg)
                return False
            time.sleep(0.2)

        msg = f"服务在 {SERVER_START_TIMEOUT:.0f} 秒内未完成启动"
        SERVICE_STATE.set("启动失败", msg, msg)
        logger.error(msg)
        return False

    def stop(self) -> None:
        with self._lock:
            SERVICE_STATE.set("停止中", "正在停止服务")
            if self.server:
                self.server.should_exit = True
            thread = self.server_thread

        if thread and thread.is_alive():
            thread.join(timeout=SERVER_STOP_TIMEOUT)

        if thread and thread.is_alive():
            msg = "服务未能在限定时间内正常停止"
            SERVICE_STATE.set("运行异常", msg, msg)
            logger.error(msg)
        else:
            SERVICE_STATE.set("已停止", "服务已停止")

    def monitor_loop(self):
        while not self.stop_event.is_set():
            time.sleep(SERVER_MONITOR_INTERVAL)
            snapshot = SERVICE_STATE.snapshot()
            if snapshot["status"] not in {"运行中", "运行异常"}:
                continue
            if self.server_thread and not self.server_thread.is_alive():
                msg = "检测到服务线程已退出"
                SERVICE_STATE.set("运行异常", msg, msg)
                logger.error(msg)
                continue
            if snapshot["status"] == "运行中" and not is_api_reachable():
                msg = "检测到服务健康检查失败"
                SERVICE_STATE.set("运行异常", msg, msg)
                logger.error(msg)

    def shutdown_monitor(self):
        self.stop_event.set()


SERVER_MANAGER = ServerManager()


class SingleInstanceGuard:
    def __init__(self, host: str = "127.0.0.1", port: int = INSTANCE_LOCK_PORT):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None

    def acquire(self) -> bool:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.bind((self.host, self.port))
            self.sock.listen(1)
            return True
        except OSError:
            if self.sock:
                self.sock.close()
            self.sock = None
            return False

    def release(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


INSTANCE_GUARD = SingleInstanceGuard()
atexit.register(INSTANCE_GUARD.release)
atexit.register(SERVER_MANAGER.shutdown_monitor)
atexit.register(ALGO_EXECUTOR.shutdown, wait=False, cancel_futures=False)


# ========================== Linux 无界面启动入口 ==========================
_LINUX_SHUTDOWN_EVENT = threading.Event()


def _linux_signal_handler(signum, _frame):
    """接收 systemd/docker/终端停止信号，在主循环中执行正常停机。"""
    logger.info("收到 Linux 停止信号: %s", signum)
    _LINUX_SHUTDOWN_EVENT.set()


def bootstrap() -> None:
    """Linux 服务器无界面启动。

    仅替换原来的 Tk/系统托盘启动层；FastAPI 接口、算法、业务规则、
    并发控制、日志和输入输出结构保持不变。
    """
    if not INSTANCE_GUARD.acquire():
        msg = "检测到程序已在运行，请勿重复启动。"
        logger.warning(msg)
        print(msg)
        return

    logger.info("Linux 无界面模式启动，目录=%s，版本=%s", APP_DIR, BUILD_VERSION)

    monitor_thread = threading.Thread(
        target=SERVER_MANAGER.monitor_loop,
        name="server-monitor",
        daemon=True,
    )
    monitor_thread.start()

    started = SERVER_MANAGER.start()
    if not started:
        snapshot = SERVICE_STATE.snapshot()
        logger.error("服务启动失败：%s", snapshot["detail"])
        INSTANCE_GUARD.release()
        return

    addresses = detect_local_ipv4_addresses()
    logger.info("接口已启动：http://0.0.0.0:%s/api/v1/optimize", API_PORT)
    for ip in addresses:
        logger.info("可访问地址：http://%s:%s/api/v1/optimize", ip, API_PORT)

    # systemd 默认使用 SIGTERM；终端 Ctrl+C 为 SIGINT。
    signal.signal(signal.SIGTERM, _linux_signal_handler)
    signal.signal(signal.SIGINT, _linux_signal_handler)

    try:
        while not _LINUX_SHUTDOWN_EVENT.is_set():
            thread = SERVER_MANAGER.server_thread
            if thread is None or not thread.is_alive():
                break
            time.sleep(0.5)
    finally:
        SERVER_MANAGER.shutdown_monitor()
        try:
            SERVER_MANAGER.stop()
        except Exception:
            logger.exception("停止服务时发生异常")
        INSTANCE_GUARD.release()
        logger.info("Linux 服务已退出")


if __name__ == "__main__":
    bootstrap()
