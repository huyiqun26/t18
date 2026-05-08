import asyncio
import atexit
import logging
import math
from collections import defaultdict
from logging.handlers import RotatingFileHandler
import os
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
from pydantic import BaseModel, ConfigDict


# ========================== 基础配置 ==========================
APP_TITLE = "铁路运输配载优化 API"
API_HOST = os.getenv("RAILWAY_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("RAILWAY_API_PORT", "2376"))
HEALTH_PATH = "/health"
LOG_FILE_NAME = "railway_service_linux.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
MAX_ALGO_WORKERS = 2
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
    organizationID: str
    organizationName: str
    Unitclass: str
    belongTo: str
    personCount: int
    componentList: List[Dict[str, Any]]
    goodsList: List[Dict[str, Any]]


class OptimizationRequest(FlexibleModel):
    systemSettings: Dict[str, Any]
    data: List[Organization]


def model_to_payload(model: Any) -> Dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        return dumper()

    dumper = getattr(model, "dict", None)
    if callable(dumper):
        return dumper()

    raise TypeError(f"不支持的请求模型类型: {type(model)}")


# ========================== 算法核心（已替换为 qifa_yingji_chaoxian_rule 结构） ==========================
class SubContainer:
    def __init__(self, box_type, length_unit, weight_empty, max_capacity, capacity_type='count', category=None):
        self.box_type = box_type
        self.length_unit = float(length_unit)
        self.weight = float(weight_empty)
        self.max_capacity = max_capacity
        self.capacity_type = capacity_type
        self.current_load = 0.0
        self.contents: List[Dict[str, Any]] = []
        self.owners = set()
        self.equip_category = category

    def add_item(self, company_id, item_info, item_weight, item_load_value):
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


def get_company_yingji_name(comp):
    def normalize_yingji(value):
        if value is None:
            return ''
        s = str(value).strip()
        mapping = {
            '1': '1', '2': '2', '3': '3',
            '一年级': '1', '二年级': '2', '三年级': '3',
            '一': '1', '二': '2', '三': '3',
            '第1年级': '1', '第2年级': '2', '第3年级': '3',
            '第一级': '1', '第二级': '2', '第三级': '3',
        }
        if s in mapping:
            return mapping[s]
        digits = ''.join(ch for ch in s if ch.isdigit())
        if digits in ('1', '2', '3'):
            return digits
        return ''

    if 'yingjiName' in comp:
        y = normalize_yingji(comp.get('yingjiName'))
        if y in ('1', '2', '3'):
            return y
        raise AlgorithmError(
            f"公司 {comp.get('organizationID', '')} 的 yingjiName={comp.get('yingjiName')} 非法，必须为字符串 '1'/'2'/'3'"
        )

    u_class = safe_int(comp.get('Unitclass'), 0)
    if u_class in (1, 2, 3):
        return str(u_class)
    return ''


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
    if getattr(box, 'box_type', '') != 'Equip_Box_Large':
        return False
    for item in getattr(box, 'contents', []):
        if item.get('type') == 'component' and normalize_is_chaoxian(item.get('is_chaoXian', '')) == '是':
            return True
    return False


def box_chaoxian_owners(box):
    owners = set()
    if getattr(box, 'box_type', '') != 'Equip_Box_Large':
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
            if y in ('1', '2', '3') and len(cids) > 0
        }
        unit_yingji_names = {
            company_yingji_name.get(cid, '') for cid in unit['owners']
            if company_yingji_name.get(cid, '') in ('1', '2', '3')
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
            if yingji_name in ('1', '2', '3'):
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
                if yingji_name in ('1', '2', '3'):
                    self.yingji_companies[yingji_name].add(cid)
            if u.get('has_chaoXian_equipment'):
                self.chaoXian_companies.update(u.get('chaoXian_owners', set()))


def run_engine(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sys_settings = raw_data.get('systemSettings', {})
        sc_limit = sys_settings.get('SC_Constraint', {'maxWeightLimit': 60000, 'maxLengthLimit': 800.0})
        person_weight = sys_settings.get('Person_Weight', {'weight_per_person': 75.0})['weight_per_person']
        box_specs = sys_settings.get('Box_Specs', {})

        max_weight_per_sc = float(sc_limit['maxWeightLimit'])
        max_length_per_sc = float(sc_limit['maxLengthLimit'])

        all_sub_containers = []
        open_person_boxes = defaultdict(list)
        open_large_boxes = defaultdict(list)
        open_small_boxes = defaultdict(list)

        companies = raw_data.get('data', [])
        company_yingji_name = {}
        company_name = {}
        for comp in companies:
            cid = comp['organizationID']
            company_yingji_name[cid] = get_company_yingji_name(comp)
            company_name[cid] = comp.get('organizationName', '')

        missing_yingji_name = [cid for cid, y in company_yingji_name.items() if y not in ('1', '2', '3')]
        if missing_yingji_name:
            print('提示：以下公司未提供有效 yingjiName="1"/"2"/"3"，装车时不参与yingjiName种类数限制：')
            print(', '.join(missing_yingji_name[:20]) + ('...' if len(missing_yingji_name) > 20 else ''))

        def add_people_to_boxes(b_type, num_people, owner_id):
            spec = box_specs.get(b_type)
            if not spec or num_people <= 0:
                return 0
            cap = spec['capacity']
            remaining = int(num_people)
            added_total = 0

            def create_person_info(count):
                return {
                    "type": "person",
                    "company_id": owner_id,
                    "box_type": b_type,
                    "count": int(count)
                }

            for box in open_person_boxes[owner_id]:
                if box.box_type == b_type and box.current_load < box.max_capacity:
                    space = box.max_capacity - box.current_load
                    to_add = min(remaining, int(space))
                    if to_add > 0:
                        box.add_item(owner_id, create_person_info(to_add), to_add * person_weight, to_add)
                        remaining -= to_add
                        added_total += to_add
                    if remaining <= 0:
                        break

            while remaining > 0:
                to_add = min(remaining, cap)
                new_box = SubContainer(b_type, spec['length_unit'], spec['weight'], cap, 'count')
                new_box.add_item(owner_id, create_person_info(to_add), to_add * person_weight, to_add)
                all_sub_containers.append(new_box)
                open_person_boxes[owner_id].append(new_box)
                remaining -= to_add
                added_total += to_add

            return added_total

        for comp in companies:
            cid = comp['organizationID']
            u_class = safe_int(comp.get('Unitclass'), 5)
            p_count = safe_int(comp.get('personCount'), 0)
            remaining_p = p_count

            if u_class == 2:
                c2_cap = box_specs.get('Person_Box_C2', {}).get('capacity', 40)
                c2_quota = 2 * c2_cap
                if remaining_p > 0:
                    to_c2 = min(remaining_p, c2_quota)
                    add_people_to_boxes('Person_Box_C2', to_c2, cid)
                    remaining_p -= to_c2
                if remaining_p > 0:
                    add_people_to_boxes('Person_Box_C3', remaining_p, cid)
            else:
                if u_class == 1:
                    mandatory = {'Person_Box_C1': 1, 'Person_Box_C2': 3, 'Person_Box_C3': 1}
                elif u_class == 3:
                    mandatory = {'Person_Box_C2': 2, 'Person_Box_C3': 1}
                elif u_class == 4:
                    mandatory = {'Person_Box_C2': 1, 'Person_Box_C3': 1}
                else:
                    mandatory = {'Person_Box_C3': 1}

                for b_type, count in mandatory.items():
                    for _ in range(count):
                        if remaining_p > 0:
                            cap = box_specs.get(b_type, {}).get('capacity', 0)
                            to_add = min(remaining_p, cap)
                            add_people_to_boxes(b_type, to_add, cid)
                            remaining_p -= to_add
                        else:
                            add_people_to_boxes(b_type, 0, cid)

                if remaining_p > 0:
                    add_people_to_boxes('Person_Box_C3', remaining_p, cid)

            comps_list = comp.get('componentList', [])
            comps_list.sort(key=lambda x: x.get('needcarLarge', 0), reverse=True)
            spec_large = box_specs.get('Equip_Box_Large')
            if spec_large is None:
                raise AlgorithmError('缺少 Equip_Box_Large 配置')

            for item in comps_list:
                name = item['componentname']
                w = float(item['componentweight'])
                occupancy = float(item.get('needcarLarge', 1.0))

                item_info = {
                    "type": "component",
                    "company_id": cid,
                    "componentname": name,
                    "componentID": item.get('componentID', ''),
                    "zzbdid": item.get('zzbdid', ''),
                    "bddxid": item.get('bddxid', ''),
                    "dxcode": item.get('dxcode', ''),
                    "is_chaoXian": normalize_is_chaoxian(item.get('is_chaoXian', '')),
                    "occupancy": occupancy
                }

                best_box = None
                min_rem = 1.0
                for box in open_large_boxes[cid]:
                    if box.current_load + occupancy <= 1.001:
                        rem = 1.0 - (box.current_load + occupancy)
                        if rem < min_rem:
                            min_rem = rem
                            best_box = box

                if best_box:
                    best_box.add_item(cid, item_info, w, occupancy)
                else:
                    new_box = SubContainer('Equip_Box_Large', spec_large['length_unit'], spec_large['weight'], 1.0, 'occupancy')
                    new_box.add_item(cid, item_info, w, occupancy)
                    all_sub_containers.append(new_box)
                    open_large_boxes[cid].append(new_box)

            goods_list = comp.get('goodsList', [])
            flat_goods = []
            for g in goods_list:
                count = safe_int(g.get('count', 1), 1)
                for _ in range(count):
                    flat_goods.append(g.copy())

            flat_goods.sort(key=lambda x: x.get('tj', 0), reverse=True)
            spec_small = box_specs.get('Equip_Box_Small')

            if spec_small:
                max_vol = float(spec_small.get('capacity_volume', 120.0))
                for item in flat_goods:
                    name = item['name']
                    w = float(item['weight'])
                    vol = float(item.get('tj', 0))
                    cat = item.get('category', '未分类')

                    item_info = {
                        "type": "goods",
                        "company_id": cid,
                        "name": name,
                        "ID": item.get('ID', ''),
                        "category": cat,
                        "count": 1
                    }

                    key = (cid, cat)
                    placed = False
                    for box in open_small_boxes[key]:
                        if box.current_load + vol <= max_vol + 1e-6:
                            box.add_item(cid, item_info, w, vol)
                            placed = True
                            break

                    if not placed:
                        new_box = SubContainer(
                            'Equip_Box_Small',
                            spec_small['length_unit'],
                            spec_small['weight'],
                            max_vol,
                            'volume',
                            category=cat
                        )
                        new_box.add_item(cid, item_info, w, vol)
                        all_sub_containers.append(new_box)
                        open_small_boxes[key].append(new_box)

        logger.info("预处理完成，生成小箱总数=%s", len(all_sub_containers))
        original_boxes = all_sub_containers.copy()

        for i, box in enumerate(original_boxes):
            if box.weight > max_weight_per_sc + 1e-6 or box.length_unit > max_length_per_sc + 1e-6:
                raise AlgorithmError(
                    f"Box_{i + 1:04d} 自身超过单车限制：weight={box.weight:.1f}, length={box.length_unit:.2f}"
                )

        def create_merged_box(box_list):
            first = box_list[0]
            if first.box_type.startswith('Person_Box'):
                merged = SubContainer(first.box_type, 0.0, 0.0, first.max_capacity, first.capacity_type)
            elif first.box_type == 'Equip_Box_Small':
                merged = SubContainer(first.box_type, 0.0, 0.0, first.max_capacity, first.capacity_type, category=first.equip_category)
            else:
                raise ValueError(f"不支持合并的箱子类型: {first.box_type}")

            merged.length_unit = sum(b.length_unit for b in box_list)
            merged.weight = sum(b.weight for b in box_list)
            merged.owners = first.owners.copy()
            merged.contents = []
            for b in box_list:
                merged.contents.extend(b.contents)
            return merged

        def can_merge(box_list):
            total_len = sum(b.length_unit for b in box_list)
            total_weight = sum(b.weight for b in box_list)
            return (total_len <= max_length_per_sc + 1e-6 and total_weight <= max_weight_per_sc + 1e-6)

        group_dict = defaultdict(list)
        for idx, box in enumerate(original_boxes):
            if box.is_mixed:
                group_dict[('mixed', idx)].append(idx)
            else:
                cid = list(box.owners)[0]
                cat = getattr(box, 'equip_category', None)
                if box.box_type == 'Equip_Box_Large':
                    group_dict[('large', idx)].append(idx)
                else:
                    group_dict[(cid, box.box_type, cat)].append(idx)

        merged_boxes = []
        merge_map = []

        for key, indices in group_dict.items():
            if key[0] in ('mixed', 'large'):
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
            def can_add_to_chunk(chunk, box_idx):
                b = all_sub_containers[box_idx]
                return (chunk['weight'] + b.weight <= max_weight_per_sc + 1e-6 and
                        chunk['length'] + b.length_unit <= max_length_per_sc + 1e-6)

            def add_box_to_chunk(chunk, box_idx):
                b = all_sub_containers[box_idx]
                chunk['box_indices'].append(box_idx)
                chunk['weight'] += b.weight
                chunk['length'] += b.length_unit
                chunk['dominant'] = dominant_ratio(chunk['weight'], chunk['length'], max_weight_per_sc, max_length_per_sc)
                chunk['chaoXian_owners'].update(box_chaoxian_owners(b))
                chunk['has_chaoXian_equipment'] = len(chunk['chaoXian_owners']) > 0

            def best_chunk_for_box(chunks, box_idx, prefer_chao_chunk=False):
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
                    if best_score is None or score > best_score:
                        best_score = score
                        best_k = k
                return best_k

            chao_indices = [i for i in box_indices if box_has_chaoxian_equipment(all_sub_containers[i])]
            normal_indices = [i for i in box_indices if i not in set(chao_indices)]
            chunks = []

            ordered_chao = sorted(
                chao_indices,
                key=lambda i: (
                    dominant_ratio(all_sub_containers[i].weight, all_sub_containers[i].length_unit,
                                   max_weight_per_sc, max_length_per_sc),
                    all_sub_containers[i].length_unit,
                    all_sub_containers[i].weight
                ),
                reverse=True
            )
            for idx in ordered_chao:
                best_k = best_chunk_for_box(chunks, idx, prefer_chao_chunk=True)
                if best_k is None:
                    chunks.append(make_unit([idx], forced_owners={cid}))
                else:
                    add_box_to_chunk(chunks[best_k], idx)

            ordered_normal = sorted(
                normal_indices,
                key=lambda i: (
                    dominant_ratio(all_sub_containers[i].weight, all_sub_containers[i].length_unit,
                                   max_weight_per_sc, max_length_per_sc),
                    all_sub_containers[i].length_unit,
                    all_sub_containers[i].weight
                ),
                reverse=True
            )
            for idx in ordered_normal:
                best_k = best_chunk_for_box(chunks, idx, prefer_chao_chunk=False)
                if best_k is None:
                    chunks.append(make_unit([idx], forced_owners={cid}))
                else:
                    add_box_to_chunk(chunks[best_k], idx)

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

        units.sort(key=lambda u: (u.get('has_chaoXian_equipment', False), u['dominant'], u['length'], u['weight']), reverse=True)
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
                    if y in ('1', '2', '3') and len(cids) > 0
                )
                if len(used_yingji_names) > 2:
                    raise AlgorithmError(f"SC_{v + 1:03d} yingjiName种类数超限: used_yingjiNames={used_yingji_names}")

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
            for i in merged_indices:
                for orig_idx in merge_map[i]:
                    orig_box = original_boxes[orig_idx]
                    owners_set.update(orig_box.owners)
                    curr_w += orig_box.weight
                    curr_l += orig_box.length_unit
                    if orig_box.is_mixed:
                        has_mixed = True

                    entity_desc = build_entities(orig_box, company_yingji_name)

                    box_yingji_names = sorted({
                        company_yingji_name.get(cid, '') for cid in orig_box.owners
                        if company_yingji_name.get(cid, '') in ('1', '2', '3')
                    })
                    box_dict = {
                        "box_id": f"Box_{orig_idx + 1:04d}",
                        "box_type": orig_box.box_type,
                        "is_mixed": orig_box.is_mixed,
                        "owners": list(orig_box.owners),
                        "yingjiName": ';'.join(box_yingji_names),
                        "is_chaoXian": "是" if box_has_chaoxian_equipment(orig_box) else ("否" if orig_box.box_type == 'Equip_Box_Large' else ""),
                        "content_desc": entity_desc,
                        "weight": round(orig_box.weight, 1),
                        "length_unit": round(orig_box.length_unit, 2)
                    }

                    if orig_box.box_type == 'Equip_Box_Small':
                        box_dict["category"] = getattr(orig_box, 'equip_category', '未分类')

                    sc_info["box_list"].append(box_dict)

            yingji_names_in_sc = sorted({
                company_yingji_name.get(cid, '') for cid in owners_set
                if company_yingji_name.get(cid, '') in ('1', '2', '3')
            })
            yingji_company_distribution = {}
            for y in yingji_names_in_sc:
                yingji_company_distribution[y] = len([cid for cid in owners_set if company_yingji_name.get(cid, '') == y])

            chao_companies_in_sc = sorted({
                entity.get('company_id', '')
                for box in sc_info['box_list']
                for entity in box.get('content_desc', [])
                if entity.get('type') == 'component' and entity.get('is_chaoXian', '') == '是'
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

        validate_output_result(res_data, company_yingji_name, max_weight_per_sc, max_length_per_sc)
        return res_data
    except AlgorithmError as exc:
        logger.warning("算法输入错误: %s", exc)
        return {"code": 1, "msg": str(exc)}
    except Exception:
        logger.exception("算法执行失败")
        return {"code": 1, "msg": "算法执行失败，请查看日志"}


def validate_output_result(res_data, company_yingji_name, max_weight_per_sc, max_length_per_sc):
    for sc in res_data.get('data', {}).get('SC_list', []):
        sid = sc.get('SC_ID', '')
        summary = sc.get('summary', {})
        total_w = float(summary.get('total_weight', 0.0))
        total_l = float(summary.get('total_length_unit', 0.0))
        if total_w > max_weight_per_sc + 1e-6:
            raise AlgorithmError(f"{sid} 出参校验超重: {total_w:.1f} > {max_weight_per_sc:.1f}")
        if total_l > max_length_per_sc + 1e-6:
            raise AlgorithmError(f"{sid} 出参校验超换长: {total_l:.2f} > {max_length_per_sc:.2f}")

        used_yingji_names = set()
        for cid in summary.get('companies_included', []):
            yingji_name = company_yingji_name.get(cid, '')
            if yingji_name in ('1', '2', '3'):
                used_yingji_names.add(yingji_name)
        if len(used_yingji_names) > 2:
            raise AlgorithmError(f"{sid} 出参校验yingjiName种类数超限: used_yingjiNames={sorted(used_yingji_names)}")


def build_entities(box, company_yingji_name=None):
    if not box.contents:
        return []

    entities = []
    if box.box_type.startswith('Person_Box'):
        merged = {}
        for item in box.contents:
            key = (item['company_id'], item['box_type'])
            if key not in merged:
                merged[key] = {
                    "type": "person",
                    "company_id": item['company_id'],
                    "yingjiName": (company_yingji_name or {}).get(item['company_id'], ''),
                    "box_type": item['box_type'],
                    "count": 0
                }
            merged[key]['count'] += item['count']
        entities = list(merged.values())

    elif box.box_type == 'Equip_Box_Large':
        for item in box.contents:
            entity = {
                "type": "component",
                "company_id": item['company_id'],
                "yingjiName": (company_yingji_name or {}).get(item['company_id'], ''),
                "componentname": item['componentname'],
                "componentID": item['componentID'],
                "is_chaoXian": normalize_is_chaoxian(item.get('is_chaoXian', '')),
                "zzbdid": item.get('zzbdid', ''),
                "bddxid": item.get('bddxid', ''),
                "dxcode": item.get('dxcode', ''),
                "occupancy": item['occupancy']
            }
            entities.append(entity)

    elif box.box_type == 'Equip_Box_Small':
        merged = {}
        for item in box.contents:
            key = (item['company_id'], item['name'], item['ID'], item['category'])
            if key not in merged:
                merged[key] = {
                    "type": "goods",
                    "company_id": item['company_id'],
                    "yingjiName": (company_yingji_name or {}).get(item['company_id'], ''),
                    "name": item['name'],
                    "ID": item['ID'],
                    "category": item['category'],
                    "count": 0
                }
            merged[key]['count'] += item.get('count', 1)
        entities = list(merged.values())

    return entities


ALGO_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_ALGO_WORKERS, thread_name_prefix="algo-worker")
ALGO_GATE = threading.BoundedSemaphore(MAX_ALGO_WORKERS)
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
