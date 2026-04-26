import asyncio
import atexit
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from functools import partial
import signal
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ortools.linear_solver import pywraplp
from pydantic import BaseModel, ConfigDict


# ========================== 基础配置 ==========================
APP_TITLE = "铁路运输配载优化 API"
API_HOST = os.getenv("RAILWAY_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("RAILWAY_API_PORT", "2376"))
HEALTH_PATH = "/health"
LOG_FILE_NAME = "railway_service_linux.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
MAX_ALGO_WORKERS = 1
DEFAULT_SOLVER_TIME_LIMIT_MS = 30000


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


sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)


def handle_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        return
    logger.exception("未捕获异常", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_uncaught_exception


def _threading_excepthook(args) -> None:
    logger.exception("线程未捕获异常", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


if hasattr(threading := __import__("threading"), "excepthook"):
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


# ========================== 算法核心 ==========================
class SubContainer:
    def __init__(self, box_type, length_unit, weight_empty, max_capacity, capacity_type="count", category=None):
        self.box_type = box_type
        self.length_unit = length_unit
        self.weight = weight_empty
        self.max_capacity = max_capacity
        self.capacity_type = capacity_type
        self.current_load = 0.0
        self.contents: List[Dict[str, Any]] = []
        self.owners = set()
        self.equip_category = category

    def add_item(self, company_id, item_info, item_weight, item_load_value):
        if self.current_load + item_load_value <= self.max_capacity + 1e-6:
            self.current_load += item_load_value
            self.weight += item_weight
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


def _get_objective_value(solver: pywraplp.Solver) -> float:
    obj = solver.Objective()
    if hasattr(obj, "Value"):
        return obj.Value()
    return obj.value()


def build_entities(box: "SubContainer") -> List[Dict[str, Any]]:
    """
    将 box.contents 转换为 4.23 结构下的实体列表。
    """
    if not box.contents:
        return []

    entities: List[Dict[str, Any]] = []

    if box.box_type.startswith("Person_Box"):
        merged: Dict[tuple, Dict[str, Any]] = {}
        for item in box.contents:
            key = (item["company_id"], item["box_type"])
            if key not in merged:
                merged[key] = {
                    "type": "person",
                    "company_id": item["company_id"],
                    "box_type": item["box_type"],
                    "count": 0,
                }
            merged[key]["count"] += int(item.get("count", 0))
        entities = list(merged.values())

    elif box.box_type == "Equip_Box_Large":
        for item in box.contents:
            entity = {
                "type": "component",
                "company_id": item["company_id"],
                "componentname": item["componentname"],
                "componentID": item.get("componentID", ""),
                "zzbdid": item.get("zzbdid", ""),
                "bddxid": item.get("bddxid", ""),
                "dxcode": item.get("dxcode", ""),
                "occupancy": item["occupancy"],
            }
            entities.append(entity)

    elif box.box_type == "Equip_Box_Small":
        merged = {}
        for item in box.contents:
            key = (item["company_id"], item["name"], item["ID"], item["category"])
            if key not in merged:
                merged[key] = {
                    "type": "goods",
                    "company_id": item["company_id"],
                    "name": item["name"],
                    "ID": item["ID"],
                    "category": item["category"],
                    "count": 0,
                }
            merged[key]["count"] += int(item.get("count", 1))
        entities = list(merged.values())

    return entities


def run_engine(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sys_settings = raw_data.get("systemSettings", {})
        sc_limit = sys_settings.get("SC_Constraint", {"maxWeightLimit": 60000, "maxLengthLimit": 800.0})
        person_weight = sys_settings.get("Person_Weight", {"weight_per_person": 75.0}).get("weight_per_person", 75.0)
        box_specs = sys_settings.get("Box_Specs", {})

        required_specs = ["Person_Box_C1", "Person_Box_C2", "Person_Box_C3", "Equip_Box_Large", "Equip_Box_Small"]
        missing_specs = [name for name in required_specs if name not in box_specs]
        if missing_specs:
            raise AlgorithmError(f"缺少箱型配置: {', '.join(missing_specs)}")

        all_sub_containers: List[SubContainer] = []
        open_person_boxes: List[SubContainer] = []
        open_large_boxes: List[SubContainer] = []
        open_small_boxes: Dict[str, List[SubContainer]] = {}

        def add_people_to_boxes(b_type: str, num_people: int, owner_id: str) -> int:
            spec = box_specs.get(b_type)
            if not spec or num_people <= 0:
                return 0

            cap = spec["capacity"]
            remaining = num_people
            added_total = 0

            def create_person_info(count: int) -> Dict[str, Any]:
                return {
                    "type": "person",
                    "company_id": owner_id,
                    "box_type": b_type,
                    "count": int(count),
                }

            for box in open_person_boxes:
                if box.box_type == b_type and box.current_load < box.max_capacity:
                    available_space = box.max_capacity - box.current_load
                    to_add = min(remaining, available_space)
                    if to_add > 0:
                        box.add_item(owner_id, create_person_info(to_add), to_add * person_weight, to_add)
                        remaining -= to_add
                        added_total += to_add
                    if remaining <= 0:
                        break

            while remaining > 0:
                to_add = min(remaining, cap)
                new_box = SubContainer(b_type, spec["length_unit"], spec["weight"], cap, "count")
                new_box.add_item(owner_id, create_person_info(to_add), to_add * person_weight, to_add)
                all_sub_containers.append(new_box)
                open_person_boxes.append(new_box)
                remaining -= to_add
                added_total += to_add

            return added_total

        companies = raw_data.get("data", [])
        for comp in companies:
            cid = comp["organizationID"]
            try:
                u_class = int(comp["Unitclass"])
            except Exception:
                u_class = 5

            remaining_p = int(comp.get("personCount", 0))

            if u_class == 2:
                c2_cap = box_specs.get("Person_Box_C2", {}).get("capacity", 40)
                c2_quota = 2 * c2_cap
                if remaining_p > 0:
                    to_c2 = min(remaining_p, c2_quota)
                    add_people_to_boxes("Person_Box_C2", to_c2, cid)
                    remaining_p -= to_c2
                if remaining_p > 0:
                    add_people_to_boxes("Person_Box_C3", remaining_p, cid)
            else:
                if u_class == 1:
                    mandatory = {"Person_Box_C1": 1, "Person_Box_C2": 3, "Person_Box_C3": 1}
                elif u_class == 3:
                    mandatory = {"Person_Box_C2": 2, "Person_Box_C3": 1}
                elif u_class == 4:
                    mandatory = {"Person_Box_C2": 1, "Person_Box_C3": 1}
                else:
                    mandatory = {"Person_Box_C3": 1}

                for b_type, count in mandatory.items():
                    for _ in range(count):
                        if remaining_p > 0:
                            cap = box_specs.get(b_type, {}).get("capacity", 0)
                            to_add = min(remaining_p, cap)
                            add_people_to_boxes(b_type, to_add, cid)
                            remaining_p -= to_add
                        else:
                            add_people_to_boxes(b_type, 0, cid)

                if remaining_p > 0:
                    add_people_to_boxes("Person_Box_C3", remaining_p, cid)

            comps_list = comp.get("componentList", [])
            comps_list.sort(key=lambda x: x.get("needcarLarge", 0), reverse=True)
            spec_large = box_specs.get("Equip_Box_Large")
            if spec_large is None:
                raise AlgorithmError("缺少 Equip_Box_Large 配置")

            for item in comps_list:
                occupancy = item.get("needcarLarge", 1.0)
                item_info = {
                    "type": "component",
                    "company_id": cid,
                    "componentname": item.get("componentname", ""),
                    "componentID": item.get("componentID", ""),
                    "zzbdid": item.get("zzbdid", ""),
                    "bddxid": item.get("bddxid", ""),
                    "dxcode": item.get("dxcode", ""),
                    "occupancy": occupancy,
                }
                item_weight = item.get("componentweight", 0)

                best_box = None
                min_rem = 1.0
                for box in open_large_boxes:
                    if box.current_load + occupancy <= 1.001:
                        rem = 1.0 - (box.current_load + occupancy)
                        if rem < min_rem:
                            min_rem = rem
                            best_box = box

                if best_box:
                    best_box.add_item(cid, item_info, item_weight, occupancy)
                else:
                    new_box = SubContainer(
                        "Equip_Box_Large",
                        spec_large["length_unit"],
                        spec_large["weight"],
                        1.0,
                        "occupancy",
                    )
                    new_box.add_item(cid, item_info, item_weight, occupancy)
                    all_sub_containers.append(new_box)
                    open_large_boxes.append(new_box)

            goods_list = comp.get("goodsList", [])
            flat_goods: List[Dict[str, Any]] = []
            for g in goods_list:
                count = int(g.get("count", 1))
                for _ in range(count):
                    flat_goods.append(g.copy())

            flat_goods.sort(key=lambda x: x.get("tj", 0), reverse=True)
            spec_small = box_specs.get("Equip_Box_Small")
            if spec_small is None:
                raise AlgorithmError("缺少 Equip_Box_Small 配置")
            max_vol = spec_small.get("capacity_volume", 120.0)

            for item in flat_goods:
                cat = item.get("category", "未分类")
                item_info = {
                    "type": "goods",
                    "company_id": cid,
                    "name": item.get("name", ""),
                    "ID": item.get("ID", ""),
                    "category": cat,
                    "count": 1,
                }

                item_weight = item.get("weight", 0)
                vol = item.get("tj", 0)

                if cat not in open_small_boxes:
                    open_small_boxes[cat] = []

                placed = False
                for box in open_small_boxes[cat]:
                    if box.current_load + vol <= max_vol + 1e-6:
                        box.add_item(cid, item_info, item_weight, vol)
                        placed = True
                        break

                if not placed:
                    new_box = SubContainer(
                        "Equip_Box_Small",
                        spec_small["length_unit"],
                        spec_small["weight"],
                        max_vol,
                        "volume",
                        category=cat,
                    )
                    new_box.add_item(cid, item_info, item_weight, vol)
                    all_sub_containers.append(new_box)
                    open_small_boxes[cat].append(new_box)

        logger.info("预处理完成，生成小箱总数=%s", len(all_sub_containers))

        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            return {"code": 1, "msg": "求解器未启动"}

        total_len = sum(b.length_unit for b in all_sub_containers)
        max_sc_len = sc_limit["maxLengthLimit"]
        est_bins = int(total_len / max_sc_len) + 20

        sc_ids = range(est_bins)
        box_ids = range(len(all_sub_containers))
        comp_map = {c["organizationID"]: idx for idx, c in enumerate(companies)}
        comp_ids = comp_map.values()

        x = {}
        y = {}
        z = {}

        for i in box_ids:
            vars_in_row = []
            for j in sc_ids:
                x[i, j] = solver.IntVar(0, 1, "")
                vars_in_row.append(x[i, j])
            solver.Add(sum(vars_in_row) == 1)

        for j in sc_ids:
            y[j] = solver.IntVar(0, 1, "")
            for i in box_ids:
                solver.Add(x[i, j] <= y[j])

            solver.Add(sum(x[i, j] * all_sub_containers[i].weight for i in box_ids) <= sc_limit["maxWeightLimit"] * y[j])
            solver.Add(sum(x[i, j] * all_sub_containers[i].length_unit for i in box_ids) <= sc_limit["maxLengthLimit"] * y[j])

        for c_idx in comp_ids:
            cid = companies[c_idx]["organizationID"]
            my_boxes = [i for i, b in enumerate(all_sub_containers) if cid in b.owners]

            sc_usage = []
            for j in sc_ids:
                z[c_idx, j] = solver.IntVar(0, 1, "")
                if my_boxes:
                    solver.Add(sum(x[i, j] for i in my_boxes) <= 1000 * z[c_idx, j])
                sc_usage.append(z[c_idx, j])

            solver.Add(sum(sc_usage) <= 2)

        for j in range(len(sc_ids) - 1):
            solver.Add(y[j] >= y[j + 1])

        solver.Minimize(sum(y[j] for j in sc_ids))
        solver.set_time_limit(300000)
        logger.info("算法开始求解，箱体数=%s，预计车次数=%s", len(all_sub_containers), est_bins)
        status = solver.Solve()

        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return {"code": 1, "msg": "计算无有效解"}

        res_data: Dict[str, Any] = {
            "code": 0,
            "msg": "success",
            "data": {
                "total_SC_used": int(_get_objective_value(solver)),
                "SC_list": [],
            },
        }

        for j in sc_ids:
            if y[j].solution_value() > 0.5:
                sc_info: Dict[str, Any] = {
                    "SC_ID": f"SC_{j + 1:03d}",
                    "summary": {},
                    "box_list": [],
                }

                owners_set = set()
                curr_w = 0.0
                curr_l = 0.0
                has_mixed = False

                for i in box_ids:
                    if x[i, j].solution_value() > 0.5:
                        box = all_sub_containers[i]
                        owners_set.update(box.owners)
                        curr_w += box.weight
                        curr_l += box.length_unit
                        if box.is_mixed:
                            has_mixed = True

                        box_dict: Dict[str, Any] = {
                            "box_id": f"Box_{i + 1:04d}",
                            "box_type": box.box_type,
                            "is_mixed": box.is_mixed,
                            "owners": list(box.owners),
                            "content_desc": build_entities(box),
                            "weight": round(box.weight, 1),
                            "length_unit": round(box.length_unit, 2),
                        }

                        if box.box_type == "Equip_Box_Small":
                            box_dict["category"] = getattr(box, "equip_category", "未分类")

                        sc_info["box_list"].append(box_dict)

                owner_list = sorted(owners_set)
                sc_info["summary"] = {
                    "companies_included": owner_list,
                    "total_weight": round(curr_w, 1),
                    "total_length_unit": round(curr_l, 2),
                    "has_mixed_box": has_mixed,
                    "description": f"包含 {len(owner_list)} 个公司: {','.join(owner_list[:3])}... 共 {len(sc_info['box_list'])} 个小箱",
                }

                res_data["data"]["SC_list"].append(sc_info)

        return res_data
    except AlgorithmError as exc:
        logger.warning("算法输入错误: %s", exc)
        return {"code": 1, "msg": str(exc)}
    except Exception:
        logger.exception("算法执行失败")
        return {"code": 1, "msg": "算法执行失败，请查看日志"}

# ========================== FastAPI ==========================
ALGO_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_ALGO_WORKERS, thread_name_prefix="algo-worker")
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
async def optimize(req: OptimizationRequest):
    try:
        payload = model_to_payload(req)
        loop = asyncio.get_running_loop()
        task = partial(run_engine, payload)
        result = await loop.run_in_executor(ALGO_EXECUTOR, task)
        if result.get("code") != 0:
            raise HTTPException(status_code=400, detail=result.get("msg", "算法执行失败"))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("接口调用失败")
        raise HTTPException(status_code=500, detail=f"接口处理失败: {exc}")


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
