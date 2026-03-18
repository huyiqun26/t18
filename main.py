import logging
import socket
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import uvicorn
from ortools.linear_solver import pywraplp
from collections import Counter
import sys
import os

# ============================
# Linux ARM 适配：关闭无用输出
# ============================
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# ============================
# 日志配置
# ============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('railway_optimization.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# ============================
# 数据模型
# ============================
class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra='allow')

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

# ============================
# 算法核心（完全不变）
# ============================
class SubContainer:
    def __init__(self, box_type, length_unit, weight_empty, max_capacity, capacity_type='count'):
        self.box_type = box_type
        self.length_unit = length_unit
        self.weight = weight_empty
        self.max_capacity = max_capacity
        self.capacity_type = capacity_type
        self.current_load = 0.0
        self.contents = []
        self.owners = set()

    def add_item(self, company_id, item_desc, item_weight, item_load_value):
        if self.current_load + item_load_value <= self.max_capacity + 1e-6:
            self.current_load += item_load_value
            self.weight += item_weight
            self.contents.append(f"{company_id}:{item_desc}")
            self.owners.add(company_id)
            return True
        return False

    @property
    def is_mixed(self):
        return len(self.owners) > 1

def run_engine(raw_data: Dict[str, Any]):
    sys_settings = raw_data.get('systemSettings', {})
    sc_limit = sys_settings.get('SC_Constraint', {'maxWeightLimit': 60000, 'maxLengthLimit': 800.0})
    person_weight = sys_settings.get('Person_Weight', {'weight_per_person': 75.0})['weight_per_person']
    box_specs = sys_settings.get('Box_Specs', {})

    all_sub_containers = []
    open_person_boxes = []
    open_large_boxes = []
    open_small_boxes = []

    def add_people_to_boxes(b_type, num_people, owner_id):
        spec = box_specs.get(b_type)
        if not spec or num_people <= 0: return 0
        cap = spec['capacity']
        remaining = num_people
        added_total = 0

        for box in open_person_boxes:
            if box.box_type == b_type and box.current_load < box.max_capacity:
                available_space = box.max_capacity - box.current_load
                to_add = min(remaining, available_space)
                if to_add > 0:
                    box.add_item(owner_id, f"{int(to_add)}人({b_type})", to_add * person_weight, to_add)
                    remaining -= to_add
                    added_total += to_add
                if remaining <= 0:
                    break

        while remaining > 0:
            to_add = min(remaining, cap)
            new_box = SubContainer(b_type, spec['length_unit'], spec['weight'], cap, 'count')
            new_box.add_item(owner_id, f"{int(to_add)}人({b_type})", to_add * person_weight, to_add)
            all_sub_containers.append(new_box)
            open_person_boxes.append(new_box)
            remaining -= to_add
            added_total += to_add
        return added_total

    companies = raw_data.get('data', [])
    for comp in companies:
        cid = comp['organizationID']
        try:
            u_class = int(comp['Unitclass'])
        except:
            u_class = 5

        p_count = comp['personCount']
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
            mandatory = {}
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
        for item in comps_list:
            name = item['componentname']
            w = item['componentweight']
            occupancy = item.get('needcarLarge', 1.0)
            desc = f"{name}(占{occupancy})"
            placed = False
            best_box = None
            min_rem = 1.0
            for box in open_large_boxes:
                if box.current_load + occupancy <= 1.001:
                    rem = 1.0 - (box.current_load + occupancy)
                    if rem < min_rem:
                        min_rem = rem
                        best_box = box
            if best_box:
                best_box.add_item(cid, desc, w, occupancy)
                placed = True
            if not placed:
                new_box = SubContainer('Equip_Box_Large', spec_large['length_unit'], spec_large['weight'], 1.0, 'occupancy')
                new_box.add_item(cid, desc, w, occupancy)
                all_sub_containers.append(new_box)
                open_large_boxes.append(new_box)

        goods_list = comp.get('goodsList', [])
        flat_goods = []
        for g in goods_list:
            count = g.get('count', 1)
            for _ in range(count):
                flat_goods.append(g)
        flat_goods.sort(key=lambda x: x.get('tj', 0), reverse=True)
        spec_small = box_specs.get('Equip_Box_Small')
        max_vol = spec_small.get('capacity_volume', 100.0)
        for item in flat_goods:
            name = item['name']
            w = item['weight']
            vol = item.get('tj', 0)
            placed = False
            for box in open_small_boxes:
                if box.current_load + vol <= max_vol:
                    box.add_item(cid, name, w, vol)
                    placed = True
                    break
            if not placed:
                new_box = SubContainer('Equip_Box_Small', spec_small['length_unit'], spec_small['weight'], max_vol, 'volume')
                new_box.add_item(cid, name, w, vol)
                all_sub_containers.append(new_box)
                open_small_boxes.append(new_box)

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return {"code": 1, "msg": "求解器未启动"}

    total_len = sum(b.length_unit for b in all_sub_containers)
    max_sc_len = sc_limit['maxLengthLimit']
    est_bins = int(total_len / max_sc_len) + 20
    SC_IDs = range(est_bins)
    Box_IDs = range(len(all_sub_containers))
    comp_map = {c['organizationID']: idx for idx, c in enumerate(companies)}
    Comp_IDs = comp_map.values()

    x = {}
    y = {}
    z = {}

    for i in Box_IDs:
        vars_in_row = []
        for j in SC_IDs:
            x[i, j] = solver.IntVar(0, 1, '')
            vars_in_row.append(x[i, j])
        solver.Add(sum(vars_in_row) == 1)

    for j in SC_IDs:
        y[j] = solver.IntVar(0, 1, '')
        for i in Box_IDs:
            solver.Add(x[i, j] <= y[j])
        solver.Add(sum(x[i, j] * all_sub_containers[i].weight for i in Box_IDs) <= sc_limit['maxWeightLimit'] * y[j])
        solver.Add(sum(x[i, j] * all_sub_containers[i].length_unit for i in Box_IDs) <= sc_limit['maxLengthLimit'] * y[j])

    for c_idx in Comp_IDs:
        cid = companies[c_idx]['organizationID']
        my_boxes = [i for i, b in enumerate(all_sub_containers) if cid in b.owners]
        sc_usage = []
        for j in SC_IDs:
            z[c_idx, j] = solver.IntVar(0, 1, '')
            if my_boxes:
                solver.Add(sum(x[i, j] for i in my_boxes) <= 1000 * z[c_idx, j])
            sc_usage.append(z[c_idx, j])
        solver.Add(sum(sc_usage) <= 2)

    for j in range(len(SC_IDs) - 1):
        solver.Add(y[j] >= y[j + 1])

    solver.Minimize(sum(y[j] for j in SC_IDs))
    status = solver.Solve()

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        res_data = {"code": 0, "msg": "success", "data": {"total_SC_used": int(solver.Objective().Value()), "SC_list": []}}
        for j in SC_IDs:
            if y[j].solution_value() > 0.5:
                sc_info = {"SC_ID": f"SC_{j + 1:03d}", "summary": {}, "box_list": []}
                owners_set = set()
                curr_w = 0
                curr_l = 0
                has_mixed = False
                for i in Box_IDs:
                    if x[i, j].solution_value() > 0.5:
                        box = all_sub_containers[i]
                        owners_set.update(box.owners)
                        curr_w += box.weight
                        curr_l += box.length_unit
                        if box.is_mixed: has_mixed = True
                        item_counts = Counter(box.contents)
                        desc_list = []
                        for item_str, count in item_counts.items():
                            if count > 1:
                                desc_list.append(f"{item_str} × {count}")
                            else:
                                desc_list.append(item_str)
                        sc_info["box_list"].append({
                            "box_id": f"Box_{i + 1:04d}",
                            "box_type": box.box_type,
                            "is_mixed": box.is_mixed,
                            "owners": list(box.owners),
                            "content_desc": desc_list,
                            "weight": round(box.weight, 1),
                            "length_unit": round(box.length_unit, 2)
                        })
                sc_info["summary"] = {
                    "companies_included": list(owners_set),
                    "total_weight": round(curr_w, 1),
                    "total_length_unit": round(curr_l, 2),
                    "has_mixed_box": has_mixed,
                    "description": f"包含 {len(owners_set)} 个公司"
                }
                res_data["data"]["SC_list"].append(sc_info)
        return res_data
    else:
        return {"code": 1, "msg": "计算无有效解"}

# ============================
# FastAPI 服务（Linux ARM 专用）
# ============================
app = FastAPI(title="铁路运输配载优化 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/optimize")
async def optimize(req: OptimizationRequest):
    logger.info("收到优化计算请求")
    try:
        raw_dict = req.model_dump()
        return run_engine(raw_dict)
    except Exception as e:
        logger.error(f"算法执行异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 即使 IP 不可达，这种方式也能返回本机在局域网中的主 IP
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

# ============================
# 启动入口（无GUI，纯服务）
# ============================
if __name__ == "__main__":
    ip = get_ip()
    print(f"服务启动成功")
    print(f"局域网地址：http://{ip}:2376")
    print(f"接口地址：http://{ip}:2376/api/v1/optimize")
    uvicorn.run(app, host="0.0.0.0", port=2376)