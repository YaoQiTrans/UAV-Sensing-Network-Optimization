import sys
import os
import time
import itertools
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse

# --- 1. 环境配置与模块导入 ---
sys.path.append(os.path.join(os.getcwd(), 'model_static'))

try:
    import data_processor, preprocess
    from config import LINK_FILE, NODE_FILE
except ImportError:
    print("Error: Could not import modules from 'model_static'. Please run this script from the project root.")
    sys.exit(1)

# --- 2. 核心辅助类：AGSLM 逻辑管理器 ---
class AGSLMManager:
    def __init__(self):
        print("--- [Init] Loading Data & Basic Topologies ---")
        # 1. 加载全量网络数据
        self.df_link = pd.read_csv(LINK_FILE)
        self.df_node = pd.read_csv(NODE_FILE)
        
        # 2. 生成路径数据
        self.df_paths = data_processor.generate_paths_from_trajectories_union()
        self.num_routes = len(self.df_paths)
        self.num_links = len(self.df_link)
        
        # 3. 基础参数计算
        df_candidates = self.df_node[self.df_node['node_id'].astype(str).str.len() >= 8].copy()
        self.candidate_nodes = df_candidates['node_id'].tolist()
        self.num_uav_nodes = len(self.candidate_nodes)
        
        # 建立 UAV ID 到 Index 的映射，方便后续处理
        self.uav_id_to_idx = {uid: i for i, uid in enumerate(self.candidate_nodes)}
        
        sub_data = {
            "df_path_sub": self.df_paths,
            "df_link_sub": self.df_link,
            "df_node_sub": self.df_node,
            "df_candidate_nodes_sub": df_candidates
        }
        
        print("   Computing basic parameters (Incidence & FOV)...")
        params = preprocess.calculate_model_parameters(sub_data)
        self.incidence_matrix = params['sub_incidence_matrix']
        self.L_j = params['L_j'] 
        
        self._build_inverted_indices()
        
    def _build_inverted_indices(self):
        print("   Building inverted indices for fast geometric query...")
        # Link -> Routes
        self.link_to_routes = {}
        coo = self.incidence_matrix.tocoo()
        for r, l in zip(coo.row, coo.col):
            if l not in self.link_to_routes: self.link_to_routes[l] = []
            self.link_to_routes[l].append(r)
            
        # UAV Node -> Routes
        self.uav_to_routes = {}
        self.path_link_sets = []
        for r in range(self.num_routes):
            self.path_link_sets.append(set(self.incidence_matrix.getrow(r).indices))
            
        for uav_node in self.candidate_nodes:
            covered_links = set(self.L_j.get(uav_node, []))
            if not covered_links: continue
            
            related_routes = []
            for r in range(self.num_routes):
                if not self.path_link_sets[r].isdisjoint(covered_links):
                    related_routes.append(r)
            if len(related_routes) > 1:
                # 使用 uav_idx 作为 key，比 raw ID 更安全
                u_idx = self.uav_id_to_idx[uav_node]
                self.uav_to_routes[u_idx] = related_routes

    def get_distinguishing_set(self, r1, r2):
        """
        返回区分 r1 和 r2 的资源集合。
        为了避免类型混淆，我们使用带标签的元组：
        - Link: ('l', link_idx)
        - UAV:  ('u', uav_idx)
        """
        # 1. Ground Sensors
        links1 = self.path_link_sets[r1]
        links2 = self.path_link_sets[r2]
        # symmetric_difference 返回的是 link index (int)
        diff_links_indices = links1.symmetric_difference(links2)
        # 打标
        dist_set = {('l', idx) for idx in diff_links_indices}
        
        # 2. Aerial Sensors
        # 遍历所有候选 UAV (使用 idx 遍历)
        for uav_node in self.candidate_nodes:
            covered_links = set(self.L_j.get(uav_node, []))
            
            subpath1 = links1.intersection(covered_links)
            subpath2 = links2.intersection(covered_links)
            
            if subpath1 != subpath2:
                u_idx = self.uav_id_to_idx[uav_node]
                dist_set.add(('u', u_idx))
                
        return dist_set

# --- 3. 筛选逻辑实现 ---

def get_geometric_pairs(mgr: AGSLMManager):
    """
    Mode B: Geometric Sieve
    """
    potential_neighbors = set()
    
    # 1. Link 共享
    for link, routes in mgr.link_to_routes.items():
        if len(routes) > 1:
            for r1, r2 in itertools.combinations(routes, 2):
                if r1 > r2: r1, r2 = r2, r1
                potential_neighbors.add((r1, r2))
                
    # 2. UAV 视场共享 (使用 uav_idx)
    for uav_idx, routes in mgr.uav_to_routes.items():
        if len(routes) > 1:
            for r1, r2 in itertools.combinations(routes, 2):
                if r1 > r2: r1, r2 = r2, r1
                potential_neighbors.add((r1, r2))
                
    return list(potential_neighbors)

def filter_logical_sets(constraints_list):
    """
    Mode C/D: Logical Sieve
    Input: List of set<TaggedTuple>
    Output: Reduced List of frozenset<TaggedTuple>
    """
    # 1. 转换为 frozenset 
    unique_sets = set()
    for s in constraints_list:
        if s:
            unique_sets.add(frozenset(s))
            
    # 2. 排序与剔除
    sorted_sets = sorted(list(unique_sets), key=len)
    kept_sets = []
    
    for current_set in sorted_sets:
        is_redundant = False
        for kept_set in kept_sets:
            if kept_set.issubset(current_set):
                is_redundant = True
                break
        if not is_redundant:
            kept_sets.append(current_set)
            
    return kept_sets

# --- 4. 实验主流程 ---

def run_agslm_experiment(mode, mgr: AGSLMManager, time_limit=3600):
    print(f"\n>>> Running AGSLM Mode: {mode}")
    t_start = time.time()
    
    # --- Step 1: Constraint Generation ---
    pairs_to_process = []
    
    if mode == 'Mode A (Baseline)':
        pairs_to_process = list(itertools.combinations(range(mgr.num_routes), 2))
        print(f"  - [A] Generated {len(pairs_to_process)} total pairs.")
        
    elif mode == 'Mode B (Geometric)':
        pairs_to_process = get_geometric_pairs(mgr)
        print(f"  - [B] Geometric Sieve retained {len(pairs_to_process)} pairs.")
        
    elif mode == 'Mode C (Logical)':
        pairs_to_process = list(itertools.combinations(range(mgr.num_routes), 2))
        print(f"  - [C] Starting with {len(pairs_to_process)} pairs for Logical Sieve...")
        
    elif mode == 'Mode D (Ours)':
        pairs_to_process = get_geometric_pairs(mgr)
        print(f"  - [D] Geometric Sieve passed {len(pairs_to_process)} pairs to Logical Sieve...")

    # 计算区分集合
    raw_constraints = []
    # 如果对数量特别大，显示进度条
    iter_obj = pairs_to_process
    # if len(pairs_to_process) > 10000:
    #     print("  Computing distinguishing sets...")
    
    for r1, r2 in iter_obj:
        dist_set = mgr.get_distinguishing_set(r1, r2)
        if dist_set:
            raw_constraints.append(dist_set)
            
    final_constraints = []
    if 'Logical' in mode or 'Ours' in mode:
        print(f"  - Performing Logical Reduction on {len(raw_constraints)} sets...")
        final_constraints = filter_logical_sets(raw_constraints)
    else:
        # 即使不做逻辑剔除，也要转成 frozenset 列表以便后续处理
        final_constraints = [frozenset(s) for s in raw_constraints]
        
    t_pre_end = time.time()
    prep_time = t_pre_end - t_start
    num_constrs = len(final_constraints)
    print(f"  - Final Constraints: {num_constrs}. Prep Time: {prep_time:.2f}s")
    
    # --- Step 2: Gurobi Solving ---
    t_solve_start = time.time()
    
    model = gp.Model(f"AGSLM_{mode}")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    
    # Variables
    z = model.addVars(mgr.num_links, vtype=GRB.BINARY, name="z")
    u = model.addVars(mgr.num_uav_nodes, vtype=GRB.BINARY, name="u")
    
    # Objective: Minimize Cost (Z=1, U=7)
    # 使用 .sum() 替代 quicksum(tupledict) 以避免警告
    model.setObjective(z.sum() + 7 * u.sum(), GRB.MINIMIZE)
    
    # 1. Coverage Constraints
    for r in range(mgr.num_routes):
        # Ground
        l_indices = list(mgr.path_link_sets[r])
        # Aerial (Find UAVs covering this route)
        # 利用倒排索引 uav_to_routes 加速查找
        u_indices = []
        for u_idx, routes in mgr.uav_to_routes.items():
            if r in routes: # 这里的 routes 列表可能包含 r
                # 这里有点小问题：uav_to_routes 是仅仅基于 link 重叠构建的吗？
                # 是的，在 _build_inverted_indices 里。如果 route link 和 uav fov 重叠，就在列表里。
                u_indices.append(u_idx)
                
        model.addConstr(
            gp.quicksum(z[l] for l in l_indices) + 
            gp.quicksum(u[k] for k in u_indices) >= 1
        )
        
    # 2. Distinguishability Constraints
    # item 是 ('l', idx) 或 ('u', idx)
    for constr_set in final_constraints:
        expr = gp.LinExpr()
        for tag, idx in constr_set:
            if tag == 'l':
                expr += z[idx]
            elif tag == 'u':
                expr += u[idx]
        model.addConstr(expr >= 1)
        
    model.optimize()
    t_solve_end = time.time()
    solve_time = t_solve_end - t_solve_start
    
    return {
        "Mode": mode,
        "Constraints": num_constrs,
        "Prep_Time": prep_time,
        "Solve_Time": solve_time,
        "Total_Time": prep_time + solve_time,
        "Obj": model.ObjVal if model.status == GRB.OPTIMAL else -1
    }

if __name__ == "__main__":
    mgr = AGSLMManager()
    
    results = []
    modes = [
        'Mode A (Baseline)', 
        'Mode B (Geometric)', 
        'Mode C (Logical)', 
        'Mode D (Ours)'
    ]
    
    print(f"\nExperiment Start. Routes: {mgr.num_routes}")
    
    for m in modes:
        res = run_agslm_experiment(m, mgr, time_limit=3600)
        results.append(res)
        
    df = pd.DataFrame(results)
    # 处理可能的除零错误（如果 Mode A 没跑完）
    base_constrs = df.iloc[0]['Constraints']
    if base_constrs > 0:
        df['Reduction'] = (1 - df['Constraints'] / base_constrs) * 100
    else:
        df['Reduction'] = 0.0
        
    print("\n=== AGSLM Ablation Results ===")
    print(df[['Mode', 'Constraints', 'Reduction', 'Prep_Time', 'Solve_Time', 'Total_Time', 'Obj']])
    
    df.to_csv("agslm_ablation_results.csv", index=False)