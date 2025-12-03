# ablation_runner_full.py
import time
import itertools
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm
import os

# 引用数据加载模块
import data_loader
import utils

# ==========================================
# 核心筛选算法
# ==========================================

def get_geometric_pairs(links_per_route, num_routes):
    """
    Mode B: Geometric Sieve
    只保留空间上有重叠（共享至少一个 Link）的路径对。
    """
    pairs = []
    # 建立倒排索引：Link -> List of Route IDs
    link_to_routes = {}
    for r_idx, links in enumerate(links_per_route):
        for link in links:
            if link not in link_to_routes:
                link_to_routes[link] = []
            link_to_routes[link].append(r_idx)
            
    # 利用倒排索引快速查找重叠对，避免 O(N^2) 遍历
    # 这是一个重要的优化，否则 Mode B 的预处理时间会和 Mode A 一样慢
    potential_neighbors = set()
    for link, routes in link_to_routes.items():
        if len(routes) > 1:
            for r1, r2 in itertools.combinations(routes, 2):
                if r1 > r2: r1, r2 = r2, r1
                potential_neighbors.add((r1, r2))
    
    return list(potential_neighbors)

def filter_logical_pairs(links_per_route, candidate_pairs):
    """
    Mode C & D: Logical Sieve (Constraint Reduction)
    对输入的 candidate_pairs 进行逻辑包含检查。
    如果 Diff(r1, r2) ⊆ Diff(r3, r4)，则 (r1, r2) 是冗余的（注意：是大的集合冗余还是小的？
    修正：在此类覆盖问题中，如果是 Set Cover 形式：sum(x) >= 1。
    如果 Set A ⊆ Set B，那么满足 Set A >= 1 必然满足 Set B >= 1。
    所以，我们要保留的是【最小】的集合 (Minimal Sets)，剔除【较大】的集合。
    """
    # 1. 转换：Pair -> Distinguishing Set (Link Indices)
    # constraint_map: key=frozenset(links), value=example_pair
    # 我们只需要保留 Link Set，不需要保留具体的 Pair（因为只要有一个 Pair 生成了这个约束集就行）
    unique_sets = set()
    
    for r1, r2 in candidate_pairs:
        diff = links_per_route[r1].symmetric_difference(links_per_route[r2])
        if diff:
            unique_sets.add(frozenset(diff))
            
    # 2. 逻辑剔除 (Keep Minimal Sets)
    # 按集合大小排序
    sorted_sets = sorted(list(unique_sets), key=len)
    kept_sets = []
    
    # 贪婪检查：如果当前集合包含任何已保留的集合，则它是冗余的（大的包含小的）
    for current_set in sorted_sets:
        is_redundant = False
        for kept_set in kept_sets:
            if kept_set.issubset(current_set):
                is_redundant = True
                break
        if not is_redundant:
            kept_sets.append(current_set)
            
    return kept_sets

# ==========================================
# 求解器框架
# ==========================================

def run_experiment_mode(df_route, df_link, indicator_matrix, mode, time_limit=3600):
    """
    运行指定模式的实验
    """
    num_routes = indicator_matrix.shape[0]
    num_links = indicator_matrix.shape[1]
    
    print(f"\n>>> Running Mode: {mode}")
    
    # 预计算路径的 Link 集合 (Set for fast operations)
    links_per_route = [set(indicator_matrix.getrow(r).indices) for r in range(num_routes)]
    
    # --- Step 1: Preprocessing (Constraint Generation) ---
    t_start_pre = time.time()
    final_constraint_sets = []
    
    if mode == 'Mode A (Baseline)':
        # 全约束：生成所有对，不筛选 (仅计算 Diff)
        # 注意：为了公平对比求解时间，我们还是要算出 Diff Set，但不做任何剔除
        pairs = list(itertools.combinations(range(num_routes), 2))
        print(f"  - Generated {len(pairs)} initial pairs.")
        for r1, r2 in pairs:
            diff = links_per_route[r1].symmetric_difference(links_per_route[r2])
            if diff:
                final_constraint_sets.append(diff)
                
    elif mode == 'Mode B (Geometric)':
        # 仅几何筛选
        pairs = get_geometric_pairs(links_per_route, num_routes)
        print(f"  - Geometric Sieve found {len(pairs)} overlapping pairs.")
        for r1, r2 in pairs:
            diff = links_per_route[r1].symmetric_difference(links_per_route[r2])
            if diff:
                final_constraint_sets.append(diff)
                
    elif mode == 'Mode C (Logical Only)':
        # 仅逻辑筛选（输入是所有对）
        pairs = list(itertools.combinations(range(num_routes), 2))
        print(f"  - Generated {len(pairs)} initial pairs. Running Logical Sieve...")
        final_constraint_sets = filter_logical_pairs(links_per_route, pairs)
        
    elif mode == 'Mode D (Ours: Geo + Log)':
        # 几何 + 逻辑
        # 1. Geometric
        pairs = get_geometric_pairs(links_per_route, num_routes)
        print(f"  - Geometric Sieve found {len(pairs)} overlapping pairs. Running Logical Sieve...")
        # 2. Logical
        final_constraint_sets = filter_logical_pairs(links_per_route, pairs)
        
    t_end_pre = time.time()
    prep_time = t_end_pre - t_start_pre
    num_constrs = len(final_constraint_sets)
    print(f"  - Final Constraints: {num_constrs}. Preprocessing Time: {prep_time:.4f}s")
    
    # --- Step 2: Solving (Gurobi) ---
    t_start_solve = time.time()
    
    model = gp.Model(f"Exp_{mode}")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = 8 # 统一线程数
    
    # 变量：z_link
    z = model.addVars(num_links, vtype=GRB.BINARY)
    model.setObjective(z.sum(), GRB.MINIMIZE)
    
    # 约束 1: 覆盖 (所有模式都必须有)
    for r_idx, links in enumerate(links_per_route):
        if links:
            model.addConstr(gp.quicksum(z[l] for l in links) >= 1)
            
    # 约束 2: 区分 (根据模式生成的集合)
    # 为了加快建模速度，批量添加
    for link_set in final_constraint_sets:
        model.addConstr(gp.quicksum(z[l] for l in link_set) >= 1)
        
    model.optimize()
    t_end_solve = time.time()
    solve_time = t_end_solve - t_start_solve
    
    obj_val = model.ObjVal if model.status == GRB.OPTIMAL else -1
    print(f"  - Solve Time: {solve_time:.4f}s. Obj: {obj_val}")
    
    return {
        "Mode": mode,
        "Num_Routes": num_routes,
        "Constraints": num_constrs,
        "Prep_Time": prep_time,
        "Solve_Time": solve_time,
        "Total_Time": prep_time + solve_time,
        "Obj_Val": obj_val
    }

# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
    # 配置
    route_file = r'data/PMEUMA_460_route.csv'
    link_file = r'data/PMEUMA_402_link.csv'
    
    print("Loading Data...")
    df_route, df_link = data_loader.load_network_data(route_file, link_file)
    matrix = data_loader.calculate_route_link_incidence(df_route, df_link)
    
    print(f"Dataset Loaded: {matrix.shape[0]} Routes, {matrix.shape[1]} Links.")
    
    results = []
    
    # 按顺序运行四种模式
    # 注意：如果 Mode A 太慢，可以考虑注释掉，或者只用小数据测试
    modes = [
        'Mode A (Baseline)', 
        'Mode B (Geometric)', 
        'Mode C (Logical Only)', 
        'Mode D (Ours: Geo + Log)'
    ]
    
    for m in modes:
        res = run_experiment_mode(df_route, df_link, matrix, m)
        results.append(res)
        
    # 保存结果
    df_res = pd.DataFrame(results)
    df_res['Reduction_Rate'] = (1 - df_res['Constraints'] / df_res.loc[0, 'Constraints']) * 100
    
    print("\n=== Final Results Summary ===")
    print(df_res[['Mode', 'Constraints', 'Reduction_Rate', 'Prep_Time', 'Solve_Time', 'Total_Time']])
    df_res.to_csv("ablation_full_network.csv", index=False)