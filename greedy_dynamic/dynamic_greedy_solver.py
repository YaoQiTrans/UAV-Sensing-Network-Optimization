import sys
import os
import time
import itertools
import heapq
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm

# --- 环境配置 ---
try:
    import config_model3 as config
    import data_processor
    import preprocess_model3
    LINK_FILE = config.LINK_FILE
    NODE_FILE = config.NODE_FILE
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path: sys.path.append(project_root)
    import config_model3 as config
    import data_processor as data_processor
    import preprocess_model3 as preprocess_model3
    LINK_FILE = config.LINK_FILE
    NODE_FILE = config.NODE_FILE

# ==========================================
# 1. 动态模型数据管理器
# ==========================================
class DynamicAGSLMManager:
    def __init__(self):
        print("--- [Init] Loading Dynamic Model Data ---")
        self.bundle_path = os.path.join(config.PREPROCESSED_DATA_DIR_MODEL3, "model3_preprocessed_data.pkl")
        self.paths_path = os.path.join(config.PREPROCESSED_DATA_DIR_MODEL3, "processed_paths_by_period_undirected.pkl")

        if self._try_load_bundle():
            print("   ✅ Loaded full preprocessed bundle.")
        else:
            self._load_or_generate_basic_data()
            self._calculate_params()

        self.periods = list(self.paths_by_period.keys())
        self.num_periods = len(self.periods)
        self.candidate_nodes = self.sub_data['df_candidate_nodes_sub']['node_id'].tolist()
        self.uav_id_to_idx = {uid: i for i, uid in enumerate(self.candidate_nodes)}
        self.link_id_to_idx = {lid: i for i, lid in enumerate(self.sub_data['df_link_sub']['link_id'])}
        
        # 预计算：每个时段每条路径的 Link Set (用于快速查询)
        # 这是一个空间换时间的优化
        print("   Pre-caching route link sets...")
        self.route_link_sets = {} # (t, r) -> set(link_indices)
        for t_idx, period in enumerate(self.periods):
            inc_mat = self.dynamic_params['inc_matrix_by_period'][period]
            for r in range(inc_mat.shape[0]):
                self.route_link_sets[(t_idx, r)] = set(inc_mat.getrow(r).indices)
                
        print(f"   Loaded {self.num_periods} periods. Candidates: {len(self.candidate_nodes)} UAV, {len(self.link_id_to_idx)} Link.")

    def _try_load_bundle(self):
        if os.path.exists(self.bundle_path):
            try:
                with open(self.bundle_path, 'rb') as f:
                    data = pickle.load(f)
                self.sub_data = data['subnetwork_data']
                self.paths_by_period = data['paths_by_period']
                self.static_params = data['static_uav_params']
                self.dynamic_params = data['dynamic_params']
                return True
            except Exception: return False
        return False

    def _load_or_generate_basic_data(self):
        try:
            self.df_link = pd.read_csv(LINK_FILE)
            self.df_node = pd.read_csv(NODE_FILE)
        except: sys.exit("Base files not found")
        
        if os.path.exists(self.paths_path):
            with open(self.paths_path, 'rb') as f: self.paths_by_period = pickle.load(f)
        else:
            self.paths_by_period = data_processor.generate_paths_for_all_periods()

    def _calculate_params(self):
        self.sub_data = preprocess_model3.create_unified_subnetwork(self.paths_by_period, self.df_link, self.df_node)
        self.static_params = preprocess_model3.calculate_static_uav_params(self.sub_data)
        self.dynamic_params = preprocess_model3.calculate_dynamic_params(self.paths_by_period, self.sub_data, self.static_params)

    def get_universe_of_pairs(self):
        """
        生成全集：包含 (p, q) 区分对 和 (p, NULL) 覆盖对。
        """
        print("--- [Step 1] Generating Universe (Distinguishability + Coverage) ---")
        universe = []
        
        total_dist_pairs = 0
        total_cov_pairs = 0
        
        for t_idx, period in enumerate(self.periods):
            inc_mat = self.dynamic_params['inc_matrix_by_period'][period]
            num_routes = inc_mat.shape[0]
            
            # 1. 覆盖约束: (t, r, -1)
            # 每一条路径都必须被“看到”，即区分于“空”
            for r in range(num_routes):
                universe.append((t_idx, r, -1))
            total_cov_pairs += num_routes
            
            # 2. 区分约束: (t, r1, r2)
            # 使用倒排索引加速几何筛选
            link_to_routes = defaultdict(list)
            coo = inc_mat.tocoo()
            for r, l in zip(coo.row, coo.col): link_to_routes[l].append(r)
            
            overlapping = set()
            for routes in link_to_routes.values():
                if len(routes) > 1:
                    for r1, r2 in itertools.combinations(routes, 2):
                        if r1 > r2: r1, r2 = r2, r1
                        overlapping.add((r1, r2))
            
            for r1, r2 in overlapping: universe.append((t_idx, r1, r2))
            total_dist_pairs += len(overlapping)
            
        print(f"   Universe Size: {len(universe)} (Cov: {total_cov_pairs}, Dist: {total_dist_pairs})")
        return universe

    def build_candidate_profiles(self, universe):
        """
        构建覆盖列表。
        """
        print("--- [Step 2] Building Profiles (including Coverage) ---")
        avi_profiles = defaultdict(set)
        uav_profiles = defaultdict(set)
        
        # 准备 UAV Link Sets (映射回子网 ID)
        orig_to_sub_idx = {lid: i for i, lid in enumerate(self.sub_data['df_link_sub']['link_id'])}
        uav_link_sets_indices = {}
        for uid, links in self.static_params['L_j'].items():
            if uid in self.uav_id_to_idx:
                indices = {orig_to_sub_idx[lid] for lid in links if lid in orig_to_sub_idx}
                if indices: uav_link_sets_indices[self.uav_id_to_idx[uid]] = indices

        for pair_idx, (t_idx, r1, r2) in enumerate(tqdm(universe, desc="Mapping")):
            links1 = self.route_link_sets[(t_idx, r1)]
            
            # 处理 r2 = -1 (覆盖约束) 的情况
            if r2 == -1:
                links2 = set() # 空路径
            else:
                links2 = self.route_link_sets[(t_idx, r2)]
            
            # 1. AVI 区分
            # 只要 Link 在 links1 XOR links2 中，就能区分
            # 对于覆盖约束 (r1, -1)，XOR 就是 links1 本身。即：只要路径上有 AVI，就被覆盖。
            for l_idx in links1.symmetric_difference(links2):
                avi_profiles[l_idx].add(pair_idx)
                
            # 2. UAV 区分
            # 只要 UAV 在 r1 和 r2 上的投影不同，就能区分
            # 对于覆盖约束 (r1, -1)，只要 UAV 覆盖了 r1 的任意部分，就能区分（看到 r1 vs 看不到）
            for u_idx, fov in uav_link_sets_indices.items():
                # 快速过滤：如果 FOV 跟 r1 和 r2 都没交集，肯定分不出
                # 对于覆盖约束，只要跟 r1 没交集就分不出
                if fov.isdisjoint(links1) and fov.isdisjoint(links2):
                    continue
                
                sp1 = links1.intersection(fov)
                sp2 = links2.intersection(fov)
                
                if sp1 != sp2:
                    uav_profiles[(u_idx, t_idx)].add(pair_idx)
                    
        return avi_profiles, uav_profiles

# ==========================================
# 2. 迭代机队扩张求解器 (Iterative Fleet Expansion)
# ==========================================
class IterativeFleetSolver:
    def __init__(self, avi_profiles, uav_profiles, total_pairs_count, costs=(1.0, 7.0), num_periods=3):
        self.avi_profiles = avi_profiles
        self.uav_profiles = uav_profiles
        self.total_pairs_count = total_pairs_count
        self.cost_avi, self.cost_uav = costs
        self.num_periods = num_periods

    def solve_for_fixed_fleet(self, fleet_size):
        current_covered = set()
        selected_avis = set()
        selected_uavs = [] 
        
        # --- Phase 1: Fill UAV Slots (Sunk Cost Greedy) ---
        if fleet_size > 0:
            for t in range(self.num_periods):
                candidates = [key for key in self.uav_profiles if key[1] == t]
                pq = []
                for key in candidates:
                    gain = len(self.uav_profiles[key])
                    if gain > 0: heapq.heappush(pq, (-gain, key))
                
                slots_left = fleet_size
                while slots_left > 0 and pq:
                    neg_gain, u_key = heapq.heappop(pq)
                    real_gain = len(self.uav_profiles[u_key].difference(current_covered))
                    if real_gain == 0: continue
                    
                    if not pq or real_gain >= -pq[0][0]:
                        current_covered.update(self.uav_profiles[u_key])
                        selected_uavs.append(u_key)
                        slots_left -= 1
                    else:
                        heapq.heappush(pq, (-real_gain, u_key))

        # --- Phase 2: Fill Residue with AVI (Lazy Greedy) ---
        residue_count = self.total_pairs_count - len(current_covered)
        if residue_count > 0:
            avi_pq = []
            for l_idx, prof in self.avi_profiles.items():
                gain = len(prof.difference(current_covered))
                if gain > 0:
                    score = gain / self.cost_avi
                    heapq.heappush(avi_pq, (-score, l_idx))
            
            while len(current_covered) < self.total_pairs_count:
                if not avi_pq: break 
                neg_score, l_idx = heapq.heappop(avi_pq)
                
                # Lazy Re-eval
                # 这里必须重新计算 Gain，因为 UAV 阶段可能已经覆盖了一部分
                gain = len(self.avi_profiles[l_idx].difference(current_covered))
                if gain == 0: continue
                
                curr_score = gain / self.cost_avi
                
                if not avi_pq or curr_score >= -avi_pq[0][0] - 1e-9:
                    current_covered.update(self.avi_profiles[l_idx])
                    selected_avis.add(l_idx)
                else:
                    heapq.heappush(avi_pq, (-curr_score, l_idx))

        usage_counts = defaultdict(int)
        for _, t in selected_uavs: usage_counts[t] += 1
        actual_fleet = max(usage_counts.values()) if usage_counts else 0
        
        # Cost 计算：UAV 成本基于实际使用的最大 Fleet Size
        total_cost = actual_fleet * self.cost_uav + len(selected_avis) * self.cost_avi
        
        return {
            "fleet_size_limit": fleet_size,
            "actual_fleet": actual_fleet,
            "avi_count": len(selected_avis),
            "uav_count": len(selected_uavs),
            "total_cost": total_cost,
            "covered": len(current_covered)
        }

    def solve_global(self, max_fleet=40):
        print(f"--- [Run] Iterative Fleet Expansion (0 to {max_fleet}) ---")
        best_solution = None
        history = []
        
        for k in tqdm(range(max_fleet + 1), desc="Enumerating"):
            res = self.solve_for_fixed_fleet(k)
            history.append(res)
            if best_solution is None or res['total_cost'] < best_solution['total_cost']:
                best_solution = res
        return best_solution, history

if __name__ == "__main__":
    mgr = DynamicAGSLMManager()
    
    # Step 1: Generate Universe (Pairs + Coverage)
    universe = mgr.get_universe_of_pairs()
    
    # Step 2: Build Profiles
    avi_prof, uav_prof = mgr.build_candidate_profiles(universe)
    
    COST_AVI = 1.0
    COST_UAV = 7.0
    
    print(f"\n>>> Solving with Iterative Fleet Expansion (Cost Ratio 1:{COST_UAV})")
    solver = IterativeFleetSolver(
        avi_prof, uav_prof, len(universe), 
        costs=(COST_AVI, COST_UAV), 
        num_periods=mgr.num_periods
    )
    
    best_res, history = solver.solve_global(max_fleet=20)
    
    print("\n=== Best Solution Found ===")
    print(f"Optimal Fleet Size: {best_res['actual_fleet']}")
    print(f"AVI Sensors: {best_res['avi_count']}")
    print(f"Total Cost: {best_res['total_cost']:.2f}")
    
    # Save detailed history for plotting
    df_hist = pd.DataFrame(history)
    print(df_hist[['fleet_size_limit', 'actual_fleet', 'avi_count', 'total_cost']])
    df_hist.to_csv("greedy_fleet_expansion_results.csv", index=False)