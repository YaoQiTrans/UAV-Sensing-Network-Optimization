import sys
import os
import time
import itertools
import heapq
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import tqdm

# --- 1. 环境配置与模块导入 (修复版) ---
# 获取当前脚本所在的目录 (即 greedy_dynamic/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录 (即 greedy_dynamic 的上一级目录)
project_root = os.path.dirname(current_dir)

# 将项目根目录、以及关键的子目录添加到系统搜索路径中
# 这样子模块内部的 'import config' 或 'import config_model3' 才能正常工作
sys.path.extend([
    project_root,
    os.path.join(project_root, 'model_static'),
    os.path.join(project_root, 'model_dynamic')
])

try:
    # 现在可以直接导入了，不用加前缀，或者加前缀也行
    # 这里为了兼容性，保留特定的引用方式，但其实因为加了路径，
    # model_static 内部的 import config 也能找到了
    from model_static import data_processor
    # 注意：这里直接从 config 导入，因为 model_static 已经在 path 里了
    import config as static_config 
    from model_dynamic import preprocess_model3
    
    # 重新绑定变量名以适配后续代码
    LINK_FILE = static_config.LINK_FILE
    NODE_FILE = static_config.NODE_FILE
    
except ImportError as e:
    print(f"Error: Could not import project modules. {e}")
    print(f"Debug: Project root set to: {project_root}")
    print(f"Debug: sys.path: {sys.path}")
    sys.exit(1)

# ... (后续代码保持不变)

# ==========================================
# 1. 动态模型数据管理器 (Dynamic AGSLM Manager)
# ==========================================
class DynamicAGSLMManager:
    """
    负责加载动态模型数据，并提供几何筛选功能。
    对应论文中的: Network Decomposition & Geometric Sieve
    """
    def __init__(self):
        print("--- [Init] Loading Dynamic Model Data ---")
        self.df_link = pd.read_csv(LINK_FILE)
        self.df_node = pd.read_csv(NODE_FILE)
        
        # 1. 加载或生成多时段路径数据
        # 这里假设预处理已经运行过，直接加载 pickle
        data_path = os.path.join(project_root, "model_dynamic", "preprocessed_data_model3", "model3_preprocessed_data.pkl")
        
        if not os.path.exists(data_path):
            print(f"Warning: Preprocessed file {data_path} not found. Running simple path generation...")
            # 简化的回退逻辑，仅用于演示代码逻辑正确性
            self.paths_by_period = data_processor.generate_paths_from_trajectories_union() # 这里暂时用静态的代替
            # 把它伪装成多时段字典
            self.paths_by_period = {"period_0": self.paths_by_period, "period_1": self.paths_by_period, "period_2": self.paths_by_period}
            # 重新计算参数 (模拟 preprocess_model3)
            self.sub_data = preprocess_model3.create_unified_subnetwork(self.paths_by_period, self.df_link, self.df_node)
            self.static_params = preprocess_model3.calculate_static_uav_params(self.sub_data)
            # Dynamic params 需要重新计算
            self.dynamic_params = preprocess_model3.calculate_dynamic_params(
                self.paths_by_period, self.sub_data, self.static_params
            )
        else:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.paths_by_period = data['paths_by_period']
                self.sub_data = data['subnetwork_data']
                self.static_params = data['static_uav_params']
                self.dynamic_params = data['dynamic_params']

        self.periods = list(self.paths_by_period.keys())
        self.num_periods = len(self.periods)
        self.candidate_nodes = self.sub_data['df_candidate_nodes_sub']['node_id'].tolist()
        self.uav_id_to_idx = {uid: i for i, uid in enumerate(self.candidate_nodes)}
        self.link_id_to_idx = {lid: i for i, lid in enumerate(self.sub_data['df_link_sub']['link_id'])}
        
        print(f"   Loaded {self.num_periods} periods. Candidates: {len(self.candidate_nodes)} UAV sites, {len(self.link_id_to_idx)} Links.")

    def get_universe_of_pairs(self):
        """
        生成所有时段中，所有“几何相关”的路径对。
        这是贪婪算法的“全集 (Universe)”。
        """
        print("--- [Step 1] Generating Geometric Universe of Pairs ---")
        universe = [] # List of (period_idx, r1, r2)
        total_pairs = 0
        
        for t_idx, period in enumerate(self.periods):
            # 获取该时段的 Incidence Matrix
            inc_mat = self.dynamic_params['inc_matrix_by_period'][period]
            num_routes = inc_mat.shape[0]
            
            # 构建倒排索引加速查找 (Link -> Routes)
            link_to_routes = defaultdict(list)
            coo = inc_mat.tocoo()
            for r, l in zip(coo.row, coo.col):
                link_to_routes[l].append(r)
                
            # 几何筛选：只考虑共享 Link 的路径对
            # (注：为了加速，这里暂略去 UAV 视场重叠的筛选，只用 Link 重叠做近似，效果通常足够好)
            overlapping_pairs = set()
            for routes in link_to_routes.values():
                if len(routes) > 1:
                    for r1, r2 in itertools.combinations(routes, 2):
                        if r1 > r2: r1, r2 = r2, r1
                        overlapping_pairs.add((r1, r2))
            
            for r1, r2 in overlapping_pairs:
                universe.append((t_idx, r1, r2))
                
            total_pairs += len(overlapping_pairs)
            print(f"   Period {period}: Found {len(overlapping_pairs)} overlapping pairs.")
            
        return universe

    def build_candidate_profiles(self, universe):
        """
        列生成：预计算每个候选者（AVI 或 UAV-Time）能覆盖 Universe 中的哪些对。
        返回:
            avi_profiles: dict {link_idx: set_of_pair_indices}
            uav_profiles: dict {(uav_node_idx, period_idx): set_of_pair_indices}
        """
        print("--- [Step 2] Building Candidate Profiles (Column Generation) ---")
        
        avi_profiles = defaultdict(set)
        uav_profiles = defaultdict(set)
        
        # 预计算：每个时段每条路径的 Link Set
        route_link_sets = {} # (t, r) -> set(link_indices)
        for t_idx, period in enumerate(self.periods):
            inc_mat = self.dynamic_params['inc_matrix_by_period'][period]
            for r in range(inc_mat.shape[0]):
                route_link_sets[(t_idx, r)] = set(inc_mat.getrow(r).indices)

        # 预计算：每个 UAV 覆盖的 Link Set
        uav_link_sets = {
            self.uav_id_to_idx[uid]: set(links) 
            for uid, links in self.static_params['L_j'].items()
        }

        # 遍历 Universe 中的每一对，分配给能区分它的传感器
        # 为了效率，我们只对 Universe 遍历一次
        for pair_idx, (t_idx, r1, r2) in enumerate(tqdm(universe, desc="Mapping Pairs to Sensors")):
            links1 = route_link_sets[(t_idx, r1)]
            links2 = route_link_sets[(t_idx, r2)]
            
            # 1. 哪些 AVI 能区分？ (Symmetric Difference)
            diff_links = links1.symmetric_difference(links2)
            for l_idx in diff_links:
                avi_profiles[l_idx].add(pair_idx)
                
            # 2. 哪些 UAV 能区分？ (在时段 t)
            # 只有当 UAV 覆盖了 r1 或 r2 时才检查
            # 优化：我们只检查覆盖了 diff_links 中任意 Link 的 UAV 即可吗？
            # 不完全是。UAV 区分依赖于 Subpath。
            # 这里简化：遍历所有 UAV (190个)，检查 Subpath
            for u_idx, fov_links in uav_link_sets.items():
                # 快速检查：如果 UAV 视场跟两条路径都无交集，肯定分不出来
                if fov_links.isdisjoint(links1) and fov_links.isdisjoint(links2):
                    continue
                
                sp1 = links1.intersection(fov_links)
                sp2 = links2.intersection(fov_links)
                
                if sp1 != sp2:
                    uav_profiles[(u_idx, t_idx)].add(pair_idx)
                    
        return avi_profiles, uav_profiles

# ==========================================
# 2. 耦合成本贪婪求解器 (Coupled-Cost Greedy Solver)
# ==========================================
class CoupledGreedySolver:
    def __init__(self, avi_profiles, uav_profiles, total_pairs_count, costs=(1.0, 7.0)):
        self.avi_profiles = avi_profiles
        self.uav_profiles = uav_profiles
        self.total_pairs_count = total_pairs_count
        self.cost_avi, self.cost_uav = costs
        
        # 状态变量
        self.covered_pairs = set() # bitset might be faster, but set is easier
        self.current_fleet_size = 0
        self.uav_usage_per_period = defaultdict(int) # t -> count
        
        # 结果存储
        self.selected_avis = set()
        self.selected_uavs = [] # List of (u_idx, t_idx)
        
        # 优先队列 (Max Heap implemented as Min Heap with negative scores)
        # item: (-score, marginal_gain, cost, id_type, id_val)
        # id_type: 'a' for AVI, 'u' for UAV
        # id_val: link_idx for AVI, (u_idx, t_idx) for UAV
        self.avi_queue = []
        self.uav_queue = []
        
        # Lazy Eval 记录：记录上次计算时的 covered_count
        self.avi_last_checked = {} 
        self.uav_last_checked = {}
        
        # UAV 成本一致性标记：记录上次计算 Score 时使用的 Fleet Size
        self.uav_calc_fleet_size = {}

    def _calc_score_avi(self, l_idx):
        """ 计算 AVI 的 ROI。Cost 恒定。"""
        # Gain = new pairs covered
        # Intersection 比较慢，我们只计算增量
        # 优化：Gain = len(profile - covered)
        dist_set = self.avi_profiles[l_idx]
        gain = len(dist_set.difference(self.covered_pairs))
        cost = self.cost_avi
        return gain / cost if cost > 0 else 0, gain, cost

    def _calc_score_uav(self, u_key):
        """ 计算 UAV 的 ROI。Cost 动态耦合。"""
        u_idx, t_idx = u_key
        dist_set = self.uav_profiles[u_key]
        gain = len(dist_set.difference(self.covered_pairs))
        
        # Dynamic Cost Logic
        current_usage = self.uav_usage_per_period[t_idx]
        if current_usage + 1 > self.current_fleet_size:
            cost = self.cost_uav # 需要买新飞机
        else:
            cost = 1e-6 # 边际成本为 0 (设为极小值避免除零)
            
        return gain / cost, gain, cost

    def initialize(self):
        print("--- [Step 3] Initializing Priority Queues ---")
        # Init AVI Queue
        for l_idx in self.avi_profiles:
            score, gain, cost = self._calc_score_avi(l_idx)
            if gain > 0:
                heapq.heappush(self.avi_queue, (-score, l_idx))
                self.avi_last_checked[l_idx] = 0 # Initial covered count was 0
        
        # Init UAV Queue
        for u_key in self.uav_profiles:
            # 初始时 fleet_size = 0, 所有 UAV 都有成本
            score, gain, cost = self._calc_score_uav(u_key)
            if gain > 0:
                heapq.heappush(self.uav_queue, (-score, u_key))
                self.uav_last_checked[u_key] = 0
                self.uav_calc_fleet_size[u_key] = 0 
                
        print(f"   Queues ready. AVI: {len(self.avi_queue)}, UAV: {len(self.uav_queue)}")

    def solve(self):
        print("--- [Step 4] Running Coupled-Cost Greedy Loop ---")
        start_time = time.time()
        
        while len(self.covered_pairs) < self.total_pairs_count:
            # 1. 获取 AVI 最佳候选 (Lazy)
            best_avi = None
            while self.avi_queue:
                neg_score, l_idx = heapq.heappop(self.avi_queue)
                # Lazy Check: 如果上次计算时的已覆盖数量 != 当前已覆盖数量，说明 Gain 可能过时了
                # 严格来说应该检查 Gain 是否变了。这里简单起见，如果 Gain 变小了就要重新 push
                # 我们重新计算一次 Gain
                current_score, gain, cost = self._calc_score_avi(l_idx)
                
                if gain == 0: continue # 没用了
                
                # 如果新分数比队列中下一个还大（或者相等），那它就是真的最大
                # 注意：这里我们简化处理，总是重新计算并 push，直到取出的就是最新的
                # 标准 Lazy: if score <= -self.avi_queue[0][0]: break (found best)
                # 这里为了代码简单，每次取出都重算。如果重算后变小了，就放回去。
                # 如果重算后跟原来一样（或者误差范围内），那就是它了。
                
                if abs(current_score - (-neg_score)) < 1e-9:
                    best_avi = (current_score, gain, cost, l_idx)
                    break
                else:
                    heapq.heappush(self.avi_queue, (-current_score, l_idx))
            
            # 2. 获取 UAV 最佳候选 (Lazy + Dynamic Cost)
            best_uav = None
            while self.uav_queue:
                neg_score, u_key = heapq.heappop(self.uav_queue)
                
                # Cost Consistency Check
                # 检查计算这个分数时使用的 fleet size 是否过期
                last_fleet_size = self.uav_calc_fleet_size.get(u_key, -1)
                
                # 无论如何重新计算 Gain 和 Cost
                current_score, gain, cost = self._calc_score_uav(u_key)
                
                if gain == 0: continue
                
                # 如果分数变了（可能是 Gain 变小，也可能是 Cost 变了），放回去
                if abs(current_score - (-neg_score)) > 1e-9:
                    self.uav_calc_fleet_size[u_key] = self.current_fleet_size
                    heapq.heappush(self.uav_queue, (-current_score, u_key))
                else:
                    best_uav = (current_score, gain, cost, u_key)
                    break
            
            # 3. 决策比较
            if not best_avi and not best_uav:
                print("   No more effective candidates!")
                break
                
            score_a = best_avi[0] if best_avi else -1
            score_u = best_uav[0] if best_uav else -1
            
            if score_a >= score_u:
                # 选 AVI
                _, gain, cost, l_idx = best_avi
                self.selected_avis.add(l_idx)
                self.covered_pairs.update(self.avi_profiles[l_idx])
                # AVI 不改变 Fleet Size，不需要 Dirty Flag
            else:
                # 选 UAV
                _, gain, cost, u_key = best_uav
                self.selected_uavs.append(u_key)
                self.covered_pairs.update(self.uav_profiles[u_key])
                
                # 更新状态
                u_idx, t_idx = u_key
                self.uav_usage_per_period[t_idx] += 1
                
                # 触发器：如果 Fleet Size 增加
                if self.uav_usage_per_period[t_idx] > self.current_fleet_size:
                    self.current_fleet_size = self.uav_usage_per_period[t_idx]
                    # print(f"   [Trigger] Fleet Size Increased to {self.current_fleet_size} by UAV at T={t_idx}")
                    # 这里不需要清空队列，因为我们在 pop 时会做 Cost Consistency Check
            
            # Logging (optional)
            if len(self.covered_pairs) % 5000 == 0:
                print(f"   Covered: {len(self.covered_pairs)}/{self.total_pairs_count} ...")

        end_time = time.time()
        print(f"--- Greedy Finished in {end_time - start_time:.2f}s ---")
        
        # 计算最终目标函数值
        total_cost = len(self.selected_avis) * self.cost_avi + self.current_fleet_size * self.cost_uav
        return total_cost

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 准备数据
    mgr = DynamicAGSLMManager()
    
    # 2. 生成 Universe (几何筛选)
    t0 = time.time()
    universe = mgr.get_universe_of_pairs()
    print(f"   Geometric Universe Size: {len(universe)} pairs.")
    
    # 3. 构建 Profile (预计算覆盖关系)
    avi_prof, uav_prof = mgr.build_candidate_profiles(universe)
    print(f"   Profiles Built. Time: {time.time() - t0:.2f}s")
    
    # 4. 运行贪婪求解器
    solver = CoupledGreedySolver(avi_prof, uav_prof, len(universe), costs=(1.0, 7.0))
    solver.initialize()
    final_obj = solver.solve()
    
    print("\n=== Solution Summary ===")
    print(f"Objective Value: {final_obj}")
    print(f"AVI Sensors: {len(solver.selected_avis)}")
    print(f"UAV Fleet Size: {solver.current_fleet_size}")
    print(f"UAV Deployments: {len(solver.selected_uavs)}")
    
    # Optional: Save Warm Start for Gurobi
    # save_warm_start(solver.selected_avis, solver.selected_uavs)