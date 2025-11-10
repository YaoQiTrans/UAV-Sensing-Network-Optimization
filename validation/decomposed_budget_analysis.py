# decomposed_budget_analysis.py
"""
Decomposes a road network and performs independent budget vs. observable routes
analysis for each subnetwork using the Castillo (2008) partial observability model.
Outputs results and plots to a 'result' subfolder.
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import scipy.sparse as sp
import networkx as nx
import itertools
import time
import os
import data_loader # User-provided data loader
import utils       # Helper functions
import matplotlib.pyplot as plt
import matplotlib

# --- 完全可观测模型求解函数 (OBSV Model Solver - Integer) ---
def solve_obsv_model(num_route_sub, num_link_sub, sub_indicator_matrix,
                     df_link_sub, component_id, time_limit=None):
    """
    Solves the OBSV model (Castillo et al., 2008b) for a given subnetwork (Integer).
    为给定的子路网求解完全可观测模型 (Castillo et al., 2008b)（整数模型）。

    Args:
        num_route_sub (int): Number of routes in the subnetwork.
        num_link_sub (int): Number of links in the subnetwork.
        sub_indicator_matrix (scipy.sparse.csr_matrix): Sub-matrix for the component.
        df_link_sub (pd.DataFrame): Filtered link DataFrame for the subnetwork.
        component_id (int): Identifier for the current subnetwork/component.
        time_limit (int, optional): Time limit for Gurobi in seconds.

    Returns:
        tuple: (optimal_sensor_original_ids, status)
               - optimal_sensor_original_ids (set): Set of original link IDs.
               - status (str): Solver status.
    """
    # print(f"--- Solving OBSV for Subnetwork {component_id} ({num_route_sub} routes, {num_link_sub} links) ---") # Reduced verbosity
    if num_route_sub == 0 or num_link_sub == 0: return set(), 'Empty'
    if num_route_sub == 1:
         if not df_link_sub.empty:
             first_link_original_id = df_link_sub['link_id'].iloc[0]
             return {first_link_original_id}, 'Optimal'
         else: return set(), 'Error'

    try:
        model = gp.Model(f"OBSV_Subnetwork_{component_id}")
        model.Params.OutputFlag = 0
        if time_limit: model.Params.TimeLimit = time_limit
        else: model.Params.TimeLimit = 3600

        z = model.addVars(num_link_sub, vtype=GRB.BINARY, name='z')
        model.setObjective(z.sum(), GRB.MINIMIZE)

        links_per_route_sub_indices = [sub_indicator_matrix.getrow(r).indices for r in range(num_route_sub)]

        # Coverage
        for r_sub_idx in range(num_route_sub):
            link_indices = links_per_route_sub_indices[r_sub_idx]
            if link_indices.size > 0:
                model.addConstr(gp.quicksum(z[link_sub_idx] for link_sub_idx in link_indices) >= 1)

        # Distinguishability
        route_pairs_sub = list(itertools.combinations(range(num_route_sub), 2))
        for r1_sub_idx, r2_sub_idx in route_pairs_sub:
            links_r1_sub = set(links_per_route_sub_indices[r1_sub_idx])
            links_r2_sub = set(links_per_route_sub_indices[r2_sub_idx])
            diff_links_sub_indices = links_r1_sub.symmetric_difference(links_r2_sub)
            if diff_links_sub_indices:
                 model.addConstr(gp.quicksum(z[link_sub_idx] for link_sub_idx in diff_links_sub_indices) >= 1)

        # Optimize
        model.optimize()

        # Process results
        if model.status == GRB.OPTIMAL:
            optimal_sensor_sub_indices = {idx for idx in range(num_link_sub) if z[idx].X > 0.9}
            optimal_sensor_original_ids = utils.map_indices_to_ids(optimal_sensor_sub_indices, df_link_sub['link_id'])
            # print(f"Subnetwork {component_id}: Optimal sensors = {len(optimal_sensor_original_ids)}") # Reduced verbosity
            return optimal_sensor_original_ids, 'Optimal'
        elif model.status == GRB.INFEASIBLE: return set(), 'Infeasible'
        elif model.status == GRB.TIME_LIMIT:
             if model.SolCount > 0:
                 optimal_sensor_sub_indices = {idx for idx in range(num_link_sub) if z[idx].X > 0.9}
                 optimal_sensor_original_ids = utils.map_indices_to_ids(optimal_sensor_sub_indices, df_link_sub['link_id'])
                 return optimal_sensor_original_ids, 'TimeLimit'
             else: return set(), 'TimeLimit_NoSol'
        else: return set(), f'Error_{model.status}'
    except gp.GurobiError as e: return set(), f'GurobiError_{e.errno}'
    except Exception as e: return set(), 'Exception'


# --- 部分可观测模型求解函数 (Partial Observability Solver - Integer) ---
def solve_castillo_partial_observability(num_route_sub, num_link_sub, sub_indicator_matrix,
                                         df_link_sub, component_id, budget, time_limit=None):
    """
    Solves the Castillo (2008) partial observability model for a subnetwork (Integer).
    为给定的子路网求解 Castillo (2008) 部分可观测模型（整数模型）。

    Args:
        num_route_sub (int): Number of routes in the subnetwork.
        num_link_sub (int): Number of links in the subnetwork.
        sub_indicator_matrix (scipy.sparse.csr_matrix): Sub-matrix for the component.
        df_link_sub (pd.DataFrame): Filtered link DataFrame.
        component_id (int): Identifier for the subnetwork.
        budget (int): Sensor budget for this subnetwork.
        time_limit (int, optional): Time limit for Gurobi in seconds.

    Returns:
        tuple: (observable_routes_count, status, sensor_original_ids)
               - observable_routes_count (int): Number of observable routes.
               - status (str): Solver status.
               - sensor_original_ids (set): Set of original link IDs for sensors.
    """
    # print(f"--- Solving Partial Obs for Subnetwork {component_id}, Budget B={budget} ---") # Reduced verbosity
    if num_route_sub == 0 or num_link_sub == 0: return 0, 'Empty', set()
    if budget == 0: return 0, 'Optimal', set()

    model_name = f"PartialObs_Sub{component_id}_B{budget}"
    try:
        model = gp.Model(model_name)
        model.Params.OutputFlag = 0
        if time_limit: model.Params.TimeLimit = time_limit
        model.Params.MIPFocus = 1
        model.Params.Heuristics = 0.8

        # Pre-computation
        links_per_route_sub_indices = [set(sub_indicator_matrix.getrow(r).indices) for r in range(num_route_sub)]
        differing_links_sub = {}
        route_pairs_sub = list(itertools.combinations(range(num_route_sub), 2))
        for r1_sub, r2_sub in route_pairs_sub:
            diff = links_per_route_sub_indices[r1_sub].symmetric_difference(links_per_route_sub_indices[r2_sub])
            differing_links_sub[(r1_sub, r2_sub)] = diff
            differing_links_sub[(r2_sub, r1_sub)] = diff

        # Variables
        z = model.addVars(num_link_sub, vtype=GRB.BINARY, name='z')
        y = model.addVars(num_route_sub, vtype=GRB.BINARY, name='y')

        # Objective
        model.setObjective(y.sum(), GRB.MAXIMIZE)

        # Constraints
        model.addConstr(z.sum() <= budget, name="Budget")

        # Covering
        for r_sub in range(num_route_sub):
            links_r = links_per_route_sub_indices[r_sub]
            if links_r:
                model.addConstr(gp.quicksum(z[link_idx] for link_idx in links_r) >= y[r_sub], name=f"Covering_{r_sub}")

        # Distinguishability
        for r_sub in range(num_route_sub):
            for r_prime_sub in range(num_route_sub):
                if r_sub == r_prime_sub: continue
                diff_links = differing_links_sub.get((r_sub, r_prime_sub), set())
                if not diff_links:
                     model.addConstr(y[r_sub] <= 0, name=f"Indistinguishable_{r_sub}_{r_prime_sub}")
                else:
                    model.addConstr(gp.quicksum(z[a_idx] for a_idx in diff_links) >= y[r_sub], name=f"Differentiate_{r_sub}_from_{r_prime_sub}")

        # Optimize
        model.optimize()

        # Process results
        status = 'Error'
        observable_routes_count = -1
        sensor_original_ids = set()

        if model.status == GRB.OPTIMAL:
            status = 'Optimal'
            observable_routes_count = int(round(y.sum().getValue()))
            sensor_sub_indices = {idx for idx in range(num_link_sub) if z[idx].X > 0.9}
            sensor_original_ids = utils.map_indices_to_ids(sensor_sub_indices, df_link_sub['link_id'])
        elif model.status == GRB.INFEASIBLE: status = 'Infeasible'
        elif model.status == GRB.TIME_LIMIT:
            status = 'TimeLimit'
            if model.SolCount > 0:
                observable_routes_count = int(round(y.sum().getValue()))
                sensor_sub_indices = {idx for idx in range(num_link_sub) if z[idx].X > 0.9}
                sensor_original_ids = utils.map_indices_to_ids(sensor_sub_indices, df_link_sub['link_id'])
            else: observable_routes_count = 0 # No feasible solution found
        else: status = f'Error_{model.status}'

        return observable_routes_count, status, sensor_original_ids

    except gp.GurobiError as e: return -1, f'GurobiError_{e.errno}', set()
    except Exception as e: return -1, 'Exception', set()


# --- 网络分解函数 (Network Decomposition Function) ---
def decompose_network(df_route, df_link, route_link_indicator):
    """
    Decomposes the network into independent subnetworks based on route-link interactions.
    基于路径-路段交互将网络分解为独立的子网络。
    (Code is the same as in previous version, omitted here for brevity,
     assuming it's available or copied from the previous response)
    (代码与先前版本相同，为简洁起见此处省略，假设其可用或从先前响应复制)
    """
    print("\n--- Decomposing Network ---")
    num_route = route_link_indicator.shape[0]
    num_link = route_link_indicator.shape[1]

    if not pd.RangeIndex(start=0, stop=num_link, step=1).equals(df_link.index):
         print("Warning: df_link index is not 0-based sequential. Resetting index.")
         df_link = df_link.reset_index(drop=True)

    print("Building route interaction graph...")
    start_graph_build = time.time()
    route_interaction_graph = nx.Graph()
    route_interaction_graph.add_nodes_from(range(num_route))

    link_route_indicator = route_link_indicator.transpose().tocsr()
    for link_idx in range(num_link):
        routes_on_link = link_route_indicator.getrow(link_idx).indices
        if len(routes_on_link) > 1:
            for u, v in itertools.combinations(routes_on_link, 2):
                if not route_interaction_graph.has_edge(u, v):
                    route_interaction_graph.add_edge(u, v)
    end_graph_build = time.time()
    print(f"Route interaction graph built in {end_graph_build - start_graph_build:.2f}s: "
          f"{route_interaction_graph.number_of_nodes()} nodes, {route_interaction_graph.number_of_edges()} edges.")

    print("Finding connected components...")
    components = list(nx.connected_components(route_interaction_graph))
    # Sort components by size (number of routes) descending
    components.sort(key=len, reverse=True)
    print(f"Found {len(components)} connected components (subnetworks).")

    subnetworks = []
    link_id_to_idx = {link_id: i for i, link_id in enumerate(df_link['link_id'])}

    for i, component_route_indices in enumerate(components):
        component_id = i + 1
        # print(f"Processing Component {component_id}/{len(components)}...") # Reduced verbosity
        component_route_indices = sorted(list(component_route_indices))

        component_link_indices = set()
        for r_idx in component_route_indices:
            component_link_indices.update(route_link_indicator.getrow(r_idx).indices)
        component_link_indices = sorted(list(component_link_indices))

        if not component_route_indices or not component_link_indices:
             # print(f"Component {component_id} has no routes or links. Skipping.") # Reduced verbosity
             continue

        if 'route_id' in df_route.columns:
            original_route_ids = df_route.iloc[component_route_indices]['route_id'].tolist()
            df_route_sub = df_route[df_route['route_id'].isin(original_route_ids)].reset_index(drop=True)
        else:
            df_route_sub = df_route.iloc[component_route_indices].reset_index(drop=True)

        df_link_sub = df_link.iloc[component_link_indices].reset_index(drop=True)
        sub_indicator_matrix = route_link_indicator[component_route_indices, :][:, component_link_indices]

        if sub_indicator_matrix.shape[0] != len(df_route_sub) or sub_indicator_matrix.shape[1] != len(df_link_sub):
             print(f"Error: Submatrix shape mismatch for component {component_id}.")
             continue

        subnetworks.append({
            'component_id': component_id,
            'num_routes': len(component_route_indices),
            'num_links': len(component_link_indices),
            'route_indices': component_route_indices,
            'link_indices': component_link_indices,
            'df_route_sub': df_route_sub,
            'df_link_sub': df_link_sub,
            'sub_indicator_matrix': sub_indicator_matrix.tocsr()
        })
        # print(f"Component {component_id}: {len(component_route_indices)} routes, {len(component_link_indices)} links.") # Reduced verbosity

    return subnetworks


# --- 主执行块 (Main Execution Block) ---
if __name__ == '__main__':
    # --- 配置 Configuration ---
    # route_file = r'data/Cerrone_route.csv'
    # link_file = r'data/Cerrone_link.csv'
    # route_file = r'data/Nguyen_Dupuis_route.csv'
    # link_file = r'data/Nguyen_Dupuis_link.csv'
    route_file = r'data/pneuma_route.csv'
    link_file = r'data/pneuma_link.csv'

    # route_file = r'data/XC_601route.csv'
    # link_file = r'data/XC_578link.csv'

    time_limit_partial = 600    # Partial observability solve time limit
    time_limit_full = 3600     # Full observability solve time limit

    # --- 定义结果文件夹 Define result folder ---
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Created result directory: {result_dir}")

    # --- 加载数据 Load Data ---
    print("Loading network data...")
    df_route, df_link = data_loader.load_network_data(route_file, link_file)

    if df_route is not None and df_link is not None:
        print("Calculating full route-link incidence matrix...")
        route_link_indicator = data_loader.calculate_route_link_incidence(df_route, df_link)

        if route_link_indicator is not None:
            num_route_total = route_link_indicator.shape[0]
            num_link_total = route_link_indicator.shape[1]
            print(f"Full network: {num_route_total} routes, {num_link_total} links.")

            # --- 分解网络 Decompose Network ---
            decomposition_start = time.time()
            subnetworks = decompose_network(df_route, df_link, route_link_indicator)
            decomposition_end = time.time()
            print(f"\nNetwork decomposition finished in {decomposition_end - decomposition_start:.2f} seconds.")
            print(f"Found {len(subnetworks)} subnetworks.")

            if not subnetworks:
                print("Network decomposition failed or resulted in no valid subnetworks.")
            else:
                # --- 分析每个子网络 Analyze Each Subnetwork ---
                all_subnetwork_budget_results = []
                plt.figure(figsize=(12, 8)) # Create figure for plotting

                # --- 设置中文字体 / Set Chinese font ---
                try:
                    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
                    matplotlib.rcParams['axes.unicode_minus'] = False
                    print("Font set to SimHei for plotting.")
                except Exception:
                     try:
                         matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
                         matplotlib.rcParams['axes.unicode_minus'] = False
                         print("Font set to Microsoft YaHei for plotting.")
                     except Exception:
                         print("Warning: Could not set Chinese font. Plot labels might not display correctly.")
                # --- ---

                analysis_start_time = time.time()

                for i, sub_info in enumerate(subnetworks):
                    comp_id = sub_info['component_id']
                    num_r_sub = sub_info['num_routes']
                    num_l_sub = sub_info['num_links']
                    sub_matrix = sub_info['sub_indicator_matrix']
                    df_l_sub = sub_info['df_link_sub']

                    print(f"\nAnalyzing Subnetwork {comp_id} ({num_r_sub} routes, {num_l_sub} links)...")

                    # 1. 求解子网络的完全可观测模型，确定最大预算
                    #    Solve full observability for the subnetwork to find its max budget
                    print(f"  Solving OBSV for Subnetwork {comp_id} to determine max budget...")
                    full_obs_sensors, full_obs_status = solve_obsv_model(
                        num_r_sub, num_l_sub, sub_matrix, df_l_sub, comp_id, time_limit=time_limit_full
                    )
                    max_budget_sub = -1
                    if full_obs_status in ['Optimal', 'TimeLimit'] and full_obs_sensors is not None:
                        max_budget_sub = len(full_obs_sensors)
                        print(f"  Subnetwork {comp_id}: Full observability requires {max_budget_sub} sensors (Status: {full_obs_status}).")
                    else:
                        print(f"  Warning: Could not determine full observability sensors for Subnetwork {comp_id} (Status: {full_obs_status}). Using num_links as max budget.")
                        max_budget_sub = num_l_sub # Fallback to number of links

                    if max_budget_sub <= 0:
                        print(f"  Subnetwork {comp_id}: Max budget is {max_budget_sub}. Skipping partial observability analysis.")
                        continue

                    # 2. 对子网络运行部分可观测性分析
                    #    Run partial observability analysis for the subnetwork
                    budget_range_sub = range(1, max_budget_sub + 1)
                    sub_results_list = []
                    print(f"  Running partial observability analysis for Subnetwork {comp_id} (Budgets 1 to {max_budget_sub})...")
                    for budget in budget_range_sub:
                        obs_count, status, sensors = solve_castillo_partial_observability(
                            num_r_sub, num_l_sub, sub_matrix, df_l_sub, comp_id, budget, time_limit=time_limit_partial
                        )
                        sub_results_list.append({
                            'ComponentID': comp_id,
                            'Budget': budget,
                            'ObservableRoutes': obs_count,
                            'Status': status
                            # 'SensorIDs': sensors # Optional: store sensor IDs if needed
                        })
                        # Simple progress indicator
                        if budget % 10 == 0 or budget == max_budget_sub:
                             print(f"    Budget {budget}/{max_budget_sub}: {obs_count} routes ({status})")


                    sub_results_df = pd.DataFrame(sub_results_list)
                    all_subnetwork_budget_results.append(sub_results_df)

                    # 3. 在图上绘制该子网络的结果
                    #    Plot results for this subnetwork on the main figure
                    plot_data = sub_results_df[sub_results_df['ObservableRoutes'] >= 0] # Plot valid results
                    label = f"子网络 {comp_id} ({num_r_sub} R, {num_l_sub} L)"
                    # Highlight the largest component (first one since sorted)
                    linestyle = '-' if i == 0 else '--'
                    linewidth = 2.5 if i == 0 else 1.0
                    alpha = 1.0 if i == 0 else 0.7
                    plt.plot(plot_data['Budget'], plot_data['ObservableRoutes'], marker='.', linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)

                analysis_end_time = time.time()
                print(f"\nAnalysis for all subnetworks finished in {analysis_end_time - analysis_start_time:.2f} seconds.")

                # --- 完成并保存绘图 Finalize and Save Plot ---
                plt.xlabel("子网络传感器预算 (Sensor Budget within Subnetwork)")
                plt.ylabel("最大可观测路径数 (Max Observable Routes in Subnetwork)")
                plt.title("各子网络预算 vs. 可观测路径数 (Castillo 2008 部分可观测模型)")
                plt.grid(True)
                plt.legend(title="子网络 (按路径数降序)")
                plot_filename_base = f"decomposed_budget_vs_routes_{time.strftime('%Y%m%d_%H%M%S')}.png"
                plot_filepath = os.path.join(result_dir, plot_filename_base)
                try:
                    plt.savefig(plot_filepath)
                    print(f"\nCombined plot saved to {plot_filepath}")
                    # plt.show()
                except Exception as e:
                    print(f"\nError saving combined plot: {e}")

                # --- 保存详细数据 Save Detailed Data (Optional) ---
                if all_subnetwork_budget_results:
                    combined_results_df = pd.concat(all_subnetwork_budget_results, ignore_index=True)
                    results_filename_base = f"decomposed_budget_details_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                    results_filepath = os.path.join(result_dir, results_filename_base)
                    try:
                        combined_results_df.to_csv(results_filepath, index=False)
                        print(f"Detailed budget analysis results saved to {results_filepath}")
                    except Exception as e:
                        print(f"\nError saving detailed results: {e}")

        else:
            print("Failed to calculate incidence matrix.")
    else:
        print("Failed to load data.")

