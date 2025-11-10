# decompose_and_solve.py
"""
Decomposes a road network based on route-link interactions and solves the
Castillo (2008) full observability (OBSV) model for each independent subnetwork.
Outputs results to a 'result' subfolder.
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import scipy.sparse as sp
import networkx as nx # For graph operations and connected components
import itertools
import time
import os         # Import os module for directory operations
import data_loader # User-provided data loader
import utils       # Helper functions

# --- 完全可观测模型求解函数 (OBSV Model Solver) ---
def solve_obsv_model(num_route_sub, num_link_sub, sub_indicator_matrix,
                     df_link_sub, component_id, time_limit=None):
    """
    Solves the OBSV model (Castillo et al., 2008b) for a given subnetwork.
    为给定的子路网求解完全可观测模型 (Castillo et al., 2008b)。

    Args:
        num_route_sub (int): Number of routes in the subnetwork. 子路网中的路径数量。
        num_link_sub (int): Number of links in the subnetwork. 子路网中的路段数量。
        sub_indicator_matrix (scipy.sparse.csr_matrix): Sub-matrix for the component. 子网络的关联矩阵。
        df_link_sub (pd.DataFrame): Filtered link DataFrame for the subnetwork,
                                    containing original 'link_id'. 子路网的过滤后的 Link DataFrame。
        component_id (int): Identifier for the current subnetwork/component. 当前子网络的标识符。
        time_limit (int, optional): Time limit for Gurobi in seconds. Gurobi 时间限制（秒）。

    Returns:
        tuple: (optimal_sensor_original_ids, status)
               - optimal_sensor_original_ids (set): Set of original link IDs for optimal sensors. 最优传感器的原始 link ID 集合。
               - status (str): Solver status. 求解器状态。
    """
    print(f"--- Solving OBSV for Subnetwork {component_id} ({num_route_sub} routes, {num_link_sub} links) ---")
    if num_route_sub == 0 or num_link_sub == 0:
        print("Subnetwork is empty, skipping.")
        return set(), 'Empty'
    if num_route_sub == 1:
         # If only one route, need one sensor on any link of that route
         print("Subnetwork has only one route. Placing one sensor.")
         # Ensure df_link_sub is not empty before accessing iloc[0]
         if not df_link_sub.empty:
             first_link_original_id = df_link_sub['link_id'].iloc[0]
             return {first_link_original_id}, 'Optimal'
         else:
             print(f"Warning: Subnetwork {component_id} has 1 route but no associated links found in df_link_sub. Cannot place sensor.")
             return set(), 'Error' # Or 'Empty' if no links means no sensor needed


    try:
        model = gp.Model(f"OBSV_Subnetwork_{component_id}")
        model.Params.OutputFlag = 0 # Suppress Gurobi output
        if time_limit:
            model.Params.TimeLimit = time_limit
        else:
             model.Params.TimeLimit = 3600 # Default 1 hour

        # Decision variable: z_a = 1 if sensor on link index a (within subproblem), 0 otherwise
        # 决策变量：z_a = 1 如果传感器在子问题中的路段索引 a 上，否则为 0
        z = model.addVars(num_link_sub, vtype=GRB.BINARY, name='z')

        # Objective: Minimize the number of AVI sensors
        # 目标：最小化 AVI 传感器数量
        model.setObjective(z.sum(), GRB.MINIMIZE)

        # --- Constraints specific to the subnetwork ---
        # --- 针对子网络的约束 ---

        # Pre-calculate links per route within the subnetwork
        # 预计算子网络中每条路径的路段（使用子矩阵的 0-based 索引）
        # Indices here are 0-based relative to sub_indicator_matrix columns
        links_per_route_sub_indices = [sub_indicator_matrix.getrow(r).indices for r in range(num_route_sub)]

        # 1. Path Coverage
        # print("Adding C1 (Path Coverage) constraints...") # Reduced verbosity
        for r_sub_idx in range(num_route_sub):
            link_indices_for_route = links_per_route_sub_indices[r_sub_idx]
            if link_indices_for_route.size > 0:
                # Use subproblem link indices (0 to num_link_sub-1)
                model.addConstr(gp.quicksum(z[link_sub_idx] for link_sub_idx in link_indices_for_route) >= 1,
                                name=f"PathCoverage_{r_sub_idx}")

        # 2. Path Distinguishability
        # print(f"Adding C2 (Path Distinguishability) constraints...") # Reduced verbosity
        route_pairs_sub = list(itertools.combinations(range(num_route_sub), 2))
        for r1_sub_idx, r2_sub_idx in route_pairs_sub:
            links_r1_sub = set(links_per_route_sub_indices[r1_sub_idx])
            links_r2_sub = set(links_per_route_sub_indices[r2_sub_idx])
            # Differing links within the subproblem's indexing
            diff_links_sub_indices = links_r1_sub.symmetric_difference(links_r2_sub)

            if diff_links_sub_indices:
                 # Use subproblem link indices
                 model.addConstr(gp.quicksum(z[link_sub_idx] for link_sub_idx in diff_links_sub_indices) >= 1,
                                 name=f"PathDistinguish_{r1_sub_idx}_{r2_sub_idx}")

        # Optimize
        # print("Optimizing OBSV model for subnetwork...") # Reduced verbosity
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        print(f"Subnetwork {component_id} optimization finished in {end_time - start_time:.2f} seconds.")

        # Process results
        if model.status == GRB.OPTIMAL:
            # Get the 0-based indices *within the subproblem*
            optimal_sensor_sub_indices = {idx for idx in range(num_link_sub) if z[idx].X > 0.9}
            # Map subproblem indices back to original link IDs using df_link_sub
            # df_link_sub's index should correspond to 0..num_link_sub-1
            optimal_sensor_original_ids = utils.map_indices_to_ids(optimal_sensor_sub_indices, df_link_sub['link_id'])
            print(f"Subnetwork {component_id}: Optimal sensors = {len(optimal_sensor_original_ids)}")
            return optimal_sensor_original_ids, 'Optimal'
        elif model.status == GRB.INFEASIBLE:
            print(f"Subnetwork {component_id}: OBSV Model is infeasible.")
            return set(), 'Infeasible'
        elif model.status == GRB.TIME_LIMIT:
             print(f"Subnetwork {component_id}: OBSV Model reached time limit.")
             if model.SolCount > 0:
                 print("Returning best integer solution found within time limit.")
                 optimal_sensor_sub_indices = {idx for idx in range(num_link_sub) if z[idx].X > 0.9}
                 optimal_sensor_original_ids = utils.map_indices_to_ids(optimal_sensor_sub_indices, df_link_sub['link_id'])
                 print(f"Subnetwork {component_id}: Suboptimal sensors = {len(optimal_sensor_original_ids)}")
                 return optimal_sensor_original_ids, 'TimeLimit'
             else:
                 print("No integer solution found within time limit.")
                 return set(), 'TimeLimit'
        else:
            print(f"Subnetwork {component_id}: OBSV Optimization ended with status: {model.status}")
            return set(), 'Error'

    except gp.GurobiError as e:
        print(f"Gurobi error occurred in solve_obsv_model for subnetwork {component_id}: {e.message}")
        return set(), 'Error'
    except Exception as e:
        print(f"An unexpected error occurred in solve_obsv_model for subnetwork {component_id}: {e}")
        import traceback
        traceback.print_exc()
        return set(), 'Error'


# --- 网络分解函数 (Network Decomposition Function) ---
def decompose_network(df_route, df_link, route_link_indicator):
    """
    Decomposes the network into independent subnetworks based on route-link interactions.
    基于路径-路段交互将网络分解为独立的子网络。

    Args:
        df_route (pd.DataFrame): Full route DataFrame. 完整的路径 DataFrame。
        df_link (pd.DataFrame): Full link DataFrame. 完整的路段 DataFrame。
        route_link_indicator (scipy.sparse.csr_matrix): Full route-link incidence matrix. 完整的关联矩阵。

    Returns:
        list: A list of dictionaries, where each dictionary represents a subnetwork
              and contains: 'component_id', 'route_indices', 'link_indices',
              'df_route_sub', 'df_link_sub', 'sub_indicator_matrix'.
              一个字典列表，每个字典代表一个子网络，包含上述键值。
              Returns empty list if decomposition fails. 如果分解失败则返回空列表。
    """
    print("\n--- Decomposing Network ---")
    num_route = route_link_indicator.shape[0]
    num_link = route_link_indicator.shape[1]

    # Ensure df_link has a 0-based sequential index for mapping
    # 确报 df_link 有一个从 0 开始的顺序索引用于映射
    if not pd.RangeIndex(start=0, stop=num_link, step=1).equals(df_link.index):
         print("Warning: df_link index is not 0-based sequential. Resetting index.")
         df_link = df_link.reset_index(drop=True)

    # Build route interaction graph
    # 构建路径交互图
    print("Building route interaction graph...")
    start_graph_build = time.time()
    route_interaction_graph = nx.Graph()
    route_interaction_graph.add_nodes_from(range(num_route)) # Use 0-based route indices

    # Efficiently find shared links using matrix transpose
    # 使用矩阵转置高效地查找共享链接
    link_route_indicator = route_link_indicator.transpose().tocsr()
    for link_idx in range(num_link):
        routes_on_link = link_route_indicator.getrow(link_idx).indices
        if len(routes_on_link) > 1:
            # Add edges between all pairs of routes sharing this link
            # 在共享此路段的所有路径对之间添加边
            for u, v in itertools.combinations(routes_on_link, 2):
                if not route_interaction_graph.has_edge(u, v):
                    route_interaction_graph.add_edge(u, v)
        # Progress indicator (optional)
        # if (link_idx + 1) % 1000 == 0:
        #     print(f"  Processed {link_idx + 1}/{num_link} links for graph...")

    end_graph_build = time.time()
    print(f"Route interaction graph built in {end_graph_build - start_graph_build:.2f}s: "
          f"{route_interaction_graph.number_of_nodes()} nodes, {route_interaction_graph.number_of_edges()} edges.")

    # Find connected components
    # 查找连通分量
    print("Finding connected components...")
    components = list(nx.connected_components(route_interaction_graph))
    print(f"Found {len(components)} connected components (subnetworks).")

    subnetworks = []
    # Map original link IDs to 0-based indices (needed for submatrix extraction)
    # 将原始 link ID 映射到 0-based 索引 (提取子矩阵时需要)
    link_id_to_idx = {link_id: i for i, link_id in enumerate(df_link['link_id'])}

    # Process each component to create subnetwork data
    # 处理每个分量以创建子网络数据
    for i, component_route_indices in enumerate(components):
        component_id = i + 1
        print(f"Processing Component {component_id}/{len(components)}...")
        component_route_indices = sorted(list(component_route_indices)) # 0-based route indices

        # Find all links used by routes in this component
        # 查找此分量中路径使用的所有路段
        component_link_indices = set()
        for r_idx in component_route_indices:
            component_link_indices.update(route_link_indicator.getrow(r_idx).indices)
        component_link_indices = sorted(list(component_link_indices)) # 0-based link indices

        if not component_route_indices or not component_link_indices:
             print(f"Component {component_id} has no routes or links. Skipping.")
             continue

        # Filter DataFrames for the subnetwork
        # 为子网络过滤 DataFrame
        # Use original route IDs for filtering df_route if available, else use indices
        # 如果有原始 route ID，则使用它过滤 df_route，否则使用索引
        if 'route_id' in df_route.columns:
            # Ensure route IDs are hashable (e.g., int or str) before using isin
            original_route_ids = df_route.iloc[component_route_indices]['route_id'].tolist()
            df_route_sub = df_route[df_route['route_id'].isin(original_route_ids)].reset_index(drop=True)
        else:
            df_route_sub = df_route.iloc[component_route_indices].reset_index(drop=True)

        # Filter df_link using 0-based indices
        # 使用 0-based 索引过滤 df_link
        df_link_sub = df_link.iloc[component_link_indices].reset_index(drop=True)

        # Extract the sub-indicator matrix
        # 提取子关联矩阵
        # Rows: component_route_indices, Columns: component_link_indices
        # Need to select rows based on component_route_indices and columns based on component_link_indices
        sub_indicator_matrix = route_link_indicator[component_route_indices, :][:, component_link_indices]

        if sub_indicator_matrix.shape[0] != len(df_route_sub) or sub_indicator_matrix.shape[1] != len(df_link_sub):
             print(f"Error: Submatrix shape mismatch for component {component_id}. "
                   f"Matrix: {sub_indicator_matrix.shape}, Routes: {len(df_route_sub)}, Links: {len(df_link_sub)}")
             continue # Skip this component on error

        subnetworks.append({
            'component_id': component_id,
            'route_indices': component_route_indices, # 0-based indices in original matrix
            'link_indices': component_link_indices,   # 0-based indices in original matrix
            'df_route_sub': df_route_sub,
            'df_link_sub': df_link_sub,
            'sub_indicator_matrix': sub_indicator_matrix.tocsr() # Ensure CSR format
        })
        print(f"Component {component_id}: {len(component_route_indices)} routes, {len(component_link_indices)} links.")

    return subnetworks


# --- 主执行块 (Main Execution Block) ---
if __name__ == '__main__':
    # --- 配置 Configuration ---

    route_file = r'data/PMEUMA_460_route.csv'
    link_file = r'data/PMEUMA_402_link.csv'

    # OBSV 求解器的时间限制（秒），可选
    # Time limit per OBSV solve (seconds), optional
    time_limit_per_solve = 3600 # e.g., 1 hour

    # --- 定义结果文件夹 Define result folder ---
    result_dir = "result"
    # 检查并创建文件夹 Check and create folder
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

            if not subnetworks:
                print("Network decomposition failed or resulted in no valid subnetworks.")
            else:
                # --- 求解每个子网络 Solve Each Subnetwork ---
                all_subnetwork_results = []
                total_sensors_needed = 0
                combined_sensor_set = set()
                solve_start = time.time()

                for sub_info in subnetworks:
                    num_route_sub = sub_info['sub_indicator_matrix'].shape[0]
                    num_link_sub = sub_info['sub_indicator_matrix'].shape[1]

                    # Call OBSV solver for the subproblem
                    # 调用 OBSV 求解器处理子问题
                    sub_sensor_ids, sub_status = solve_obsv_model(
                        num_route_sub,
                        num_link_sub,
                        sub_info['sub_indicator_matrix'],
                        sub_info['df_link_sub'], # Pass filtered df_link for this subproblem
                        sub_info['component_id'],
                        time_limit=time_limit_per_solve
                    )

                    # Store results for this subnetwork
                    # 存储此子网络的结果
                    sub_result = {
                        'component_id': sub_info['component_id'],
                        'num_routes': num_route_sub,
                        'num_links': num_link_sub,
                        'status': sub_status,
                        'num_sensors': len(sub_sensor_ids),
                        'sensor_ids': sub_sensor_ids # Contains original link IDs
                    }
                    all_subnetwork_results.append(sub_result)

                    # Aggregate results
                    # 汇总结果
                    if sub_status in ['Optimal', 'TimeLimit']:
                        total_sensors_needed += len(sub_sensor_ids)
                        combined_sensor_set.update(sub_sensor_ids)
                    else:
                         print(f"Warning: Subnetwork {sub_info['component_id']} solve failed or was infeasible. Results might be incomplete.")

                solve_end = time.time()
                print(f"\nSolving all subnetworks finished in {solve_end - solve_start:.2f} seconds.")

                # --- 输出摘要 Output Summary ---
                print("\n--- Decomposition and Solving Summary ---")
                print(f"Number of independent subnetworks found: {len(subnetworks)}")
                print("\nSubnetwork Details:")
                for res in all_subnetwork_results:
                    print(f"  Subnetwork {res['component_id']}: {res['num_routes']} routes, {res['num_links']} links -> Status: {res['status']}, Sensors: {res['num_sensors']}") # Sensors: {res['sensor_ids']}

                print(f"\nTotal sensors needed (sum over optimal subnetworks): {total_sensors_needed}")
                print(f"Combined optimal sensor set size (union): {len(combined_sensor_set)}")
                # print(f"Combined optimal sensor set (Original Link IDs): {sorted(list(combined_sensor_set))}") # Can be very long

                # 可以选择将结果保存到文件
                # Optionally save results to file
                try:
                    # --- 构建输出文件路径 Build output file paths ---
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    summary_filename_base = f"decomposition_summary_{timestamp}.csv"
                    combined_filename_base = f"combined_sensors_{timestamp}.txt"

                    summary_filepath = os.path.join(result_dir, summary_filename_base)
                    combined_filepath = os.path.join(result_dir, combined_filename_base)
                    # --- ---

                    summary_df = pd.DataFrame(all_subnetwork_results)
                    # Convert sensor_ids set to string for CSV compatibility
                    summary_df['sensor_ids_str'] = summary_df['sensor_ids'].apply(lambda x: str(sorted(list(x))))
                    summary_df.drop(columns=['sensor_ids']).to_csv(summary_filepath, index=False)
                    print(f"\nSubnetwork summary saved to {summary_filepath}")

                    # Save combined sensor list
                    with open(combined_filepath, 'w') as f:
                         f.write(f"Total Sensors: {len(combined_sensor_set)}\n")
                         f.write("Sensor Link IDs:\n")
                         for sensor_id in sorted(list(combined_sensor_set)):
                             f.write(f"{sensor_id}\n")
                    print(f"Combined sensor list saved to {combined_filepath}")

                except Exception as e:
                    print(f"\nError saving results: {e}")


        else:
            print("Failed to calculate incidence matrix.")
    else:
        print("Failed to load data.")

