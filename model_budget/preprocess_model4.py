# preprocess_model4.py
"""
Preprocessing Script for Model 4.
Extends the Model 3 preprocessor to calculate path benefit weights (w_pt).
"""
import os
import pandas as pd
import numpy as np
import itertools
import scipy.sparse
import pickle

# Import configurations from the new config file
try:
    from config_model4 import (
        LINK_FILE, NODE_FILE, PREPROCESSED_DATA_DIR_MODEL4,
        UAV_FOV_WIDTH, UAV_FOV_HEIGHT, CANDIDATE_NEST_NODES, NEST_SERVICE_RADIUS,
        PATH_WEIGHT_STRATEGY
    )
    from data_processor import generate_paths_for_all_periods
except ImportError:
    print("❌ Error: Could not import from config_model4.py or data_processor.py.")
    exit()

# Functions from preprocess_model3.py can be reused directly
# For brevity, we assume they are present here.
# load_processed_paths, create_unified_subnetwork, calculate_static_uav_params,
# calculate_dynamic_params, calculate_nest_params

def load_processed_paths(data_dir):
    """Loads the processed path data."""
    filepath = os.path.join("preprocessed_data_model4", "processed_paths_by_period_undirected.pkl")
    if not os.path.exists(filepath):
        print(f"Path file not found at {filepath}. Running data_processor now...")
        generate_paths_for_all_periods()

    try:
        with open(filepath, 'rb') as f:
            processed_data = pickle.load(f)
        print(f"✅ Successfully loaded processed path data from: {filepath}")
        return processed_data
    except FileNotFoundError:
        print(f"❌ Critical Error: Processed path file not found at {filepath}.")
        return None

def create_unified_subnetwork(paths_by_period, df_link, df_node):
    print("\n--- Creating a unified subnetwork for all time periods ---")
    all_links_in_paths = set()
    for _, df_paths in paths_by_period.items():
        for path_set in df_paths['path_links_set']:
            all_links_in_paths.update(path_set)
    df_link_sub = df_link[df_link['link_id'].isin(all_links_in_paths)].copy().reset_index(drop=True)
    all_nodes_in_links = set(df_link_sub['u_node_id']).union(set(df_link_sub['v_node_id']))
    df_node_sub = df_node[df_node['node_id'].isin(all_nodes_in_links)].copy().reset_index(drop=True)
    df_candidate_nodes_sub = df_node_sub[df_node_sub['node_id'].astype(str).str.len() >= 8].copy()
    print(f"Unified subnetwork created: {len(df_link_sub)} links, {len(df_node_sub)} nodes, {len(df_candidate_nodes_sub)} candidate UAV locations")
    return {"df_link_sub": df_link_sub, "df_node_sub": df_node_sub, "df_candidate_nodes_sub": df_candidate_nodes_sub}

def calculate_static_uav_params(subnetwork_data):
    print("\n--- Calculating static UAV parameters (L_j: FOV Coverage) ---")
    df_link_sub, df_node_sub, candidate_nodes = subnetwork_data['df_link_sub'], subnetwork_data['df_node_sub'], subnetwork_data['df_candidate_nodes_sub']
    node_id_to_coords = {row['node_id']: (row['x_coord'], row['y_coord']) for _, row in df_node_sub.iterrows()}
    uav_fov_coverage = {}
    for _, uav_node in candidate_nodes.iterrows():
        uav_node_id, uav_pos_x, uav_pos_y = uav_node['node_id'], uav_node['x_coord'], uav_node['y_coord']
        fov_x_min, fov_x_max = uav_pos_x - UAV_FOV_WIDTH / 2, uav_pos_x + UAV_FOV_WIDTH / 2
        fov_y_min, fov_y_max = uav_pos_y - UAV_FOV_HEIGHT / 2, uav_pos_y + UAV_FOV_HEIGHT / 2
        covered_link_ids = set()
        for _, link in df_link_sub.iterrows():
            start_coords, end_coords = node_id_to_coords.get(link['u_node_id']), node_id_to_coords.get(link['v_node_id'])
            if (start_coords and fov_x_min <= start_coords[0] <= fov_x_max and fov_y_min <= start_coords[1] <= fov_y_max) or \
               (end_coords and fov_x_min <= end_coords[0] <= fov_x_max and fov_y_min <= end_coords[1] <= fov_y_max):
                covered_link_ids.add(link['link_id'])
        uav_fov_coverage[uav_node_id] = list(covered_link_ids)
    print("✅ L_j calculation complete.")
    return {"L_j": uav_fov_coverage}

def calculate_dynamic_params(paths_by_period, subnetwork_data, static_uav_params):
    print("\n--- Calculating dynamic, time-dependent parameters (delta_ap_t, d_jpq_t) ---")
    df_link_sub, candidate_nodes = subnetwork_data['df_link_sub'], subnetwork_data['df_candidate_nodes_sub']
    L_j = static_uav_params['L_j']
    link_id_to_idx_map = {link_id: i for i, link_id in enumerate(df_link_sub['link_id'])}
    num_links_sub = len(df_link_sub)
    inc_matrix_by_period, d_jpq_by_period = {}, {}
    for period_name, df_paths in paths_by_period.items():
        print(f"\n  Processing for period: '{period_name}'...")
        num_routes_period = len(df_paths)
        rows, cols, data = [], [], []
        for r_idx, path_row in df_paths.iterrows():
            for link_id in path_row['path_links_set']:
                if link_id in link_id_to_idx_map:
                    l_idx = link_id_to_idx_map[link_id]
                    rows.append(r_idx)
                    cols.append(l_idx)
                    data.append(1)
        inc_matrix = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(num_routes_period, num_links_sub))
        inc_matrix_by_period[period_name] = inc_matrix
        d_jpq_for_period = {}
        for _, uav_node in candidate_nodes.iterrows():
            uav_node_id = uav_node['node_id']
            d_matrix = scipy.sparse.dok_matrix((num_routes_period, num_routes_period), dtype=np.int8)
            links_in_fov_ids = set(L_j.get(uav_node_id, []))
            if links_in_fov_ids:
                traces = {r_idx: path_row['path_links_set'].intersection(links_in_fov_ids) for r_idx, path_row in df_paths.iterrows()}
                for r1_idx, r2_idx in itertools.combinations(range(num_routes_period), 2):
                    if traces[r1_idx] != traces[r2_idx]:
                        d_matrix[r1_idx, r2_idx] = 1
            d_jpq_for_period[uav_node_id] = d_matrix.tocsr()
        d_jpq_by_period[period_name] = d_jpq_for_period
    return {"inc_matrix_by_period": inc_matrix_by_period, "d_jpq_by_period": d_jpq_by_period}

def calculate_nest_params(subnetwork_data, df_node_full):
    print("\n--- Calculating Nest Service Association Parameter (w_hj) ---")
    candidate_uav_nodes = subnetwork_data['df_candidate_nodes_sub']
    node_coords_map = df_node_full.set_index('node_id')[['x_coord', 'y_coord']].to_dict('index')
    valid_nest_nodes = [h for h in CANDIDATE_NEST_NODES if h in node_coords_map]
    nest_coords = {h: node_coords_map[h] for h in valid_nest_nodes}
    uav_site_coords = {j: node_coords_map[j] for j in candidate_uav_nodes['node_id'] if j in node_coords_map}
    nest_ids = list(nest_coords.keys())
    uav_site_ids = list(uav_site_coords.keys())
    nest_id_to_idx = {nid: i for i, nid in enumerate(nest_ids)}
    uav_site_id_to_idx = {nid: i for i, nid in enumerate(uav_site_ids)}
    num_nests = len(nest_coords)
    num_uav_sites = len(uav_site_coords)
    w_hj = np.zeros((num_nests, num_uav_sites), dtype=int)
    for h_idx, h_id in enumerate(nest_ids):
        h_coord = np.array([nest_coords[h_id]['x_coord'], nest_coords[h_id]['y_coord']])
        for j_idx, j_id in enumerate(uav_site_ids):
            j_coord = np.array([uav_site_coords[j_id]['x_coord'], uav_site_coords[j_id]['y_coord']])
            distance = np.linalg.norm(h_coord - j_coord)
            if distance <= NEST_SERVICE_RADIUS:
                w_hj[h_idx, j_idx] = 1
    print(f"✅ w_hj calculation complete. Shape: {w_hj.shape}")
    nest_params = {
        "w_hj": w_hj,
        "nest_ids": nest_ids,
        "uav_site_ids_for_nest_calc": uav_site_ids
    }
    return nest_params


# --- NEW FUNCTION FOR MODEL 4 ---
def calculate_path_weights(paths_by_period):
    """
    Calculates the benefit weight for observing each path in each time period.

    Args:
        paths_by_period (dict): Dictionary with period names as keys and
                                path DataFrames as values.

    Returns:
        dict: A dictionary containing the path weights w_pt.
              Format: {'period_name': [weight_p1, weight_p2, ...]}
    """
    print("\n--- Calculating Path Benefit Weights (w_pt) ---")
    path_weights = {}
    for period_name, df_paths in paths_by_period.items():
        if PATH_WEIGHT_STRATEGY == 'uniform':
            # 策略1: 统一权重，所有路径权重为1
            weights = [1.0] * len(df_paths)
            print(f"  - Period '{period_name}': Assigned uniform weight of 1.0 to {len(weights)} paths.")
        elif PATH_WEIGHT_STRATEGY == 'flow_based':
            # 策略2: 基于流量的权重 (可以取消注释以使用)
            # weights = df_paths['hourly_flow'].tolist()
            # print(f"  - Period '{period_name}': Assigned flow-based weights to {len(weights)} paths.")
            # 暂时保持统一权重
            weights = [1.0] * len(df_paths)
            print(f"  - Period '{period_name}': (Flow-based selected, but using uniform for now).")
        else:
            raise ValueError(f"Unknown PATH_WEIGHT_STRATEGY: {PATH_WEIGHT_STRATEGY}")

        path_weights[period_name] = weights
    print("✅ Path benefit weights calculation complete.")
    return {"path_weights": path_weights}


def save_preprocessed_data(data, file_path):
    """Saves the final preprocessed data dictionary using pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"\n✅ All preprocessed data for Model 4 saved to: {file_path}")
    except Exception as e:
        print(f"❌ Error saving preprocessed data: {e}")

if __name__ == "__main__":
    print("--- Running Full Preprocessing Workflow for Model 4 (B-DSLM) ---")

    # Reuse the path data generated for Model 3
    paths_by_period = load_processed_paths(PREPROCESSED_DATA_DIR_MODEL4)

    if paths_by_period:
        try:
            df_link_full = pd.read_csv(LINK_FILE)
            df_node_full = pd.read_csv(NODE_FILE)
        except FileNotFoundError as e:
            print(f"❌ Critical Error: Base network file not found. {e}")
            exit()

        # Standard parameter calculation
        subnetwork_data = create_unified_subnetwork(paths_by_period, df_link_full, df_node_full)
        static_uav_params = calculate_static_uav_params(subnetwork_data)
        dynamic_params = calculate_dynamic_params(paths_by_period, subnetwork_data, static_uav_params)
        nest_params = calculate_nest_params(subnetwork_data, df_node_full)

        # NEW step for Model 4
        path_weight_params = calculate_path_weights(paths_by_period)

        # Combine all data and save
        final_preprocessed_data = {
            "subnetwork_data": subnetwork_data,
            "paths_by_period": paths_by_period,
            "static_uav_params": static_uav_params,
            "dynamic_params": dynamic_params,
            "nest_params": nest_params,
            "path_weight_params": path_weight_params # Add new params
        }

        output_filepath = os.path.join(PREPROCESSED_DATA_DIR_MODEL4, "model4_preprocessed_data.pkl")
        save_preprocessed_data(final_preprocessed_data, output_filepath)

        print("\n--- Preprocessing for Model 4 successfully completed! ---")
