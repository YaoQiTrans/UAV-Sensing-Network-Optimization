# preprocess.py
"""
Preprocessing Script for Static Models from Trajectory Data.
This version uses a unified path set from multiple time periods.
"""
import os
import time
import pandas as pd
import numpy as np
import itertools
import scipy.sparse
import pickle

from config import (
    LINK_FILE, NODE_FILE, PREPROCESSED_DATA_DIR,
    UAV_FOV_WIDTH, UAV_FOV_HEIGHT
)
# Use the new data processing function
from data_processor import generate_paths_from_trajectories_union
from utils import save_preprocessed_data

def create_subnetwork_from_paths(df_unified_paths, df_link_full, df_node_full):
    """
    Creates a subnetwork containing all links and nodes from the given path set.
    """
    print("\n--- Creating subnetwork from the unified path set ---")
    
    all_links_in_paths = set()
    for path_set in df_unified_paths['path_links_set']:
        all_links_in_paths.update(path_set)
            
    df_link_sub = df_link_full[df_link_full['link_id'].isin(all_links_in_paths)].copy().reset_index(drop=True)
    
    all_nodes_in_links = set(df_link_sub['u_node_id']).union(set(df_link_sub['v_node_id']))
    df_node_sub = df_node_full[df_node_full['node_id'].isin(all_nodes_in_links)].copy().reset_index(drop=True)
    
    df_candidate_nodes_sub = df_node_sub[df_node_sub['node_id'].astype(str).str.len() >= 8].copy()

    print(f"Subnetwork created:")
    print(f"  - {len(df_unified_paths)} paths")
    print(f"  - {len(df_link_sub)} links")
    print(f"  - {len(df_node_sub)} nodes")
    print(f"  - {len(df_candidate_nodes_sub)} candidate UAV locations")

    return {
        "df_path_sub": df_unified_paths,
        "df_link_sub": df_link_sub,
        "df_node_sub": df_node_sub,
        "df_candidate_nodes_sub": df_candidate_nodes_sub
    }

def calculate_model_parameters(subnetwork_data):
    """
    Calculates path-link incidence matrix (delta_ap), UAV FOV coverage (L_j),
    and path distinguishability (d_jpq).
    """
    print("\n--- Calculating all model parameters (delta_ap, L_j, d_jpq) ---")
    df_path_sub = subnetwork_data['df_path_sub']
    df_link_sub = subnetwork_data['df_link_sub']
    df_node_sub = subnetwork_data['df_node_sub']
    candidate_nodes = subnetwork_data['df_candidate_nodes_sub']
    
    num_paths = len(df_path_sub)
    num_links = len(df_link_sub)
    link_id_to_idx_map = {link_id: i for i, link_id in enumerate(df_link_sub['link_id'])}

    # 1. Calculate Path-Link Incidence Matrix (delta_ap)
    rows, cols, data = [], [], []
    for r_idx, path_row in df_path_sub.iterrows():
        for link_id in path_row['path_links_set']:
            if link_id in link_id_to_idx_map:
                l_idx = link_id_to_idx_map[link_id]
                rows.append(r_idx)
                cols.append(l_idx)
                data.append(1)
    incidence_matrix = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(num_paths, num_links))
    print(f"1. Path-Link Incidence Matrix (delta_ap) created, shape: {incidence_matrix.shape}")

    # 2. Calculate UAV FOV Coverage (L_j)
    node_id_to_coords = {row['node_id']: (row['x_coord'], row['y_coord']) for _, row in df_node_sub.iterrows()}
    L_j = {} 
    for _, uav_node in candidate_nodes.iterrows():
        uav_node_id = uav_node['node_id']
        uav_pos_x, uav_pos_y = uav_node['x_coord'], uav_node['y_coord']
        fov_x_min, fov_x_max = uav_pos_x - UAV_FOV_WIDTH / 2, uav_pos_x + UAV_FOV_WIDTH / 2
        fov_y_min, fov_y_max = uav_pos_y - UAV_FOV_HEIGHT / 2, uav_pos_y + UAV_FOV_HEIGHT / 2
        
        covered_link_ids = set()
        for _, link in df_link_sub.iterrows():
            start_coords = node_id_to_coords.get(link['u_node_id'])
            end_coords = node_id_to_coords.get(link['v_node_id'])
            if (start_coords and (fov_x_min <= start_coords[0] <= fov_x_max) and (fov_y_min <= start_coords[1] <= fov_y_max)) or \
               (end_coords and (fov_x_min <= end_coords[0] <= fov_x_max) and (fov_y_min <= end_coords[1] <= fov_y_max)):
                covered_link_ids.add(link['link_id'])
        
        covered_link_indices = [link_id_to_idx_map[lid] for lid in covered_link_ids if lid in link_id_to_idx_map]
        L_j[uav_node_id] = covered_link_indices
    print("2. UAV FOV Coverage (L_j) calculation complete.")

    # 3. Calculate UAV Path Distinguishability (d_jpq)
    start_time = time.time()
    d_jpq = {}
    for _, uav_node in candidate_nodes.iterrows():
        uav_node_id = uav_node['node_id']
        d_matrix = scipy.sparse.dok_matrix((num_paths, num_paths), dtype=np.int8)
        
        covered_indices = L_j.get(uav_node_id, [])
        idx_to_link_id_map = {i: link_id for link_id, i in link_id_to_idx_map.items()}
        links_in_fov_ids = {idx_to_link_id_map[idx] for idx in covered_indices}

        if not links_in_fov_ids:
            d_jpq[uav_node_id] = d_matrix.tocsr()
            continue

        traces = {
            r_idx: path_row['path_links_set'].intersection(links_in_fov_ids)
            for r_idx, path_row in df_path_sub.iterrows()
        }
        for r1, r2 in itertools.combinations(range(num_paths), 2):
            if traces[r1] != traces[r2]:
                d_matrix[r1, r2] = 1
        d_jpq[uav_node_id] = d_matrix.tocsr()
    print(f"3. UAV Path Distinguishability (d_jpq) calculated in {time.time() - start_time:.2f} seconds.")

    return {
        "sub_incidence_matrix": incidence_matrix,
        "L_j": L_j,
        "d_jpq": d_jpq
    }

def run_full_preprocessing():
    """Main function to execute the entire preprocessing pipeline."""
    preprocessed_file = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed_data_bundle.pkl")
    if os.path.exists(preprocessed_file):
        print(f"✅ Found fully preprocessed data file: {preprocessed_file}. Skipping.")
        return

    # Step 1: Generate the unified Top-K path set from all trajectory data
    df_unified_paths = generate_paths_from_trajectories_union()
    if df_unified_paths is None or df_unified_paths.empty:
        print("❌ Preprocessing failed: Could not generate paths from trajectory data.")
        return

    # Step 2: Load full network data
    try:
        df_link_full = pd.read_csv(LINK_FILE)
        df_node_full = pd.read_csv(NODE_FILE)
    except FileNotFoundError as e:
        print(f"❌ Critical Error: Base network file not found. {e}")
        return

    # Step 3: Create subnetwork and calculate parameters
    subnetwork_data = create_subnetwork_from_paths(df_unified_paths, df_link_full, df_node_full)


    # # Step 3: Create subnetwork and calculate parameters
    # # --- START OF MODIFICATION ---
    # # Comment out the original subnetwork creation
    # # subnetwork_data = create_subnetwork_from_paths(df_unified_paths, df_link_full, df_node_full)

    # print("\n--- [BENCHMARK MODE] SKIPPING subnetwork creation. Using FULL network graph ---")
    
    # # Manually build the 'subnetwork_data' dict using the FULL dataframes
    # df_candidate_nodes_full = df_node_full[df_node_full['node_id'].astype(str).str.len() >= 8].copy()
    # subnetwork_data = {
    #     "df_path_sub": df_unified_paths,
    #     "df_link_sub": df_link_full,      # <-- Use the full link df
    #     "df_node_sub": df_node_full,      # <-- Use the full node df
    #     "df_candidate_nodes_sub": df_candidate_nodes_full # <-- Use candidates from full df
    # }
    # print(f"  - {len(df_unified_paths)} paths")
    # print(f"  - {len(df_link_full)} links (Full Network)")
    # print(f"  - {len(df_node_full)} nodes (Full Network)")
    # print(f"  - {len(df_candidate_nodes_full)} candidate UAV locations (Full Network)")
    
    # # --- END OF MODIFICATION ---

    model_params = calculate_model_parameters(subnetwork_data)
    
    
    # Step 4: Bundle and save all data
    final_data_bundle = {**subnetwork_data, **model_params}
    save_preprocessed_data(final_data_bundle, preprocessed_file)
    print("\n--- ✅ Full preprocessing workflow completed successfully! ---")

if __name__ == "__main__":
    run_full_preprocessing()
