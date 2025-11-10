# data_processor.py
"""
Data Processor for Static Models.

This script processes raw trajectory data from MULTIPLE specified time periods
to generate a unified, high-flow path set for static model analysis.
The final path set is the UNION of the Top-K paths from each period.
"""
import os
import pandas as pd
from tqdm import tqdm
import pickle
import ast

from config import (
    LINK_FILE, TRAJECTORY_DATA_DIRS, NUM_TOP_PATHS_PER_PERIOD, MIN_PATH_LENGTH,
    PREPROCESSED_DATA_DIR
)

def safe_str_to_list(s):
    """Safely converts a string representation of a list to a list."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

def reconstruct_unordered_path(df_traj, link_uv_to_id_map):
    """
    Reconstructs an UNORDERED path (frozenset of link IDs) from a trajectory.
    """
    if df_traj.empty or 'matchlink' not in df_traj.columns:
        return None
    
    node_pairs = [tuple(safe_str_to_list(l)) for l in df_traj['matchlink'] if safe_str_to_list(l)]
    link_ids = {link_uv_to_id_map.get(tuple(sorted(uv))) for uv in node_pairs}
    link_ids.discard(None)
    
    if len(link_ids) >= MIN_PATH_LENGTH:
        return frozenset(link_ids)
    return None

def generate_paths_from_trajectories_union():
    """
    Main function to process trajectories from all specified periods and
    generate a single DataFrame containing the UNION of their Top-K paths.
    """
    print("--- Starting Path Generation from Union of Trajectory Data ---")
    
    output_path_file = os.path.join(PREPROCESSED_DATA_DIR, "top_paths_union.pkl")
    if os.path.exists(output_path_file):
        print(f"✅ Found existing unified path file: {output_path_file}. Skipping generation.")
        with open(output_path_file, 'rb') as f:
            return pickle.load(f)

    # 1. Load link data to create an UNDIRECTED link-to-ID map
    try:
        df_link = pd.read_csv(LINK_FILE)
        link_uv_to_id_map = {
            tuple(sorted((row.u_node_id, row.v_node_id))): row.link_id 
            for _, row in df_link.iterrows()
        }
        print(f"✅ Loaded {len(df_link)} links to create undirected map.")
    except FileNotFoundError:
        print(f"❌ Critical Error: Link file not found at {LINK_FILE}")
        return None

    # 2. Iterate through each time period to collect Top-K paths
    all_top_paths_sets = set()
    for period_name, traj_dir in TRAJECTORY_DATA_DIRS.items():
        print(f"\n--- Processing Time Period: {period_name} ---")
        if not os.path.isdir(traj_dir):
            print(f"⚠️ Warning: Directory not found for '{period_name}': {traj_dir}. Skipping.")
            continue

        path_counts = {}
        traj_files = [os.path.join(traj_dir, f) for f in os.listdir(traj_dir) if f.endswith('.csv')]
        if not traj_files:
            print(f"⚠️ Warning: No trajectory files found in {traj_dir}. Skipping.")
            continue

        print(f"Reconstructing paths for '{period_name}'...")
        for filepath in tqdm(traj_files, desc=f"Processing {period_name}"):
            try:
                df_traj = pd.read_csv(filepath)
                unordered_path = reconstruct_unordered_path(df_traj, link_uv_to_id_map)
                if unordered_path:
                    path_counts[unordered_path] = path_counts.get(unordered_path, 0) + 1
            except Exception:
                continue
        
        if not path_counts:
            print(f"⚠️ Warning: No valid paths generated for {period_name}. Skipping.")
            continue

        path_list = [{'path_links_set': k, 'flow_count': v} for k, v in path_counts.items()]
        df_period_paths = pd.DataFrame(path_list)
        
        num_to_select = min(NUM_TOP_PATHS_PER_PERIOD, len(df_period_paths))
        df_top_paths_period = df_period_paths.nlargest(num_to_select, 'flow_count')
        
        print(f"Generated {len(df_period_paths)} unique paths, selected Top {len(df_top_paths_period)}.")
        
        # Add the frozensets of the top paths to our master set
        for path_set in df_top_paths_period['path_links_set']:
            all_top_paths_sets.add(path_set)

    # 3. Create the final unified DataFrame from the set of all top paths
    if not all_top_paths_sets:
        print("❌ Error: No paths were collected from any time period. Aborting.")
        return None

    final_path_list = [{'path_links_set': path} for path in all_top_paths_sets]
    df_unified_paths = pd.DataFrame(final_path_list).reset_index(drop=True)
    
    print("\n--------------------------------------------------")
    print(f"Total unique paths in the final union set: {len(df_unified_paths)}")
    print("--------------------------------------------------")

    # 4. Save the final unified DataFrame
    try:
        with open(output_path_file, 'wb') as f:
            pickle.dump(df_unified_paths, f)
        print(f"\n✅ Unified Top-K paths data saved to: {output_path_file}")
    except Exception as e:
        print(f"❌ Error saving unified path data: {e}")

    return df_unified_paths

if __name__ == "__main__":
    generate_paths_from_trajectories_union()
