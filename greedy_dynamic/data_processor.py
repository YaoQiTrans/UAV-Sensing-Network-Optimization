# data_processor.py
"""
Data Processor for Experiment Extension 1.

This script is responsible for processing raw trajectory data from multiple 
time periods into structured, high-flow path sets.

*** Undirected Graph Version ***
- Paths are defined as UNORDERED sets of link IDs (frozenset).
- The network is treated as an undirected graph.
"""
import os
import pandas as pd
from tqdm import tqdm
import pickle
import ast

# Import configurations from the new config file
try:
    from config_model3 import (
        LINK_FILE, TRAJECTORY_DATA_DIRS, NUM_TOP_PATHS, MIN_PATH_LENGTH,
        PREPROCESSED_DATA_DIR_MODEL3
    )
except ImportError:
    print("❌ Error: Could not import from config_model3.py. Make sure the file exists.")
    exit()

def safe_str_to_list(s):
    """Safely converts a string representation of a list to a list."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

def reconstruct_unordered_path(df_traj, link_uv_to_id_map):
    """
    Reconstructs an UNORDERED path (frozenset of link IDs) from a trajectory.
    The network is treated as an undirected graph.

    Args:
        df_traj (pd.DataFrame): DataFrame for a single vehicle's trajectory.
        link_uv_to_id_map (dict): Mapping from sorted (u, v) node tuples to link_id.

    Returns:
        frozenset: An unordered frozenset of link IDs representing the path, or None.
    """
    if df_traj.empty or 'matchlink' not in df_traj.columns:
        return None
    
    node_pairs = [tuple(safe_str_to_list(l)) for l in df_traj['matchlink'] if safe_str_to_list(l)]
    
    # Map node pairs to link IDs using the undirected map (sorted tuples)
    link_ids = {link_uv_to_id_map.get(tuple(sorted(uv))) for uv in node_pairs}
    link_ids.discard(None) # Remove None if a link was not found
    
    if len(link_ids) >= MIN_PATH_LENGTH:
        return frozenset(link_ids)
    return None

def generate_paths_for_all_periods():
    """
    Main function to process trajectories for all time periods and generate
    the corresponding top-k path sets based on an undirected graph model.
    """
    print("--- Starting Data Processing for Extension 1 (Undirected Graph Mode) ---")
    
    # 1. Load link data to create an UNDIRECTED link-to-ID map
    try:
        df_link = pd.read_csv(LINK_FILE)
        # For an undirected graph, the key is the sorted tuple of nodes
        link_uv_to_id_map = {
            tuple(sorted((row.u_node_id, row.v_node_id))): row.link_id 
            for _, row in df_link.iterrows()
        }
        print(f"✅ Loaded {len(df_link)} links to create undirected map.")
    except FileNotFoundError:
        print(f"❌ Critical Error: Link file not found at {LINK_FILE}")
        return None

    all_periods_path_data = {}

    # 2. Iterate through each defined time period
    for period_name, traj_dir in TRAJECTORY_DATA_DIRS.items():
        print(f"\n--- Processing Time Period: {period_name} ---")
        if not os.path.isdir(traj_dir):
            print(f"⚠️ Warning: Directory not found for '{period_name}': {traj_dir}. Skipping.")
            continue

        path_counts = {}
        total_observation_seconds = 0
        
        traj_files = [os.path.join(traj_dir, f) for f in os.listdir(traj_dir) if f.endswith('.csv')]
        if not traj_files:
            print(f"⚠️ Warning: No trajectory files found in {traj_dir}. Skipping.")
            continue
            
        print("Calculating total observation duration...")
        for filepath in tqdm(traj_files, desc=f"Scanning time in {period_name}"):
            try:
                df_time = pd.read_csv(filepath, usecols=['time'])
                if not df_time.empty:
                    total_observation_seconds = max(total_observation_seconds, df_time['time'].max())
            except Exception:
                continue

        if total_observation_seconds == 0:
            print(f"⚠️ Warning: Could not determine observation duration for {period_name}. Skipping.")
            continue
        
        total_observation_hours = total_observation_seconds / 3600.0
        print(f"Total duration for '{period_name}': {total_observation_hours:.2f} hours.")

        # 3. Reconstruct paths (as unordered sets) and count flows
        print("Reconstructing paths and counting flows...")
        for filepath in tqdm(traj_files, desc=f"Processing paths in {period_name}"):
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

        # 4. Convert to DataFrame, calculate hourly flow, and filter top paths
        path_list = [
            {'path_links_set': k, 'hourly_flow': v / total_observation_hours} # Changed column name
            for k, v in path_counts.items()
        ]
        df_period_paths = pd.DataFrame(path_list)
        
        num_to_select = min(NUM_TOP_PATHS, len(df_period_paths))
        df_top_paths = df_period_paths.nlargest(num_to_select, 'hourly_flow').reset_index(drop=True)
        
        print(f"Generated {len(df_period_paths)} unique paths, selected Top {len(df_top_paths)}.")
        all_periods_path_data[period_name] = df_top_paths

    # 5. Save the final dictionary of DataFrames
    output_filename = os.path.join(PREPROCESSED_DATA_DIR_MODEL3, "processed_paths_by_period_undirected.pkl")
    try:
        with open(output_filename, 'wb') as f:
            pickle.dump(all_periods_path_data, f)
        print(f"\n✅ All periods processed. Data saved to: {output_filename}")
    except Exception as e:
        print(f"❌ Error saving processed data: {e}")

    return all_periods_path_data

if __name__ == "__main__":
    generate_paths_for_all_periods()
