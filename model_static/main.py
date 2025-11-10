# main.py
"""
Main Execution Script
Orchestrates the experimental workflow for static models using paths
generated directly from trajectory data.
"""
import os
import sys
import subprocess

from config import (
    PREPROCESSED_DATA_DIR, COST_GROUND_SENSOR, UAV_GROUND_COST_RATIO, RESULTS_DIR
)
from utils import load_preprocessed_data
from solver import (
    solve_uav_only_model,
    solve_air_ground_model,
    solve_baseline_castillo
)
from visualizer import plot_deployment

def save_solution_to_file(solution, model_name, filename, cost_ratio):
    """Saves a detailed solution to a text file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"****** Solution for: {model_name} ******\n")
        f.write(f"Data Source: Trajectory-based (Top {solution.get('num_paths', 'N/A')} paths)\n")
        f.write(f"Cost Ratio (UAV:AVI): {cost_ratio}:1\n\n")
        
        total_cost = solution.get('total_cost', 'N/A')
        f.write(f"Optimal Total Deployment Cost: {total_cost:.2f}\n")

        ground_ids = sorted(list(solution.get('ground_sensor_ids', set())))
        uav_ids = sorted(list(solution.get('uav_node_ids', set())))
        
        f.write(f"Number of Ground Sensors: {len(ground_ids)}\n")
        f.write(f"Number of UAVs: {len(uav_ids)}\n\n")
        
        if ground_ids:
            f.write(f"Ground Sensor Locations (Link IDs):\n{ground_ids}\n\n")
        if uav_ids:
            f.write(f"UAV Deployment Locations (Node IDs):\n{uav_ids}\n")
            
    print(f"✅ Solution for {model_name} saved to: {filepath}")

def run_experiment_from_trajectory():
    """Runs the full experiment for all static models."""
    # 1. Preprocessing Check
    preprocessed_file = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed_data_bundle.pkl")
    if not os.path.exists(preprocessed_file):
        print("--- Preprocessed data not found, running preprocess.py... ---")
        try:
            # The new preprocess.py handles the entire workflow
            subprocess.run([sys.executable, "preprocess.py"], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ Preprocessing script failed: {e}")
            return
    
    # 2. Load Data
    print("\n--- Loading preprocessed data bundle... ---")
    data = load_preprocessed_data(preprocessed_file)
    if not data:
        print("❌ Failed to load preprocessed data, experiment aborted.")
        return
        
    # Prepare data subsets for solvers
    subnetwork_data = {
        "df_path_sub": data["df_path_sub"],
        "df_link_sub": data["df_link_sub"],
        "df_node_sub": data["df_node_sub"],
        "df_candidate_nodes_sub": data["df_candidate_nodes_sub"], 
        "sub_incidence_matrix": data["sub_incidence_matrix"]
    }
    uav_params = {
        "L_j": data["L_j"],
        "d_jpq": data["d_jpq"]
    }
    
    cost_uav = COST_GROUND_SENSOR * UAV_GROUND_COST_RATIO
    print(f"\n--- Starting experiments with Cost Ratio (UAV:Ground) = {UAV_GROUND_COST_RATIO}:1 ---")

    # --- Solve, Save, and Plot Each Model ---

    # Model 1: UAV-Only Location Model (ULM)
    ulm_solution, ulm_status = solve_uav_only_model(subnetwork_data, uav_params, cost_uav)
    if ulm_status == "Optimal":
        ulm_solution['num_paths'] = len(subnetwork_data['df_path_sub'])
        print_summary("UAV-Only Model (ULM)", ulm_solution)
        save_solution_to_file(ulm_solution, "UAV-Only Model (ULM)", f"solution_ulm_ratio_{UAV_GROUND_COST_RATIO}.txt", UAV_GROUND_COST_RATIO)
        plot_deployment(ulm_solution, subnetwork_data, UAV_GROUND_COST_RATIO, "UAV-Only Model (ULM)", "ulm")
    else:
        print(f"--- [Result] UAV-Only Model (ULM) failed, status: {ulm_status} ---")

    # Model 2: Air-Ground Sensor Location Model (AG-SLM)
    ag_solution, ag_status = solve_air_ground_model(subnetwork_data, uav_params, COST_GROUND_SENSOR, cost_uav)
    if ag_status == "Optimal":
        ag_solution['num_paths'] = len(subnetwork_data['df_path_sub'])
        print_summary("Air-Ground Model (AG-SLM)", ag_solution)
        save_solution_to_file(ag_solution, "Air-Ground Model (AG-SLM)", f"solution_agslm_ratio_{UAV_GROUND_COST_RATIO}.txt", UAV_GROUND_COST_RATIO)
        plot_deployment(ag_solution, subnetwork_data, UAV_GROUND_COST_RATIO, "Air-Ground Model (AG-SLM)", "agslm")
    else:
        print(f"--- [Result] Air-Ground Model (AG-SLM) failed, status: {ag_status} ---")

    # Baseline: Ground-Only Model
    baseline_solution, baseline_status = solve_baseline_castillo(subnetwork_data, COST_GROUND_SENSOR)
    if baseline_status == "Optimal":
        baseline_solution['num_paths'] = len(subnetwork_data['df_path_sub'])
        print_summary("Baseline Ground-Only Model", baseline_solution)
        save_solution_to_file(baseline_solution, "Baseline Ground-Only Model", "solution_baseline.txt", UAV_GROUND_COST_RATIO)
        plot_deployment(baseline_solution, subnetwork_data, UAV_GROUND_COST_RATIO, "Baseline Ground-Only Model", "baseline")
    else:
        print(f"--- [Result] Baseline Ground-Only Model failed, status: {baseline_status} ---")

def print_summary(model_name, solution):
    """Prints a formatted summary of a solution."""
    print(f"\n--- [Result] {model_name} ---")
    print(f"  - Optimal Total Cost: {solution.get('total_cost', 'N/A'):.2f}")
    print(f"  - Number of Ground Sensors: {len(solution.get('ground_sensor_ids', set()))}")
    print(f"  - Number of UAVs: {len(solution.get('uav_node_ids', set()))}")

if __name__ == "__main__":
    run_experiment_from_trajectory()
