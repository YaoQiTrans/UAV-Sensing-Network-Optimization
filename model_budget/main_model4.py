# main_model4.py
"""
Main Execution Script for Model 4 (B-DSLM).
This script iterates through a list of budgets, solves the model for each,
and saves the results, including detailed observable path information.
"""
import os
import pickle
import time
import pandas as pd

try:
    from config_model4 import (
        PREPROCESSED_DATA_DIR_MODEL4, RESULTS_DIR_MODEL4, LINK_FILE, NODE_FILE,
        BUDGETS # Import the list of budgets
    )
    from preprocess_model4 import (
        load_processed_paths, create_unified_subnetwork,
        calculate_static_uav_params, calculate_dynamic_params,
        calculate_nest_params, calculate_path_weights, save_preprocessed_data
    )
    from solver_model4 import solve_model4
    from visualizer_model4 import plot_model4_solution # Import the new visualizer
except ImportError as e:
    print(f"❌ Critical Error: Could not import a required module. Details: {e}")
    exit()

def save_solution_to_file(solution, filename):
    """Saves the detailed solution for Model 4 to a text file."""
    filepath = os.path.join(RESULTS_DIR_MODEL4, filename)
    budget = solution['budget']
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"  Model 4: B-DSLM - Solution for Budget = {budget}\n")
            f.write("="*60 + "\n\n")

            f.write(f"Budget Constraint: {budget:.2f}\n")
            f.write(f"Maximized Total Benefit (Observed Paths): {solution['total_benefit']:.2f}\n")
            f.write(f"Actual Total Investment Cost: {solution['total_cost']:.2f}\n")
            f.write(f"  - AVI Investment Cost: {solution['cost_breakdown']['avi_investment_cost']:.2f}\n")
            f.write(f"  - UAV Fleet Investment Cost: {solution['cost_breakdown']['uav_investment_cost']:.2f}\n")
            f.write(f"  - Nest Investment Cost: {solution['cost_breakdown']['nest_investment_cost']:.2f}\n\n")

            f.write(f"Required UAV Fleet Size (U_max): {solution['solved_uav_fleet_size']}\n\n")

            f.write("-" * 60 + "\n")
            f.write("Static Infrastructure Deployment\n")
            f.write("-" * 60 + "\n")
            f.write(f"Number of AVIs: {len(solution['avi_sensor_link_ids'])}\n")
            f.write(f"  - Link IDs: {sorted(list(solution['avi_sensor_link_ids']))}\n\n")
            f.write(f"Number of Nests: {len(solution['built_nest_ids'])}\n")
            f.write(f"  - Node IDs: {sorted(list(solution['built_nest_ids']))}\n\n")

            f.write("-" * 60 + "\n")
            f.write("Dynamic UAV Deployment by Period\n")
            f.write("-" * 60 + "\n")
            for period, node_ids in solution['uav_deployment_by_period'].items():
                f.write(f"Period: {period.replace('_', ' ').title()}\n")
                f.write(f"  - Number of UAVs Used: {len(node_ids)}\n")
                f.write(f"  - Deployed at Node IDs: {sorted(list(node_ids))}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("Path Observability by Period\n")
            f.write("-" * 60 + "\n")
            for period, path_indices in solution['observed_paths_by_period'].items():
                f.write(f"Period: {period.replace('_', ' ').title()}\n")
                f.write(f"  - Number of Paths Observed: {len(path_indices)}\n")
                # f.write(f"  - Observed Path Indices: {sorted(path_indices)}\n\n") # Optional: too verbose

        print(f"✅ Detailed solution for budget {budget} saved to: {filepath}")
    except Exception as e:
        print(f"❌ Error saving solution file for budget {budget}: {e}")

def save_observable_paths_to_csv(solution, preprocessed_data, budget, timestamp):
    """
    Saves the observability status of all candidate paths to CSV files for a given solution.
    For each time period, a CSV is generated containing all candidate paths, with a new
    'Observable' column indicating if the path was observed (1) or not (0).
    """
    results_dir = RESULTS_DIR_MODEL4
    paths_by_period = preprocessed_data['paths_by_period']
    observed_paths_by_period = solution['observed_paths_by_period']

    print("\n--- Generating observable path CSV files ---")
    try:
        # Iterate over each time period (e.g., 'morning_peak')
        for period_name, df_paths in paths_by_period.items():
            # Create a copy to avoid modifying the original preprocessed data in memory
            df_output = df_paths.copy()

            # Get the indices of observed paths for the current period from the solution
            observed_indices = observed_paths_by_period.get(period_name, [])

            # Add the 'Observable' column and default its value to 0 (not observed)
            df_output['Observable'] = 0

            # If there are observed paths, set 'Observable' to 1 for those specific paths using their indices
            if observed_indices:
                df_output.loc[observed_indices, 'Observable'] = 1
            
            # The 'path_links_set' column contains frozenset objects, which are not ideal for CSV.
            # Convert them to a string representation of a list for better readability.
            df_output['path_links_set'] = df_output['path_links_set'].apply(lambda x: str(sorted(list(x))))

            # Construct a unique filename for the CSV and save it
            csv_filename = f"model4_paths_budget_{budget}_{period_name}_{timestamp}.csv"
            filepath = os.path.join(results_dir, csv_filename)
            df_output.to_csv(filepath, index_label='path_index')
            
            print(f"  - ✅ Saved path observability data for '{period_name}' to: {filepath}")

    except Exception as e:
        print(f"❌ Error saving observable path CSV files for budget {budget}: {e}")


def run_model4_workflow():
    """Executes the full end-to-end workflow for Model 4."""
    print("="*80)
    print("      STARTING MODEL 4 WORKFLOW (B-DSLM)")
    print("="*80)

    preprocessed_file = os.path.join(PREPROCESSED_DATA_DIR_MODEL4, "model4_preprocessed_data.pkl")

    if os.path.exists(preprocessed_file):
        print(f"\n--- Found existing preprocessed data. Loading from file... ---")
        with open(preprocessed_file, 'rb') as f:
            preprocessed_data = pickle.load(f)
        print("✅ Preprocessed data loaded successfully.")
    else:
        print("\n--- Preprocessed data not found. Starting full data processing workflow... ---")
        paths_by_period = load_processed_paths(PREPROCESSED_DATA_DIR_MODEL4)
        if not paths_by_period: return
        try:
            df_link_full = pd.read_csv(LINK_FILE)
            df_node_full = pd.read_csv(NODE_FILE)
        except FileNotFoundError as e:
            print(f"❌ Critical Error: Base network file not found. {e}")
            return
        
        subnetwork_data = create_unified_subnetwork(paths_by_period, df_link_full, df_node_full)
        static_uav_params = calculate_static_uav_params(subnetwork_data)
        dynamic_params = calculate_dynamic_params(paths_by_period, subnetwork_data, static_uav_params)
        nest_params = calculate_nest_params(subnetwork_data, df_node_full)
        path_weight_params = calculate_path_weights(paths_by_period)
        
        preprocessed_data = {
            "subnetwork_data": subnetwork_data, "paths_by_period": paths_by_period,
            "static_uav_params": static_uav_params, "dynamic_params": dynamic_params,
            "nest_params": nest_params, "path_weight_params": path_weight_params
        }
        save_preprocessed_data(preprocessed_data, preprocessed_file)
        print("✅ Full preprocessing workflow for Model 4 completed.")

    # --- Loop through each budget and solve ---
    for budget in BUDGETS:
        solution, status = solve_model4(preprocessed_data, budget)

        if status in ["Optimal", "Time Limit"] and solution:
            print(f"\n\n--- [SUCCESS] Solution Found for Budget {budget} (Status: {status})! ---")
            print(f"  - Maximized Benefit (Observed Paths): {solution['total_benefit']:.2f}")
            print(f"  - Actual Cost: {solution['total_cost']:.2f} (Budget: {budget})")
            print(f"  - Required UAV Fleet Size: {solution['solved_uav_fleet_size']}")
            print(f"  - Static AVI Sensors Deployed: {len(solution['avi_sensor_link_ids'])}")
            print(f"  - Nests Built: {len(solution['built_nest_ids'])}")

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            solution_filename = f"model4_solution_budget_{budget}_{timestamp}.txt"
            plot_filename = f"model4_solution_budget_{budget}_{timestamp}.svg"
            
            # --- Save all results for the current solution ---
            save_solution_to_file(solution, solution_filename)
            plot_model4_solution(solution, preprocessed_data, plot_filename)
            # Call the new function to save observable path details to CSV
            save_observable_paths_to_csv(solution, preprocessed_data, budget, timestamp)

        else:
            print(f"\n\n--- [FAILURE] Could not find a valid solution for Budget {budget}. Solver status: {status} ---")

    print("\n" + "="*80)
    print("      MODEL 4 WORKFLOW FINISHED")
    print("="*80)

if __name__ == "__main__":
    run_model4_workflow()
