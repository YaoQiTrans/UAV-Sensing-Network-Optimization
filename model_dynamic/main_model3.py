# main_model3.py
"""
Main Execution Script for Model 3 (DSLM with Nests).
"""
import os
import pickle
import time
import pandas as pd

try:
    from config_model3 import (
        PREPROCESSED_DATA_DIR_MODEL3, RESULTS_DIR_MODEL3, LINK_FILE, NODE_FILE
    )
    from preprocess_model3 import (
        load_processed_paths, create_unified_subnetwork,
        calculate_static_uav_params, calculate_dynamic_params,
        save_preprocessed_data
    )
    from solver_model3 import solve_model3
    from visualizer_model3 import plot_model3_solution
except ImportError as e:
    print(f"❌ Critical Error: Could not import a required module. Details: {e}")
    exit()

def save_solution_to_file(solution, filename):
    """Saves the detailed solution for Model 3 to a text file."""
    filepath = os.path.join(RESULTS_DIR_MODEL3, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("    Model 3: DSLM with Nests - Solution\n")
            f.write("="*60 + "\n\n")

            f.write(f"Total Minimized Investment Cost: {solution['total_cost']:.2f}\n")
            f.write(f"  - AVI Investment Cost: {solution['cost_breakdown']['avi_investment_cost']:.2f}\n")
            f.write(f"  - UAV Fleet Investment Cost: {solution['cost_breakdown']['uav_investment_cost']:.2f}\n")
            # f.write(f"  - Nest Investment Cost: {solution['cost_breakdown']['nest_investment_cost']:.2f}\n\n")

            f.write(f"Required UAV Fleet Size (U_max): {solution['solved_uav_fleet_size']}\n\n")

            f.write("-" * 60 + "\n")
            f.write("Static Infrastructure Deployment\n")
            f.write("-" * 60 + "\n")
            f.write(f"Number of AVIs: {len(solution['avi_sensor_link_ids'])}\n")
            f.write(f"  - Link IDs: {sorted(list(solution['avi_sensor_link_ids']))}\n\n")
            # f.write(f"Number of Nests: {len(solution['built_nest_ids'])}\n")
            # f.write(f"  - Node IDs: {sorted(list(solution['built_nest_ids']))}\n\n")

            f.write("-" * 60 + "\n")
            f.write("Dynamic UAV Deployment by Period\n")
            f.write("-" * 60 + "\n")
            for period, node_ids in solution['uav_deployment_by_period'].items():
                f.write(f"Period: {period.replace('_', ' ').title()}\n")
                f.write(f"  - Number of UAVs Used: {len(node_ids)}\n")
                f.write(f"  - Deployed at Node IDs: {sorted(list(node_ids))}\n\n")

        print(f"✅ Detailed solution saved to: {filepath}")
    except Exception as e:
        print(f"❌ Error saving solution file: {e}")

def run_model3_workflow():
    """Executes the full end-to-end workflow for Model 3."""
    print("="*80)
    print("      STARTING MODEL 3 WORKFLOW (DSLM WITH NESTS)")
    print("="*80)

    preprocessed_file = os.path.join(PREPROCESSED_DATA_DIR_MODEL3, "model3_preprocessed_data.pkl")

    if os.path.exists(preprocessed_file):
        print(f"\n--- Found existing preprocessed data. Loading from file... ---")
        with open(preprocessed_file, 'rb') as f:
            preprocessed_data = pickle.load(f)
        print("✅ Preprocessed data loaded successfully.")
    else:
        print("\n--- Preprocessed data not found. Starting full data processing workflow... ---")
        paths_by_period = load_processed_paths("preprocessed_data_ext1")
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
        # nest_params = calculate_nest_params(subnetwork_data, df_node_full) # New step
        
        preprocessed_data = {
            "subnetwork_data": subnetwork_data, "paths_by_period": paths_by_period,
            "static_uav_params": static_uav_params, "dynamic_params": dynamic_params
            # "nest_params": nest_params
        }
        save_preprocessed_data(preprocessed_data, preprocessed_file)
        print("✅ Full preprocessing workflow for Model 3 completed.")

    solution, status = solve_model3(preprocessed_data)

    if status in ["Optimal", "Time Limit"] and solution:
        print(f"\n\n--- [SUCCESS] Solution Found (Status: {status})! ---")
        print(f"  - Total Minimized Investment Cost: {solution['total_cost']:.2f}")
        print(f"  - Required UAV Fleet Size: {solution['solved_uav_fleet_size']}")
        print(f"  - Static AVI Sensors Deployed: {len(solution['avi_sensor_link_ids'])}")
        # print(f"  - Nests Built: {len(solution['built_nest_ids'])}")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_solution_to_file(solution, f"model3_solution_{timestamp}.txt")
        plot_model3_solution(solution, preprocessed_data)
    else:
        print(f"\n\n--- [FAILURE] Could not find a valid solution. Solver status: {status} ---")

    print("\n" + "="*80)
    print("      MODEL 3 WORKFLOW FINISHED")
    print("="*80)

if __name__ == "__main__":
    run_model3_workflow()
