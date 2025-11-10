# solver_model3.py
"""
Model Solver for Model 3: Dynamic Sensor Location Model (DSLM) with Nest Selection.
"""
import gurobipy as gp
from gurobipy import GRB
import time
import itertools
import pickle
import os

try:
    from config_model3 import (
        COST_AVI_INVESTMENT, COST_UAV_INVESTMENT,
        UAV_FLEET_SIZE, GUROBI_TIME_LIMIT
    )
except ImportError:
    print("❌ Error: Could not import from config_model3.py.")
    exit()

def solve_model3(preprocessed_data):
    """
    Solves the DSLM with Nest Selection.

    Args:
        preprocessed_data (dict): The complete data package from preprocess_model3.py.

    Returns:
        tuple: (solution, status)
    """
    print("\n--- Solving Model 3 (DSLM with Nests) ---")
    start_time = time.time()

    # --- Unpack data ---
    sub_data = preprocessed_data['subnetwork_data']
    paths_by_period = preprocessed_data['paths_by_period']
    static_uav_params = preprocessed_data['static_uav_params']
    dynamic_params = preprocessed_data['dynamic_params']
    # nest_params = preprocessed_data['nest_params'] # New nest data

    df_link_sub = sub_data['df_link_sub']
    df_candidate_uav_nodes = sub_data['df_candidate_nodes_sub']
    
    L_j = static_uav_params['L_j']
    inc_matrix_by_period = dynamic_params['inc_matrix_by_period']
    d_jpq_by_period = dynamic_params['d_jpq_by_period']
    
    # w_hj = nest_params['w_hj']
    # nest_ids = nest_params['nest_ids']
    
    time_periods = list(paths_by_period.keys())
    num_links = len(df_link_sub)
    num_candidate_uav_nodes = len(df_candidate_uav_nodes)
    # num_candidate_nests = len(nest_ids)
    
    candidate_uav_node_ids = df_candidate_uav_nodes['node_id'].tolist()
    uav_node_id_to_idx = {nid: i for i, nid in enumerate(candidate_uav_node_ids)}
    
    try:
        model = gp.Model("DSLM_with_Nests")
        model.Params.OutputFlag = 1
        model.Params.TimeLimit = GUROBI_TIME_LIMIT

        # --- Decision Variables ---
        z = model.addVars(num_links, vtype=GRB.BINARY, name="z_a")
        u = model.addVars(num_candidate_uav_nodes, len(time_periods), vtype=GRB.BINARY, name="u_jt")
        U_max = model.addVar(vtype=GRB.INTEGER, name="U_max", lb=0)
        # y = model.addVars(num_candidate_nests, vtype=GRB.BINARY, name="y_h") # NEW: Nest selection

        # --- Objective Function (Updated for Model 3) ---
        avi_cost = gp.quicksum(z[l] * COST_AVI_INVESTMENT for l in range(num_links))
        uav_fleet_cost = U_max * COST_UAV_INVESTMENT
        # nest_investment_cost = gp.quicksum(y[h] * COST_NEST_INVESTMENT for h in range(num_candidate_nests))
        
        model.setObjective(avi_cost + uav_fleet_cost, GRB.MINIMIZE) # Updated Objective
        print("Objective function set for AVI + UAV Fleet + Nest investment.")

        # --- Constraints ---
        
        # C1: Fleet Size Definition
        for t_idx, period_name in enumerate(time_periods):
            model.addConstr(gp.quicksum(u[j, t_idx] for j in range(num_candidate_uav_nodes)) <= U_max,
                            name=f"Define_Umax_{period_name}")
        print("  - Added fleet size definition constraints (U_max).")

        # C2 & C3: Time-of-Day Path Coverage & Distinguishability
        for t_idx, period_name in enumerate(time_periods):
            print(f"  - Adding path constraints for time period: '{period_name}'")
            inc_matrix = inc_matrix_by_period[period_name]
            d_jpq_period = d_jpq_by_period[period_name]
            num_routes_period = inc_matrix.shape[0]

            for r in range(num_routes_period):
                links_on_route_indices = inc_matrix.getrow(r).indices
                ground_coverage = gp.quicksum(z[l] for l in links_on_route_indices)
                uav_coverage = gp.quicksum(
                    u[uav_node_id_to_idx[node_id], t_idx] 
                    for node_id in candidate_uav_node_ids 
                    if any(df_link_sub.iloc[l_idx]['link_id'] in L_j.get(node_id, []) for l_idx in links_on_route_indices)
                )
                model.addConstr(ground_coverage + uav_coverage >= 1, name=f"PathCoverage_{period_name}_{r}")

            for r1, r2 in itertools.combinations(range(num_routes_period), 2):
                links_r1 = set(inc_matrix.getrow(r1).indices)
                links_r2 = set(inc_matrix.getrow(r2).indices)
                diff_links = links_r1.symmetric_difference(links_r2)
                ground_distinguish = gp.quicksum(z[l] for l in diff_links)
                uav_distinguish = gp.quicksum(
                    u[uav_node_id_to_idx[node_id], t_idx] 
                    for node_id in candidate_uav_node_ids 
                    if d_jpq_period[node_id][r1, r2] > 0
                )
                model.addConstr(ground_distinguish + uav_distinguish >= 1, name=f"PathDistinguish_{period_name}_{r1}_{r2}")
        
        # # C4: UAV Deployment-Nest Service Association (NEW for Model 3)
        # # Ensure alignment between UAV candidate nodes in the main model and the ones used for w_hj calculation
        # uav_site_ids_for_nest_calc = nest_params['uav_site_ids_for_nest_calc']
        # uav_site_map_for_nest = {nid: i for i, nid in enumerate(uav_site_ids_for_nest_calc)}

        # for j_model_idx, j_model_id in enumerate(candidate_uav_node_ids):
        #     if j_model_id in uav_site_map_for_nest:
        #         j_whj_idx = uav_site_map_for_nest[j_model_id]
        #         nest_service_sum = gp.quicksum(w_hj[h, j_whj_idx] * y[h] for h in range(num_candidate_nests))
        #         for t_idx in range(len(time_periods)):
        #             model.addConstr(u[j_model_idx, t_idx] <= nest_service_sum, name=f"NestService_{j_model_id}_{t_idx}")
        #     else:
        #         # If a UAV site somehow isn't in the w_hj calculation, it can't be used.
        #         for t_idx in range(len(time_periods)):
        #              model.addConstr(u[j_model_idx, t_idx] == 0, name=f"NestService_Unserviced_{j_model_id}_{t_idx}")
        # print("  - Added UAV deployment-nest service association constraints.")

        # C5: UAV Fleet Size Upper Bound
        model.addConstr(U_max <= UAV_FLEET_SIZE, name="FleetSize_UpperBound")

        # --- Solve the model ---
        print("\n--- Starting Gurobi optimization ---")
        model.optimize()
        end_time = time.time()
        print(f"--- Solver finished in {end_time - start_time:.2f} seconds ---")

        # --- Parse and return solution ---
        solution = {}
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            if model.status == GRB.TIME_LIMIT:
                print("⚠️ Warning: Solver reached the time limit. Solution may be suboptimal.")
            solution['total_cost'] = model.ObjVal
            
            # AVI solution
            optimal_z_indices = {i for i, var in z.items() if var.X > 0.5}
            solution['avi_sensor_link_ids'] = set(df_link_sub.iloc[list(optimal_z_indices)]['link_id'])
            
            # UAV solution
            uav_deployment_by_period = {p: set() for p in time_periods}
            for (j, t), var in u.items():
                if var.X > 0.5:
                    uav_deployment_by_period[time_periods[t]].add(candidate_uav_node_ids[j])
            solution['uav_deployment_by_period'] = uav_deployment_by_period
            
            # Nest solution (NEW)
            # solution['built_nest_ids'] = {nest_ids[h] for h, var in y.items() if var.X > 0.5}
            
            # Cost breakdown and fleet size
            solved_fleet_size = round(U_max.X)
            solution['solved_uav_fleet_size'] = solved_fleet_size
            solution['cost_breakdown'] = {
                'avi_investment_cost': len(solution['avi_sensor_link_ids']) * COST_AVI_INVESTMENT,
                'uav_investment_cost': solved_fleet_size * COST_UAV_INVESTMENT
                # 'nest_investment_cost': len(solution['built_nest_ids']) * COST_NEST_INVESTMENT
            }
            
            return solution, "Optimal" if model.status == GRB.OPTIMAL else "Time Limit"
        else:
            print(f"❌ Solver did not find an optimal solution. Status code: {model.status}")
            return None, f"Solver Status: {model.status}"

    except gp.GurobiError as e:
        print(f"❌ Gurobi error occurred: {e.message}")
        return None, "Gurobi Error"
    except Exception as e:
        import traceback
        print(f"❌ An unexpected error occurred in the solver: {e}")
        traceback.print_exc()
        return None, "Exception"
