# solver_model4.py
"""
Model Solver for Model 4: Budget-Constrained Dynamic Sensor Location Model (B-DSLM).
"""
import gurobipy as gp
from gurobipy import GRB
import time
import itertools

try:
    from config_model4 import (
        COST_AVI_INVESTMENT, COST_UAV_INVESTMENT, COST_NEST_INVESTMENT,
        UAV_FLEET_SIZE, GUROBI_TIME_LIMIT
    )
except ImportError:
    print("❌ Error: Could not import from config_model4.py.")
    exit()

def solve_model4(preprocessed_data, budget):
    """
    Solves the B-DSLM for a given budget.

    Args:
        preprocessed_data (dict): The complete data package from preprocess_model4.py.
        budget (float): The total investment budget B.

    Returns:
        tuple: (solution, status)
    """
    print(f"\n--- Solving Model 4 (B-DSLM) for Budget: {budget} ---")
    start_time = time.time()

    # --- Unpack data ---
    sub_data = preprocessed_data['subnetwork_data']
    paths_by_period = preprocessed_data['paths_by_period']
    static_uav_params = preprocessed_data['static_uav_params']
    dynamic_params = preprocessed_data['dynamic_params']
    nest_params = preprocessed_data['nest_params']
    path_weight_params = preprocessed_data['path_weight_params']

    df_link_sub = sub_data['df_link_sub']
    df_candidate_uav_nodes = sub_data['df_candidate_nodes_sub']
    
    L_j = static_uav_params['L_j']
    inc_matrix_by_period = dynamic_params['inc_matrix_by_period']
    d_jpq_by_period = dynamic_params['d_jpq_by_period']
    w_hj = nest_params['w_hj']
    nest_ids = nest_params['nest_ids']
    path_weights = path_weight_params['path_weights']
    
    time_periods = list(paths_by_period.keys())
    num_links = len(df_link_sub)
    num_candidate_uav_nodes = len(df_candidate_uav_nodes)
    num_candidate_nests = len(nest_ids)
    
    candidate_uav_node_ids = df_candidate_uav_nodes['node_id'].tolist()
    uav_node_id_to_idx = {nid: i for i, nid in enumerate(candidate_uav_node_ids)}
    
    try:
        model = gp.Model(f"B_DSLM_Budget_{budget}")
        model.Params.OutputFlag = 1
        model.Params.TimeLimit = GUROBI_TIME_LIMIT

        # --- Decision Variables ---
        z = model.addVars(num_links, vtype=GRB.BINARY, name="z_a")
        u = model.addVars(num_candidate_uav_nodes, len(time_periods), vtype=GRB.BINARY, name="u_jt")
        U_max = model.addVar(vtype=GRB.INTEGER, name="U_max", lb=0)
        y_h = model.addVars(num_candidate_nests, vtype=GRB.BINARY, name="y_h")
        
        # NEW decision variable for Model 4: y_pt
        # y_pt[t_idx, p_idx] = 1 if path p in period t is observed
        y_pt = {}
        for t_idx, period_name in enumerate(time_periods):
            num_routes_period = paths_by_period[period_name].shape[0]
            for p_idx in range(num_routes_period):
                y_pt[t_idx, p_idx] = model.addVar(vtype=GRB.BINARY, name=f"y_{period_name}_{p_idx}")

        # --- Objective Function (Maximizing Benefit) ---
        total_benefit = gp.quicksum(
            path_weights[time_periods[t_idx]][p_idx] * y_pt[t_idx, p_idx]
            for t_idx, p_idx in y_pt.keys()
        )
        model.setObjective(total_benefit, GRB.MAXIMIZE)
        print("Objective function set to MAXIMIZE total path benefit.")

        # --- Constraints ---
        
        # C1: Total Budget Constraint (NEW)
        avi_cost = gp.quicksum(z[l] * COST_AVI_INVESTMENT for l in range(num_links))
        uav_fleet_cost = U_max * COST_UAV_INVESTMENT
        nest_investment_cost = gp.quicksum(y_h[h] * COST_NEST_INVESTMENT for h in range(num_candidate_nests))
        model.addConstr(avi_cost + uav_fleet_cost + nest_investment_cost <= budget, name="TotalBudget")
        print(f"  - Added total budget constraint (<= {budget}).")

        # C2: Fleet Size Definition (Same as Model 3)
        for t_idx, period_name in enumerate(time_periods):
            model.addConstr(gp.quicksum(u[j, t_idx] for j in range(num_candidate_uav_nodes)) <= U_max,
                            name=f"Define_Umax_{period_name}")
        print("  - Added fleet size definition constraints (U_max).")

        # C3 & C4: Conditional Path Coverage & Distinguishability (MODIFIED)
        for t_idx, period_name in enumerate(time_periods):
            print(f"  - Adding conditional path constraints for time period: '{period_name}'")
            inc_matrix = inc_matrix_by_period[period_name]
            d_jpq_period = d_jpq_by_period[period_name]
            num_routes_period = inc_matrix.shape[0]

            # C3: Conditional Path Coverage
            for p in range(num_routes_period):
                links_on_route_indices = inc_matrix.getrow(p).indices
                ground_coverage = gp.quicksum(z[l] for l in links_on_route_indices)
                uav_coverage = gp.quicksum(
                    u[uav_node_id_to_idx[node_id], t_idx] 
                    for node_id in candidate_uav_node_ids 
                    if any(df_link_sub.iloc[l_idx]['link_id'] in L_j.get(node_id, []) for l_idx in links_on_route_indices)
                )
                # Coverage is only required if we decide to observe this path (y_pt = 1)
                model.addConstr(ground_coverage + uav_coverage >= y_pt[t_idx, p], name=f"CondPathCoverage_{period_name}_{p}")

            # C4: Conditional Path Distinguishability
            for p1, p2 in itertools.combinations(range(num_routes_period), 2):
                links_p1 = set(inc_matrix.getrow(p1).indices)
                links_p2 = set(inc_matrix.getrow(p2).indices)
                diff_links = links_p1.symmetric_difference(links_p2)
                ground_distinguish = gp.quicksum(z[l] for l in diff_links)
                uav_distinguish = gp.quicksum(
                    u[uav_node_id_to_idx[node_id], t_idx] 
                    for node_id in candidate_uav_node_ids 
                    if d_jpq_period[node_id][p1, p2] > 0
                )
                # Distinguishability is only required if we decide to observe BOTH paths
                model.addConstr(ground_distinguish + uav_distinguish >= y_pt[t_idx, p1] + y_pt[t_idx, p2] - 1, 
                                name=f"CondPathDistinguish_{period_name}_{p1}_{p2}")
        
        # C5: UAV Deployment-Nest Service Association (Same as Model 3)
        uav_site_ids_for_nest_calc = nest_params['uav_site_ids_for_nest_calc']
        uav_site_map_for_nest = {nid: i for i, nid in enumerate(uav_site_ids_for_nest_calc)}
        for j_model_idx, j_model_id in enumerate(candidate_uav_node_ids):
            if j_model_id in uav_site_map_for_nest:
                j_whj_idx = uav_site_map_for_nest[j_model_id]
                nest_service_sum = gp.quicksum(w_hj[h, j_whj_idx] * y_h[h] for h in range(num_candidate_nests))
                for t_idx in range(len(time_periods)):
                    model.addConstr(u[j_model_idx, t_idx] <= nest_service_sum, name=f"NestService_{j_model_id}_{t_idx}")
            else:
                for t_idx in range(len(time_periods)):
                     model.addConstr(u[j_model_idx, t_idx] == 0, name=f"NestService_Unserviced_{j_model_id}_{t_idx}")
        print("  - Added UAV deployment-nest service association constraints.")

        # C6: UAV Fleet Size Upper Bound (Same as Model 3)
        model.addConstr(U_max <= UAV_FLEET_SIZE, name="FleetSize_UpperBound")

        # --- Solve the model ---
        print("\n--- Starting Gurobi optimization ---")
        model.optimize()
        end_time = time.time()
        print(f"--- Solver finished in {end_time - start_time:.2f} seconds ---")

        # --- Parse and return solution ---
        solution = {}
        if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            if model.status == GRB.TIME_LIMIT:
                print("⚠️ Warning: Solver reached the time limit. Solution may be suboptimal.")
            
            solution['budget'] = budget
            solution['total_benefit'] = model.ObjVal
            
            # Observed paths
            observed_paths = {p: [] for p in time_periods}
            total_paths_observed = 0
            for (t_idx, p_idx), var in y_pt.items():
                if var.X > 0.5:
                    observed_paths[time_periods[t_idx]].append(p_idx)
                    total_paths_observed += 1
            solution['observed_paths_by_period'] = observed_paths
            solution['total_paths_observed'] = total_paths_observed

            # Deployment solution
            solution['avi_sensor_link_ids'] = {df_link_sub.iloc[i]['link_id'] for i, var in z.items() if var.X > 0.5}
            uav_deployment = {p: set() for p in time_periods}
            for (j, t), var in u.items():
                if var.X > 0.5:
                    uav_deployment[time_periods[t]].add(candidate_uav_node_ids[j])
            solution['uav_deployment_by_period'] = uav_deployment
            solution['built_nest_ids'] = {nest_ids[h] for h, var in y_h.items() if var.X > 0.5}
            
            # Cost breakdown and fleet size
            solved_fleet_size = round(U_max.X)
            solution['solved_uav_fleet_size'] = solved_fleet_size
            cost_avi = len(solution['avi_sensor_link_ids']) * COST_AVI_INVESTMENT
            cost_uav = solved_fleet_size * COST_UAV_INVESTMENT
            cost_nest = len(solution['built_nest_ids']) * COST_NEST_INVESTMENT
            solution['cost_breakdown'] = {
                'avi_investment_cost': cost_avi,
                'uav_investment_cost': cost_uav,
                'nest_investment_cost': cost_nest
            }
            solution['total_cost'] = cost_avi + cost_uav + cost_nest
            
            return solution, "Optimal" if model.status == GRB.OPTIMAL else "Time Limit"
        else:
            print(f"❌ Solver did not find a valid solution. Status code: {model.status}")
            return None, f"Solver Status: {model.status}"

    except gp.GurobiError as e:
        print(f"❌ Gurobi error occurred: {e.message}")
        return None, "Gurobi Error"
    except Exception as e:
        import traceback
        print(f"❌ An unexpected error occurred in the solver: {e}")
        traceback.print_exc()
        return None, "Exception"
