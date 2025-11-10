# solver.py
"""
Model Solvers
Contains Gurobi implementations for all models:
1. UAV-Only Location Model (ULM)
2. Air-Ground Sensor Location Model (AG-SLM)
3. Baseline Ground-Only Model (Castillo et al., 2008)
"""
import gurobipy as gp
from gurobipy import GRB
import time
import itertools
from config import GUROBI_TIME_LIMIT
import traceback

def solve_uav_only_model(sub_data, uav_params, cost_uav):
    """
    Solves the UAV-Only Location Model (ULM).
    """
    print("\n--- Solving Model 1: UAV-Only Location Model (ULM)... ---")
    start_time = time.time()

    inc_matrix = sub_data['sub_incidence_matrix']
    df_candidate_nodes_sub = sub_data['df_candidate_nodes_sub']
    
    num_routes, _ = inc_matrix.shape
    candidate_nodes = df_candidate_nodes_sub['node_id'].tolist()
    num_candidate_nodes = len(candidate_nodes)
    node_id_to_idx = {nid: i for i, nid in enumerate(candidate_nodes)}

    L_j = uav_params['L_j']
    d_jpq = uav_params['d_jpq']

    try:
        model = gp.Model("UAV_Only_ULM")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = GUROBI_TIME_LIMIT

        u = model.addVars(num_candidate_nodes, vtype=GRB.BINARY, name="u")
        objective = gp.quicksum(u[j] * cost_uav for j in range(num_candidate_nodes))
        model.setObjective(objective, GRB.MINIMIZE)

        # Constraint 1: Path Coverage
        for r in range(num_routes):
            path_links_indices = set(inc_matrix.getrow(r).indices)
            if not path_links_indices: continue

            uav_coverage = gp.LinExpr()
            for node_id, j_idx in node_id_to_idx.items():
                links_in_fov_indices = set(L_j.get(node_id, []))
                if not path_links_indices.isdisjoint(links_in_fov_indices):
                    uav_coverage += u[j_idx]
            model.addConstr(uav_coverage >= 1, name=f"PathCoverage_{r}")

        # Constraint 2: Path Distinguishability
        for r1, r2 in itertools.combinations(range(num_routes), 2):
            uav_distinguish = gp.LinExpr()
            for node_id, j_idx in node_id_to_idx.items():
                # FIX: Check for key existence directly instead of evaluating the matrix object.
                if node_id in d_jpq and d_jpq[node_id][r1, r2] > 0:
                    uav_distinguish += u[j_idx]
            model.addConstr(uav_distinguish >= 1, name=f"PathDistinguish_{r1}_{r2}")

        model.optimize()
        end_time = time.time()
        print(f"  Solver finished in {end_time - start_time:.2f} seconds.")
        
        solution = {}
        if model.status == GRB.OPTIMAL:
            solution['total_cost'] = model.ObjVal
            optimal_u_indices = {i for i, var in u.items() if var.X > 0.5}
            solution['uav_node_ids'] = set(df_candidate_nodes_sub.iloc[list(optimal_u_indices)]['node_id'])
            solution['ground_sensor_ids'] = set()
            return solution, "Optimal"
        else:
            return None, f"Solver Status: {model.status}"

    except gp.GurobiError as e:
        print(f"❌ Gurobi Error during ULM solution: {e}")
        return None, "Gurobi Error"
    except Exception as e:
        print(f"❌ An unexpected error occurred during ULM solution: {e}")
        traceback.print_exc()
        return None, "Exception"


def solve_air_ground_model(sub_data, uav_params, cost_ground, cost_uav):
    """
    Solves the air-ground collaborative sensor location model (AG-SLM).
    """
    print("\n--- Solving Model 2: Air-Ground Sensor Location Model (AG-SLM)... ---")
    start_time = time.time()

    inc_matrix = sub_data['sub_incidence_matrix']
    df_link_sub = sub_data['df_link_sub']
    df_candidate_nodes_sub = sub_data['df_candidate_nodes_sub']
    
    num_routes, num_links = inc_matrix.shape
    candidate_nodes = df_candidate_nodes_sub['node_id'].tolist()
    num_candidate_nodes = len(candidate_nodes)
    node_id_to_idx = {nid: i for i, nid in enumerate(candidate_nodes)}

    L_j = uav_params['L_j']
    d_jpq = uav_params['d_jpq']

    try:
        model = gp.Model("Air_Ground_SLM")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = GUROBI_TIME_LIMIT

        z = model.addVars(num_links, vtype=GRB.BINARY, name="z")
        u = model.addVars(num_candidate_nodes, vtype=GRB.BINARY, name="u")

        objective = gp.quicksum(z[l] * cost_ground for l in range(num_links)) + \
                    gp.quicksum(u[j] * cost_uav for j in range(num_candidate_nodes))
        model.setObjective(objective, GRB.MINIMIZE)

        # Constraint 1: Path Coverage
        for r in range(num_routes):
            path_links_indices = set(inc_matrix.getrow(r).indices)
            if not path_links_indices: continue
            
            ground_coverage = gp.quicksum(z[l] for l in path_links_indices)
            
            uav_coverage = gp.LinExpr()
            for node_id, j_idx in node_id_to_idx.items():
                links_in_fov_indices = set(L_j.get(node_id, []))
                if not path_links_indices.isdisjoint(links_in_fov_indices):
                    uav_coverage += u[j_idx]
            model.addConstr(ground_coverage + uav_coverage >= 1, name=f"PathCoverage_{r}")

        # Constraint 2: Path Distinguishability
        for r1, r2 in itertools.combinations(range(num_routes), 2):
            links_r1 = set(inc_matrix.getrow(r1).indices)
            links_r2 = set(inc_matrix.getrow(r2).indices)
            diff_links = links_r1.symmetric_difference(links_r2)
            ground_distinguish = gp.quicksum(z[l] for l in diff_links)

            uav_distinguish = gp.LinExpr()
            for node_id, j_idx in node_id_to_idx.items():
                # FIX: Check for key existence directly instead of evaluating the matrix object.
                if node_id in d_jpq and d_jpq[node_id][r1, r2] > 0:
                    uav_distinguish += u[j_idx]
            model.addConstr(ground_distinguish + uav_distinguish >= 1, name=f"PathDistinguish_{r1}_{r2}")

        model.optimize()
        end_time = time.time()
        print(f"  Solver finished in {end_time - start_time:.2f} seconds.")
        
        solution = {}
        if model.status == GRB.OPTIMAL:
            solution['total_cost'] = model.ObjVal
            optimal_z_indices = {i for i, var in z.items() if var.X > 0.5}
            solution['ground_sensor_ids'] = set(df_link_sub.iloc[list(optimal_z_indices)]['link_id'])
            optimal_u_indices = {i for i, var in u.items() if var.X > 0.5}
            solution['uav_node_ids'] = set(df_candidate_nodes_sub.iloc[list(optimal_u_indices)]['node_id'])
            return solution, "Optimal"
        else:
            return None, f"Solver Status: {model.status}"

    except gp.GurobiError as e:
        print(f"❌ Gurobi Error during AG-SLM solution: {e}")
        return None, "Gurobi Error"
    except Exception as e:
        print(f"❌ An unexpected error occurred during AG-SLM solution: {e}")
        traceback.print_exc()
        return None, "Exception"


def solve_baseline_castillo(sub_data, cost_ground):
    """
    Solves the classic Castillo OBSV model (ground sensors only).
    """
    print("\n--- Solving Baseline: Ground-Only Model... ---")
    start_time = time.time()
    
    inc_matrix = sub_data['sub_incidence_matrix']
    num_routes, num_links = inc_matrix.shape
    df_link_sub = sub_data['df_link_sub']

    try:
        model = gp.Model("Baseline_OBSV")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = GUROBI_TIME_LIMIT

        z = model.addVars(num_links, vtype=GRB.BINARY, name="z")
        model.setObjective(gp.quicksum(z[l] * cost_ground for l in range(num_links)), GRB.MINIMIZE)

        for r in range(num_routes):
            links_on_route = inc_matrix.getrow(r).indices
            if len(links_on_route) > 0:
                model.addConstr(gp.quicksum(z[l] for l in links_on_route) >= 1)

        for r1, r2 in itertools.combinations(range(num_routes), 2):
            links_r1 = set(inc_matrix.getrow(r1).indices)
            links_r2 = set(inc_matrix.getrow(r2).indices)
            diff_links = links_r1.symmetric_difference(links_r2)
            if diff_links:
                model.addConstr(gp.quicksum(z[l] for l in diff_links) >= 1)

        model.optimize()
        end_time = time.time()
        print(f"  Solver finished in {end_time - start_time:.2f} seconds.")

        solution = {}
        if model.status == GRB.OPTIMAL:
            solution['total_cost'] = model.ObjVal
            optimal_indices = {i for i, var in z.items() if var.X > 0.5}
            solution['ground_sensor_ids'] = set(df_link_sub.iloc[list(optimal_indices)]['link_id'])
            solution['uav_node_ids'] = set()
            return solution, "Optimal"
        else:
            return None, f"Solver Status: {model.status}"

    except gp.GurobiError as e:
        print(f"❌ Gurobi Error during baseline solution: {e}")
        return None, "Gurobi Error"
    except Exception as e:
        print(f"❌ An unexpected error occurred during baseline solution: {e}")
        traceback.print_exc()
        return None, "Exception"
