# data_loader.py
"""
Module for loading and preprocessing network data (Refined).
负责加载和预处理网络数据、路径数据，计算关联矩阵（精炼版）。
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import utils
import time

def load_network_data(route_file, link_file):
    """
    Loads route and link data from specified files.
    从指定文件加载路径和路段数据。

    Args:
        route_file (str): Path to the route data file (CSV or Excel).
        link_file (str): Path to the link data file (CSV or Excel).

    Returns:
        tuple: (df_route, df_link) - DataFrames for routes and links.
               None, None if loading fails.
    """
    try:
        # print(f"Attempting to load route data from: {route_file}")
        if route_file.endswith('.csv'):
            df_route = pd.read_csv(route_file)
        elif route_file.endswith('.xlsx'):
            df_route = pd.read_excel(route_file)
        else:
            print(f"Error: Unsupported route file format for {route_file}. Please use CSV or XLSX.")
            return None, None
        # print(f"Successfully loaded route data. Shape: {df_route.shape}")

        # print(f"Attempting to load link data from: {link_file}")
        if link_file.endswith('.csv'):
            df_link = pd.read_csv(link_file)
        elif link_file.endswith('.xlsx'):
            df_link = pd.read_excel(link_file)
        else:
             print(f"Error: Unsupported link file format for {link_file}. Please use CSV or XLSX.")
             return None, None
        # print(f"Successfully loaded link data. Shape: {df_link.shape}")

        # --- Data Cleaning and Validation ---
        link_col_map = {'u_node_id': 'u', 'start_node': 'u',
                        'v_node_id': 'v', 'end_node': 'v'}
        df_link.rename(columns=link_col_map, inplace=True, errors='ignore') # Ignore errors if columns don't exist

        required_route_cols = ['route_node']
        required_link_cols = ['u', 'v']
        missing_route_cols = [col for col in required_route_cols if col not in df_route.columns]
        missing_link_cols = [col for col in required_link_cols if col not in df_link.columns]

        if missing_route_cols:
            print(f"Error: Required columns missing in route file: {missing_route_cols}")
            return None, None
        if missing_link_cols:
            print(f"Error: Required columns missing in link file: {missing_link_cols}")
            return None, None

        # Assign unique IDs if not present (using index as default)
        if 'route_id' not in df_route.columns:
             print("Assigning 'route_id' based on DataFrame index.")
             df_route['route_id'] = df_route.index
        if 'link_id' not in df_link.columns:
             print("Assigning 'link_id' based on DataFrame index.")
             df_link['link_id'] = df_link.index

        # Ensure IDs are unique (optional check)
        # if not df_route['route_id'].is_unique:
        #     print("Warning: 'route_id' column contains duplicate values.")
        # if not df_link['link_id'].is_unique:
        #      print("Warning: 'link_id' column contains duplicate values.")

        print(f"Data loaded: {len(df_route)} routes and {len(df_link)} links.")
        return df_route, df_link

    except FileNotFoundError:
        print(f"Error: File not found. Check paths: {route_file}, {link_file}")
        return None, None
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        return None, None

def calculate_route_link_incidence(df_route, df_link):
    """
    Calculates the sparse route-link incidence matrix (delta).
    计算稀疏的路径-路段关联矩阵 (delta)。

    Args:
        df_route (pd.DataFrame): DataFrame containing route data with 'route_node' and 'route_id'.
        df_link (pd.DataFrame): DataFrame containing link data with 'u', 'v', and 'link_id'.

    Returns:
        scipy.sparse.csr_matrix: The route-link incidence matrix.
                                  Rows correspond to routes (indexed by 0-based position in df_route),
                                  columns to links (indexed by 0-based position in df_link).
                                  Value is 1 if link is on route, 0 otherwise.
    """
    num_route = len(df_route)
    num_link = len(df_link)
    row_indices, col_indices, data = [], [], [] # Use 0-based indices for matrix construction

    # Create a mapping from (u, v) node pairs to the 0-based index of the link in df_link
    link_uv_to_matrix_idx_map = {(u, v): i for i, (u, v) in enumerate(zip(df_link['u'], df_link['v']))}
    # print(f"Created link_uv_to_matrix_idx_map with {len(link_uv_to_matrix_idx_map)} entries.")

    calculation_start_time = time.time()
    errors_parsing_route = 0

    # Iterate using 0-based index for rows
    for route_matrix_idx, route_data in df_route.iterrows():
        node_list_repr = route_data['route_node']
        node_list = utils.str_to_list(node_list_repr) # Use utility function

        if node_list is None or not isinstance(node_list, list):
            errors_parsing_route += 1
            continue # Skip this route if parsing failed

        links_on_this_route_cols = set() # Store 0-based column indices
        if len(node_list) > 1:
            for i in range(len(node_list) - 1):
                try:
                    u, v = int(node_list[i]), int(node_list[i+1])
                    link_key = (u, v)

                    if link_key in link_uv_to_matrix_idx_map:
                        link_matrix_idx = link_uv_to_matrix_idx_map[link_key] # Get 0-based column index

                        if link_matrix_idx not in links_on_this_route_cols:
                            row_indices.append(route_matrix_idx)
                            col_indices.append(link_matrix_idx)
                            data.append(1)
                            links_on_this_route_cols.add(link_matrix_idx)
                except (ValueError, TypeError):
                    # Silently skip invalid node pairs within a route
                    pass

        # Optional: Progress indicator for very large datasets
        # if (route_matrix_idx + 1) % 1000 == 0:
        #     print(f"  Processed {route_matrix_idx + 1}/{num_route} routes...")

    calculation_end_time = time.time()
    print(f"Incidence matrix calculation loop took {calculation_end_time - calculation_start_time:.2f} seconds.")
    if errors_parsing_route > 0:
        print(f"Warning: Failed to parse 'route_node' for {errors_parsing_route} routes.")

    if not row_indices:
        print("Warning: No valid route-link incidences found. Returning empty matrix.")
        return sp.csr_matrix((num_route, num_link), dtype=int)

    # Create the sparse matrix using 0-based indices
    matrix_shape = (num_route, num_link)
    route_link_indicator = sp.csr_matrix((data, (row_indices, col_indices)), shape=matrix_shape, dtype=int)

    return route_link_indicator

def get_differing_links_set(route_id_1, route_id_2, df_route, df_link, route_link_indicator):
    """
    Finds the set of *original* link IDs where two routes differ.
    查找两个路径不同的 *原始* 路段 ID 集合 D(r1, r2)。

    Args:
        route_id_1 (int): Original ID of the first route (from df_route['route_id']).
        route_id_2 (int): Original ID of the second route (from df_route['route_id']).
        df_route (pd.DataFrame): DataFrame containing route data with 'route_id'.
        df_link (pd.DataFrame): DataFrame containing link data with 'link_id'.
        route_link_indicator (scipy.sparse.csr_matrix): Route-link incidence matrix
                                                        (rows/cols indexed by 0-based position).

    Returns:
        set: A set containing the *original* link IDs (from df_link['link_id'])
             where the routes differ. Returns an empty set if route IDs are not found
             or an error occurs.
    """
    try:
        # Find the 0-based row index corresponding to the original route IDs
        row_idx_1 = df_route.index[df_route['route_id'] == route_id_1].tolist()[0]
        row_idx_2 = df_route.index[df_route['route_id'] == route_id_2].tolist()[0]

        # Get the rows from the sparse matrix
        row1 = route_link_indicator.getrow(row_idx_1)
        row2 = route_link_indicator.getrow(row_idx_2)

        # Find non-zero column indices (0-based matrix indices for links)
        links_idx_1 = set(row1.indices)
        links_idx_2 = set(row2.indices)

        # Symmetric difference gives 0-based column indices of differing links
        diff_link_indices = links_idx_1.symmetric_difference(links_idx_2)

        # Map 0-based column indices back to original link IDs using df_link
        # Assumes the order in df_link corresponds to the matrix columns
        if not diff_link_indices:
            return set() # Return empty set if no difference

        # Get the original link IDs corresponding to these indices
        original_link_ids = df_link.iloc[list(diff_link_indices)]['link_id'].tolist()

        # Convert numpy types (like np.int32) to standard Python int if necessary
        diff_original_link_ids = {int(link_id) for link_id in original_link_ids}

        return diff_original_link_ids

    except IndexError:
        print(f"Error: Route ID {route_id_1} or {route_id_2} not found in df_route.")
        return set()
    except Exception as e:
        print(f"Error getting differing links set for routes {route_id_1}, {route_id_2}: {e}")
        return set()


# Example Usage (can be placed in __main__ block)
if __name__ == '__main__':

    # Use the provided CSV file names (adjust paths if needed)
    route_file = 'data\\Nguyen_Dupuis_route.csv'
    link_file = 'data\\Nguyen_Dupuis_link.csv'

    df_route, df_link = load_network_data(route_file, link_file)

    if df_route is not None and df_link is not None:
        print("Route Data Head:")
        print(df_route.head())
        print("Link Data Head:")
        print(df_link.head())

        print("Calculating route-link incidence matrix...")
        route_link_indicator = calculate_route_link_incidence(df_route, df_link)

        if route_link_indicator is not None:
            print(f"Calculated Route-Link Incidence Matrix (Shape: {route_link_indicator.shape})")
            print(f"Number of non-zero elements: {route_link_indicator.nnz}")

            # Example of getting differing links using original IDs
            if len(df_route) >= 2:
                 # Use the actual route_id values from the DataFrame
                 r1_original_id = df_route['route_id'].iloc[0] # Example: Route ID 1
                 r2_original_id = df_route['route_id'].iloc[1] # Example: Route ID 2
                 # Pass df_link to map indices back to original IDs
                 diff_set = get_differing_links_set(r1_original_id, r2_original_id, df_route, df_link, route_link_indicator)
                 print(f"Links where route {r1_original_id} and route {r2_original_id} differ (Original IDs): {diff_set}")
                 # Expected output for your example data: {2, 3, 6}
        else:
            print("\nFailed to calculate route-link incidence matrix.")

    else:
        print("\nFailed to load data.")
