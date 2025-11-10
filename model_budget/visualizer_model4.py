# visualizer_model4.py
"""
Visualization Module for Model 4 (B-DSLM).
Generates a multi-plot figure to show deployment and *observed paths*
for a given budget.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import time
import os
import numpy as np

try:
    from config_model4 import (
        UAV_FOV_WIDTH, UAV_FOV_HEIGHT, RESULTS_DIR_MODEL4,
        TIME_PERIOD_COLORS, NEST_COLOR, SERVICE_LINE_COLOR
    )
except ImportError:
    print("❌ Error: Could not import from config_model4.py.")
    exit()

def plot_model4_solution(solution, preprocessed_data, filename):
    """
    Visualizes the deployment and observed paths for the B-DSLM solution.
    """
    print(f"\n--- Generating plot for the Model 4 solution (Budget: {solution['budget']}) ---")

    # --- Unpack data ---
    sub_data = preprocessed_data['subnetwork_data']
    paths_by_period = preprocessed_data['paths_by_period']
    df_link_sub = sub_data['df_link_sub']
    df_node_sub = sub_data['df_node_sub']
    node_coords_map = {row['node_id']: (row['x_coord'], row['y_coord']) for _, row in df_node_sub.iterrows()}
    link_id_to_nodes = {row['link_id']: (row['u_node_id'], row['v_node_id']) for _, row in df_link_sub.iterrows()}

    # --- Create base graph ---
    G = nx.Graph()
    G.add_nodes_from(node_coords_map.keys())
    for link_id, nodes in link_id_to_nodes.items():
        G.add_edge(nodes[0], nodes[1], link_id=link_id)

    # --- Get solution elements ---
    time_periods = list(solution['uav_deployment_by_period'].keys())
    avi_link_ids = solution.get('avi_sensor_link_ids', set())
    built_nest_ids = solution.get('built_nest_ids', set())
    avi_edges = [(u, v) for u, v, data in G.edges(data=True) if data['link_id'] in avi_link_ids]

    # --- Create Subplots ---
    fig, axes = plt.subplots(1, len(time_periods), figsize=(30, 12), sharex=True, sharey=True)
    if len(time_periods) == 1: axes = [axes]
    
    fig_title = (f"B-DSLM Deployment | Budget: ${solution['budget']}$ | "
                 f"Total Benefit (Observed Paths): ${solution['total_benefit']:.0f}$ | "
                 f"Actual Cost: ${solution['total_cost']:.2f}$")
    fig.suptitle(fig_title, fontsize=22, weight='bold')

    for ax, period in zip(axes, time_periods):
        ax.set_title(period.replace('_', ' ').title(), fontsize=18)
        
        # 1. Draw base network
        nx.draw_networkx_nodes(G, node_coords_map, ax=ax, node_size=20, node_color='white', edgecolors='gray', linewidths=0.5)
        nx.draw_networkx_edges(G, node_coords_map, ax=ax, width=0.8, edge_color='#e0e0e0', alpha=0.7)

        # 2. Highlight candidate paths vs observed paths (NEW)
        all_period_paths = paths_by_period[period]['path_links_set']
        observed_path_indices = solution['observed_paths_by_period'][period]
        
        # Draw all candidate paths in a light gray
        for path_links in all_period_paths:
            path_edges = [link_id_to_nodes[link_id] for link_id in path_links if link_id in link_id_to_nodes]
            nx.draw_networkx_edges(G, node_coords_map, ax=ax, edgelist=path_edges, width=1.5, edge_color='#f0f0f0', alpha=0.9)
            
        # Draw observed paths in a standout color
        for path_idx in observed_path_indices:
            path_links = all_period_paths.iloc[path_idx]
            path_edges = [link_id_to_nodes[link_id] for link_id in path_links if link_id in link_id_to_nodes]
            nx.draw_networkx_edges(G, node_coords_map, ax=ax, edgelist=path_edges, width=3.5, edge_color=TIME_PERIOD_COLORS[period], alpha=0.6)

        # 3. Draw static AVI sensors
        # FIX: Changed 'style' to 'linestyle'
        nx.draw_networkx_edges(G, node_coords_map, ax=ax, edgelist=avi_edges, width=3.0, edge_color='#2ca02c', alpha=0.9, style='dashed')

        # 4. Draw built nests
        for nest_id in built_nest_ids:
            if nest_id in node_coords_map:
                pos = node_coords_map[nest_id]
                ax.plot(pos[0], pos[1], '*', markersize=20, markerfacecolor=NEST_COLOR, markeredgecolor='black')

        # 5. Draw dynamic UAVs for the current period
        deployed_uavs_period = solution['uav_deployment_by_period'][period]
        period_color = TIME_PERIOD_COLORS.get(period, 'gray')
        
        for uav_id in deployed_uavs_period:
            if uav_id in node_coords_map:
                uav_pos = node_coords_map[uav_id]
                ax.plot(uav_pos[0], uav_pos[1], 'P', markersize=14, markerfacecolor=period_color, markeredgecolor='black')
                fov_rect = patches.Rectangle(
                    (uav_pos[0] - UAV_FOV_WIDTH / 2, uav_pos[1] - UAV_FOV_HEIGHT / 2),
                    UAV_FOV_WIDTH, UAV_FOV_HEIGHT, linewidth=1.5,
                    edgecolor=period_color, facecolor='none', alpha=0.8, linestyle='--'
                )
                ax.add_patch(fov_rect)
        
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        ax.set_aspect('equal', adjustable='box')
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- Create a shared legend ---
    legend_elements = [
        plt.Line2D([0], [0], color='#e0e0e0', lw=2, label='Network Link'),
        plt.Line2D([0], [0], color='gray', lw=3, alpha=0.4, label='Candidate Path'),
        plt.Line2D([0], [0], color='#FF6347', lw=3, label='Observed Path (Morning)'),
        plt.Line2D([0], [0], color='#4169E1', lw=3, label='Observed Path (Off-Peak)'),
        plt.Line2D([0], [0], color='#FFA500', lw=3, label='Observed Path (Evening)'),
        # FIX: Changed 'style' to 'linestyle'
        plt.Line2D([0], [0], color='#2ca02c', lw=3, linestyle='dashed', label='Link with AVI Sensor'),
        plt.Line2D([0], [0], marker='*', color='w', label='Built Nest', markerfacecolor=NEST_COLOR, markeredgecolor='k', markersize=15),
        plt.Line2D([0], [0], marker='P', color='w', label='Deployed UAV', markerfacecolor='gray', markeredgecolor='k', markersize=12),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Save the plot ---
    filepath = os.path.join(RESULTS_DIR_MODEL4, filename)
    try:
        fig.savefig(filepath, format='svg', bbox_inches='tight', dpi=300)
        print(f"✅ Plot saved successfully to: {filepath}")
    except Exception as e:
        print(f"❌ Could not save the plot. Error: {e}")
    plt.close(fig) # Close the figure to free up memory
