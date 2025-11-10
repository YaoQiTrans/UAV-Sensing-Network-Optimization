# visualizer_model3.py
"""
Visualization Module for Model 3 (DSLM with Nests).
Generates a multi-plot figure to show deployment per time period.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import time
import os
import numpy as np

try:
    from config_model3 import (
        UAV_FOV_WIDTH, UAV_FOV_HEIGHT, RESULTS_DIR_MODEL3,
        TIME_PERIOD_COLORS
    )
except ImportError:
    print("❌ Error: Could not import from config_model3.py.")
    exit()

def plot_model3_solution(solution, preprocessed_data):
    """
    Visualizes the final deployment solution for Model 3.
    Creates one subplot for each time period.
    """
    print("\n--- Generating plot for the Model 3 solution ---")

    # --- Unpack data ---
    sub_data = preprocessed_data['subnetwork_data']
    df_link_sub = sub_data['df_link_sub']
    df_node_sub = sub_data['df_node_sub']
    node_coords_map = {row['node_id']: (row['x_coord'], row['y_coord']) for _, row in df_node_sub.iterrows()}

    # --- Create base graph ---
    G = nx.Graph()
    G.add_nodes_from(node_coords_map.keys())
    for _, link in df_link_sub.iterrows():
        G.add_edge(link['u_node_id'], link['v_node_id'], link_id=link['link_id'])

    # --- Get solution elements ---
    time_periods = list(solution['uav_deployment_by_period'].keys())
    avi_link_ids = solution.get('avi_sensor_link_ids', set())
    # built_nest_ids = solution.get('built_nest_ids', set())
    avi_edges = [(u, v) for u, v, data in G.edges(data=True) if data['link_id'] in avi_link_ids]

    # --- Create Subplots ---
    fig, axes = plt.subplots(1, len(time_periods), figsize=(30, 12), sharex=True, sharey=True)
    if len(time_periods) == 1: axes = [axes] # Ensure axes is iterable for a single period
    fig.suptitle('Dynamic Air-Ground Deployment with Nests by Time Period', fontsize=24, weight='bold')

    for ax, period in zip(axes, time_periods):
        ax.set_title(period.replace('_', ' ').title(), fontsize=18)
        
        # 1. Draw base network
        nx.draw_networkx_nodes(G, node_coords_map, ax=ax, node_size=20, node_color='white', edgecolors='gray', linewidths=0.5)
        nx.draw_networkx_edges(G, node_coords_map, ax=ax, width=0.8, edge_color='lightgray', alpha=0.7)

        # 2. Draw static AVI sensors (same on all plots)
        nx.draw_networkx_edges(G, node_coords_map, ax=ax, edgelist=avi_edges, width=3.0, edge_color='#2ca02c', alpha=0.8)

        # # 3. Draw built nests (same on all plots)
        # for nest_id in built_nest_ids:
        #     if nest_id in node_coords_map:
        #         pos = node_coords_map[nest_id]
        #         ax.plot(pos[0], pos[1], '*', markersize=20, markerfacecolor=NEST_COLOR, markeredgecolor='black')

        # 4. Draw dynamic UAVs for the current period
        deployed_uavs_period = solution['uav_deployment_by_period'][period]
        period_color = TIME_PERIOD_COLORS.get(period, 'gray')
        
        for uav_id in deployed_uavs_period:
            if uav_id in node_coords_map:
                uav_pos = node_coords_map[uav_id]
                
                # Draw UAV marker
                ax.plot(uav_pos[0], uav_pos[1], 'P', markersize=14, markerfacecolor=period_color, markeredgecolor='black')
                
                # Draw FOV
                fov_rect = patches.Rectangle(
                    (uav_pos[0] - UAV_FOV_WIDTH / 2, uav_pos[1] - UAV_FOV_HEIGHT / 2),
                    UAV_FOV_WIDTH, UAV_FOV_HEIGHT, linewidth=1.5,
                    edgecolor=period_color, facecolor=period_color, alpha=0.15
                )
                ax.add_patch(fov_rect)

                # # 5. Draw service lines from nests to UAVs
                # # Find the closest serving nest and draw a line
                # uav_coord = np.array(uav_pos)
                # min_dist = float('inf')
                # serving_nest_pos = None
                # for nest_id in built_nest_ids:
                #     if nest_id in node_coords_map:
                #         nest_coord = np.array(node_coords_map[nest_id])
                #         dist = np.linalg.norm(uav_coord - nest_coord)
                #         if dist < min_dist:
                #             min_dist = dist
                #             serving_nest_pos = node_coords_map[nest_id]
                
                # if serving_nest_pos:
                #     ax.plot([serving_nest_pos[0], uav_pos[0]], [serving_nest_pos[1], uav_pos[1]],
                #             linestyle='--', color=SERVICE_LINE_COLOR, linewidth=1.2)

        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        ax.set_aspect('equal', adjustable='box')
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- Create a shared legend ---
    legend_elements = [
        plt.Line2D([0], [0], color='lightgray', lw=2, label='Network Link'),
        plt.Line2D([0], [0], color='#2ca02c', lw=3, label='Link with AVI Sensor'),
        # plt.Line2D([0], [0], marker='*', color='w', label='Built Nest', markerfacecolor=NEST_COLOR, markeredgecolor='k', markersize=15),
        # plt.Line2D([0], [0], linestyle='--', color=SERVICE_LINE_COLOR, label='Nest-UAV Service Link'),
        plt.Line2D([0], [0], marker='P', color='w', markersize=0, label='\nUAV Deployment:'),
    ]
    for period, color in TIME_PERIOD_COLORS.items():
        legend_elements.append(plt.Line2D([0], [0], marker='P', color='w', markerfacecolor=color, markeredgecolor='k', markersize=12, label=f'{period.replace("_", " ").title()}'))
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=len(legend_elements), fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for title and legend

    # --- Save the plot ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"model3_solution_{timestamp}.svg"
    filepath = os.path.join(RESULTS_DIR_MODEL3, filename)
    
    try:
        fig.savefig(filepath, format='svg', bbox_inches='tight', dpi=300)
        print(f"✅ Plot saved successfully to: {filepath}")
    except Exception as e:
        print(f"❌ Could not save the plot. Error: {e}")

    plt.show()
