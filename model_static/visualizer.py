# visualizer.py
"""
Visualization Module
Contains a generic function for plotting any deployment solution.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import time
import os
import textwrap
from config import UAV_FOV_WIDTH, UAV_FOV_HEIGHT, RESULTS_DIR

def plot_deployment(solution, sub_data, cost_ratio, model_name, file_prefix):
    """
    Visualizes a sensor deployment solution and saves the plot to the results folder.
    This function is generic and can plot ground-only, uav-only, or air-ground solutions.

    Args:
        solution (dict): The solution dictionary from the solver.
        sub_data (dict): Dictionary containing the largest subnetwork's data.
        cost_ratio (float): The cost ratio used for the experiment.
        model_name (str): The name of the model for the plot title.
        file_prefix (str): A prefix for the output filename (e.g., 'ulm', 'agslm').
    """
    print(f"\n--- Generating plot for {model_name}... ---")

    df_link_sub = sub_data['df_link_sub']
    df_node_sub = sub_data['df_node_sub']

    # --- Create and draw the subnetwork graph ---
    G = nx.Graph()
    # FIX: Use the correct column names 'x_coord' and 'y_coord' from the source CSV.
    pos = {row['node_id']: (row['x_coord'], row['y_coord']) for _, row in df_node_sub.iterrows()}
    G.add_nodes_from(pos.keys())
    for _, link in df_link_sub.iterrows():
        # FIX: Use the correct column names 'u_node_id' and 'v_node_id'.
        G.add_edge(link['u_node_id'], link['v_node_id'], link_id=link['link_id'])

    # --- Plotting setup ---
    fig, ax = plt.subplots(figsize=(20, 16))
    plt.style.use('default')

    # 1. Draw base network
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=25, node_color='white', edgecolors='black', linewidths=1.0)
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.8, edge_color='gray', alpha=0.7)

    # 2. Highlight deployed AVI sensors (if any)
    avi_link_ids = solution.get('ground_sensor_ids', set())
    if avi_link_ids:
        avi_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('link_id') in avi_link_ids]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=avi_edges, width=3.0, edge_color='green')

    # 3. Mark deployed UAVs and their FOV (if any)
    uav_node_ids = solution.get('uav_node_ids', set())
    uav_node_plotted = False
    uav_fov_plotted = False
    if uav_node_ids:
        node_coords_map = df_node_sub.set_index('node_id')
        for node_id in uav_node_ids:
            if node_id in node_coords_map.index:
                node_info = node_coords_map.loc[node_id]
                # FIX: Use 'x_coord' and 'y_coord' here as well.
                node_pos = (node_info['x_coord'], node_info['y_coord'])
                
                ax.plot(node_pos[0], node_pos[1], 'P', markersize=15, markerfacecolor='gold', markeredgecolor='black', 
                        label='UAV Deployment Node' if not uav_node_plotted else "")
                uav_node_plotted = True

                fov_rect = patches.Rectangle(
                    (node_pos[0] - UAV_FOV_WIDTH / 2, node_pos[1] - UAV_FOV_HEIGHT / 2),
                    UAV_FOV_WIDTH, UAV_FOV_HEIGHT,
                    linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.15,
                    label='UAV Field of View' if not uav_fov_plotted else ""
                )
                ax.add_patch(fov_rect)
                uav_fov_plotted = True

    # --- Aesthetics and Info ---
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)
    ax.set_aspect('equal', adjustable='box')
    for spine in ax.spines.values():
        spine.set_visible(False)

    total_cost = solution.get('total_cost', 'N/A')
    cost_str = f"{total_cost:.2f}" if isinstance(total_cost, float) else str(total_cost)
    
    full_title = (
        f"Deployment for: {model_name}\n"
        f"UAV:Ground Cost Ratio = {int(cost_ratio)}:1 | Total Cost = {cost_str}"
    )
    ax.set_title(full_title, fontsize=18, pad=20)
    
    legend_elements = [
        plt.Line2D([0], [0], color='gray', lw=1, label='Network Link'),
        plt.Line2D([0], [0], marker='o', color='w', label='Network Node', 
                   markerfacecolor='white', markeredgecolor='black', markersize=8),
    ]
    if avi_link_ids:
        legend_elements.append(plt.Line2D([0], [0], color='green', lw=3, label=f'Link with AVI ({len(avi_link_ids)})'))
    if uav_node_ids:
        legend_elements.append(plt.Line2D([0], [0], marker='P', color='w', label=f'UAV Node ({len(uav_node_ids)})',
                                           markerfacecolor='gold', markersize=15, markeredgecolor='black'))
        legend_elements.append(patches.Patch(facecolor='red', edgecolor='red', alpha=0.3, label='UAV Field of View'))
        
    ax.legend(handles=legend_elements, loc='best', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{file_prefix}_ratio_{int(cost_ratio)}_{timestamp}.svg"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    try:
        fig.savefig(filepath, format='svg', bbox_inches='tight')
        print(f"✅ Plot saved successfully to: {filepath}")
    except Exception as e:
        print(f"❌ Could not save the plot. Error: {e}")

    plt.close(fig)
