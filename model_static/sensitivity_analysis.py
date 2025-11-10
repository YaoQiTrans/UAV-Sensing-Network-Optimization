# sensitivity_analysis.py
"""
敏感性分析脚本

功能：
本脚本旨在分析无人机与地面传感器（AVI）的成本比率（R）变化对
空地协同传感器部署模型（AGSLM）总成本的影响。

工作流程：
1. 定义一组预设的成本比率 R (R = 单架无人机成本 / 单个AVI成本)。
2. 检查必要的预处理数据是否存在，如果不存在，则自动运行 `preprocess.py`。
3. 加载预处理后的网络和模型参数。
4. 遍历每个预设的 R 值：
   a. 根据当前的 R 值计算无人机的成本。
   b. 调用 `solver.py` 中的 `solve_air_ground_model` 函数求解 AGSLM 模型。
   c. 记录求解状态、总成本、部署的无人机数量和地面传感器数量。
5. 将所有 R 值下的分析结果汇总，并保存到一个 CSV 文件中，以便后续分析和可视化。

使用方法：
直接在终端中运行此脚本即可：
python sensitivity_analysis.py
"""
import os
import sys
import subprocess
import pandas as pd
import time

# 导入现有项目模块中的配置和函数
from config import (
    PREPROCESSED_DATA_DIR, COST_GROUND_SENSOR, RESULTS_DIR
)
from utils import load_preprocessed_data
from solver import solve_air_ground_model

def run_sensitivity_analysis():
    """
    执行完整的敏感性分析流程。
    """
    # --- 1. 定义敏感性分析参数 ---
    # 您可以在此列表中修改、添加或删除要测试的成本比率 R
    # R = Cost_UAV / Cost_AVI
    cost_ratios_R = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    
    print("--- 启动无人机成本敏感性分析 ---")
    print(f"将要测试的成本比率 (R) 列表: {cost_ratios_R}")

    # 用于存储每一次运行结果的列表
    analysis_results = []

    # --- 2. 检查并加载预处理数据 ---
    # 这部分逻辑与 main.py 保持一致，确保数据准备就绪
    preprocessed_file = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed_data_bundle.pkl")
    if not os.path.exists(preprocessed_file):
        print("\n--- 未找到预处理数据，正在调用 preprocess.py... ---")
        try:
            # 运行预处理脚本
            subprocess.run([sys.executable, "preprocess.py"], check=True)
            print("--- 预处理完成 ---")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ 运行预处理脚本失败: {e}")
            print("❌ 敏感性分析中止。")
            return

    print("\n--- 正在加载预处理数据包... ---")
    data = load_preprocessed_data(preprocessed_file)
    if not data:
        print("❌ 加载预处理数据失败，分析中止。")
        return

    # 将加载的数据分拆为求解器所需的输入格式
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

    # --- 3. 循环遍历 R 值，运行 AGSLM 模型 ---
    total_runs = len(cost_ratios_R)
    for i, r in enumerate(cost_ratios_R):
        print(f"\n--- [{i+1}/{total_runs}] 正在为成本比率 R = {r} 求解 AGSLM 模型... ---")
        
        # 根据比率 R 计算当前无人机成本
        cost_uav = COST_GROUND_SENSOR * r

        # 调用核心求解器函数
        solution, status = solve_air_ground_model(
            subnetwork_data, uav_params, COST_GROUND_SENSOR, cost_uav
        )

        # --- 4. 记录当次运行的结果 ---
        if status == "Optimal":
            num_uavs = len(solution.get('uav_node_ids', set()))
            num_ground = len(solution.get('ground_sensor_ids', set()))
            total_cost = solution.get('total_cost', -1.0)
            
            current_result = {
                'cost_ratio_R': r,
                'total_cost': total_cost,
                'num_uavs': num_uavs,
                'num_ground_sensors': num_ground,
                'status': status
            }
            analysis_results.append(current_result)
            print(f"  ✅ 求解成功！状态: {status}，总成本: {total_cost:.2f}")
            print(f"  部署方案: {num_uavs} 架无人机, {num_ground} 个地面传感器")
        else:
            # 如果求解失败或未找到最优解，也记录下来
            current_result = {
                'cost_ratio_R': r,
                'total_cost': float('nan'), # 使用 NaN 表示无效成本
                'num_uavs': -1,
                'num_ground_sensors': -1,
                'status': status
            }
            analysis_results.append(current_result)
            print(f"  ❌ 求解失败或未找到最优解。状态: {status}")

    # --- 5. 保存最终的分析结果 ---
    if not analysis_results:
        print("\n--- 分析未产生任何结果，程序结束。 ---")
        return

    # 将结果列表转换为 Pandas DataFrame
    df_results = pd.DataFrame(analysis_results)
    
    # 定义并创建输出文件路径
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"sensitivity_analysis_agslm_results_{timestamp}.csv"
    output_path = os.path.join(RESULTS_DIR, output_filename)

    try:
        # 保存为 CSV 文件
        df_results.to_csv(output_path, index=False, float_format='%.2f')
        print(f"\n--- 敏感性分析全部完成！ ---")
        print(f"✅ 结果已成功保存至: {output_path}")
        print("\n最终结果概览:")
        print(df_results.to_string(index=False))
    except Exception as e:
        print(f"❌ 保存结果文件时出错: {e}")

if __name__ == "__main__":
    run_sensitivity_analysis()
