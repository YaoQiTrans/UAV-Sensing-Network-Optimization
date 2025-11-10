# calculate_path_density.py
"""
此脚本用于计算每个时间段内，各个路段的路径密度。
路径密度定义为在该时间段内，有多少条TOP200的路径经过了该路段。

工作流程:
1. 加载由 data_processor.py 生成的预处理路径数据。
2. 遍历每个时间段 (morning_peak, off_peak, evening_peak)。
3. 对每个时间段，统计所有TOP200路径，计算每个link_id出现的次数。
4. 将结果保存为三个独立的CSV文件，每个文件包含 'link_id' 和 'path_density' 两列。
5. 只有 path_density > 0 的路段才会被输出。
"""

import os
import pandas as pd
import pickle
from collections import Counter

# 尝试从配置文件导入路径，如果失败则使用默认值
try:
    from config_model3 import (
        PREPROCESSED_DATA_DIR_MODEL3,
        RESULTS_DIR_MODEL3,
        TRAJECTORY_DATA_DIRS
    )
    print("✅ 成功从 config_model3.py 加载配置。")
except ImportError:
    print("⚠️ 未找到 config_model3.py，将使用默认目录。")
    PREPROCESSED_DATA_DIR_MODEL3 = "preprocessed_data_model3"
    RESULTS_DIR_MODEL3 = "results_model3"
    TRAJECTORY_DATA_DIRS = {
        'morning_peak': 'data/trajectory/20181030_dX_0800_0830',
        'off_peak': 'data/trajectory/20181029_dX_0830_0900',
        'evening_peak': 'data/trajectory/20181029_dX_0900_0930',
    }

def calculate_and_save_path_density():
    """
    主函数，执行路径密度的计算和保存。
    """
    print("\n--- 开始计算路段的路径密度 ---")

    # 1. 定义并检查输入文件路径
    processed_paths_file = os.path.join(
        PREPROCESSED_DATA_DIR_MODEL3, 
        "processed_paths_by_period_undirected.pkl"
    )

    if not os.path.exists(processed_paths_file):
        print(f"❌ 错误: 输入文件未找到: {processed_paths_file}")
        print("   请先运行 'data_processor.py' 来生成此文件。")
        return

    # 2. 加载预处理的路径数据
    try:
        with open(processed_paths_file, 'rb') as f:
            paths_by_period = pickle.load(f)
        print(f"✅ 成功从 {processed_paths_file} 加载路径数据。")
    except Exception as e:
        print(f"❌ 加载文件时出错: {e}")
        return

    # 确保输出目录存在
    os.makedirs(RESULTS_DIR_MODEL3, exist_ok=True)

    # 3. 遍历每个时间段进行计算
    time_periods = TRAJECTORY_DATA_DIRS.keys()
    for period in time_periods:
        print(f"\n--- 正在处理时间段: {period} ---")
        
        if period not in paths_by_period:
            print(f"⚠️ 警告: 在加载的数据中未找到时间段 '{period}' 的数据，已跳过。")
            continue

        df_paths = paths_by_period[period]
        
        if 'path_links_set' not in df_paths.columns:
            print(f"❌ 错误: 在时间段 '{period}' 的DataFrame中未找到 'path_links_set' 列。")
            continue
            
        # 使用 collections.Counter 高效地统计每个link_id的出现次数
        link_counts = Counter()
        
        # 遍历该时段的每一条路径
        for path_set in df_paths['path_links_set']:
            link_counts.update(path_set)
            
        if not link_counts:
            print(f"ℹ️ 在时间段 '{period}' 中没有可处理的路径。")
            continue
            
        # 4. 将统计结果转换为DataFrame
        df_density = pd.DataFrame(
            link_counts.items(), 
            columns=['link_id', 'path_density']
        ).sort_values(by='link_id').reset_index(drop=True)

        print(f"   计算完成: 共 {len(df_density)} 个路段具有路径密度。")
        
        # 5. 保存为CSV文件
        output_filename = f"path_density_{period}.csv"
        output_filepath = os.path.join(RESULTS_DIR_MODEL3, output_filename)
        
        try:
            df_density.to_csv(output_filepath, index=False)
            print(f"✅ 成功保存路径密度文件到: {output_filepath}")
        except Exception as e:
            print(f"❌ 保存文件时出错: {e}")

    print("\n--- 所有时间段的路径密度计算完毕！ ---")


if __name__ == "__main__":
    calculate_and_save_path_density()
