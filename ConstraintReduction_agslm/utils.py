# utils.py
"""
Utility Functions Module
Contains various general-purpose helper functions for the project.
"""
import ast
import pickle
import pandas as pd
import scipy.sparse

def str_to_list(node_list_repr):
    """
    Safely converts a string representation of a node list into a Python list.

    Args:
        node_list_repr (str): 节点列表的字符串表示形式 (e.g., "[1, 2, 3]").

    Returns:
        list: The parsed list of nodes, or None if parsing fails.
    """
    if not isinstance(node_list_repr, str):
        return None
    try:
        if node_list_repr.startswith('[') and node_list_repr.endswith(']'):
            node_list = ast.literal_eval(node_list_repr)
            if isinstance(node_list, list):
                return [int(node) for node in node_list]
    except (ValueError, SyntaxError):
        return None
    return None

def save_preprocessed_data(data, file_path):
    """
    Saves preprocessed data using pickle.

    Args:
        data (dict): A dictionary containing the data to save.
        file_path (str): The path to the file.
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ 数据已成功保存至: {file_path}")
    except Exception as e:
        print(f"❌ 保存数据时出错: {e}")

def load_preprocessed_data(file_path):
    """
    Loads preprocessed data from a pickle file.

    Args:
        file_path (str): The path to the file to load.

    Returns:
        dict: The loaded data, or None if it fails.
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ 数据已成功从以下位置加载: {file_path}")
        return data
    except FileNotFoundError:
        print(f"⚠️ 预处理数据文件未找到: {file_path}")
        return None
    except Exception as e:
        print(f"❌ 加载数据时出错: {e}")
        return None

def map_indices_to_ids(indices_set, id_series):
    """
    Maps a set of 0-based indices back to their original IDs.
    
    Args:
        indices_set (set): Set of 0-based indices.
        id_series (pd.Series): A pandas Series containing the original IDs.
    
    Returns:
        set: A set containing the corresponding original IDs.
    """
    if not indices_set:
        return set()
    return set(id_series.iloc[list(indices_set)].tolist())
