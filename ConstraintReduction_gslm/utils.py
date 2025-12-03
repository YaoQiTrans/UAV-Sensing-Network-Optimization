# utils.py
"""
辅助函数模块。
"""
import ast
import pandas as pd

def str_to_list(node_list_repr):
    """
    将表示节点列表的字符串安全地转换为 Python 列表。
    Safely converts a string representation of a node list into a Python list.

    Args:
        node_list_repr (str): 节点列表的字符串表示形式 (e.g., "[1, 2, 3]" or "1-2-3").

    Returns:
        list: 解析后的节点列表，如果解析失败则返回 None。
              The parsed list of nodes, or None if parsing fails.
    """
    if not isinstance(node_list_repr, str):
        # print(f"Warning: Input is not a string: {node_list_repr}")
        return None
    try:
        # 尝试使用 ast.literal_eval 解析 "[1, 2, 3]" 格式
        # Try parsing "[1, 2, 3]" format using ast.literal_eval
        if node_list_repr.startswith('[') and node_list_repr.endswith(']'):
            node_list = ast.literal_eval(node_list_repr)
            if isinstance(node_list, list):
                # 确保列表中的所有元素都是整数或可以转换为整数
                # Ensure all elements in the list are integers or can be converted
                return [int(node) for node in node_list]
            else:
                # print(f"Warning: Parsed result is not a list: {node_list}")
                return None
        # 尝试解析 "1-2-3" 格式
        # Try parsing "1-2-3" format
        elif '-' in node_list_repr:
            node_list = [int(node.strip()) for node in node_list_repr.split('-')]
            return node_list
        # 尝试解析 "1 2 3" 格式 (以空格分隔)
        # Try parsing "1 2 3" format (space-separated)
        elif ' ' in node_list_repr:
             node_list = [int(node.strip()) for node in node_list_repr.split()]
             return node_list
        # 如果是单个数字
        # If it's a single number
        elif node_list_repr.isdigit():
             return [int(node_list_repr)]
        else:
            # print(f"Warning: Unrecognized node list format: {node_list_repr}")
            return None
    except (ValueError, SyntaxError, TypeError) as e:
        # print(f"Error parsing node list string '{node_list_repr}': {e}")
        return None

def map_indices_to_ids(indices_set, id_series):
    """
    将一组 0-based 索引映射回其原始 ID。
    Maps a set of 0-based indices back to their original IDs.

    Args:
        indices_set (set): 包含 0-based 索引的集合。
                           Set containing 0-based indices.
        id_series (pd.Series): 包含原始 ID 的 Pandas Series，其索引应与 0-based 索引对应。
                               Pandas Series containing original IDs, whose index should correspond
                               to the 0-based indices.

    Returns:
        set: 包含相应原始 ID 的集合。
             Set containing the corresponding original IDs.
    """
    if not indices_set:
        return set()
    try:
        # 使用 .iloc 进行基于整数位置的索引
        # Use .iloc for integer-location based indexing
        original_ids = id_series.iloc[list(indices_set)].tolist()
        # 确保 ID 是标准 Python 类型
        # Ensure IDs are standard Python types
        return {int(id_) if pd.notna(id_) else id_ for id_ in original_ids}
    except IndexError:
        print(f"Error: One or more indices in {indices_set} are out of bounds for the id_series.")
        # 返回在范围内可以映射的 ID
        # Return IDs that can be mapped within the bounds
        valid_indices = {idx for idx in indices_set if 0 <= idx < len(id_series)}
        if not valid_indices:
            return set()
        original_ids = id_series.iloc[list(valid_indices)].tolist()
        return {int(id_) if pd.notna(id_) else id_ for id_ in original_ids}
    except Exception as e:
        print(f"An unexpected error occurred during index-to-ID mapping: {e}")
        return set()

