# config_model3.py
"""
Configuration File for Model 3: Dynamic Sensor Location Model (DSLM)
with Nest Selection.
"""
import os

# --- Base Data File Paths ---
DATA_DIR = "C:/YaoQiTrans/03_Resources/实验/CODE_DSLP/UAV-Sensing-Network-Optimization/data"
LINK_FILE = os.path.join(DATA_DIR, "pneuma_link-1.csv")
NODE_FILE = os.path.join(DATA_DIR, "pneuma_node.csv")

# --- Time-of-Day Trajectory Data (Same as before) ---
# 确保这些路径指向您的轨迹数据文件夹
TRAJECTORY_DATA_DIRS = {
    'morning_peak': os.path.join(DATA_DIR, 'trajectory', '20181030_dX_0800_0830'),
    'off_peak':     os.path.join(DATA_DIR, 'trajectory', '20181029_dX_0830_0900'),
    'evening_peak': os.path.join(DATA_DIR, 'trajectory', '20181029_dX_0900_0930'),
}

# --- Path Generation Parameters (Same as before) ---
NUM_TOP_PATHS = 200
MIN_PATH_LENGTH = 2

# --- Output Directories for Model 3 ---
PREPROCESSED_DATA_DIR_MODEL3 = "preprocessed_data_model3"
RESULTS_DIR_MODEL3 = "results_model3"

# --- UAV Parameters ---
UAV_FOV_WIDTH = 300.0
UAV_FOV_HEIGHT = 300.0
UAV_FLEET_SIZE = 20 # Upper bound K for the fleet size

# --- Nest (机巢) Parameters ---
# 您提供的机巢候选点
# CANDIDATE_NEST_NODES = [95663527, 97788195, 97797102, 250691723, 250714051, 599116897]
# 机巢服务半径 (米)
# NEST_SERVICE_RADIUS = 500.0

# --- Cost Parameters (c_a, c_U, c_h) ---
COST_AVI_INVESTMENT = 1.0
COST_UAV_INVESTMENT = 5.0 # c_U: 购买一架无人机的成本
# c_h: 建设一个机巢的成本 (您可以根据实际情况调整此值)
# COST_NEST_INVESTMENT = 10.0

# --- Solver Parameters ---
GUROBI_TIME_LIMIT = 3600 # Gurobi求解时间限制 (秒)

# --- Visualization Parameters ---
TIME_PERIOD_COLORS = {
    'morning_peak': '#FF6347', # Tomato
    'off_peak':     '#4169E1', # RoyalBlue
    'evening_peak': '#FFA500', # Orange
}
# 新增机巢和其服务范围的可视化颜色
# NEST_COLOR = '#8B4513'      # SaddleBrown (for nest marker)
# SERVICE_LINE_COLOR = '#A9A9A9' # DarkGray (for service lines)

# --- Create output directories ---
os.makedirs(PREPROCESSED_DATA_DIR_MODEL3, exist_ok=True)
os.makedirs(RESULTS_DIR_MODEL3, exist_ok=True)

print("✅ Configuration for Model 3 (DSLM with Nests) loaded.")
