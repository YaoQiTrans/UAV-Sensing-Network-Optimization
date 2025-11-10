# config_model4.py
"""
Configuration File for Model 4: Budget-Constrained Dynamic Sensor
Location Model (B-DSLM).
"""
import os

# --- Base Data File Paths (No change) ---
DATA_DIR = "data"
LINK_FILE = os.path.join(DATA_DIR, "pneuma_link-1.csv")
NODE_FILE = os.path.join(DATA_DIR, "pneuma_node.csv")

# --- Time-of-Day Trajectory Data (No change) ---
TRAJECTORY_DATA_DIRS = {
    'morning_peak': os.path.join(DATA_DIR, 'trajectory', '20181030_dX_0800_0830'),
    'off_peak':     os.path.join(DATA_DIR, 'trajectory', '20181029_dX_0830_0900'),
    'evening_peak': os.path.join(DATA_DIR, 'trajectory', '20181029_dX_0900_0930'),
}

# --- Path Generation Parameters (No change) ---
NUM_TOP_PATHS = 200
MIN_PATH_LENGTH = 2

# --- Output Directories for Model 4 ---
PREPROCESSED_DATA_DIR_MODEL4 = "preprocessed_data_model4"
RESULTS_DIR_MODEL4 = "results_model4"

# --- UAV Parameters (No change) ---
UAV_FOV_WIDTH = 300.0
UAV_FOV_HEIGHT = 300.0
UAV_FLEET_SIZE = 20 # Upper bound K for the fleet size

# --- Nest Parameters (No change) ---
CANDIDATE_NEST_NODES = [95663527, 97788195, 97797102, 250691723, 250714051, 599116897]
NEST_SERVICE_RADIUS = 500.0

# --- Cost Parameters (No change) ---
COST_AVI_INVESTMENT = 1.0
COST_UAV_INVESTMENT = 5.0
COST_NEST_INVESTMENT = 10.0

# --- NEW: Model 4 Specific Parameters ---
# Model 3的最优成本是94，所以我们测试一系列小于94的预算值
BUDGETS = [80, 50, 20] 

# 路径观测效益权重的策略
# 'uniform': 所有路径权重为1 (当前采用)
# 'flow_based': 使用路径的小时流量作为权重 (未来可以扩展)
PATH_WEIGHT_STRATEGY = 'uniform'


# --- Solver Parameters ---
GUROBI_TIME_LIMIT = 3600 # Gurobi求解时间限制 (秒)

# --- Visualization Parameters (No change) ---
TIME_PERIOD_COLORS = {
    'morning_peak': '#FF6347', # Tomato
    'off_peak':     '#4169E1', # RoyalBlue
    'evening_peak': '#FFA500', # Orange
}
NEST_COLOR = '#8B4513'      # SaddleBrown
SERVICE_LINE_COLOR = '#A9A9A9' # DarkGray

# --- Create output directories ---
os.makedirs(PREPROCESSED_DATA_DIR_MODEL4, exist_ok=True)
os.makedirs(RESULTS_DIR_MODEL4, exist_ok=True)

print("✅ Configuration for Model 4 (B-DSLM) loaded.")
print(f"   - Budgets to test: {BUDGETS}")
print(f"   - Path weight strategy: {PATH_WEIGHT_STRATEGY}")
