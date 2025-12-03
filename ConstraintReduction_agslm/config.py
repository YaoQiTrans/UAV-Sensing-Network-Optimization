# config.py
"""
Configuration File
Manages all global parameters, file paths, and settings for the project.
This version is updated to process a UNION of paths from multiple trajectory datasets.
"""
import os

# --- Base Data File Paths ---
DATA_DIR = "data"
LINK_FILE = os.path.join(DATA_DIR, "pneuma_link-1.csv")
NODE_FILE = os.path.join(DATA_DIR, "pneuma_node.csv")

# --- Trajectory Data for Path Generation ---
# We will take the union of Top-K paths from all these time periods.
TRAJECTORY_DATA_DIRS = {
    'morning_peak': os.path.join(DATA_DIR, 'trajectory', '20181030_dX_0800_0830'),
    'off_peak':     os.path.join(DATA_DIR, 'trajectory', '20181029_dX_0830_0900'),
    'evening_peak': os.path.join(DATA_DIR, 'trajectory', '20181029_dX_0900_0930'),
}

# --- Path Generation Parameters ---
# We will extract the Top 200 paths from EACH period before taking the union.
NUM_TOP_PATHS_PER_PERIOD = 200
# Minimum number of links for a trajectory to be considered a valid path.
MIN_PATH_LENGTH = 2

# --- Output Directories ---
# New directories to avoid conflicts with previous experiments.
PREPROCESSED_DATA_DIR = "preprocessed_data_from_traj_union"
RESULTS_DIR = "results_from_traj_union"

# --- UAV Parameters ---
UAV_FOV_WIDTH = 300.0
UAV_FOV_HEIGHT = 300.0

# --- Cost Parameters ---
COST_GROUND_SENSOR = 1.0
UAV_GROUND_COST_RATIO = 7.0

# --- Solver Parameters ---
GUROBI_TIME_LIMIT = 3600  # 1 hour

# --- Miscellaneous ---
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("✅ Configuration for static models (from UNION of trajectory data) loaded.")
