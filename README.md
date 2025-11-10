# Optimal Layout of a UAV-Integrated Sensing Network for Time-Varying Traffic Flow Observability

This repository contains the official code implementation for the paper: "Optimal Layout of a UAV-Integrated Sensing Network for Time-Varying Traffic Flow Observability".

**Authors:** Qi Cao, Yao Qi, Gang Ren
**Status:** Submitted to *Transportation Science*.

## Repository Structure

The repository is organized into folders corresponding to the models presented in the paper.

⚠️ Due to relative imports (e.g., `from config_model4 import ...`), each model is self-contained. You must `cd` into the specific model's directory before running any scripts.

```
UAV-Sensing-Network-Optimization/
├── model_static/           # Static ASLM (Sec 4.1) & AGSLM (Sec 4.2)
├── model_dynamic/          # Dynamic D-AGSLM (Sec 4.3)
├── model_budget/           # Budget-Constrained BD-AGSLM (Sec 4.4)
└── validation/             # Computational Performance Validation (Sec 6.5)
```

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/YourUsername/UAV-Sensing-Network-Optimization.git
    cd UAV-Sensing-Network-Optimization
    ```

2.  Install the required Python packages (see `requirements.txt` for details):

    ```bash
    pip install -r requirements.txt
    ```

3.  **Solver Requirement:**
    This project relies on **Gurobi 11.0** (as used in the paper) or a compatible version. You must have a valid Gurobi license installed.

## Data

This study is based on the PNEUMA open-source dataset.

The raw PNEUMA trajectory data (used by `data_processor.py`) consists of tens of thousands of individual CSV files and is too large to be included in this repository.

Therefore, this repository is designed to run using the final preprocessed `.pkl` files. These `.pkl` files are the direct output of our `data_processor.py` and `preprocess.py` scripts and serve as the direct input for the main solver scripts.

## How to Run

As stated above, you must be *inside* the specific model's directory to run it.

### 1\. Static Models (AGSLM, ASLM, GSLM)

Corresponds to Section 4.1, 4.2, and 6.2 of the paper.

```bash
cd model_static
python main.py
```

  * This script will run the static models as defined in `config.py`.
  * Results are saved in `model_static/results_from_traj_union/`.

### 2\. Dynamic Model (D-AGSLM)

Corresponds to Section 4.3 and 6.3 of the paper.

```bash
cd model_dynamic
python main_model3.py
```

  * This script runs the dynamic, cost-minimization model.
  * Results will be saved in the directory specified in `config_model3.py`.

### 3\. Budget-Constrained Model (BD-AGSLM)

Corresponds to Section 4.4 and 6.4 of the paper.

```bash
cd model_budget
python main_model4.py
```

  * This script runs the benefit-maximization model for the list of budgets specified in `config_model4.py` (e.g., `BUDGETS = [80, 50, 20]`).
  * Results, plots, and CSVs of observable paths are saved in `model_budget/results_model4/`.

### 4\. Validation (Decomposition)

Corresponds to Section 5 and 6.5 of the paper.

```bash
cd validation
python decompose_and_solve.py
```

  * This runs the computational performance benchmark described in the paper.
  * The benchmark data and original code are available at: https://github.com/yuanjian24/sensor_location

## How to Cite

If you use this code in your research, please cite our paper.

```bibtex
To be submitted...
```