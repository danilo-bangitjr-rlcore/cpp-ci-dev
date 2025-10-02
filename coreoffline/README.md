# CoreOffline
CoreOffline provides tools for processing historical data, generating reports, and training RL agents without online interaction.

## Installation

### Requirements
- Python 3.13.0 (exact version required)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Setup

1. Clone the repository and navigate to the `coreoffline` directory:
```bash
cd coreoffline
```

2. Install dependencies using `uv`:
```bash
uv sync
```


## Project Structure

```
coreoffline/
├── coreoffline/           # Main package directory
│   ├── scripts/           # Executable scripts for various workflows
│   │   ├── behaviour_clone.py
│   │   ├── create_data_report.py
│   │   ├── create_transition_report.py
│   │   ├── generate_tag_configs.py
│   │   └── run_offline_training.py
│   └── utils/             # Utility modules and helper functions
└── tests/                 # Test suite

```

## Scripts

The `scripts/` directory contains executable Python modules for different offline workflows. All scripts use configuration files (YAML) to specify their behavior. The structure of these configs can be found in `coreoffline/utils/config.py`.

### `behaviour_clone.py`

Trains behavior cloning models to imitate historical actions using supervised learning.

**Purpose:** Evaluate how well different models (baseline, linear regression, MLP) can predict historical actions from observations.

**Usage:**
```bash
python -m coreoffline.scripts.behaviour_clone --config path/to/config.yaml
```

**Outputs:**
- Scatter plots in `outputs/plots/behaviour_clone/`

---

### `create_data_report.py`

Generates reports on sensor data quality and statistics.

**Purpose:** Analyze raw or processed data to identify data quality issues, sensor ranges, correlations, and goal violations.

**Usage:**
```bash
python -m coreoffline.scripts.create_data_report --config path/to/config.yaml
```

**Outputs:**
- `outputs/sensor_report.csv` - Statistical summaries
- `outputs/cross_correlation.csv` - Correlation matrix
- `outputs/goal_violations.csv` - Goal violation events
- `outputs/zone_violation_statistics.csv` - Zone boundary violations
- `outputs/plots/` - Various plots for each sensor

---

### `create_transition_report.py`

Generates reports focused on RL transitions (state, action, reward, next_state).

**Purpose:** Analyze the quality and distribution of RL transitions generated from historical data, useful for debugging reward functions and transition generation.


**Usage:**
```bash
python -m coreoffline.scripts.create_transition_report --config path/to/config.yaml
```

**Outputs:**
- Transition statistics and visualizations

---

### `generate_tag_configs.py`

Automatically generates tag configuration files from database statistics.

**Purpose:** Bootstrap configuration files by analyzing historical data in the database to determine appropriate operating and expected ranges for each tag.

**Usage:**
```bash
python -m coreoffline.scripts.generate_tag_configs --config path/to/config.yaml
```

**Outputs:**
- `tags.yaml` - Generated tag configuration file with ranges

---

### `run_offline_training.py`

Trains a reinforcement learning agent using offline (batch) data.

**Purpose:** Perform offline RL training on pre-collected transitions without environment interaction, then optionally evaluate on held-out periods.

**Usage:**
```bash
python -m coreoffline.scripts.run_offline_training --config path/to/config.yaml
```
