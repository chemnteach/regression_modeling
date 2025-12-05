"""
Configuration file for RAPID (Regression Analysis Pipeline with Intelligent Data).

This module contains all configuration constants, thresholds, and settings
used throughout the feature reduction and modeling pipeline.

PEP 8 Compliance:
    - All constants use UPPER_CASE_WITH_UNDERSCORES naming convention
    - Module-level constants grouped by functionality
    - Descriptive comments for all threshold values

Author: RAPID Development Team
Last Updated: 2025-12-01
"""

import os
import psutil
from multiprocessing import cpu_count

# =============================================================================
# Directory and File Paths
# =============================================================================

# Directory for log files
LOG_DIR = 'logs'

# Directory for output data files
DATA_DIR = 'data'

# Directory for output figures and plots
FIGURES_DIR = 'figures'


# =============================================================================
# Parallel Processing Configuration (Auto-Detected)
# =============================================================================

def detect_gpu_availability():
    """
    Detect if GPU is available for XGBoost, LightGBM, and CatBoost.
    
    Returns
    -------
    dict
        GPU availability information including device type and recommendations.
    """
    gpu_info = {
        'available': False,
        'device': 'cpu',
        'xgboost_method': 'hist',  # CPU default
        'lightgbm_device': 'cpu',
        'catboost_task': 'CPU',
        'recommendation': 'CPU only'
    }
    
    # Try to detect CUDA GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['device'] = 'cuda'
            gpu_info['xgboost_method'] = 'gpu_hist'
            gpu_info['lightgbm_device'] = 'gpu'
            gpu_info['catboost_task'] = 'GPU'
            gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
            gpu_info['recommendation'] = 'GPU acceleration enabled'
            return gpu_info
    except (ImportError, RuntimeError):
        pass
    
    # Fallback: Try XGBoost GPU detection with actual test
    try:
        import xgboost as xgb
        # Perform lightweight GPU test - will fail if GPU not available
        dmat = xgb.DMatrix([[1, 2]], label=[1])
        params = {'tree_method': 'gpu_hist', 'verbosity': 0}
        bst = xgb.train(params, dmat, num_boost_round=1)
        
        # GPU test succeeded
        gpu_info['available'] = True
        gpu_info['device'] = 'gpu'
        gpu_info['xgboost_method'] = 'gpu_hist'
        gpu_info['lightgbm_device'] = 'gpu'
        gpu_info['catboost_task'] = 'GPU'
        gpu_info['recommendation'] = 'GPU detected (via XGBoost)'
    except (ImportError, Exception):
        # GPU not available or XGBoost not installed
        pass
    
    return gpu_info


def detect_hardware_capabilities():
    """
    Auto-detect system hardware and recommend parallelization settings.
    
    Adapts to different hardware profiles:
    - LAPTOP: <20 GB RAM, conservative parallelization
    - WORKSTATION: 20-64 GB RAM, balanced approach
    - SERVER: 64+ GB RAM, aggressive parallelization
    
    Also detects GPU availability for accelerated training.
    
    Returns
    -------
    dict
        Hardware capabilities including CPU cores, RAM, GPU, and recommended
        parallelization settings.
    """
    # Get system information
    total_cores = cpu_count()  # Physical + logical cores
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # Detect GPU
    gpu_info = detect_gpu_availability()
    
    # Conservative estimate: reserve cores for OS
    if total_cores <= 4:  # Laptop
        usable_cores = max(1, total_cores - 1)  # Reserve 1 core
    elif total_cores <= 16:  # Workstation
        usable_cores = max(1, total_cores - 2)  # Reserve 2 cores
    else:  # Server (32+ cores)
        usable_cores = max(1, total_cores - 4)  # Reserve 4 cores for OS/overhead
    
    # Memory-based job limit (estimate RAM per parallel job)
    # Account for: base data + model training + cross-validation overhead
    if total_ram_gb < 20:  # Laptop (16 GB)
        max_jobs_by_memory = max(1, int(available_ram_gb / 2.5))  # ~2.5 GB per job
        profile = "LAPTOP"
        ram_per_job_gb = 2.5
    elif total_ram_gb < 64:  # Workstation (32-64 GB)
        max_jobs_by_memory = max(1, int(available_ram_gb / 2))  # ~2 GB per job
        profile = "WORKSTATION"
        ram_per_job_gb = 2.0
    else:  # Server (128+ GB)
        max_jobs_by_memory = max(1, int(available_ram_gb / 1.5))  # ~1.5 GB per job
        profile = "SERVER"
        ram_per_job_gb = 1.5
    
    # Final recommendation: min of CPU-based and memory-based limits
    recommended_jobs = min(usable_cores, max_jobs_by_memory)
    
    # Cap at reasonable max (diminishing returns beyond 24-32 parallel jobs)
    if profile == "SERVER":
        recommended_jobs = min(recommended_jobs, 28)  # Leave 4 cores for overhead
    
    return {
        'total_cores': total_cores,
        'usable_cores': usable_cores,
        'total_ram_gb': round(total_ram_gb, 1),
        'available_ram_gb': round(available_ram_gb, 1),
        'ram_per_job_gb': ram_per_job_gb,
        'max_jobs_by_memory': max_jobs_by_memory,
        'recommended_jobs': recommended_jobs,
        'profile': profile,
        'gpu_available': gpu_info['available'],
        'gpu_device': gpu_info['device'],
        'gpu_name': gpu_info.get('gpu_name', 'N/A')
    }


# Auto-detect hardware on module import
_HARDWARE = detect_hardware_capabilities()
_GPU = detect_gpu_availability()

# Number of parallel jobs for scikit-learn operations
# -1 = use all cores, 1 = no parallelization, N = use N cores
# Auto-set based on hardware detection
N_JOBS = _HARDWARE['recommended_jobs']

# Backend for parallel processing ('loky', 'threading', 'multiprocessing')
# 'loky' is default and most robust for scikit-learn
PARALLEL_BACKEND = 'loky'

# GPU Configuration for XGBoost, LightGBM, and CatBoost
USE_GPU = _GPU['available']  # Auto-detected
XGBOOST_TREE_METHOD = _GPU['xgboost_method']  # 'gpu_hist' or 'hist'
LIGHTGBM_DEVICE = _GPU['lightgbm_device']  # 'gpu' or 'cpu'
CATBOOST_TASK_TYPE = _GPU['catboost_task']  # 'GPU' or 'CPU'

# Random state for reproducibility
RANDOM_STATE = 42

# Test set size for train-test split
TEST_SIZE = 0.20  # 20%

# Number of cross-validation folds
CV_FOLDS = 5

# Maximum number of features to select (None = no limit)
MAX_FEATURES = None

# Hyperparameter search iterations (adapt based on hardware)
# More iterations = better optimization but slower
# With parallelization, servers can handle many more iterations efficiently
if _HARDWARE['profile'] == "LAPTOP":
    HYPERPARAM_SEARCH_ITER = 10  # Quick search (10 configs x 5 folds = 50 fits)
elif _HARDWARE['profile'] == "WORKSTATION":
    HYPERPARAM_SEARCH_ITER = 20  # Balanced (20 configs x 5 folds = 100 fits)
else:  # SERVER (128 GB RAM, 32 cores)
    HYPERPARAM_SEARCH_ITER = 100  # Thorough search (100 configs x 5 folds = 500 fits)
    # With 28 parallel jobs, 500 fits completes in ~18 sequential batches
    PARALLEL_PRE_DISPATCH = 'all'  # Aggressive


# =============================================================================
# Missing Data Thresholds
# =============================================================================

# Maximum percentage of missing data before column is automatically dropped
MAX_MISSING_DATA = 0.40  # 40%

# Threshold for low missing data (use simple imputation: median/mode)
LOW_MISSING_THRESHOLD = 0.05  # 5%

# Threshold for medium missing data (use KNN imputation)
MEDIUM_MISSING_THRESHOLD = 0.20  # 20%

# Threshold for high missing data (use iterative imputation MICE)
HIGH_MISSING_THRESHOLD = 0.40  # 40%


# =============================================================================
# Categorical Encoding Thresholds
# =============================================================================

# Maximum unique categories for one-hot encoding
ONE_HOT_ENCODING_MAX_CATEGORIES = 10

# Maximum unique categories for label encoding
LABEL_ENCODING_MAX_CATEGORIES = 50

# Threshold percentage for high cardinality columns (likely IDs)
HIGH_CARDINALITY_THRESHOLD = 0.95  # 95% unique values


# =============================================================================
# Feature Selection and Variance Thresholds
# =============================================================================

# Minimum variance threshold for feature selection
LOW_VARIANCE_THRESHOLD = 0.01  # 1%

# Correlation threshold for feature importance
CORRELATION_THRESHOLD = 0.10  # 10%

# Correlation threshold for removing highly correlated features (multicollinearity)
FEATURE_CORRELATION_THRESHOLD = 0.95  # 95%

# Minimum ratio of features to retain after selection
MIN_FEATURE_RETENTION_RATIO = 0.20  # 20%

# Cumulative importance threshold for feature selection
OPTIMIZATION_CDF_THRESHOLD = 0.95  # 95%

# Number of trees for feature importance estimation
FEATURE_IMPORTANCE_N_ESTIMATORS = 100

# Models to use for feature importance calculation
# Available: 'RF' (RandomForest), 'XGB' (XGBoost), 'GBT' (GradientBoosting), 
#            'CB' (CatBoost), 'LGB' (LightGBM), 'ETR' (ExtraTrees)
# Note: LGB and ETR available but historically added little value
FEATURE_IMPORTANCE_MODELS = ['RF', 'XGB', 'GBT', 'CB']

# Maximum iterations for iterative imputer
ITERATIVE_IMPUTER_MAX_ITER = 10

# R² cutoff threshold for base model selection
CUTOFF_R2 = 0.30  # 30%


# =============================================================================
# Visualization Settings
# =============================================================================

# Number of top features to display in importance plots
FEATURE_IMPORTANCE_TOP_N = 20

# Figure size for feature importance plots
FEATURE_IMPORTANCE_FIGSIZE = (10, 8)

# Whether to use square root scale for importance visualization
FEATURE_IMPORTANCE_USE_SQRT_SCALE = True

# Number of bins for histograms
HISTOGRAM_BINS = 50


# =============================================================================
# Data Cleaning Flags
# =============================================================================

# Whether to automatically remove date-like columns
REMOVE_DATE_COLUMNS = True

# Whether to automatically remove duplicate rows
REMOVE_DUPLICATE_ROWS = True

# Whether to automatically remove low variance columns
REMOVE_LOW_VARIANCE_COLS = True


# =============================================================================
# Machine Learning Model Parameters
# =============================================================================

# Random state for reproducibility
RANDOM_STATE = 42

# Test set size for train-test split
TEST_SIZE = 0.20  # 20%

# Number of cross-validation folds
CV_FOLDS = 5

# Maximum number of features to select (None = no limit)
MAX_FEATURES = None


# =============================================================================
# Logging Configuration
# =============================================================================

# Logging level for file handler (DEBUG, INFO, WARNING, ERROR, CRITICAL)
FILE_LOG_LEVEL = 'DEBUG'

# Logging level for console handler
CONSOLE_LOG_LEVEL = 'INFO'

# Log file format
LOG_FORMAT_FILE = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Console log format (simpler)
LOG_FORMAT_CONSOLE = '%(levelname)s: %(message)s'

# Date format for log timestamps
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


# =============================================================================
# Display and Output Settings
# =============================================================================

# Maximum number of columns to display in summaries
MAX_DISPLAY_COLUMNS = 20

# Number of decimal places for floating point display
DECIMAL_PLACES = 4

# Width of separator lines in console output
SEPARATOR_WIDTH = 80

# Separator character
SEPARATOR_CHAR = '='


# =============================================================================
# Model Evaluation Metrics
# =============================================================================

# Primary metric for model evaluation (RÂ², MAE, MSE, RMSE)
PRIMARY_METRIC = 'RÂ²'

# Whether to include stacking ensemble analysis
RUN_STACKING_ANALYSIS = True


# =============================================================================
# Data Type Detection Patterns
# =============================================================================

# Patterns for detecting date-like columns (regex patterns)
DATE_PATTERNS = [
    r'date',
    r'time',
    r'year',
    r'month',
    r'day',
    r'dt',
    r'timestamp'
]

# Patterns for detecting ID-like columns
ID_PATTERNS = [
    r'id$',
    r'^id',
    r'_id$',
    r'identifier',
    r'key$'
]


# =============================================================================
# Validation Settings
# =============================================================================

# Minimum number of rows required for analysis
MIN_ROWS = 10

# Minimum number of numeric columns required for regression
MIN_NUMERIC_COLS = 1

# Maximum percentage of identical values before flagging as potential issue
MAX_IDENTICAL_VALUE_THRESHOLD = 0.99  # 99%


# =============================================================================
# Helper Functions
# =============================================================================

def validate_config() -> bool:
    """
    Validate configuration values for logical consistency.
    
    Returns
    -------
    bool
        True if configuration is valid, False otherwise.
        
    Raises
    ------
    ValueError
        If configuration values are logically inconsistent.
    """
    # Validate missing data thresholds are in correct order
    if not (LOW_MISSING_THRESHOLD < MEDIUM_MISSING_THRESHOLD < HIGH_MISSING_THRESHOLD):
        raise ValueError(
            "Missing data thresholds must be in order: "
            "LOW < MEDIUM < HIGH"
        )
    
    # Validate encoding thresholds
    if ONE_HOT_ENCODING_MAX_CATEGORIES >= LABEL_ENCODING_MAX_CATEGORIES:
        raise ValueError(
            "ONE_HOT_ENCODING_MAX_CATEGORIES must be less than "
            "LABEL_ENCODING_MAX_CATEGORIES"
        )
    
    # Validate test size
    if not 0 < TEST_SIZE < 1:
        raise ValueError("TEST_SIZE must be between 0 and 1")
    
    return True


def print_config() -> None:
    """Print current configuration settings in a formatted display."""
    print(f"{SEPARATOR_CHAR * SEPARATOR_WIDTH}")
    print("âš™ï¸  DATA PREPROCESSING CONFIGURATION")
    print(f"{SEPARATOR_CHAR * SEPARATOR_WIDTH}")
    
    # Hardware detection
    print(f"\nðŸ–¥ï¸  Hardware Profile: {_HARDWARE['profile']}")
    print(f"   â€¢ Total CPU cores: {_HARDWARE['total_cores']} "
          f"(using {_HARDWARE['usable_cores']} for ML)")
    print(f"   â€¢ Parallel jobs: {N_JOBS} (auto-detected)")
    print(f"   â€¢ Total RAM: {_HARDWARE['total_ram_gb']:.1f} GB "
          f"({_HARDWARE['available_ram_gb']:.1f} GB available)")
    print(f"   â€¢ RAM per job: ~{_HARDWARE['ram_per_job_gb']:.1f} GB")
    print(f"   â€¢ Hyperparameter iterations: {HYPERPARAM_SEARCH_ITER}")
    if _HARDWARE['profile'] == "SERVER":
        total_fits = HYPERPARAM_SEARCH_ITER * CV_FOLDS
        print(f"   â€¢ Expected CV fits per model: {total_fits:,} "
              f"({HYPERPARAM_SEARCH_ITER} configs Ã— {CV_FOLDS} folds)")
    
    # GPU detection
    if _GPU['available']:
        print(f"\nðŸŽ® GPU Acceleration: ENABLED")
        if 'gpu_name' in _GPU and _GPU['gpu_name'] != 'N/A':
            print(f"   â€¢ GPU Device: {_GPU['gpu_name']}")
        print(f"   â€¢ XGBoost: {XGBOOST_TREE_METHOD}")
        print(f"   â€¢ LightGBM: {LIGHTGBM_DEVICE}")
        print(f"   â€¢ CatBoost: {CATBOOST_TASK_TYPE}")
        print(f"   â€¢ Expected speedup: 2-6x on gradient boosting models")
    else:
        print(f"\nðŸŽ® GPU Acceleration: Not Available (CPU only)")
    
    print(f"\nðŸ“Š Missing Data Handling:")
    print(f"   â€¢ Columns with >{MAX_MISSING_DATA:.0%} missing â†’ "
          f"AUTOMATICALLY DROPPED")
    print(f"   â€¢ <{LOW_MISSING_THRESHOLD:.0%} missing â†’ "
          f"Simple imputation (median/mode)")
    print(f"   â€¢ {LOW_MISSING_THRESHOLD:.0%}-{MEDIUM_MISSING_THRESHOLD:.0%} "
          f"missing â†’ KNN imputation")
    print(f"   â€¢ {MEDIUM_MISSING_THRESHOLD:.0%}-{HIGH_MISSING_THRESHOLD:.0%} "
          f"missing â†’ Iterative imputation (MICE)")
    
    print(f"\nðŸ§¹ Automatic Cleaning (no prompts):")
    print(f"   â€¢ Date columns: {'REMOVE' if REMOVE_DATE_COLUMNS else 'KEEP'}")
    print(f"   â€¢ Duplicate rows: "
          f"{'REMOVE' if REMOVE_DUPLICATE_ROWS else 'KEEP'}")
    print(f"   â€¢ Low variance ({LOW_VARIANCE_THRESHOLD:.0%}): "
          f"{'REMOVE' if REMOVE_LOW_VARIANCE_COLS else 'KEEP'}")
    print(f"   â€¢ High cardinality ({HIGH_CARDINALITY_THRESHOLD:.0%}): REMOVE")
    
    print(f"\nâœ… Configuration loaded successfully!")
    print(f"{SEPARATOR_CHAR * SEPARATOR_WIDTH}")


# Validate configuration on import
if __name__ != '__main__':
    validate_config()
