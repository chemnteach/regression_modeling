# %% [markdown]
# # RAPID: Regression Analysis Pipeline with Intelligent Data Preprocessing
# 
# **Version:** 2.0  
# **Last Updated:** 2025-12-01  
# **PEP 8 Compliant** | **Type Hints** | **Professional Logging** | **GPU Accelerated**
# 
# ---
# 
# ## 📋 Overview
# 
# This notebook provides a comprehensive, production-ready workflow for preparing data for machine learning and feature reduction. It follows Python best practices including PEP 8 style guidelines, comprehensive logging, and modular configuration.
# 
# ### ⚡ New in Version 2.0
# 
# - **Auto-Adaptive Parallelization**: Automatically detects hardware (laptop/workstation/server) and optimizes parallel processing
# - **GPU Acceleration**: 2-6x speedup on XGBoost, LightGBM, and CatBoost with automatic CUDA detection
# - **Centralized Configuration**: All settings managed in `config.py` with validation
# - **Comprehensive Logging**: Dual-handler system (file + console) with full audit trails
# 
# ---
# 
# ## 🎯 Workflow Steps
# 
# 1. **Environment Setup** → Install required packages and configure proxy
# 2. **Import Libraries** → Load all necessary Python modules with type hints
# 3. **Configure Logging** → Set up file and console logging handlers
# 4. **Display Configuration** → Show hardware profile, GPU status, and preprocessing parameters
# 5. **Load Data** → Interactive CSV file selection with encoding fallback
# 6. **Quality Control** → Remove columns exceeding missing data thresholds
# 7. **Data Exploration** → Comprehensive dataset analysis and reporting
# 8. **Missing Data Imputation** → Intelligent strategy-based imputation
# 9. **String Preprocessing** → Categorical encoding and cleaning
# 10. **Feature Selection** → Automated feature importance and reduction
# 11. **Model Training** → Multiple regression algorithms with GPU acceleration and parallel CV
# 12. **Results Export** → Generate Excel reports and save artifacts
# 
# ---
# 
# ## 💡 Usage Tips
# 
# - **Execute cells sequentially** from top to bottom
# - **Check configuration** display for hardware detection and GPU status
# - **Check logs** in the `logs/` directory for detailed execution history
# - **Modify settings** in `config.py` rather than hardcoding values
# - **Review outputs** after each major step before proceeding
# - **Save checkpoints** by exporting data at intermediate stages
# 
# ---
# 
# ## 📂 Project Structure
# 
# ```
# regression_modeling/
# ├── Feature Reduction.ipynb    # Main pipeline (this file)
# ├── config.py                  # Configuration constants & hardware detection
# ├── excel_reporter.py          # Report generation utilities
# ├── logs/                      # Execution logs (timestamped)
# ├── data/                      # Output datasets
# └── figures/                   # Generated plots
# ```
# 
# ---
# 
# ## 🚀 Performance Features
# 
# - **Hardware-Adaptive**: 2-28 parallel jobs based on CPU cores and RAM
# - **GPU Acceleration**: Automatic CUDA detection for gradient boosting models
# - **Smart Iteration Scaling**: 10-100 hyperparameter iterations based on hardware profile
# - **Memory-Aware**: Prevents OOM errors with intelligent job scheduling
# 

# %% [markdown]
# 

# %% [markdown]
# ## 📦 Step 0.1: Install Necessary Packages
# 
# This cell handles package installation with proxy support. Configure your proxy settings in a `.env` file or set `PROXY_URL` directly if behind a corporate firewall.

# %%
# =============================================================================
# PROXY CONFIGURATION
# =============================================================================
# Optional: Set this if you're behind a corporate proxy
# Better: Configure HTTP_PROXY in your .env file or system environment
PROXY_URL = None  # Example: "http://proxy-dmz.intel.com:912"

import os
import sys
import subprocess

# Auto-detect proxy: manual setting > .env file > system environment > no proxy
proxy = PROXY_URL or os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')

if proxy:
    print(f"📡 Using proxy: {proxy}")
    proxy_args = ["--proxy", proxy]
else:
    print("🌐 No proxy configured - direct connection")
    proxy_args = []

# =============================================================================
# PACKAGE INSTALLATION
# =============================================================================
print("\n⏳ Upgrading pip...")
subprocess.check_call([sys.executable, "-m", "pip", "install"] + proxy_args + ["--upgrade", "pip"])

print("\n⏳ Installing required packages...")
packages = ["matplotlib", "pandas", "scikit-learn", "seaborn", "xgboost", "lightgbm", 
            "shap", "catboost", "numpy", "scipy", "ipywidgets", "typing"]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + proxy_args + packages)

print("\n✅ All packages installed successfully!")

# %% [markdown]
# ## 📚 Step 0.2: Import Required Libraries
# 
# Import all necessary Python libraries for data processing, machine learning, and visualization. All imports follow PEP 8 conventions with proper grouping.

# %%
# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# THIRD-PARTY DATA PROCESSING IMPORTS
# =============================================================================
import joblib
import numpy as np
import pandas as pd

# =============================================================================
# VISUALIZATION LIBRARIES
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# SCIKIT-LEARN: PREPROCESSING AND IMPUTATION
# =============================================================================
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

# =============================================================================
# SCIKIT-LEARN: MODEL SELECTION AND EVALUATION
# =============================================================================
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split
)
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =============================================================================
# SCIKIT-LEARN: PREPROCESSING AND FEATURE SELECTION
# =============================================================================
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler
)
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression
)

# =============================================================================
# SCIKIT-LEARN: REGRESSION MODELS
# =============================================================================
from sklearn.ensemble import (
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# =============================================================================
# GRADIENT BOOSTING LIBRARIES
# =============================================================================
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# =============================================================================
# SUPPRESS WARNINGS FOR CLEANER OUTPUT
# =============================================================================
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")
print(f"📊 Pandas version: {pd.__version__}")
print(f"🔢 NumPy version: {np.__version__}")
print(f"🤖 Scikit-learn available")


# %% [markdown]
# ## 🔧 Step 0.3: Configure Logging System
# 
# Set up comprehensive logging with both file and console handlers. Logs are saved to `logs/feature_reduction_TIMESTAMP.log` for audit trails and debugging.

# %%
"""
Configure logging system for the RAPID pipeline.

This cell sets up comprehensive logging with both file and console handlers,
following Python logging best practices.
"""
import logging
import os
from datetime import datetime

# Import configuration constants
import config

# Create log directory if it doesn't exist
os.makedirs(config.LOG_DIR, exist_ok=True)

# Generate timestamp for log file
LOG_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILENAME = os.path.join(
    config.LOG_DIR,
    f'feature_reduction_{LOG_TIMESTAMP}.log'
)

# Configure root logger
logger = logging.getLogger('FeatureReduction')
logger.setLevel(getattr(logging, config.FILE_LOG_LEVEL))

# Remove existing handlers to avoid duplicates
logger.handlers.clear()

# File handler - detailed logging
file_handler = logging.FileHandler(LOG_FILENAME, encoding='utf-8')
file_handler.setLevel(getattr(logging, config.FILE_LOG_LEVEL))
file_formatter = logging.Formatter(
    config.LOG_FORMAT_FILE,
    datefmt=config.LOG_DATE_FORMAT
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler - user-friendly output
console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, config.CONSOLE_LOG_LEVEL))
console_formatter = logging.Formatter(config.LOG_FORMAT_CONSOLE)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Log initialization
logger.info("=" * config.SEPARATOR_WIDTH)
logger.info("RAPID - Regression Analysis Pipeline Initialized")
logger.info("=" * config.SEPARATOR_WIDTH)
logger.info(f"Log file: {LOG_FILENAME}")
logger.info(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * config.SEPARATOR_WIDTH)

# Log configuration settings
logger.debug("Configuration loaded:")
logger.debug(f"  MAX_MISSING_DATA: {config.MAX_MISSING_DATA}")
logger.debug(f"  LOW_MISSING_THRESHOLD: {config.LOW_MISSING_THRESHOLD}")
logger.debug(f"  MEDIUM_MISSING_THRESHOLD: {config.MEDIUM_MISSING_THRESHOLD}")
logger.debug(f"  HIGH_MISSING_THRESHOLD: {config.HIGH_MISSING_THRESHOLD}")
logger.debug(
    f"  ONE_HOT_ENCODING_MAX_CATEGORIES: "
    f"{config.ONE_HOT_ENCODING_MAX_CATEGORIES}"
)
logger.debug(
    f"  LABEL_ENCODING_MAX_CATEGORIES: "
    f"{config.LABEL_ENCODING_MAX_CATEGORIES}"
)

print(f"\n✅ Logging configured successfully!")
print(f"📝 Log file: {LOG_FILENAME}")


# %%
# =============================================================================
# Import Configuration Constants into Global Namespace
# =============================================================================
"""
Import all configuration constants from config.py into the global namespace
for backward compatibility with existing code.

This allows code to reference constants directly (e.g., MAX_MISSING_DATA)
rather than requiring the config prefix (e.g., config.MAX_MISSING_DATA).
"""

# Missing Data Thresholds
MAX_MISSING_DATA = config.MAX_MISSING_DATA
LOW_MISSING_THRESHOLD = config.LOW_MISSING_THRESHOLD
MEDIUM_MISSING_THRESHOLD = config.MEDIUM_MISSING_THRESHOLD
HIGH_MISSING_THRESHOLD = config.HIGH_MISSING_THRESHOLD

# Categorical Encoding Thresholds
ONE_HOT_ENCODING_MAX_CATEGORIES = config.ONE_HOT_ENCODING_MAX_CATEGORIES
LABEL_ENCODING_MAX_CATEGORIES = config.LABEL_ENCODING_MAX_CATEGORIES
HIGH_CARDINALITY_THRESHOLD = config.HIGH_CARDINALITY_THRESHOLD

# Feature Selection Thresholds
LOW_VARIANCE_THRESHOLD = config.LOW_VARIANCE_THRESHOLD
CORRELATION_THRESHOLD = config.CORRELATION_THRESHOLD

# Data Cleaning Flags
REMOVE_DATE_COLUMNS = config.REMOVE_DATE_COLUMNS
REMOVE_DUPLICATE_ROWS = config.REMOVE_DUPLICATE_ROWS
REMOVE_LOW_VARIANCE_COLS = config.REMOVE_LOW_VARIANCE_COLS

# Machine Learning Parameters
RANDOM_STATE = config.RANDOM_STATE
TEST_SIZE = config.TEST_SIZE
CV_FOLDS = config.CV_FOLDS
MAX_FEATURES = config.MAX_FEATURES

# Parallel Processing & GPU Configuration
N_JOBS = config.N_JOBS
HYPERPARAM_SEARCH_ITER = config.HYPERPARAM_SEARCH_ITER
USE_GPU = config.USE_GPU
XGBOOST_TREE_METHOD = config.XGBOOST_TREE_METHOD
LIGHTGBM_DEVICE = config.LIGHTGBM_DEVICE
CATBOOST_TASK_TYPE = config.CATBOOST_TASK_TYPE

# Display Settings
SEPARATOR_WIDTH = config.SEPARATOR_WIDTH
SEPARATOR_CHAR = config.SEPARATOR_CHAR

# String Processing (if defined in config - these may need to be added)
CONVERT_STRING_NULLS = True
STRING_NULL_VALUES = [
    'N/A', 'NA', 'n/a', 'na',
    'NULL', 'null', 'Null',
    'None', 'none',
    'NaN', 'nan',
    '', ' ',
    'missing', 'Missing', 'MISSING',
    '--', '---',
    'unknown', 'Unknown', 'UNKNOWN'
]

print("✅ Configuration constants imported into global namespace")
print(f"   MAX_MISSING_DATA: {MAX_MISSING_DATA}")
print(f"   RANDOM_STATE: {RANDOM_STATE}")
print(f"   N_JOBS (parallel cores): {N_JOBS}")
print(f"   GPU Acceleration: {'ENABLED' if USE_GPU else 'DISABLED'}")


# %% [markdown]
# ## ⚙️ Step 0.4: Display Configuration Settings
# 
# Review all preprocessing parameters and thresholds loaded from `config.py`. Modify values in the config file to customize behavior.

# %%
# Display configuration to user
config.print_config()

# Log configuration display
logger.info("Configuration displayed to user")


# %%
# Execute the combined load and select workflow
base_data, dependent_var, column_info = load_and_select_target()

# Stacking is always enabled
RUN_STACKING_ANALYSIS = True

# Log data loading
logger.info(f"Data loaded: {base_data.shape[0]:,} rows × {base_data.shape[1]} columns")
logger.info(f"Target variable: {dependent_var}")

# %% [markdown]
# ## 🔍 Step 2: Missing Data Quality Control
# 
# Analyze all columns for missing data and automatically remove those exceeding the `MAX_MISSING_DATA` threshold (default: 40%). This prevents downstream issues with severely incomplete features.

# %%
"""
Missing Data Quality Control Module

This module analyzes and automatically removes columns from a pandas DataFrame that exceed
a specified threshold of missing data. It's designed to clean datasets before feature
engineering and machine learning model development.

Dependencies:
    - pandas (for DataFrame operations)
    - MAX_MISSING_DATA: Global variable defining the threshold (default: 0.5 = 50%)
    - base_data: Global pandas DataFrame containing the dataset to analyze

Usage:
    Ensure MAX_MISSING_DATA and base_data are defined before running this code block.
    The code will automatically remove columns exceeding the missing data threshold.

Example:
    MAX_MISSING_DATA = 0.5  # Remove columns with 50%+ missing data
    base_data = pd.read_csv('your_dataset.csv')
    # Run this code block to clean the dataset
"""

# ==============================================================================
# MISSING DATA ANALYSIS AND CLEANUP
# ==============================================================================

# Check for columns with MAX_MISSING_DATA or more missing data and remove them
if 'base_data' in locals():
    """
    Safety check to ensure the base_data DataFrame exists in the local scope
    before attempting to analyze it. Prevents NameError exceptions.
    """
    
    # -------------------------------------------------------------------------
    # STEP 1: Calculate missing data statistics
    # -------------------------------------------------------------------------
    
    # Count missing values (NaN, None, null) for each column
    missing_data = base_data.isnull().sum()
    
    # Get total number of rows for percentage calculations
    total_rows = len(base_data)
    
    # Store original column count for reporting
    original_columns = len(base_data.columns)
    
    # -------------------------------------------------------------------------
    # STEP 2: Calculate missing data percentages
    # -------------------------------------------------------------------------
    
    # Convert raw missing counts to percentages for each column
    # Formula: (missing_count / total_rows) * 100
    missing_percentages = (missing_data / total_rows) * 100
    
    # Convert the threshold from decimal (0.5) to percentage (50.0) for display
    max_missing_threshold = MAX_MISSING_DATA * 100
    
    # -------------------------------------------------------------------------
    # STEP 3: Identify columns exceeding the threshold
    # -------------------------------------------------------------------------
    
    # Find columns with missing data >= threshold
    # Uses >= to include columns with exactly the threshold amount of missing data
    high_missing_cols = missing_percentages[missing_percentages >= max_missing_threshold]
    
    # -------------------------------------------------------------------------
    # STEP 4: Display analysis results
    # -------------------------------------------------------------------------
    
    print(f"📊 Missing Data Analysis:")
    print(f"Original columns: {original_columns}")
    print(f"Columns with >={max_missing_threshold}% missing data: {len(high_missing_cols)}")
    
    # -------------------------------------------------------------------------
    # STEP 5: Process columns for removal (if any found)
    # -------------------------------------------------------------------------
    
    if len(high_missing_cols) > 0:
        """
        If columns exceed the missing data threshold, remove them from the dataset
        """
        
        # Display which columns will be removed, sorted by missing percentage (highest first)
        print(f"\n🗑️ Columns being removed:")
        for col, missing_pct in high_missing_cols.sort_values(ascending=False).items():
            print(f"  {col}: {missing_pct:.1f}% missing")
        
        # Extract column names from the pandas Series index
        columns_to_drop = high_missing_cols.index.tolist()
        
        # Remove the problematic columns from the DataFrame
        # Note: This modifies the global base_data variable
        base_data = base_data.drop(columns=columns_to_drop)
        
        # Report the results of the cleanup operation
        print(f"\n✅ Removed {len(columns_to_drop)} columns")
        print(f"Remaining columns: {len(base_data.columns)}")
        print(f"Data shape after cleanup: {base_data.shape}")
        
    else:
        """
        If no columns exceed the threshold, report successful data quality
        """
        print(f"\n✅ No columns have {max_missing_threshold}% or more missing data!")
        print(f"Data shape: {base_data.shape}")
        
else:
    """
    Error handling: Inform user that the required DataFrame doesn't exist
    """
    print("❌ Please load your data first using: base_data = select_csv_file()")

# %% [markdown]
# ## 🧹 Step 3: Automated Data Cleaning
# 
# Execute comprehensive data cleaning including:
# - Duplicate row removal
# - Low variance feature elimination
# - Date column handling
# - High cardinality ID detection
# 
# All actions are logged and configurable via `config.py`.

# %%
"""
Comprehensive Data Exploration Script
=====================================

This script performs a thorough exploration and analysis of a pandas DataFrame
to understand data structure, quality, and potential issues before analysis.

Requirements:
    - pandas library
    - A DataFrame named 'base_data' in the local namespace

"""


def preprocess_string_nulls(
    df: pd.DataFrame, 
    string_null_values: List[str]
) -> Tuple[pd.DataFrame, int, Dict[str, int]]:
    """
    Convert string representations of null values to pandas NaN.
    
    Parameters:
        df (pd.DataFrame): DataFrame to process
        string_null_values (list): List of string values to treat as nulls
    
    Returns:
        tuple: (processed_df, total_conversions, conversion_summary_dict)
    """
    total_conversions = 0
    conversion_summary = {}
    
    # Process only object (string) columns
    object_columns = df.select_dtypes(include=['object']).columns
    
    for col in object_columns:
        col_data = df[col]
        mask = col_data.isin(string_null_values)
        col_conversions = mask.sum()
        
        if col_conversions > 0:
            df[col] = col_data.replace(string_null_values, np.nan)
            conversion_summary[col] = col_conversions
            total_conversions += col_conversions
    
def analyze_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:


def analyze_dataset_overview(df):
    """
    Provide high-level dataset overview statistics.
    
    Parameters:
        df (pd.DataFrame): DataFrame to analyze
    
    Returns:
        dict: Overview statistics including shape, memory, duplicates
    """
    overview = {
        'shape': df.shape,
        'rows': df.shape[0],
        'cols': df.shape[1],
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicate_count': df.duplicated().sum(),
        'duplicate_pct': (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0
    }
def analyze_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:


def analyze_data_types(df):
    """
    Categorize and analyze columns by data type.
    
    Parameters:
        df (pd.DataFrame): DataFrame to analyze
    
    Returns:
        dict: Dictionary with column lists by type (float_cols, int_cols, object_cols, etc.)
    """
    return {
        'float_cols': df.select_dtypes(include=['float64', 'float32']).columns.tolist(),
        'int_cols': df.select_dtypes(include=['int64', 'int32', 'int16', 'int8']).columns.tolist(),
        'object_cols': df.select_dtypes(include=['object']).columns.tolist(),
        'datetime_cols': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'bool_cols': df.select_dtypes(include=['bool']).columns.tolist()
def analyze_missing_values(df: pd.DataFrame) -> Tuple[pd.Series, float]:


def analyze_missing_values(df):
    """
    Analyze missing value patterns across the dataset.
    
    Parameters:
        df (pd.DataFrame): DataFrame to analyze
    
    Returns:
        tuple: (missing_data_series, total_missing_pct)
    """
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    total_missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
    
def analyze_object_columns(df: pd.DataFrame, object_cols: List[str]) -> Dict[str, Dict[str, Any]]:


def analyze_object_columns(df, object_cols):
    """
    Analyze object (string) columns for uniqueness and common values.
    
    Parameters:
        df (pd.DataFrame): DataFrame to analyze
        object_cols (list): List of object column names
    
    Returns:
        dict: Analysis results for each object column
    """
    object_analysis = {}
    
    for col in object_cols:
        try:
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            
            if total_count > 0:
                uniqueness_ratio = unique_count / total_count
                top_values = df[col].value_counts().head(3)
                
                object_analysis[col] = {
                    'unique_count': unique_count,
                    'uniqueness_ratio': uniqueness_ratio,
                    'top_values': top_values.to_dict()
                }
        except Exception as e:
            object_analysis[col] = {'error': str(e)}
    
def detect_mixed_type_columns(df: pd.DataFrame, object_cols: List[str]) -> List[str]:


def detect_mixed_type_columns(df, object_cols):
    """
    Detect columns that contain mixed numeric and text values.
    
    Parameters:
        df (pd.DataFrame): DataFrame to analyze
        object_cols (list): List of object column names to check
    
    Returns:
        list: List of column names with mixed types
    """
    mixed_cols = []
    
    for col in object_cols:
        try:
            all_values = df[col].dropna()
            if len(all_values) > 0:
                # Count numeric-convertible values
                numeric_count = sum(1 for val in all_values if _is_numeric_value(val))
                
                # Mixed if some but not all are numeric
                if 0 < numeric_count < len(all_values):
                    mixed_cols.append(col)
        except Exception:
            pass
    
def _is_numeric_value(val: Any) -> bool:


def _is_numeric_value(val):
    """Helper function to check if a value can be converted to float."""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
def detect_data_quality_issues(
    df: pd.DataFrame, 
    object_cols: List[str], 
    numeric_cols: List[str], 
    high_card_threshold: float
) -> Dict[str, List]:
    """
    Detect potential data quality issues.
    
    Parameters:
        df (pd.DataFrame): DataFrame to analyze
        object_cols (list): Object column names
        numeric_cols (list): Numeric column names
        high_card_threshold (float): Threshold for high cardinality detection
    
    Returns:
        dict: Dictionary with 'high_cardinality' and 'low_variance' column lists
    """
    issues = {
        'high_cardinality_cols': [],
        'low_variance_cols': []
    }
    
    # Check for high cardinality in object columns
    for col in object_cols:
        try:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > high_card_threshold:
                issues['high_cardinality_cols'].append((col, unique_ratio))
        except Exception:
            pass
    
    # Check for constant values in numeric columns
    for col in numeric_cols:
        try:
            if df[col].nunique() == 1:
                issues['low_variance_cols'].append(col)
        except Exception:
def print_dataset_overview(overview: Dict[str, Any]) -> None:
    
    return issues


def print_dataset_overview(overview):
    """Print dataset overview section."""
    print(f"\n📊 Dataset Overview:")
    print(f"Shape: {overview['rows']:,} rows × {overview['cols']:,} columns")
    print(f"Memory usage: {overview['memory_mb']:.2f} MB")
    
    if overview['duplicate_count'] > 0:
        print(f"⚠️  Duplicate rows found: {overview['duplicate_count']:,} ({overview['duplicate_pct']:.1f}%)")
def print_data_types(type_info: Dict[str, List[str]]) -> None:
    else:
        print("✅ No duplicate rows found")


def print_data_types(type_info):
    """Print data types analysis section."""
    print("🔍 Data Types Analysis:")
    dtype_counts = {}
    
    if type_info['float_cols']:
        dtype_counts['float64'] = len(type_info['float_cols'])
    if type_info['int_cols']:
        dtype_counts['int64'] = len(type_info['int_cols'])
    if type_info['object_cols']:
        dtype_counts['object'] = len(type_info['object_cols'])
    if type_info['datetime_cols']:
        dtype_counts['datetime64'] = len(type_info['datetime_cols'])
    if type_info['bool_cols']:
        dtype_counts['bool'] = len(type_info['bool_cols'])
def print_missing_values(missing_data: pd.Series, df: pd.DataFrame) -> None:
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")


def print_missing_values(missing_data, df):
    """Print missing values analysis section."""
    print("❗ Missing Values Analysis:")
    
    if len(missing_data) > 0:
        print(f"Columns with missing values: {len(missing_data)}")
        print("\nTop 20 columns with most missing values:")
        for col, missing_count in missing_data.head(20).items():
            missing_pct = (missing_count / len(df)) * 100
def print_columns_by_type(type_info: Dict[str, List[str]]) -> None:
    else:
        print("✅ No missing values found!")


def print_columns_by_type(type_info):
    """Print categorized column lists."""
    print("📋 Columns by Data Type:")
    
    # Print float columns
    print(f"\n🔢 Float columns ({len(type_info['float_cols'])}):")
    if type_info['float_cols']:
        for i, col in enumerate(type_info['float_cols']):
            if i % 3 == 0:
                print()
            print(f"  {col:<30}", end="")
        print()
    
    # Print integer columns
    print(f"\n🔢 Integer columns ({len(type_info['int_cols'])}):")
    if type_info['int_cols']:
        for i, col in enumerate(type_info['int_cols']):
            if i % 3 == 0:
                print()
            print(f"  {col:<30}", end="")
        print()
    
    # Print object columns
    print(f"\n📝 Object/String columns ({len(type_info['object_cols'])}):")
    if type_info['object_cols']:
        for i, col in enumerate(type_info['object_cols']):
            if i % 3 == 0:
                print()
            print(f"  {col:<30}", end="")
        print()
    
    # Print datetime columns
    if type_info['datetime_cols']:
        print(f"\n📅 DateTime columns ({len(type_info['datetime_cols'])}):")
        for i, col in enumerate(type_info['datetime_cols']):
            if i % 3 == 0:
                print()
            print(f"  {col:<30}", end="")
        print()
    
    # Print boolean columns
    if type_info['bool_cols']:
        print(f"\n✅ Boolean columns ({len(type_info['bool_cols'])}):")
        for i, col in enumerate(type_info['bool_cols']):
            if i % 3 == 0:
                print()
            print(f"  {col:<30}", end="")
        print()


def print_numeric_statistics(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    """Print numeric statistics section."""
    if numeric_cols:
        print("📈 Quick Numeric Statistics:")
        print(f"Found {len(numeric_cols)} numeric columns")
        
        sample_cols = numeric_cols[:5]
        if sample_cols:
            print(f"\nSample statistics (first {len(sample_cols)} numeric columns):")
            try:
                stats = df[sample_cols].describe()
                print(stats.round(2))
            except Exception as e:
                print(f"Error calculating statistics: {e}")
                
            if len(numeric_cols) > 5:
                print(f"\n... and {len(numeric_cols) - 5} more numeric columns")


def print_object_analysis(object_analysis: Dict[str, Dict[str, Any]], df: pd.DataFrame) -> None:
    """Print object column analysis section."""
    print("📝 Object Column Analysis:")
    print(f"Analyzing all {len(object_analysis)} object columns...")
    
    for col, analysis in object_analysis.items():
        if 'error' in analysis:
            print(f"  {col}: Error analyzing column - {analysis['error']}")
        else:
            print(f"  {col}: {analysis['unique_count']:,} unique values "
                  f"({analysis['uniqueness_ratio']:.1%} unique)")
            
            # Show top values
            top_items = list(analysis['top_values'].items())[:3]
            formatted_values = [f'{val} ({count})' for val, count in top_items]
            print(f"    Top values: {', '.join(formatted_values)}")


def print_mixed_columns(mixed_cols: List[str], df: pd.DataFrame) -> None:
    """Print mixed data type detection section."""
    print("⚠️  Columns with Mixed/Problematic Data:")
    
    if mixed_cols:
        for col in mixed_cols:
            unique_vals = df[col].nunique()
            all_values = df[col].dropna()
            numeric_count = sum(1 for val in all_values if _is_numeric_value(val))
            print(f"  {col}: {unique_vals:,} unique values "
                  f"(mixed numeric/text - {numeric_count:,}/{len(all_values):,} numeric)")
    else:
        print("✅ No obviously mixed-type columns detected")


def print_quality_issues(issues: Dict[str, List], df: pd.DataFrame) -> None:
    """Print data quality issues section."""
    print("🔍 Potential Data Quality Issues:")
    issues_found = False
    
    # High cardinality columns
    if issues['high_cardinality_cols']:
        issues_found = True
        print("  High cardinality columns (may be IDs or need special handling):")
        for col, ratio in issues['high_cardinality_cols']:
            print(f"    {col}: {ratio:.1%} unique values")
    
    # Low variance columns
    if issues['low_variance_cols']:
        issues_found = True
        print("  Constant value columns (no variance):")
        for col in issues['low_variance_cols']:
            try:
                constant_value = df[col].iloc[0]
                print(f"    {col}: constant value = {constant_value}")
            except Exception as e:
                print(f"    {col}: Error retrieving constant value - {e}")
    
    if not issues_found:
        print("✅ No obvious data quality issues detected")


def comprehensive_data_exploration() -> None:
    """
    Performs comprehensive data exploration on the 'base_data' DataFrame.
    
    This function analyzes data structure, types, missing values, duplicates,
    and potential quality issues. Results are stored in a global 'column_info'
    dictionary for subsequent analysis.
    
    Features:
        - Dataset overview (shape, memory usage, duplicates)
        - Data type analysis and categorization
        - Missing value detection and quantification
        - Numeric statistics for sample columns
        - Object column uniqueness analysis
        - Mixed data type detection
        - Data quality issue identification
    
    Global Variables Created:
        column_info (dict): Comprehensive metadata about the dataset
    
    Returns:
        None: Prints analysis results and creates global column_info variable
        
    Raises:
        NameError: If 'base_data' DataFrame is not found in local namespace
        AttributeError: If 'base_data' is not a valid pandas DataFrame
    """
    
    global base_data, column_info
    if 'base_data' not in globals():
    # Validate DataFrame existence
    if 'base_data' not in globals():
        print("❌ base_data DataFrame not found. Please load your data first.")
        return
    
    if not hasattr(base_data, 'shape'):
        print("❌ base_data is not a valid pandas DataFrame. Please load your data first.")
        return
    
    print("=" * 80)
    print("📊 COMPREHENSIVE DATA EXPLORATION")
    print("=" * 80)
    
    # String null preprocessing
    if 'CONVERT_STRING_NULLS' in globals() and CONVERT_STRING_NULLS:
        print(f"\n🔧 Preprocessing: Converting string null values to pandas null...")
        
        if 'STRING_NULL_VALUES' not in globals():
            print("❌ STRING_NULL_VALUES constant not found. Please run the constants section first.")
            return
        
        print(f"✅ Using {len(STRING_NULL_VALUES)} configured string null patterns")
        
        base_data, total_conv, conv_summary = preprocess_string_nulls(base_data, STRING_NULL_VALUES)
        
        if total_conv > 0:
            print(f"✅ Converted {total_conv:,} string null values to pandas NaN in {len(conv_summary)} columns")
            if len(conv_summary) <= 10:
                for col, count in conv_summary.items():
                print(f"   Top 5 columns with most conversions:")
            else:
                print(f"   Top 5 columns with most conversions:")
                sorted_conv = sorted(conv_summary.items(), key=lambda x: x[1], reverse=True)
                for col, count in sorted_conv[:5]:
                    print(f"   {col}: {count:,} conversions")
                print(f"   ... and {len(conv_summary) - 5} more columns")
        else:
            print("✅ No string null values found to convert")
    else:
        print(f"\n🔧 String null conversion disabled (CONVERT_STRING_NULLS = {CONVERT_STRING_NULLS if 'CONVERT_STRING_NULLS' in globals() else 'not set'})")
    
    # Dataset overview
    overview = analyze_dataset_overview(base_data)
    # Dataset overview
    overview = analyze_dataset_overview(base_data)
    print_dataset_overview(overview)
    # Data types analysis
    type_info = analyze_data_types(base_data)
    # Data types analysis
    type_info = analyze_data_types(base_data)
    print_data_types(type_info)
    # Missing values analysis
    missing_data, total_missing_pct = analyze_missing_values(base_data)
    # Missing values analysis
    missing_data, total_missing_pct = analyze_missing_values(base_data)
    print_missing_values(missing_data, base_data)
    print("\n" + "=" * 80 + "\n")
    
    # Columns by type
    print_columns_by_type(type_info)
    # Numeric statistics
    numeric_cols = type_info['float_cols'] + type_info['int_cols']
    # Numeric statistics
    numeric_cols = type_info['float_cols'] + type_info['int_cols']
    print_numeric_statistics(base_data, numeric_cols)
    print("\n" + "=" * 80 + "\n")
    
    # Object column analysis
    if type_info['object_cols']:
        object_analysis = analyze_object_columns(base_data, type_info['object_cols'])
        print_object_analysis(object_analysis, base_data)
        print("\n" + "=" * 80 + "\n")
    mixed_cols = detect_mixed_type_columns(base_data, type_info['object_cols'])
    # Mixed type detection
    mixed_cols = detect_mixed_type_columns(base_data, type_info['object_cols'])
    print_mixed_columns(mixed_cols, base_data)
    print("\n" + "=" * 80 + "\n")
    
    # Data quality issues
    issues = detect_data_quality_issues(
        base_data, 
        type_info['object_cols'], 
        numeric_cols,
        HIGH_CARDINALITY_THRESHOLD if 'HIGH_CARDINALITY_THRESHOLD' in globals() else 0.8
    )
    print_quality_issues(issues, base_data)
    print("\n" + "=" * 80 + "\n")
    
    # Compile results
    column_info = {
        'shape': base_data.shape,
        'memory_mb': overview['memory_mb'],
        'duplicates': overview['duplicate_count'],
        'float_cols': type_info['float_cols'],
        'int_cols': type_info['int_cols'],
        'object_cols': type_info['object_cols'],
        'datetime_cols': type_info['datetime_cols'],
        'bool_cols': type_info['bool_cols'],
        'mixed_cols': mixed_cols,
        'high_cardinality_cols': [col for col, _ in issues['high_cardinality_cols']],
        'low_variance_cols': issues['low_variance_cols'],
        'missing_data': missing_data,
        'total_missing_pct': total_missing_pct
    }
    
    # Summary
    print("🔍 Data exploration complete!")
    print(f"Summary: {base_data.shape[0]:,} rows, {base_data.shape[1]:,} cols, "
          f"{column_info['total_missing_pct']:.1f}% missing data")
    print("Results stored in 'column_info' variable for further analysis.")
    print("=" * 80)
# ============================================================================

# ============================================================================
# EXECUTION BLOCK
# ============================================================================

print("✅ Comprehensive data exploration function loaded!")

# Check if base_data exists and provide status
if 'base_data' in globals():
    if hasattr(base_data, 'shape'):
        print(f"✅ base_data found: {base_data.shape[0]:,} rows × {base_data.shape[1]:,} columns")
        print("🚀 Running comprehensive data exploration...")
        comprehensive_data_exploration()
    else:
        print("⚠️ base_data exists but is not a valid DataFrame")
        print("📊 To run analysis manually: comprehensive_data_exploration()")
else:
    print("⚠️ base_data not found. Please load your data first.")
    print("📊 To run analysis manually: comprehensive_data_exploration()")


print(f"\n🔧 Current configuration:")
    print(f"   - HIGH_CARDINALITY_THRESHOLD = {HIGH_CARDINALITY_THRESHOLD}")    print(f"   - REMOVE_DUPLICATE_ROWS = {REMOVE_DUPLICATE_ROWS}")

if 'HIGH_CARDINALITY_THRESHOLD' in globals():
if 'REMOVE_DUPLICATE_ROWS' in globals():if 'REMOVE_DUPLICATE_ROWS' in globals():

    print(f"   - HIGH_CARDINALITY_THRESHOLD = {HIGH_CARDINALITY_THRESHOLD}")    print(f"   - REMOVE_DUPLICATE_ROWS = {REMOVE_DUPLICATE_ROWS}")

# %% [markdown]
# ## Intelligent Imputation Function
# 
# Define the intelligent imputation strategy function that will be used in the strategic workflow below.

# %%
def calculate_missing_percentage(df: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
    """
    Calculate missing data percentage for specified columns.
    
    Parameters:
        df (pd.DataFrame): DataFrame to analyze
        columns (list): Column names to calculate missing percentages for
    
    Returns:
        dict: Column name -> missing percentage mapping
    """
    missing_pct = {}
    for col in columns:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct[col] = missing_count / len(df)
    return missing_pct


def categorize_by_missing_severity(
    missing_cols: pd.Series, 
    df: pd.DataFrame, 
    low_threshold: float, 
    medium_threshold: float, 
    high_threshold: float
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Categorize columns by missing data severity based on configured thresholds.
    
    Parameters:
        missing_cols (pd.Series): Series with column names and missing counts
        df (pd.DataFrame): Source DataFrame for calculating percentages
        low_threshold (float): Threshold for low missing data
        medium_threshold (float): Threshold for medium missing data
        high_threshold (float): Threshold for high missing data
    
    Returns:
        dict: Categorized columns by severity level
    """
    strategies = {
        'low_missing': [],      # Simple imputation
        'medium_missing': [],   # Advanced imputation
        'high_missing': [],     # Consider dropping or advanced methods
        'very_high_missing': [] # Recommend dropping
    }
    
    for col in missing_cols.index:
        missing_pct = missing_cols[col] / len(df)
        
        if missing_pct < low_threshold:
            strategies['low_missing'].append((col, missing_pct))
        elif missing_pct < medium_threshold:
            strategies['medium_missing'].append((col, missing_pct))
        elif missing_pct < high_threshold:
            strategies['high_missing'].append((col, missing_pct))
def apply_simple_imputation(
    df: pd.DataFrame, 
    columns_with_pct: List[Tuple[str, float]], 
    numeric_cols: List[str], 
    categorical_cols: List[str], 
    target_var: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Apply simple imputation (median/mode) for columns with low missing data.
    
    Parameters:
        df (pd.DataFrame): DataFrame to impute
        columns_with_pct (list): List of (column_name, missing_pct) tuples
        numeric_cols (list): List of numeric column names
        categorical_cols (list): List of categorical column names
        target_var (str): Target variable to exclude from imputation
    
    Returns:
        tuple: (imputed_df, methods_used_dict, columns_imputed_list)
    """
    imputed_df = df.copy()
    methods_used = {}
    columns_imputed = []
    
    for col, missing_pct in columns_with_pct:
        if col == target_var:
            continue
            
        if col in numeric_cols:
            median_value = imputed_df[col].median()
            imputed_df[col] = imputed_df[col].fillna(median_value)
            method = f"median ({median_value:.3f})"
            
        elif col in categorical_cols:
            mode_value = imputed_df[col].mode()
            if len(mode_value) > 0:
                imputed_df[col] = imputed_df[col].fillna(mode_value[0])
                method = f"mode ('{mode_value[0]}')"
            else:
                imputed_df[col] = imputed_df[col].fillna('Unknown')
                method = "constant ('Unknown')"
        
        methods_used[col] = method
def apply_advanced_imputation(
    df: pd.DataFrame, 
    columns_with_pct: List[Tuple[str, float]], 
    numeric_cols: List[str], 
    categorical_cols: List[str], 
    missing_data: pd.Series, 
    target_var: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Apply advanced imputation (KNN/iterative) for columns with medium missing data.
    
    Parameters:
        df (pd.DataFrame): DataFrame to impute
        columns_with_pct (list): List of (column_name, missing_pct) tuples
        numeric_cols (list): List of numeric column names
        categorical_cols (list): List of categorical column names
        missing_data (pd.Series): Series with missing counts per column
        target_var (str): Target variable to exclude from imputation
    Returns:
    Returns:
        tuple: (imputed_df, methods_used_dict, columns_imputed_list)
    """
    imputed_df = df.copy()
    methods_used = {}
    columns_imputed = []
    
    # Separate numeric and categorical
    medium_numeric = [col for col, _ in columns_with_pct if col in numeric_cols and col != target_var]
    medium_categorical = [col for col, _ in columns_with_pct if col in categorical_cols and col != target_var]
    
    # KNN Imputation for numeric columns
    if medium_numeric:
        print(f"  🔢 KNN Imputation for {len(medium_numeric)} numeric columns:")
        
        # Select features for KNN (use other numeric columns with low missing)
        knn_features = [col for col in numeric_cols if col not in medium_numeric and missing_data[col] < len(df) * 0.1]
        
        if len(knn_features) >= 2:
            try:
                knn_data = imputed_df[knn_features + medium_numeric].copy()
                knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
                knn_imputed = knn_imputer.fit_transform(knn_data)
                
                for i, col in enumerate(medium_numeric):
                    col_idx = knn_features.index(col) if col in knn_features else len(knn_features) + medium_numeric.index(col)
                    imputed_df[col] = knn_imputed[:, col_idx]
                    methods_used[col] = "KNN (k=5)"
                    columns_imputed.append(col)
                    print(f"    ✅ {col}: KNN imputation")
                    
            except Exception as e:
                print(f"    ⚠️ KNN failed, falling back to median imputation: {e}")
                for col in medium_numeric:
                    median_value = imputed_df[col].median()
                    imputed_df[col] = imputed_df[col].fillna(median_value)
                    methods_used[col] = f"median fallback ({median_value:.3f})"
                    columns_imputed.append(col)
        else:
            print(f"    ⚠️ Insufficient features for KNN, using median")
            for col in medium_numeric:
                median_value = imputed_df[col].median()
                imputed_df[col] = imputed_df[col].fillna(median_value)
                methods_used[col] = f"median ({median_value:.3f})"
                columns_imputed.append(col)
    
    # Frequent value imputation for categorical
    if medium_categorical:
        print(f"  📝 Frequent Value Imputation for {len(medium_categorical)} categorical columns:")
        for col in medium_categorical:
            value_counts = imputed_df[col].value_counts()
            if len(value_counts) > 0:
                most_frequent = value_counts.index[0]
                imputed_df[col] = imputed_df[col].fillna(most_frequent)
def apply_high_missing_imputation(
    df: pd.DataFrame, 
    columns_with_pct: List[Tuple[str, float]], 
    numeric_cols: List[str], 
    categorical_cols: List[str], 
    missing_data: pd.Series, 
    target_var: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Apply advanced methods for columns with high missing data (20-40%).
    
    Parameters:
        df (pd.DataFrame): DataFrame to impute
        columns_with_pct (list): List of (column_name, missing_pct) tuples
        numeric_cols (list): List of numeric column names
        categorical_cols (list): List of categorical column names
        missing_data (pd.Series): Series with missing counts per column
        target_var (str): Target variable to exclude from imputation
    Returns:
    Returns:
        tuple: (imputed_df, methods_used_dict, columns_imputed_list)
    """
    imputed_df = df.copy()
    methods_used = {}
    columns_imputed = []
    
    for col, missing_pct in columns_with_pct:
        if col == target_var:
            continue
            
        print(f"  {col}: {missing_pct:.1%} missing")
        
        if col in numeric_cols:
            try:
                # Use iterative imputation (MICE-like)
                iter_features = [c for c in numeric_cols if missing_data[c] < len(df) * 0.3 and c != col]
                
                if len(iter_features) >= 3:
                    iter_data = imputed_df[[col] + iter_features].copy()
                    iter_imputer = IterativeImputer(random_state=42, max_iter=10)
                    iter_imputed = iter_imputer.fit_transform(iter_data)
                    imputed_df[col] = iter_imputed[:, 0]
                    methods_used[col] = "iterative_imputation"
                    columns_imputed.append(col)
                    print(f"    ✅ Applied iterative imputation")
                else:
                    median_value = imputed_df[col].median()
                    imputed_df[col] = imputed_df[col].fillna(median_value)
                    methods_used[col] = f"median_fallback ({median_value:.3f})"
                    columns_imputed.append(col)
                    print(f"    ⚠️ Used median fallback")
                    
            except Exception as e:
                median_value = imputed_df[col].median()
                imputed_df[col] = imputed_df[col].fillna(median_value)
                methods_used[col] = f"median_error_fallback ({median_value:.3f})"
                columns_imputed.append(col)
                print(f"    ⚠️ Error in advanced imputation, used median: {e}")
        
        elif col in categorical_cols:
            imputed_df[col] = imputed_df[col].fillna('Missing_Imputed')
def intelligent_imputation_strategy(
    df: pd.DataFrame, 
    column_info: Dict[str, Any], 
    target_var: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Intelligent imputation strategy based on data analysis and missing patterns.
    
    This function analyzes missing data patterns and applies appropriate imputation
    methods based on data types, missing percentages, and relationships with target variable.
    
    Parameters:
        df (pd.DataFrame): DataFrame to impute
        column_info (dict): Results from comprehensive_data_exploration
        target_var (str): Name of target variable (to avoid imputing it)
    
    Returns:
        pd.DataFrame: DataFrame with imputed values
        dict: Imputation report with methods used
    """
        pd.DataFrame: DataFrame with imputed values
    print("🔧 INTELLIGENT IMPUTATION STRATEGY")
    print("=" * 60)
    print(f"✅ Using configured thresholds: Low={LOW_MISSING_THRESHOLD:.0%}, Medium={MEDIUM_MISSING_THRESHOLD:.0%}, High={HIGH_MISSING_THRESHOLD:.0%}")
    
    imputed_df = df.copy()
    imputation_report = {
        'original_missing': df.isnull().sum().sum(),
        'methods_used': {},
        'columns_imputed': [],
        'columns_skipped': [],
        'imputation_summary': {}
    }
        'columns_skipped': [],
    # Get missing data analysis
    missing_data = df.isnull().sum()
    missing_cols = missing_data[missing_data > 0].sort_values(ascending=False)
    # Get missing data analysis
    if len(missing_cols) == 0:
        print("✅ No missing values found!")
        return imputed_df, imputation_report
    
    print(f"📊 Found {len(missing_cols)} columns with missing values")
    print(f"Total missing values: {missing_data.sum():,}")
    print(f"📊 Found {len(missing_cols)} columns with missing values")
    # Categorize columns by data type
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variable from imputation if specified
    if target_var:
        if target_var in numeric_cols:
            numeric_cols.remove(target_var)
        if target_var in categorical_cols:
            categorical_cols.remove(target_var)
        print(f"🎯 Protecting target variable '{target_var}' from imputation")
    
    # Analyze missing data patterns
    print(f"\n📋 Missing Data Analysis by Column Type:")
    numeric_missing = [col for col in missing_cols.index if col in numeric_cols]
    categorical_missing = [col for col in missing_cols.index if col in categorical_cols]
    print(f"  Numeric columns with missing values: {len(numeric_missing)}")
    print(f"  Categorical columns with missing values: {len(categorical_missing)}")
    
    # Categorize by missing severity using helper function
    print(f"\n🎯 Imputation Strategy Selection:")
    print(f"  📊 Low Missing (<{LOW_MISSING_THRESHOLD:.0%}): Simple imputation (median/mode)")
    print(f"  🧠 Medium Missing ({LOW_MISSING_THRESHOLD:.0%}-{MEDIUM_MISSING_THRESHOLD:.0%}): Advanced imputation (KNN/iterative)")
    print(f"  ⚠️ High Missing ({MEDIUM_MISSING_THRESHOLD:.0%}-{HIGH_MISSING_THRESHOLD:.0%}): Advanced methods or consider dropping")
    print(f"  🚨 Very High Missing (>{HIGH_MISSING_THRESHOLD:.0%}): Recommend dropping")
    print(f"  🧠 Medium Missing ({LOW_MISSING_THRESHOLD:.0%}-{MEDIUM_MISSING_THRESHOLD:.0%}): Advanced imputation (KNN/iterative)")
    strategies = categorize_by_missing_severity(
        missing_cols, df, LOW_MISSING_THRESHOLD, MEDIUM_MISSING_THRESHOLD, HIGH_MISSING_THRESHOLD
    )
    
    # Report strategy assignments
    for strategy, cols in strategies.items():
        if cols:
            print(f"\n  {strategy.replace('_', ' ').title()}: {len(cols)} columns")
            for col, pct in cols[:3]:
                print(f"    {col}: {pct:.1%} missing")
            if len(cols) > 3:
                print(f"    ... and {len(cols)-3} more")
    
    # Apply imputation methods
    print(f"\n🔄 Applying Imputation Methods:")
    
    # Low Missing - Simple Imputation
    if strategies['low_missing']:
        print(f"\n📊 Simple Imputation (Low Missing < {LOW_MISSING_THRESHOLD:.0%}):")
        imputed_df, methods, cols = apply_simple_imputation(
            imputed_df, strategies['low_missing'], numeric_cols, categorical_cols, target_var
        )
        imputation_report['methods_used'].update(methods)
        imputation_report['columns_imputed'].extend(cols)
    
    # Medium Missing - Advanced Imputation
    if strategies['medium_missing']:
        print(f"\n🧠 Advanced Imputation (Medium Missing {LOW_MISSING_THRESHOLD:.0%}-{MEDIUM_MISSING_THRESHOLD:.0%}):")
        imputed_df, methods, cols = apply_advanced_imputation(
            imputed_df, strategies['medium_missing'], numeric_cols, categorical_cols, missing_data, target_var
        )
        imputation_report['methods_used'].update(methods)
        imputation_report['columns_imputed'].extend(cols)
    
    # High Missing - Advanced Methods
    if strategies['high_missing']:
        print(f"\n⚠️ High Missing Data ({MEDIUM_MISSING_THRESHOLD:.0%}-{HIGH_MISSING_THRESHOLD:.0%}) - Requires Decision:")
        imputed_df, methods, cols = apply_high_missing_imputation(
            imputed_df, strategies['high_missing'], numeric_cols, categorical_cols, missing_data, target_var
        )
        imputation_report['methods_used'].update(methods)
        imputation_report['columns_imputed'].extend(cols)
    
    # Very High Missing - Recommend Dropping
    if strategies['very_high_missing']:
        print(f"\n🚨 Very High Missing Data (>{HIGH_MISSING_THRESHOLD:.0%}) - Consider Dropping:")
        for col, missing_pct in strategies['very_high_missing']:
            print(f"  {col}: {missing_pct:.1%} missing - Consider removing this column")
            imputation_report['columns_skipped'].append(col)
    
    # Final Report
    final_missing = imputed_df.isnull().sum().sum()
    imputation_report['final_missing'] = final_missing
    imputation_report['missing_reduced'] = imputation_report['original_missing'] - final_missing
    
    print(f"\n" + "=" * 60)
    print("✅ IMPUTATION COMPLETE!")
    print(f"Original missing values: {imputation_report['original_missing']:,}")
    print(f"Final missing values: {final_missing:,}")
    print(f"Missing values imputed: {imputation_report['missing_reduced']:,}")
    print(f"Columns imputed: {len(imputation_report['columns_imputed'])}")
    print(f"Columns skipped (high missing): {len(imputation_report['columns_skipped'])}")
    
    if final_missing > 0:
        remaining_missing = imputed_df.isnull().sum()
        remaining_missing = remaining_missing[remaining_missing > 0]
        print(f"\n⚠️ Remaining missing values in:")
        for col, count in remaining_missing.items():
            pct = count / len(imputed_df) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
    
    print("=" * 60)
    
    return imputed_df, imputation_report


# Usage example
def apply_imputation_workflow():
    """
    Apply imputation as part of the preprocessing workflow
    """
    if 'base_data' in globals() and 'column_info' in globals():
        print("🔧 Applying intelligent imputation strategy...")
        
        # Apply imputation before automated cleaning
        imputed_data, imputation_report = intelligent_imputation_strategy(
            base_data, column_info, target_var=None
        )
        
        print(f"\n📊 Imputation Summary:")
        print(f"  Columns imputed: {len(imputation_report['columns_imputed'])}")
        print(f"  Missing values reduced: {imputation_report['missing_reduced']:,}")
        print(f"  Methods used: {len(set(imputation_report['methods_used'].values()))}")
        
        return imputed_data, imputation_report
    else:
        print("❌ Please run comprehensive_data_exploration() first")
        return None, None


print("🔧 Imputation functions ready!")
        print("❌ Please run comprehensive_data_exploration() first")
        print("❌ Please run comprehensive_data_exploration() first")
        return None, None
        return None, Noneprint("2. imputed_data, report = apply_imputation_workflow()")

print("💡 Usage options:")
        return None, None
        return None, None

print("2. imputed_data, report = apply_imputation_workflow()")

print("1. imputed_data, report = intelligent_imputation_strategy(base_data, column_info)")


print("🔧 Imputation functions ready!")
print("🔧 Imputation functions ready!")print("2. imputed_data, report = apply_imputation_workflow()")

print("2. imputed_data, report = apply_imputation_workflow()")
print("🔧 Imputation functions ready!")
print("🔧 Imputation functions ready!")
print("💡 Usage options:")
print("💡 Usage options:")print("2. imputed_data, report = apply_imputation_workflow()")

print("💡 Usage options:")
print("💡 Usage options:")
print("1. imputed_data, report = intelligent_imputation_strategy(base_data, column_info)")
print("1. imputed_data, report = intelligent_imputation_strategy(base_data, column_info)")print("1. imputed_data, report = intelligent_imputation_strategy(base_data, column_info)")
print("1. imputed_data, report = intelligent_imputation_strategy(base_data, column_info)")

# %% [markdown]
# ## 🎯 Strategic Imputation (Recommended Decision Point)
# 
# **Perfect timing!** Now that we have the `intelligent_imputation_strategy` function defined above and complete data analysis results, this is the optimal point to apply intelligent imputation before automated cleaning. This preserves valuable columns that might otherwise be dropped.

# %%
# RECOMMENDED WORKFLOW: Strategic Imputation at Decision Point 1
# ================================================================

def execute_recommended_imputation_workflow():
    """
    Execute the recommended imputation strategy at the optimal decision point.
    
    This runs after comprehensive_data_exploration() but before automated_data_cleaning()
    to preserve valuable columns that might otherwise be dropped due to missing data.
    """
    
    print("🚀 EXECUTING RECOMMENDED IMPUTATION WORKFLOW")
    print("=" * 70)
    
    # Step 1: Validate prerequisites
    if 'base_data' not in globals():
        print("❌ base_data not found. Please run the file selection cell first.")
        return None, None
    
    if 'column_info' not in globals():
        print("❌ column_info not found. Please run comprehensive_data_exploration() first.")
        return None, None
    
    print("✅ Prerequisites met - data and analysis results available")
    print(f"📊 Working with dataset: {base_data.shape}")
    
    # Step 2: Apply strategic imputation
    print(f"\n🎯 Step 1: Applying intelligent imputation strategy...")
    print("📋 This will preserve valuable columns before automated cleaning")
    
    try:
        # Apply the intelligent imputation strategy
        imputed_data, imputation_report = intelligent_imputation_strategy(
            base_data, 
            column_info, 
            target_var=None  # Target not selected yet
        )
        
        print(f"\n✅ Imputation completed successfully!")
        
        # Report imputation results
        print(f"\n📈 Imputation Results Summary:")
        print(f"  Original missing values: {imputation_report['original_missing']:,}")
        print(f"  Final missing values: {imputation_report['final_missing']:,}")
        print(f"  Values imputed: {imputation_report['missing_reduced']:,}")
        print(f"  Columns processed: {len(imputation_report['columns_imputed'])}")
        print(f"  Columns skipped (high missing): {len(imputation_report['columns_skipped'])}")
        
        # Show methods used
        if imputation_report['methods_used']:
            method_counts = {}
            for method in imputation_report['methods_used'].values():
                method_type = method.split('(')[0].strip()  # Extract method name
                method_counts[method_type] = method_counts.get(method_type, 0) + 1
            
            print(f"\n🔧 Imputation Methods Applied:")
            for method, count in method_counts.items():
                print(f"  {method}: {count} columns")
        
        # Data quality comparison
        print(f"\n📊 Data Quality Improvement:")
        
        # Calculate missing values instead of completeness for clearer reporting
        original_missing_count = base_data.isnull().sum().sum()
        new_missing_count = imputed_data.isnull().sum().sum()
        total_cells = base_data.shape[0] * base_data.shape[1]
        
        original_missing_pct = (original_missing_count / total_cells) * 100
        new_missing_pct = (new_missing_count / total_cells) * 100
        
        print(f"  Original missing values: {original_missing_count:,} ({original_missing_pct:.3f}% of data)")
        print(f"  After imputation: {new_missing_count:,} ({new_missing_pct:.3f}% of data)")
        print(f"  Missing values eliminated: {original_missing_count - new_missing_count:,}")
        
        if new_missing_count == 0:
            print(f"  ✅ All missing values successfully imputed!")
        
        return imputed_data, imputation_report
        
    except Exception as e:
        print(f"❌ Error during imputation: {e}")
        print("💡 You can still proceed with the original data and automated cleaning")
        return None, None

# Execute the recommended workflow
print("🎯 Ready to execute recommended imputation strategy!")
print("✅ intelligent_imputation_strategy function is now defined above and ready to use!")
print("\n📋 This is the optimal decision point because:")
print("  ✅ Complete data analysis is available (string nulls converted)")
print("  ✅ Can preserve valuable columns before automated cleaning")
print("  ✅ Advanced imputation methods can use all available information")
print("  ✅ Results will inform subsequent cleaning decisions")

print(f"\n💡 Execute the workflow:")
print("imputed_data, imputation_report = execute_recommended_imputation_workflow()")

# Uncomment the line below to run automatically:
imputed_data, imputation_report = execute_recommended_imputation_workflow()

# %% [markdown]
# ## ✅ Implemented: Recommended Imputation Strategy
# 
# **Great choice!** I've implemented the recommended imputation strategy at Decision Point 1. Here's what's now available:
# 
# ### 🎯 **Strategic Imputation (Decision Point 1)**
# - **Location**: Right here - after comprehensive analysis, before automated cleaning
# - **Function**: `execute_recommended_imputation_workflow()`
# - **Benefits**: Preserves valuable columns, uses complete analysis results, informs cleaning decisions
# 
# ### 🔄 **Updated Complete Workflow**  
# The `complete_data_preprocessing_workflow()` function now includes:
# 1. **Data Loading & Validation**
# 2. **Comprehensive Analysis** (includes string null conversion)
# 3. **Strategic Imputation** ← NEW! (Decision Point 1)
# 4. **Automated Cleaning** (now works on imputed data)
# 5. **Target Selection Prep**
# 6. **Quality Reporting**
# 
# ### 🚀 **Ready to Execute Options**
# 
# **Option 1: Strategic Imputation Only**
# ```python
# # Execute just the recommended imputation step
# imputed_data, imputation_report = execute_recommended_imputation_workflow()
# ```
# 
# **Option 2: Complete Enhanced Workflow**  
# ```python
# # Execute the complete workflow with strategic imputation
# cleaned_data, results = complete_data_preprocessing_workflow()
# ```
# 
# ### 💡 **Why This Strategy Works**
# - ✅ **Optimal Timing**: Has complete data analysis but before cleaning decisions
# - ✅ **Data Preservation**: Saves columns that might otherwise be dropped
# - ✅ **Intelligent Methods**: Uses advanced imputation based on data characteristics  
# - ✅ **Informed Decisions**: Imputation results guide subsequent cleaning
# - ✅ **Flexible**: Can still proceed with original workflow if needed
# 
# **Execute the cell below to proceed with your recommended strategy!**

# %% [markdown]
# ## 📋 When to Consider Imputation: Decision Framework
# 
# ### 🎯 **Key Decision Points in Your Pipeline**
# 
# **Point 1: After Comprehensive Analysis (Pre-Cleaning)**
# - **Best for**: Preserving valuable columns that would otherwise be dropped
# - **Use when**: Columns have 20-40% missing but contain important information
# - **Methods**: KNN, Iterative imputation, domain-specific approaches
# 
# **Point 2: During Automated Cleaning (Alternative to Dropping)**  
# - **Best for**: Integrated workflow with automatic decision making
# - **Use when**: You want a complete end-to-end automated pipeline
# - **Methods**: Intelligent strategy selection based on missing percentage
# 
# **Point 3: After Target Selection (Pre-Modeling)**
# - **Best for**: Final cleanup before feature engineering
# - **Use when**: Small amounts of missing data remain after cleaning
# - **Methods**: Simple imputation (median, mode) for remaining gaps
# 
# ### 🧠 **Imputation vs Deletion Decision Matrix**
# 
# | Missing % | Numeric Columns | Categorical Columns | Recommendation |
# |-----------|----------------|-------------------|----------------|
# | < 5% | Median | Mode | ✅ **Simple Imputation (Auto)** |
# | 5-20% | KNN Imputation | Most Frequent | ✅ **KNN Imputation (Auto)** |
# | 20-40% | MICE/Iterative | Missing Category | ✅ **Iterative Imputation (Auto)** |
# | > 40% | Drop Column | Drop Column | ❌ **Auto-Drop** (unless target) |
# 
# ### 🔧 **Method Selection Guidelines**
# 
# **Numeric Data**:
# - **Median**: Robust to outliers and skewed distributions (always used for < 5% missing)
# - **KNN**: Use when similar observations can inform missing values
# - **Iterative (MICE)**: Use when multiple variables have missing data patterns
# 
# **Categorical Data**:
# - **Mode**: Use for ordinal data or when most frequent makes sense
# - **Missing Category**: Use when "missing" itself is informative
# - **Predictive**: Use when other variables can predict the category
# 
# ### ⚖️ **Trade-offs to Consider**
# 
# **Advantages of Imputation**:
# - ✅ Preserves sample size
# - ✅ Retains potentially valuable features
# - ✅ Avoids bias from complete case analysis
# - ✅ Better model performance with sufficient data
# 
# **Disadvantages of Imputation**:
# - ❌ Can introduce bias if done incorrectly
# - ❌ May reduce variance artificially
# - ❌ Computational overhead for advanced methods
# - ❌ Risk of overfitting to imputation model
# 
# ### 🚨 **When NOT to Impute**
# 
# 1. **Missing Not at Random (MNAR)**: When missingness is informative
# 2. **High Missing %**: When > 50% of values are missing
# 3. **ID Columns**: When missing values indicate invalid records  
# 4. **Time Dependencies**: When imputation would violate temporal relationships
# 5. **Domain Constraints**: When imputed values would be impossible/invalid

# %% [markdown]
# ## Complete Workflow Execution
# 
# Execute this cell to run the complete data preprocessing workflow automatically.

# %%
def automated_data_cleaning(df, column_info, interactive=True):
    """Perform systematic data cleaning based on configuration constants.
    
    Executes multiple cleaning operations including removal of columns with
    excessive missing data, duplicate rows, low variance columns, datetime
    columns, and high cardinality columns (likely IDs). Protects the target
    variable (PROTECTED_COLUMN) from all removal operations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to clean.
    column_info : dict
        Results from comprehensive_data_exploration (currently unused but
        reserved for future enhancements).
    interactive : bool, default=True
        If True, prompts user for confirmation before removing columns.
        If False, performs all operations automatically.
    
    Returns
    -------
    tuple of (pd.DataFrame, dict)
        cleaned_df : pd.DataFrame
            Cleaned DataFrame with problematic columns/rows removed.
        cleaning_summary : dict
            Summary dictionary containing:
            - 'original_shape': Original (rows, cols)
            - 'final_shape': Final (rows, cols)
            - 'columns_removed': List of removed column names
            - 'rows_removed': Count of removed rows
            - 'operations': List of operation descriptions
    
    Notes
    -----
    Respects global constants: MAX_MISSING_DATA, REMOVE_DUPLICATE_ROWS,
    REMOVE_LOW_VARIANCE_COLS, LOW_VARIANCE_THRESHOLD, REMOVE_DATE_COLUMNS,
    HIGH_CARDINALITY_THRESHOLD, and PROTECTED_COLUMN.
    """
    
    print("🧹 AUTOMATED DATA CLEANING")
    print("=" * 60)
    
    cleaned_df = df.copy()
    cleaning_summary = {
        'original_shape': df.shape,
        'columns_removed': [],
        'rows_removed': 0,
        'operations': []
    }
    
    # ========================================================================
    # STEP 1: Remove columns with excessive missing data
    # ========================================================================
    print(f"\n📊 Step 1: Checking for columns with >{MAX_MISSING_DATA:.0%} missing data...")
    
    missing_pct = cleaned_df.isnull().sum() / len(cleaned_df)
    high_missing_cols = missing_pct[missing_pct > MAX_MISSING_DATA].index.tolist()
    
    # Protect target variable from removal
    if 'PROTECTED_COLUMN' in globals() and PROTECTED_COLUMN in high_missing_cols:
        print(f"   🛡️  Protecting target variable: {PROTECTED_COLUMN}")
        high_missing_cols.remove(PROTECTED_COLUMN)
    
    if high_missing_cols:
        print(f"   Found {len(high_missing_cols)} columns exceeding threshold:")
        for col in high_missing_cols[:10]:
            pct = missing_pct[col]
            print(f"      {col}: {pct:.1%} missing")
        if len(high_missing_cols) > 10:
            print(f"      ... and {len(high_missing_cols)-10} more")
        
        remove = True
        if interactive:
            response = input(f"\n   Remove these {len(high_missing_cols)} columns? (y/n): ").lower().strip()
            remove = (response == 'y')
        
        if remove:
            cleaned_df = cleaned_df.drop(columns=high_missing_cols)
            cleaning_summary['columns_removed'].extend(high_missing_cols)
            cleaning_summary['operations'].append(f"Removed {len(high_missing_cols)} columns with >{MAX_MISSING_DATA:.0%} missing data")
            print(f"   ✅ Removed {len(high_missing_cols)} high-missing columns")
        else:
            print(f"   ⏭️  Skipped removal")
    else:
        print(f"   ✅ No columns exceed {MAX_MISSING_DATA:.0%} missing data threshold")
    
    # ========================================================================
    # STEP 2: Remove duplicate rows
    # ========================================================================
    if REMOVE_DUPLICATE_ROWS:
        print(f"\n🔄 Step 2: Checking for duplicate rows...")
        
        duplicates = cleaned_df.duplicated().sum()
        if duplicates > 0:
            print(f"   Found {duplicates} duplicate rows ({duplicates/len(cleaned_df):.1%})")
            
            remove = True
            if interactive:
                response = input(f"   Remove duplicate rows? (y/n): ").lower().strip()
                remove = (response == 'y')
            
            if remove:
                initial_rows = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                rows_removed = initial_rows - len(cleaned_df)
                cleaning_summary['rows_removed'] += rows_removed
                cleaning_summary['operations'].append(f"Removed {rows_removed} duplicate rows")
                print(f"   ✅ Removed {rows_removed} duplicate rows")
            else:
                print(f"   ⏭️  Skipped removal")
        else:
            print(f"   ✅ No duplicate rows found")
    else:
        print(f"\n🔄 Step 2: Duplicate removal disabled (REMOVE_DUPLICATE_ROWS = False)")
    
    # ========================================================================
    # STEP 3: Remove low variance columns
    # ========================================================================
    if REMOVE_LOW_VARIANCE_COLS:
        print(f"\n📉 Step 3: Checking for low variance columns (≥{LOW_VARIANCE_THRESHOLD:.0%} same values)...")
        
        low_var_cols = []
        for col in cleaned_df.columns:
            if len(cleaned_df[col]) > 0:
                most_common_pct = cleaned_df[col].value_counts(normalize=True).iloc[0] if len(cleaned_df[col].value_counts()) > 0 else 0
                if most_common_pct >= LOW_VARIANCE_THRESHOLD:
                    low_var_cols.append(col)
        
        # Protect target variable from removal
        if 'PROTECTED_COLUMN' in globals() and PROTECTED_COLUMN in low_var_cols:
            print(f"   🛡️  Protecting target variable: {PROTECTED_COLUMN}")
            low_var_cols.remove(PROTECTED_COLUMN)
        
        if low_var_cols:
            print(f"   Found {len(low_var_cols)} low variance columns:")
            for col in low_var_cols[:10]:
                most_common_pct = cleaned_df[col].value_counts(normalize=True).iloc[0]
                print(f"      {col}: {most_common_pct:.1%} same value")
            if len(low_var_cols) > 10:
                print(f"      ... and {len(low_var_cols)-10} more")
            
            remove = True
            if interactive:
                response = input(f"\n   Remove these {len(low_var_cols)} columns? (y/n): ").lower().strip()
                remove = (response == 'y')
            
            if remove:
                cleaned_df = cleaned_df.drop(columns=low_var_cols)
                cleaning_summary['columns_removed'].extend(low_var_cols)
                cleaning_summary['operations'].append(f"Removed {len(low_var_cols)} low variance columns")
                print(f"   ✅ Removed {len(low_var_cols)} low variance columns")
            else:
                print(f"   ⏭️  Skipped removal")
        else:
            print(f"   ✅ No low variance columns found")
    else:
        print(f"\n📉 Step 3: Low variance removal disabled (REMOVE_LOW_VARIANCE_COLS = False)")
    
    # ========================================================================
    # STEP 4: Remove datetime columns
    # ========================================================================
    if REMOVE_DATE_COLUMNS:
        print(f"\n📅 Step 4: Checking for datetime columns...")
        
        datetime_cols = cleaned_df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Protect target variable from removal
        if 'PROTECTED_COLUMN' in globals() and PROTECTED_COLUMN in datetime_cols:
            print(f"   🛡️  Protecting target variable: {PROTECTED_COLUMN}")
            datetime_cols.remove(PROTECTED_COLUMN)
        
        if datetime_cols:
            print(f"   Found {len(datetime_cols)} datetime columns: {datetime_cols}")
            
            remove = True
            if interactive:
                response = input(f"   Remove datetime columns? (y/n): ").lower().strip()
                remove = (response == 'y')
            
            if remove:
                cleaned_df = cleaned_df.drop(columns=datetime_cols)
                cleaning_summary['columns_removed'].extend(datetime_cols)
                cleaning_summary['operations'].append(f"Removed {len(datetime_cols)} datetime columns")
                print(f"   ✅ Removed {len(datetime_cols)} datetime columns")
            else:
                print(f"   ⏭️  Skipped removal")
        else:
            print(f"   ✅ No datetime columns found")
    else:
        print(f"\n📅 Step 4: Datetime removal disabled (REMOVE_DATE_COLUMNS = False)")
    
    # ========================================================================
    # STEP 5: Remove high cardinality columns (likely IDs)
    # ========================================================================
    print(f"\n🔢 Step 5: Checking for high cardinality columns (≥{HIGH_CARDINALITY_THRESHOLD:.0%} unique values)...")
    
    high_card_cols = []
    for col in cleaned_df.columns:
        unique_ratio = cleaned_df[col].nunique() / len(cleaned_df)
        if unique_ratio >= HIGH_CARDINALITY_THRESHOLD:
            high_card_cols.append(col)
    
    # Protect target variable from removal
    if 'PROTECTED_COLUMN' in globals() and PROTECTED_COLUMN in high_card_cols:
        print(f"   🛡️  Protecting target variable: {PROTECTED_COLUMN}")
        high_card_cols.remove(PROTECTED_COLUMN)
    
    if high_card_cols:
        print(f"   Found {len(high_card_cols)} high cardinality columns (likely IDs):")
        for col in high_card_cols[:10]:
            unique_ratio = cleaned_df[col].nunique() / len(cleaned_df[col])
            print(f"      {col}: {unique_ratio:.1%} unique values")
        if len(high_card_cols) > 10:
            print(f"      ... and {len(high_card_cols)-10} more")
        
        remove = True
        if interactive:
            response = input(f"\n   Remove these {len(high_card_cols)} columns? (y/n): ").lower().strip()
            remove = (response == 'y')
        
        if remove:
            cleaned_df = cleaned_df.drop(columns=high_card_cols)
            cleaning_summary['columns_removed'].extend(high_card_cols)
            cleaning_summary['operations'].append(f"Removed {len(high_card_cols)} high cardinality columns")
            print(f"   ✅ Removed {len(high_card_cols)} high cardinality columns")
        else:
            print(f"   ⏭️  Skipped removal")
    else:
        print(f"   ✅ No high cardinality columns found")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    cleaning_summary['final_shape'] = cleaned_df.shape
    
    print(f"\n" + "=" * 60)
    print("✅ AUTOMATED DATA CLEANING COMPLETE!")
    print(f"   Original shape: {cleaning_summary['original_shape']}")
    print(f"   Final shape: {cleaning_summary['final_shape']}")
    print(f"   Columns removed: {len(cleaning_summary['columns_removed'])}")
    print(f"   Rows removed: {cleaning_summary['rows_removed']}")
    print(f"   Operations performed: {len(cleaning_summary['operations'])}")
    
    if cleaning_summary['operations']:
        remaining_missing = cleaned_df.isnull().sum().sum()
    
    if remaining_missing > 0:
        print(f"\n   ⚠️  Remaining missing values: {remaining_missing}")
    else:
        print(f"\n   ✅ No missing values remaining")
    
    print("=" * 60)
    
    return cleaned_df, cleaning_summary
    print(f"\n   ⚠️  Remaining missing values: {remaining_missing}")
    
    

    print("=" * 60)

    print(f"\n   ✅ No missing values remaining")    
    print("=" * 60)


# %%
# Complete Data Preprocessing Workflow
# ====================================

def complete_data_preprocessing_workflow():
    """
    Complete workflow that combines all preprocessing steps in the correct order.
    Now includes the recommended strategic imputation at Decision Point 1.
    """
    
    print("🚀 COMPLETE DATA PREPROCESSING WORKFLOW (WITH STRATEGIC IMPUTATION)")
    print("=" * 70)
    
    # Step 1: Check if data is loaded
    if 'base_data' not in globals():
        print("❌ No data loaded. Please run the file selection cell first.")
        return None, None, None
    
    print(f"✅ Data loaded: {base_data.shape}")
    
    # Step 2: Run comprehensive exploration (includes string null handling)
    print("\n📊 Step 1: Running comprehensive data exploration...")
    comprehensive_data_exploration()
    
    # Step 3: Apply strategic imputation (RECOMMENDED DECISION POINT)
    print("\n🎯 Step 2: Applying strategic imputation (Decision Point 1)...")
    try:
        imputed_data, imputation_report = intelligent_imputation_strategy(
            base_data, column_info, target_var=None
        )
        print(f"✅ Strategic imputation completed!")
        print(f"   Missing values reduced: {imputation_report['missing_reduced']:,}")
        print(f"   Columns processed: {len(imputation_report['columns_imputed'])}")
        
        # Use imputed data for subsequent steps
        data_for_cleaning = imputed_data
        
    except Exception as e:
        print(f"⚠️ Imputation failed: {e}")
        print("   Proceeding with original data...")
        data_for_cleaning = base_data
        imputation_report = None
    
    # Step 4: Perform automated cleaning on imputed data
    print("\n🧹 Step 3: Performing automated data cleaning...")
    cleaned_data, cleaning_summary = automated_data_cleaning(data_for_cleaning, column_info, interactive=False)
    
    # Step 5: Target variable selection
    print(f"\n🎯 Step 4: Select target variable...")
    print("Available numeric columns for target selection:")
    numeric_cols = cleaned_data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
    for i, col in enumerate(numeric_cols[:10], 1):  # Show first 10
        print(f"  {i}. {col}")
    if len(numeric_cols) > 10:
        print(f"  ... and {len(numeric_cols)-10} more columns")
    
    print(f"\n💡 Next step: Run select_dependent_variable(cleaned_data) to choose your target")
    
    # Step 6: Final quality check
    print(f"\n🔍 Step 5: Final quality check...")
    print(f"   Original shape: {base_data.shape}")
    if imputation_report:
        print(f"   After imputation: {data_for_cleaning.shape}")
    print(f"   Final cleaned shape: {cleaned_data.shape}")
    print(f"   Columns removed: {len(cleaning_summary['columns_removed'])}")
    print(f"   Rows removed: {cleaning_summary['rows_removed']}")
    
    # Check for remaining issues
    remaining_missing = cleaned_data.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"   ⚠️  Remaining missing values: {remaining_missing}")
    else:
        print(f"   ✅ No missing values remaining")
    
    # Data type summary
    dtype_summary = cleaned_data.dtypes.value_counts().to_dict()
    print(f"   📝 Final data types: {dtype_summary}")
    
    print(f"\n✅ Enhanced preprocessing complete with strategic imputation!")
    print("📋 Next steps:")
    print("   1. Run: dependent_var = select_dependent_variable(cleaned_data)")
    print("   2. Run: model_ready_data, report = advanced_string_preprocessing_for_modeling(cleaned_data, dependent_var)")
    
    # Return both cleaning and imputation results
    results = {
        'cleaned_data': cleaned_data,
        'cleaning_summary': cleaning_summary,
        'column_info': column_info,
        'imputation_report': imputation_report
    }
    
    return cleaned_data, results

# Execute complete workflow with strategic imputation
print("🔧 Ready to run enhanced preprocessing workflow with strategic imputation!")
print("\n🎯 This workflow includes:")
print("   ✅ Data loading and validation")
print("   ✅ Comprehensive analysis with string null conversion") 
print("   ✅ Strategic imputation at Decision Point 1 (RECOMMENDED)")
print("   ✅ Automated data cleaning")
print("   ✅ Quality reporting and next steps")
print("\nUncomment the line below to execute:")
print("# cleaned_data, results = complete_data_preprocessing_workflow()")

# %%
cleaned_data, results = complete_data_preprocessing_workflow()

# %% [markdown]
# ## Advanced String Preprocessing for Modeling
# 
# After selecting your target variable, this section converts all categorical data to numeric formats suitable for machine learning algorithms.

# %%
def remove_date_like_columns(
    df: pd.DataFrame, 
    string_cols: List[str], 
    remove_dates: bool
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Remove columns that appear to be date-related.
    
    Parameters:
        df (pd.DataFrame): DataFrame to process
        string_cols (list): List of string column names
        remove_dates (bool): Whether to remove date-like columns
    
    Returns:
        tuple: (processed_df, remaining_string_cols, removed_cols)
    """
    if not remove_dates:
        return df, string_cols, []
    
    date_keywords = ['date', 'time', 'datetime', 'timestamp', 'dt_', '_dt']
    date_like_cols = [col for col in string_cols if any(keyword in col.lower() for keyword in date_keywords)]
    
    if date_like_cols:
        print(f"🗓️  Removing {len(date_like_cols)} date-like string columns:")
        for col in date_like_cols:
            print(f"  - {col}")
            string_cols.remove(col)
        df = df.drop(columns=date_like_cols)
def clean_string_columns(df: pd.DataFrame, string_cols: List[str]) -> pd.DataFrame:
    return df, string_cols, date_like_cols


def clean_string_columns(df, string_cols):
    """
    Clean string columns by standardizing whitespace and case.
    
    Parameters:
        df (pd.DataFrame): DataFrame to process
        string_cols (list): List of string column names
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    print(f"\n🧹 Step 1: Additional string cleaning...")
    
    for col in string_cols:
        initial_nulls = df[col].isnull().sum()
        
        # Strip whitespace and standardize case
        df[col] = df[col].astype(str).str.strip().str.lower()
        
        # Convert empty strings to NaN
        df[col] = df[col].replace('', np.nan)
        df[col] = df[col].replace('nan', np.nan)
        
        final_nulls = df[col].isnull().sum()
        new_nulls = final_nulls - initial_nulls
        
        if new_nulls > 0:
            print(f"  {col}: Created {new_nulls} additional nulls from empty/whitespace strings")
def categorize_columns_by_cardinality(
    df: pd.DataFrame, 
    string_cols: List[str], 
    low_threshold: int, 
    high_threshold: int
) -> Dict[str, List[str]]:
    """
    Categorize string columns by cardinality for appropriate encoding.
    
    Parameters:
        df (pd.DataFrame): DataFrame to analyze
        string_cols (list): List of string column names
        low_threshold (int): Max unique values for one-hot encoding
        high_threshold (int): Max unique values for label encoding
    Returns:
    Returns:
        dict: Dictionary with 'low', 'high', and 'very_high' cardinality column lists
    """
    print(f"\n🏷️  Step 2: Analyzing categorical columns for encoding...")
    
    categorized = {
        'low': [],      # One-hot encoding
        'high': [],     # Label encoding
        'very_high': [] # Consider removing
    }
    
    for col in string_cols:
        if col in df.columns:
            unique_count = df[col].nunique()
            total_count = df[col].count()
            if total_count == 0:
            if total_count == 0:
                continue
                
            cardinality_ratio = unique_count / total_count
            
            if unique_count <= low_threshold:
                categorized['low'].append(col)
                print(f"  {col}: {unique_count} unique values → Good for One-Hot Encoding")
            elif unique_count <= high_threshold:
                categorized['high'].append(col)
                print(f"  {col}: {unique_count} unique values → Consider Target/Label Encoding")
            else:
                categorized['very_high'].append(col)
def apply_one_hot_encoding(
    df: pd.DataFrame, 
    columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Apply one-hot encoding to low cardinality columns.
    
    Parameters:
        df (pd.DataFrame): DataFrame to process
        columns (list): List of column names to encode
    
    Returns:
        tuple: (processed_df, encoding_map_dict)
    """
    if not columns:
        return df, {}
    
    print(f"\n📊 One-hot encoding {len(columns)} low-cardinality columns...")
    
    encoding_maps = {}
    
    for col in columns:
        if col in df.columns:
            # Create dummy variables
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            
            # Add to dataframe and remove original
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
            
            # Track the encoding
            encoding_maps[col] = {
                'method': 'one_hot',
                'new_columns': list(dummies.columns)
            }
            
            print(f"  ✅ {col} → {len(dummies.columns)} dummy columns")
    
def apply_label_encoding(
    df: pd.DataFrame, 
    columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Apply label encoding to medium cardinality columns.
    
    Parameters:
        df (pd.DataFrame): DataFrame to process
        columns (list): List of column names to encode
    
    Returns:
        tuple: (processed_df, encoding_map_dict)
    """
    if not columns:
        return df, {}
    
    print(f"\n🏷️  Label encoding {len(columns)} medium-cardinality columns...")
    
    from sklearn.preprocessing import LabelEncoder
    
    encoding_maps = {}
    
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
        if col in df.columns:
            # Handle missing values
            non_null_mask = df[col].notna()
            
            if non_null_mask.sum() > 0:
                df.loc[non_null_mask, f'{col}_encoded'] = le.fit_transform(df.loc[non_null_mask, col])
                df[f'{col}_encoded'] = df[f'{col}_encoded'].astype('Int64')
                df.loc[non_null_mask, f'{col}_encoded'] = le.fit_transform(df.loc[non_null_mask, col])
                # Drop original column
                df = df.drop(columns=[col])
                
                # Track the encoding
                encoding_maps[col] = {
                    'method': 'label_encoding',
                    'encoder': le,
                    'new_column': f'{col}_encoded'
                }
                
                print(f"  ✅ {col} → {col}_encoded ({len(le.classes_)} categories)")
    
def handle_high_cardinality_columns(
    df: pd.DataFrame, 
    columns: List[str], 
    interactive: bool = True
) -> Tuple[pd.DataFrame, int]:
    """
    Handle very high cardinality columns (likely IDs).
    
    Parameters:
        df (pd.DataFrame): DataFrame to process
        columns (list): List of high cardinality column names
        interactive (bool): Whether to prompt user for removal
    
    Returns:
        tuple: (processed_df, removed_count)
    """
    if not columns:
        return df, 0
    
    print(f"\n⚠️  {len(columns)} very high-cardinality columns detected:")
    for col in columns:
        if col in df.columns:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values (likely ID column)")
    
    remove = True
    if interactive:
        response = input(f"\nRemove these {len(columns)} high-cardinality columns? (y/n): ").lower().strip()
        remove = (response == 'y')
    
    if remove:
        df = df.drop(columns=[col for col in columns if col in df.columns])
        print(f"  ✅ Removed {len(columns)} high-cardinality columns")
        return df, len(columns)
def handle_remaining_object_columns(
    df: pd.DataFrame, 
    dependent_var: Optional[str] = None, 
    interactive: bool = True
) -> pd.DataFrame:
    """
    Handle any remaining object columns after encoding.
    
    Parameters:
        df (pd.DataFrame): DataFrame to process
        dependent_var (str): Dependent variable to preserve
        interactive (bool): Whether to prompt user for action
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    print(f"\n🧹 Step 4: Final cleanup...")
    remaining_object_cols = df.select_dtypes(include=['object']).columns.tolist()
    remaining_object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if dependent_var and dependent_var in remaining_object_cols:
        remaining_object_cols.remove(dependent_var)
    
    if not remaining_object_cols:
        return df
    print(f"⚠️  Warning: {len(remaining_object_cols)} object columns remain:")
    print(f"⚠️  Warning: {len(remaining_object_cols)} object columns remain:")
    for col in remaining_object_cols:
        print(f"  {col}: {df[col].nunique()} unique values")
    
    convert = True
    if interactive:
        response = input("Convert remaining object columns to numeric hash codes? (y/n): ").lower().strip()
        convert = (response == 'y')
    
    if convert:
        for col in remaining_object_cols:
            df[f'{col}_hash'] = df[col].astype(str).apply(lambda x: hash(x) % (10**8))
            df = df.drop(columns=[col])
            print(f"  ✅ {col} → {col}_hash (numeric)")
def print_preprocessing_summary(report: Dict[str, Any], df: pd.DataFrame) -> None:
    return df


def print_preprocessing_summary(report, df):
    """
    Print final preprocessing summary.
    
    Parameters:
        report (dict): Preprocessing report dictionary
        df (pd.DataFrame): Final processed DataFrame
    """
    print(f"\n" + "=" * 60)
    print("✅ ADVANCED STRING PREPROCESSING COMPLETE!")
    print(f"Original shape: {report['original_shape']}")
    print(f"Final shape: {report['final_shape']}")
    print(f"String columns processed: {len(report['columns_modified'])}")
    print(f"Operations performed: {len(report['operations'])}")
    
    # Data type summary
    final_dtypes = df.dtypes.value_counts()
    print(f"\nFinal data types:")
    for dtype, count in final_dtypes.items():
        print(f"  {dtype}: {count} columns")
    
    # Check for remaining missing values
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        print(f"\n⚠️  Remaining missing values: {total_missing}")
        cols_with_missing = df.isnull().sum()
        cols_with_missing = cols_with_missing[cols_with_missing > 0]
        for col, missing_count in cols_with_missing.head(5).items():
            pct = (missing_count / len(df)) * 100
            print(f"  {col}: {missing_count} ({pct:.1f}%)")
    else:
        print("\n✅ No missing values remaining!")
def advanced_string_preprocessing_for_modeling(
    df: pd.DataFrame, 
    dependent_var: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Preprocess string columns for machine learning modeling.
    
    Performs advanced string column handling including whitespace cleaning,
    categorical encoding (one-hot for low cardinality, label encoding for
    medium cardinality), and removal of high cardinality ID-like columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process.
    dependent_var : str, optional
        Name of dependent variable column to preserve from encoding/removal.
    
    Returns
    -------
    tuple of (pd.DataFrame, dict)
        processed_df : pd.DataFrame
            DataFrame with all string columns encoded as numeric.
        preprocessing_report : dict
            Report containing:
            - 'original_shape': Original dimensions
            - 'final_shape': Final dimensions
            - 'operations': List of operations performed
            - 'columns_modified': List of modified column names
            - 'encoding_maps': Dictionary mapping columns to encoding details
    
    Notes
    -----
    Encoding strategy:
    - ≤10 unique values: One-hot encoding
    - 11-50 unique values: Label encoding
    - >50 unique values: Flagged for removal (likely IDs)
    """
    """
    print("🔧 ADVANCED STRING PREPROCESSING FOR MODELING")
    print("=" * 60)
    
    processed_df = df.copy()
    report = {
        'original_shape': df.shape,
        'operations': [],
        'columns_modified': [],
        'encoding_maps': {}
    }
    
    # Get string/object columns (excluding the dependent variable)
    string_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
    if dependent_var and dependent_var in string_cols:
        string_cols.remove(dependent_var)
    
    # Remove date-like columns
    processed_df, string_cols, removed_dates = remove_date_like_columns(
        processed_df, 
        string_cols, 
        REMOVE_DATE_COLUMNS if 'REMOVE_DATE_COLUMNS' in globals() else True
    )
    if removed_dates:
        report['operations'].append(f"Removed {len(removed_dates)} date-like string columns")
    
    print(f"📝 Found {len(string_cols)} string columns to process")
    
    # Clean string columns
    processed_df = clean_string_columns(processed_df, string_cols)
    report['operations'].append("Cleaned whitespace and standardized case")
    
    # Categorize by cardinality
    categorized = categorize_columns_by_cardinality(
        processed_df,
        string_cols,
        ONE_HOT_ENCODING_MAX_CATEGORIES if 'ONE_HOT_ENCODING_MAX_CATEGORIES' in globals() else 10,
        LABEL_ENCODING_MAX_CATEGORIES if 'LABEL_ENCODING_MAX_CATEGORIES' in globals() else 50
    )
    )
    # Apply encoding strategies
    print(f"\n🔄 Step 3: Applying encoding strategies...")
    
    # One-hot encoding
    processed_df, one_hot_maps = apply_one_hot_encoding(processed_df, categorized['low'])
    report['encoding_maps'].update(one_hot_maps)
    report['columns_modified'].extend(categorized['low'])
    
    # Label encoding
    processed_df, label_maps = apply_label_encoding(processed_df, categorized['high'])
    report['encoding_maps'].update(label_maps)
    report['columns_modified'].extend(categorized['high'])
    
    # Handle high cardinality
    processed_df, removed_count = handle_high_cardinality_columns(
        processed_df, 
        categorized['very_high'],
        interactive=True
    )
    if removed_count > 0:
        report['operations'].append(f"Removed {removed_count} high-cardinality columns")
        report['operations'].append(f"Removed {removed_count} high-cardinality columns")
    # Handle remaining object columns
    processed_df = handle_remaining_object_columns(processed_df, dependent_var, interactive=True)
    
    # Finalize report
    report['final_shape'] = processed_df.shape
    report['columns_added'] = processed_df.shape[1] - df.shape[1] + len(report['columns_modified'])
    report['columns_added'] = processed_df.shape[1] - df.shape[1] + len(report['columns_modified'])
    # Print summary
    print_preprocessing_summary(report, processed_df)
    
    return processed_df, report


# Quick usage example
def prepare_data_for_modeling():
    """
    Complete pipeline: exploration → cleaning → string preprocessing
    """
    if 'cleaned_data' in globals() and 'dependent_var' in globals():
        # Check if dependent_var was removed during cleaning and restore it if needed
        if dependent_var not in cleaned_data.columns:
            print(f"\n⚠️  Warning: Dependent variable '{dependent_var}' was removed during cleaning")
            if dependent_var in imputed_data.columns:
                print(f"✅ Restoring '{dependent_var}' from imputed_data")
                cleaned_data[dependent_var] = imputed_data[dependent_var]
            else:
                print(f"❌ ERROR: Cannot restore '{dependent_var}' - not found in imputed_data either!")
                return None, None
        
        print("🚀 Running advanced string preprocessing on cleaned data...")

        model_ready_data, preprocessing_report = advanced_string_preprocessing_for_modeling(
        print("🚀 Running advanced string preprocessing on cleaned data...")
        print("🚀 Running advanced string preprocessing on cleaned data...")
        model_ready_data, preprocessing_report = advanced_string_preprocessing_for_modeling(
        model_ready_data, preprocessing_report = advanced_string_preprocessing_for_modeling(
        
        print("🚀 Running advanced string preprocessing on cleaned data...")        )

            cleaned_data, dependent_var
        model_ready_data, preprocessing_report = advanced_string_preprocessing_for_modeling(
        model_ready_data, preprocessing_report = advanced_string_preprocessing_for_modeling(
            cleaned_data, dependent_var
            cleaned_data, dependent_var
        print("🚀 Running advanced string preprocessing on cleaned data...")
        model_ready_data, preprocessing_report = advanced_string_preprocessing_for_modeling(        )

        )
            cleaned_data, dependent_var
            cleaned_data, dependent_var
        )
        )
        model_ready_data, preprocessing_report = advanced_string_preprocessing_for_modeling(
            cleaned_data, dependent_var            cleaned_data, dependent_var

        )        )

# %% [markdown]
# ## 📖 Complete Workflow Documentation
# 
# ### 🎯 Summary: End-to-End Data Preprocessing Pipeline
# 
# This notebook implements a comprehensive 4-stage preprocessing pipeline:
# 
# ### **Stage 1: Environment Setup**
# - **Package Installation**: Intel proxy-compatible package installation
# - **Library Imports**: All necessary libraries for data processing and modeling
# - **Configuration**: Global constants controlling all preprocessing behaviors
# 
# ### **Stage 2: Data Loading & Initial Quality**
# - **File Selection**: Interactive CSV file picker with encoding handling
# - **Missing Data Removal**: Automatic removal of columns exceeding missing data threshold
# - **Basic Validation**: Shape, memory usage, and data type overview
# 
# ### **Stage 3: Comprehensive Analysis & Cleaning**
# - **String Null Conversion**: Converts 20+ string null patterns to pandas NaN
# - **Data Exploration**: 10-section analysis covering all data quality aspects
# - **Automated Cleaning**: Systematic removal of problematic columns based on analysis
# - **Quality Reporting**: Detailed reporting of all transformations applied
# 
# ### **Stage 4: Modeling Preparation**
# - **Target Selection**: Interactive selection of dependent variable
# - **Categorical Encoding**: Smart encoding based on cardinality (configurable via constants)
#   - ≤ONE_HOT_ENCODING_MAX_CATEGORIES (default: 10) → One-hot encoding
#   - ≤LABEL_ENCODING_MAX_CATEGORIES (default: 50) → Label encoding  
#   - >LABEL_ENCODING_MAX_CATEGORIES → Flagged for removal (likely IDs)
# - **Final Cleanup**: Ensures all data is numeric and model-ready
# 
# ### � **Quick Start Guide**
# 
# **Option 1: Step-by-Step Execution**
# ```python
# # 1. Load your data
# base_data = select_csv_file()
# 
# # 2. Run comprehensive analysis (includes string null handling)
# comprehensive_data_exploration()
# 
# # 3. Clean the data
# cleaned_data, summary = automated_data_cleaning(base_data, column_info)
# 
# # 4. Select target variable
# dependent_var = select_dependent_variable(cleaned_data)
# 
# # 5. Prepare for modeling
# model_ready_data, report = advanced_string_preprocessing_for_modeling(cleaned_data, dependent_var)
# ```
# 
# **Option 2: Automated Workflow**
# ```python
# # Run the complete workflow automatically
# cleaned_data, summary, analysis = complete_data_preprocessing_workflow()
# 
# # Then select target and finalize
# dependent_var = select_dependent_variable(cleaned_data)
# model_ready_data, report = advanced_string_preprocessing_for_modeling(cleaned_data, dependent_var)
# ```
# 
# ### ⚙️ **Configuration Options**
# Modify these constants to control preprocessing behavior:
# - `MAX_MISSING_DATA = 0.5` → Remove columns with >50% missing data
# - `HIGH_CARDINALITY_THRESHOLD = 0.8` → ID detection threshold
# - `REMOVE_DUPLICATE_ROWS = True` → Automatic duplicate removal
# - `REMOVE_LOW_VARIANCE_COLS = True` → Remove constant columns
# - `CONVERT_STRING_NULLS = True` → Enable string null conversion
# 
# ### 🎉 **Expected Results**
# After completing the pipeline, you'll have:
# - ✅ **Clean dataset** with no missing values or duplicates
# - ✅ **Numeric data only** (suitable for ML algorithms)
# - ✅ **Proper encoding** of categorical variables
# - ✅ **Selected target variable** for modeling
# - ✅ **Detailed reports** of all transformations applied
# - ✅ **Model-ready data** for feature reduction and machine learning
# 
# The final `model_ready_data` DataFrame will be ready for feature selection, dimensionality reduction, and machine learning model training.

# %% [markdown]
# # Feature Selection & Model Training
# 
# Now that data is preprocessed, we'll perform feature selection and train models.

# %% [markdown]
# ## Configuration Constants for Feature Selection

# %%
# =============================================================================
# FEATURE SELECTION AND MODEL TRAINING CONFIGURATION
# =============================================================================
# These constants control the feature selection and machine learning workflow.
# Adjust these values based on your dataset size and computational resources.

# Train/Test Split Configuration
VALIDATION_SIZE = 0.2            # Proportion of data held out for final validation (20%)
SEED = 42                        # Random seed for reproducibility across runs

# Cross-Validation Settings
NUM_FOLDS = 10                   # Number of folds for k-fold cross-validation
                                 # Higher = more robust estimates but slower training

# Hyperparameter Optimization
HYPER_PARAMETER_ITER = 100       # Number of random parameter combinations to test
                                 # Higher = better optimization but longer runtime

# Model Selection Thresholds
CUTOFF_R2 = 0.3                  # Minimum R² score for model to proceed to optimization
                                 # Models below this threshold are discarded

# Feature Reduction Thresholds
CORRELATION_THRESHOLD = 0.95     # Remove features with ≥ this correlation
                                 # Keeps feature most correlated with target
                                 # Range: 0.0-1.0, typical values: 0.90-0.95

# Visualization Settings  
HISTOGRAM_BINS = 100             # Number of bins for residual histograms
OPTIMIZATION_CDF_THRESHOLD = 0.95  # Percentile threshold for CDF analysis (95%)

# Legacy parameter (not used in rank-based feature selection)
IMPORTANCE_MULTIPLYER = 10       # Retained for backwards compatibility

print("Feature Selection Configuration:")
print(f"  Validation Size: {VALIDATION_SIZE}")
print(f"  Random Seed: {SEED}")
print(f"  CV Folds: {NUM_FOLDS}")
print(f"  Correlation Threshold: {CORRELATION_THRESHOLD}")
print(f"  R² Cutoff: {CUTOFF_R2}")
print(f"  Hyperparameter Iterations: {HYPER_PARAMETER_ITER}")

# Log model configuration
logger.info("Feature selection and model training configuration loaded")
logger.debug(f"VALIDATION_SIZE: {VALIDATION_SIZE}")
logger.debug(f"SEED: {SEED}")
logger.debug(f"NUM_FOLDS: {NUM_FOLDS}")
logger.debug(f"CORRELATION_THRESHOLD: {CORRELATION_THRESHOLD}")
logger.debug(f"CUTOFF_R2: {CUTOFF_R2}")
logger.debug(f"HYPER_PARAMETER_ITER: {HYPER_PARAMETER_ITER}")

# %% [markdown]
# ## Define Base Learners and Hyperparameter Grids

# %%
# =============================================================================
# BASE LEARNER MODELS
# =============================================================================
# Collection of regression models to evaluate
# Each tuple: (abbreviated_name, model_instance)
# Models are configured for parallel processing where supported (n_jobs=-1)
base_learners = [
    ('CB', CatBoostRegressor(verbose=0)),                 # CatBoost - handles categorical features well
    ('LGB', LGBMRegressor(verbose=-1, n_jobs=-1)),       # LightGBM - fast gradient boosting
    ('XGB', XGBRegressor(n_jobs=-1)),                     # XGBoost - popular gradient boosting
    ('RF', RandomForestRegressor(n_jobs=-1)),             # Random Forest - ensemble of trees
    ('GBT', GradientBoostingRegressor()),                 # Sklearn Gradient Boosting
    ('KNN', KNeighborsRegressor()),                       # K-Nearest Neighbors - distance-based
    ('ETR', ExtraTreesRegressor(n_jobs=-1)),              # Extra Trees - randomized ensemble
    ('Bag', BaggingRegressor(n_jobs=-1))                 # Bagging - variance reduction ensemble
]

# =============================================================================
# HYPERPARAMETER DISTRIBUTIONS
# =============================================================================
# Parameter distributions for RandomizedSearchCV optimization
# Each model has tunable parameters that affect performance
# uniform(low, high): continuous distribution from low to low+high
# randint(low, high): integer distribution from low to high-1
param_dict = {
    'KNN': {
        'model__n_neighbors': randint(3, 30),
        'model__weights': ['uniform', 'distance'],
        'model__p': randint(1, 3)
    },
    'XGB': {
        'model__learning_rate': uniform(0.01, 0.29),
        'model__n_estimators': randint(100, 1000),
        'model__max_depth': randint(3, 10),
        'model__min_child_weight': randint(1, 10),
        'model__gamma': uniform(0, 0.5),
        'model__subsample': uniform(0.5, 0.5),           # [0.5, 1.0]
        'model__colsample_bytree': uniform(0.5, 0.5),    # [0.5, 1.0]
        'model__reg_alpha': uniform(0, 1.0),             # [0, 1.0]
        'model__reg_lambda': uniform(0.1, 0.9)           # [0.1, 1.0]
    },
    'ETR': {
        'model__n_estimators': randint(100, 500),
        'model__max_depth': [None, 3, 5, 7, 10, 15, 20],
        'model__min_samples_split': randint(2, 21),
        'model__min_samples_leaf': randint(1, 21),
        'model__max_features': ['sqrt', 'log2', None, 1.0],
        'model__bootstrap': [True, False]
    },
    'RF': {
        'model__n_estimators': randint(100, 1000),
        'model__max_depth': [None, 3, 5, 7, 10, 15, 20],
        'model__min_samples_split': randint(2, 21),
        'model__min_samples_leaf': randint(1, 21),
        'model__max_features': ['sqrt', 'log2', None, 1.0],
        'model__bootstrap': [True, False]
    },
    'GBT': {
        'model__n_estimators': randint(100, 500),
        'model__learning_rate': uniform(0.01, 0.29),
        'model__max_depth': randint(3, 11),
        'model__min_samples_split': randint(2, 21),
        'model__min_samples_leaf': randint(1, 21),
        'model__max_features': ['sqrt', 'log2', None, 1.0],
        'model__subsample': uniform(0.5, 0.5)            # [0.5, 1.0]
    },
    'Bag': {
        'model__n_estimators': randint(10, 200),
        'model__max_samples': uniform(0.1, 0.9),         # [0.1, 1.0]
        'model__max_features': uniform(0.1, 0.9),        # [0.1, 1.0]
        'model__bootstrap': [True, False],
        'model__bootstrap_features': [True, False]
    },
    'LGB': {
        'model__n_estimators': randint(50, 1000),
        'model__learning_rate': uniform(0.01, 0.29),
        'model__num_leaves': randint(20, 300),
        'model__max_depth': randint(-1, 21),
        'model__min_child_samples': randint(10, 31),
        'model__subsample': uniform(0.1, 0.9),           # [0.1, 1.0]
        'model__colsample_bytree': uniform(0.1, 0.9),    # [0.1, 1.0] - was causing error!
        'model__reg_alpha': uniform(0.0, 1.0),           # [0, 1.0]
        'model__reg_lambda': uniform(0.0, 1.0)           # [0, 1.0]
    },
    'CB': {
        'model__iterations': randint(50, 1000),
        'model__reg_alpha': uniform(0.0, 1.0),
        'model__depth': randint(3, 10),
        'model__l2_leaf_reg': randint(1, 10),
        'model__border_count': randint(50, 255),
        'model__subsample': uniform(0.5, 0.5)            # [0.5, 1.0]
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def combine_distributions(fixed_params: dict, conditional_params: list) -> list:
    """
    Combine fixed and conditional hyperparameter distributions.
    
    This function is used for models like CatBoost where certain parameters
    (e.g., bagging_temperature) are only valid when other parameters have
    specific values (e.g., bootstrap_type='Bayesian').
    
    Parameters
    ----------
    fixed_params : dict
        Base hyperparameter distributions that apply to all conditions.
    conditional_params : list of dict
        List of conditional parameter combinations. Each dict contains
        parameters that are conditional on specific settings.
    
    Returns
    -------
    list of dict
        Combined parameter distributions for RandomizedSearchCV.
        
    Examples
    --------
    >>> fixed = {'model__depth': randint(3, 10)}
    >>> conditional = [
    ...     {'model__bootstrap_type': ['Bayesian'], 'model__bagging_temperature': uniform(0, 1)},
    ...     {'model__bootstrap_type': ['Bernoulli']}
    ... ]
    >>> combined = combine_distributions(fixed, conditional)
    """
    combined_params = []
    
    # Iterate through each conditional parameter set
    for condition in conditional_params:
        # Create a copy of fixed params to avoid mutation
        current_params = fixed_params.copy()
        # Merge conditional parameters
        current_params.update(condition)
        combined_params.append(current_params)
    
    return combined_params


def get_feature_importance_models() -> list:
    """Get tree-based models for feature importance analysis.
    
    Returns a list of tuples containing model names and instances of
    tree-based regressors that provide feature_importances_ attribute.
    These models are used to identify the most predictive features.
    
    Returns
    -------
    list of tuple
        List of (model_name, model_instance) tuples.
        Models included: RandomForest, XGBoost, GradientBoosting, CatBoost.
        
    Notes
    -----
    All models are configured for parallel processing (n_jobs=-1).
    CatBoost is set to verbose=0 to suppress training output.
    These models are used for feature selection, not final predictions.
    """
    return [
        ('RF', RandomForestRegressor(n_jobs=-1)),      # Random Forest
        ('XGB', XGBRegressor(n_jobs=-1)),              # XGBoost
        ('GBT', GradientBoostingRegressor()),          # Gradient Boosting
        ('CB', CatBoostRegressor(verbose=0))           # CatBoost
    ]

# =============================================================================
# MODEL PROGRESSION TRACKING
# =============================================================================
# Dictionary to track R² scores at each stage of the modeling pipeline
# Structure: {model_name: {'initial_r2': ..., 'baseline_cv_r2': ..., 'optimized_cv_r2': ..., 'validation_r2': ..., 'stacking_r2': ...}}
model_progression = {}

print(f"✅ Configured {len(base_learners)} base learner models")
print(f"✅ Defined hyperparameter grids for {len(param_dict)} models")
print(f"✅ Initialized model progression tracker")

# Log model configuration
logger.info(f"Configured {len(base_learners)} base learners: {[name for name, _ in base_learners]}")
logger.debug(f"Hyperparameter grids defined for: {list(param_dict.keys())}")

# %% [markdown]
# ## Step 1: Remove Highly Correlated Features
# 
# Removes features with correlation ≥0.95 to reduce multicollinearity. When two features are highly correlated, the one with higher correlation to the target variable is retained.

# %%
print("🔍 CORRELATION-BASED FEATURE REDUCTION")
print("=" * 60)

# Prepare working dataset with proper data types
working_data = cleaned_data.copy()

# Check for any remaining string columns
string_cols = working_data.select_dtypes(include=['object']).columns.tolist()
if string_cols:
    print(f"\n⚠️  Warning: Found {len(string_cols)} string columns that need to be removed:")
    for col in string_cols[:5]:  # Show first 5
        print(f"  - {col}")
    if len(string_cols) > 5:
        print(f"  ... and {len(string_cols) - 5} more")
    
    # Remove all string columns (they should have been encoded already)
    print(f"\n🗑️  Removing {len(string_cols)} string columns...")
    working_data = working_data.drop(columns=string_cols)

# Ensure target variable is numeric (required for correlation analysis)
working_data[dependent_var] = working_data[dependent_var].astype('float64')

print(f"\nInitial shape: {working_data.shape}")
print(f"Data types: {working_data.dtypes.value_counts().to_dict()}")

# =============================================================================
# Save dataset before feature reduction
# =============================================================================
# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Generate timestamp for unique filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Helper function to save plots
def save_plot(name, timestamp=timestamp):
    """Save current plot to data folder with timestamp."""
    filename = f'data/{name}_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  💾 Saved: {filename}")

# Save the fully preprocessed data before feature selection
before_feature_reduction_path = f'data/final_modeling_data_before_feature_reduction_{timestamp}.csv'
working_data.to_csv(before_feature_reduction_path, index=False)
print(f"\n💾 Saved pre-feature-selection data to: {before_feature_reduction_path}")
print(f"   Shape: {working_data.shape}")

# Calculate pairwise correlation matrix for all numeric features
# Pearson correlation coefficient ranges from -1 to 1
print(f"\n🔢 Calculating correlation matrix for {len(working_data.columns)} numeric columns...")
corr_matrix = working_data.corr()

# Calculate absolute correlation between each feature and target variable
# We use absolute values because both positive and negative correlations indicate relationship strength
target_corr = corr_matrix[dependent_var].abs().drop(dependent_var)

# =============================================================================
# Identify highly correlated features to remove
# =============================================================================
# Strategy: For each pair of highly correlated features, keep the one that
# has stronger correlation with the target variable and drop the other.
# This reduces multicollinearity while preserving predictive information.
to_drop = set()

# Nested loop to check all pairwise correlations
for i in range(len(corr_matrix.columns)):
    feature_i = corr_matrix.columns[i]
    # Skip if already marked for removal or is the target variable
    if feature_i in to_drop or feature_i == dependent_var:
        continue
    
    # Check correlations with all subsequent features (avoid checking same pair twice)
    for j in range(i + 1, len(corr_matrix.columns)):
        feature_j = corr_matrix.columns[j]
        # Skip if already marked for removal or is the target variable
        if feature_j in to_drop or feature_j == dependent_var:
            continue
        
        # Check if correlation exceeds threshold (using absolute value)
        if abs(corr_matrix.iloc[i, j]) >= CORRELATION_THRESHOLD:
            # Keep the feature more correlated with target, drop the other
            if target_corr[feature_i] > target_corr[feature_j]:
                to_drop.add(feature_j)
                print(f"  Dropping {feature_j} (corr={corr_matrix.iloc[i, j]:.3f} with {feature_i})")
            else:
                to_drop.add(feature_i)
                print(f"  Dropping {feature_i} (corr={corr_matrix.iloc[i, j]:.3f} with {feature_j})")
                break  # feature_i is dropped, no need to check its other correlations

print("=" * 60)

# Remove the identified correlated features from dataset
working_data_reduced = working_data.drop(columns=to_drop)
print(f"\n✅ Removed {len(to_drop)} highly correlated features")
print(f"Final shape: {working_data_reduced.shape}")


# %% [markdown]
# ## Step 2: Split Data for Training
# 
# Create train/validation split for feature selection and model training.

# %%
# =============================================================================
# Prepare features and target for modeling
# =============================================================================
# Separate independent variables (features) from dependent variable (target)
independent = working_data_reduced.drop([dependent_var], axis=1)
dependent = working_data_reduced[dependent_var]
independent_columns = independent.columns
features = np.array(independent_columns)  # Convert to array for easier indexing

# Create train/validation split
# Training set: used for model training and cross-validation
# Validation set: held-out data for final model evaluation
X_train, X_validation, Y_train, Y_validation = train_test_split(
    independent, dependent, 
    test_size=VALIDATION_SIZE,  # Typically 0.2 (20% validation, 80% training) 
    random_state=SEED
)

print("📊 DATA SPLIT SUMMARY")
print("=" * 60)
print(f"Total samples: {len(working_data_reduced)}")
print(f"Training samples: {len(X_train)} ({(1-VALIDATION_SIZE)*100:.0f}%)")
print(f"Validation samples: {len(X_validation)} ({VALIDATION_SIZE*100:.0f}%)")
print(f"Number of features: {len(features)}")
print(f"Target variable: {dependent_var}")
print("=" * 60)

# %% [markdown]
# ## Step 3: Feature Importance-Based Selection
# 
# ### Methodology
# This step uses an ensemble approach to identify the most predictive features:
# 
# **1. Multiple Model Perspectives**: We train 4 different tree-based models (RF, XGB, GBT, CB) because each algorithm may identify different important features based on its internal mechanics.
# 
# **2. Rank Aggregation**: Instead of using arbitrary thresholds, we:
#    - Rank features within each model (1 = most important)
#    - Average the ranks across all models
#    - Select features with the best average ranks
# 
# **3. Why This Works**:
#    - **Reduces bias**: No single model dictates feature selection
#    - **Captures consensus**: Features important across multiple models are more likely to be truly predictive
#    - **Robust to outliers**: Rank-based approach is less sensitive to extreme importance values
#    - **Interpretable**: Mean rank is easy to understand and explain
# 
# **4. Selection Criterion**: We use cumulative importance analysis to keep features that account for 95% of the total importance. This ensures we:
#    - Retain all meaningfully predictive features
#    - Remove low-importance noise features
#    - Adapt selection to the actual importance distribution (not a fixed percentage)
#    - Keep a minimum of 20% of features as a safety net

# %%
print("🎯 FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

num_features = len(features)
feature_importance_models = get_feature_importance_models()

# Store ranks from each model
feature_ranks = {feature: [] for feature in features}
all_importances = {feature: [] for feature in features}

print(f"Analyzing {num_features} features across {len(feature_importance_models)} models...\n")

# Train each model and collect feature ranks
for name, mod in feature_importance_models:
    print(f"Training {name}...")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', mod)
    ])
    
    pipeline.fit(X_train, Y_train)
    
    # Extract and normalize feature importances from trained model
    if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
        importance = pipeline.named_steps['model'].feature_importances_
        # Normalize importances to sum to 1.0 for comparability across models
        normalized_importances = importance / np.sum(importance)
    else:
        raise ValueError(f"Model {name} does not have feature importances")
    
    # Calculate ranks for each feature (1 = most important, N = least important)
    # np.argsort(-normalized_importances) gives indices in descending order
    # np.argsort(argsort) converts indices to ranks
    ranks = np.argsort(np.argsort(-normalized_importances)) + 1
    
    # Store ranks and importances for aggregation across models
    for i, feature in enumerate(features):
        feature_ranks[feature].append(ranks[i])
        all_importances[feature].append(normalized_importances[i])
    
    # Visualize top features for this model
    top_n = min(20, len(features))
    top_indices = np.argsort(normalized_importances)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_indices)), normalized_importances[top_indices], color='steelblue', alpha=0.8)
    plt.yticks(range(len(top_indices)), [features[i] for i in top_indices], fontsize=9)
    plt.xlabel('Normalized Importance', fontsize=11)
    plt.title(f'Top {top_n} Features - {name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"  ✓ {name} complete\n")

# =============================================================================
# Aggregate feature statistics across all models
# =============================================================================
# Calculate mean rank and importance for each feature
# Lower mean rank = more consistently important across models
feature_statistics = []
for feature in features:
    # mean_rank: Average rank across 4 models (1 = most important, N = least important)
    #   - Example: If feature ranks [2, 1, 3, 2] across models, mean_rank = 2.0
    #   - LOWER IS BETTER
    mean_rank = np.mean(feature_ranks[feature])
    
    # mean_importance: Average normalized importance score (0.0 to 1.0)
    #   - Represents percentage of total predictive power
    #   - Example: 0.15 means this feature contributes 15% to predictions
    #   - HIGHER IS BETTER
    mean_importance = np.mean(all_importances[feature])
    
    # std_rank: Standard deviation of ranks across 4 models
    #   - Measures consensus: Low std (<2) = models agree, High std (>5) = models disagree
    #   - Example: std_rank=0.5 means all models ranked it similarly
    #   - LOWER IS BETTER (indicates agreement)
    std_rank = np.std(feature_ranks[feature])
    
    feature_statistics.append({
        'feature': feature,
        'mean_rank': mean_rank,
        'std_rank': std_rank,
        'mean_importance': mean_importance
    })

# Sort features by mean importance (higher = better)
feature_statistics = sorted(feature_statistics, key=lambda x: x['mean_importance'], reverse=True)

# Calculate cumulative importance to find natural cutoff
total_importance = sum([f['mean_importance'] for f in feature_statistics])
cumulative_importance = 0
cumulative_threshold = 0.95  # Keep features that account for 95% of total importance

# Select features until we reach cumulative threshold
selected_features = []
for feat_stat in feature_statistics:
    cumulative_importance += feat_stat['mean_importance']
    selected_features.append(feat_stat)
    
    if cumulative_importance / total_importance >= cumulative_threshold:
        break

# Ensure we keep at least 20% of features (safety net for sparse importance distributions)
min_features = max(int(np.ceil(num_features * 0.20)), 10)
if len(selected_features) < min_features:
    selected_features = feature_statistics[:min_features]
    print(f"⚠️  Note: Using minimum threshold of {min_features} features")

feature_names = [f['feature'] for f in selected_features]

print("=" * 60)
print(f"✅ Feature Selection Complete")
print(f"Initial features: {num_features}")
print(f"Selected features: {len(feature_names)} ({len(feature_names)/num_features*100:.1f}%)")
print(f"Cumulative importance captured: {sum([f['mean_importance'] for f in selected_features]) / total_importance * 100:.1f}%")
print(f"Removed features: {num_features - len(feature_names)}")
print("=" * 60)

# Create summary DataFrame with detailed column calculations
feature_df = pd.DataFrame(selected_features)

# Calculate rank percentile: Higher is better (100% = rank 1, 0% = worst rank)
feature_df['rank_percentile'] = (1 - feature_df['mean_rank'] / num_features) * 100

# Reorder columns for logical presentation
feature_df = feature_df[['feature', 'mean_rank', 'std_rank', 'mean_importance', 'rank_percentile']]

print("\n📋 FEATURE IMPORTANCE COLUMNS EXPLAINED:")
print("=" * 80)
print("  • feature:          Feature name from dataset")
print("  • mean_rank:        Average rank across 4 models (LOWER = MORE IMPORTANT)")
print("                      Range: 1 to N, where 1 = most important feature")
print("  • std_rank:         Rank consistency (LOWER = MORE CONSENSUS)")
print("                      Low std (<2) = models agree, High std (>5) = models disagree")
print("  • mean_importance:  Average normalized importance (HIGHER = MORE IMPORTANT)")
print("                      Range: 0.0 to 1.0, sum of all features ≈ 1.0")
print("  • rank_percentile:  Percentile rank (HIGHER = BETTER)")
print("                      100% = best feature, 50% = median, 0% = worst")
print("=" * 80)

# Save to CSV
feature_df.to_csv('feature_importance_scores.csv', index=False)
print("\n📁 Saved feature statistics to 'feature_importance_scores.csv'")

# Display top 10 features
print("\n📊 Top 10 Features by Mean Rank:")
print(feature_df.head(10).to_string(index=False))

# %% [markdown]
# ## Step 4: Update Dataset with Selected Features
# 
# Recreate train/validation split with reduced feature set.

# %%
# Create final dataset with selected features
independent = working_data_reduced[feature_names]
dependent = working_data_reduced[dependent_var]
features = feature_names

# =============================================================================
# Save dataset after feature selection
# =============================================================================
# Reconstruct full dataset with selected features and target
final_modeling_data = working_data_reduced[feature_names + [dependent_var]]

# Save to data directory with timestamp (reuse timestamp from above for consistency)
post_feature_selection_path = f'data/final_modeling_data_post_feature_selection_{timestamp}.csv'
final_modeling_data.to_csv(post_feature_selection_path, index=False)
print(f"\n💾 Saved post-feature-selection data to: {post_feature_selection_path}")
print(f"   Shape: {final_modeling_data.shape}")
print(f"   Features: {len(feature_names)} selected features + 1 target variable\n")

# Re-split with selected features
X_train, X_validation, Y_train, Y_validation = train_test_split(
    independent, dependent,
    test_size=VALIDATION_SIZE,
    random_state=SEED
)

print("📊 UPDATED DATASET SUMMARY")
print("=" * 60)
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_validation)}")
print(f"Selected features: {len(feature_names)}")
print("\nTop 10 features by mean rank:")
for i, feat_info in enumerate(selected_features[:10], 1):
    print(f"  {i:2d}. {feat_info['feature']}: rank={feat_info['mean_rank']:.2f}, importance={feat_info['mean_importance']:.4f}")
print("=" * 60)

# Log feature selection results
logger.info(f"Feature selection complete: {len(feature_names)} features selected")
logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_validation)}")
logger.debug(f"Top 5 features: {[f['feature'] for f in selected_features[:5]]}")

# %% [markdown]
# ## Step 5: Initial Model Screening
# 
# Test all base learners and filter by R² threshold.

# %%
print("🔍 INITIAL MODEL SCREENING")
print("=" * 60)
print(f"R² cutoff threshold: {CUTOFF_R2}\n")

cutoff_models = []

for est_name, est in base_learners:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', est)
    ])
    
    pipeline.fit(X_train, Y_train)
    test_score = pipeline.score(X_validation, Y_validation)
    
    # Track initial R² in model_progression
    if est_name not in model_progression:
        model_progression[est_name] = {}
    model_progression[est_name]['initial_r2'] = test_score
    
    status = "✅ PASS" if test_score >= CUTOFF_R2 else "❌ FAIL"
    print(f"{est_name:8s}: R² = {test_score:.4f} {status}")
    
    if test_score >= CUTOFF_R2:
        cutoff_models.append((est_name, est))

print("\n" + "=" * 60)
print(f"Models passing threshold: {len(cutoff_models)}/{len(base_learners)}")
print(f"Selected models: {[name for name, _ in cutoff_models]}")
print("=" * 60)

# Log initial screening results
logger.info(f"Initial screening: {len(cutoff_models)}/{len(base_learners)} models passed R² threshold ({CUTOFF_R2})")
logger.info(f"Selected models: {[name for name, _ in cutoff_models]}")

# %% [markdown]
# ## Step 6: Hyperparameter Optimization
# 
# ### Methodology
# Optimize hyperparameters for models that passed the R² threshold using RandomizedSearchCV:
# 
# **1. Randomized Search Strategy**: Instead of exhaustive grid search, we randomly sample from parameter distributions. This is more efficient and often finds good solutions faster.
# 
# **2. Cross-Validation**: Each parameter combination is evaluated using 10-fold cross-validation to ensure robust performance estimates and avoid overfitting.
# 
# **3. CatBoost Special Handling**: CatBoost's `bootstrap_type` parameter requires conditional logic:
#    - When `bootstrap_type='Bayesian'`, we can set `bagging_temperature`
#    - For other bootstrap types, `bagging_temperature` must be excluded
#    
# **4. Model Re-evaluation**: After optimization, models are checked again against the R² threshold. Any that fail to improve are removed.
# 
# **5. Model Update**: The base learner instances are updated with their optimized hyperparameters for final evaluation.

# %%
print("⚙️ HYPERPARAMETER OPTIMIZATION")
print("=" * 60)
print(f"Optimizing {len(cutoff_models)} models with {HYPER_PARAMETER_ITER} iterations each")
print(f"Cross-validation folds: {NUM_FOLDS}\n")

# Define R² scorer for optimization objective
# Higher R² = better model performance (closer to 1.0)
scorer = make_scorer(r2_score)

# Dictionary to store optimized models and their performance metrics
best_models = {}

# Store baseline performance for comparison
baseline_scores = {}

# =============================================================================
# Calculate baseline scores first (with default hyperparameters)
# =============================================================================
print("📊 Calculating baseline scores with default parameters...")
for model_name, model in cutoff_models:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', clone(model))  # Use clone to avoid modifying original
    ])
    # Use cross-validation for fair comparison
    cv_scores = cross_val_score(pipeline, X_train, Y_train, cv=NUM_FOLDS, scoring=scorer)
    baseline_scores[model_name] = cv_scores.mean()
    
    # Track baseline CV R² in model_progression
    if model_name not in model_progression:
        model_progression[model_name] = {}
    model_progression[model_name]['baseline_cv_r2'] = baseline_scores[model_name]
    
    print(f"  {model_name}: Baseline CV R² = {baseline_scores[model_name]:.4f}")
print()

# =============================================================================
# Optimize hyperparameters for each model that passed initial screening
# =============================================================================
for model_name, model in cutoff_models:
    print(f"{'='*60}")
    print(f"Starting {model_name}...")
    print(f"{'='*60}")
    
    # Skip models without defined hyperparameter grids
    if model_name not in param_dict:
        print(f"⚠️  No hyperparameters defined for {model_name}, skipping.\n")
        continue
    
    # Create pipeline with feature scaling (required for many algorithms)
    # StandardScaler: Transforms features to have mean=0 and variance=1
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Handle CatBoost conditional parameters
    if model_name == 'CB':
        # Fixed parameters (always included)
        fixed_params = {
            'model__iterations': randint(50, 1000),
            'model__learning_rate': uniform(0.01, 0.3),
            'model__depth': randint(3, 10),
            'model__l2_leaf_reg': randint(1, 10),
            'model__border_count': randint(50, 255)
        }
        
        # Conditional parameters based on bootstrap_type
        conditional_params = [
            {
                'model__bootstrap_type': ['Bayesian'],
                'model__bagging_temperature': uniform(0.0, 0.9)
            },
            {
                'model__bootstrap_type': ['Bernoulli', 'MVS', 'No']
                # No bagging_temperature for these types
            }
        ]
        
        # Combine into list of parameter sets
        param_distributions = combine_distributions(fixed_params, conditional_params)
    else:
        param_distributions = param_dict[model_name]
    
    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=HYPER_PARAMETER_ITER,
        scoring=scorer,
        cv=NUM_FOLDS,
        verbose=2,
        random_state=SEED,
        n_jobs=-1,
        error_score='raise'
    )
    
    # Fit and find best hyperparameters
    print(f"Training with {HYPER_PARAMETER_ITER} random parameter combinations...")
    random_search.fit(X_train, Y_train)
    
    # Compare optimized vs baseline performance
    optimized_score = random_search.best_score_
    baseline_score = baseline_scores.get(model_name, 0)
    
    # Only use optimized model if it's actually better than baseline
    if optimized_score >= baseline_score:
        # Store optimized results
        best_models[model_name] = {
            'best_estimator': random_search.best_estimator_,
            'best_score': optimized_score,
            'best_params': random_search.best_params_,
            'used_optimization': True
        }
        # Track optimized CV R² in model_progression
        model_progression[model_name]['optimized_cv_r2'] = optimized_score
        
        improvement = optimized_score - baseline_score
        print(f"\n✅ {model_name} optimization improved performance")
        print(f"   Baseline CV R²: {baseline_score:.4f}")
        print(f"   Optimized CV R²: {optimized_score:.4f}")
        print(f"   Improvement: +{improvement:.4f}")
    else:
        # Keep default model - optimization made it worse
        default_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', clone(model))
        ])
        default_pipeline.fit(X_train, Y_train)
        best_models[model_name] = {
            'best_estimator': default_pipeline,
            'best_score': baseline_score,
            'best_params': model.get_params(),
            'used_optimization': False
        }
        # Track that we kept baseline (optimized was worse)
        model_progression[model_name]['optimized_cv_r2'] = baseline_score  # Used baseline
        
        degradation = baseline_score - optimized_score
        print(f"\n⚠️  {model_name} optimization degraded performance - keeping defaults")
        print(f"   Baseline CV R²: {baseline_score:.4f}")
        print(f"   Optimized CV R²: {optimized_score:.4f}")
        print(f"   Avoided degradation: {degradation:.4f}")
    
    # Check if model meets performance threshold
    final_score = best_models[model_name]['best_score']
    meets_threshold = final_score >= CUTOFF_R2
    status = "✅ Meets threshold" if meets_threshold else f"⚠️  Below threshold ({CUTOFF_R2})"
    print(f"   Status: {status}\n")

print("=" * 60)
print("OPTIMIZATION SUMMARY")
print("=" * 60)
optimized_count = 0
default_count = 0
for model_name, model_info in best_models.items():
    status = "✅" if model_info['best_score'] >= CUTOFF_R2 else "❌"
    opt_status = "optimized" if model_info.get('used_optimization', True) else "defaults"
    print(f"{model_name:12s}: R² = {model_info['best_score']:.4f} {status} ({opt_status})")
    # Log each model's optimization result
    logger.info(f"Hyperparameter optimization - {model_name}: R² = {model_info['best_score']:.4f} ({opt_status})")
    if model_info.get('used_optimization', True):
        optimized_count += 1
    else:
        default_count += 1
print("=" * 60)
print(f"Models using optimized params: {optimized_count}")
print(f"Models using default params: {default_count}")
print("=" * 60)

logger.info(f"Optimization complete: {optimized_count} optimized, {default_count} using defaults")

# %% [markdown]
# ### Update Model Instances with Best Parameters
# 
# Replace the original model instances with optimized versions.

# %%
print("🔄 UPDATING MODEL INSTANCES")
print("=" * 60)

# Update cutoff_models with optimized parameters
updated_models = []

for model_name, model_info in best_models.items():
    best_estimator = model_info['best_estimator']
    best_params = best_estimator.named_steps['model'].get_params()
    best_score = model_info['best_score']
    
    # Check if model still meets threshold
    if best_score < CUTOFF_R2:
        print(f"❌ {model_name}: R² = {best_score:.4f} (below threshold, removing)")
        continue
    
    # Find the model in cutoff_models and update it
    for i, (est_name, est) in enumerate(cutoff_models):
        if est_name == model_name:
            # Get model class and create new instance with best params
            model_class = est.__class__
            updated_model = model_class(**best_params)
            updated_models.append((est_name, updated_model))
            print(f"✅ {model_name}: Updated with optimized parameters (R² = {best_score:.4f})")
            break

# Replace cutoff_models with updated models
cutoff_models = updated_models

print("=" * 60)
print(f"Final model count: {len(cutoff_models)}")
print(f"Models: {[name for name, _ in cutoff_models]}")
print("=" * 60)

# %% [markdown]
# ## Step 7: Model Evaluation & Visualization
# 
# ### Methodology
# Comprehensive evaluation of optimized models:
# 
# **1. Cross-Validation Performance**: Use 10-fold CV on training data to get robust performance estimates with mean and standard deviation of R² scores.
# 
# **2. Validation Set Predictions**: Test on held-out validation data to assess generalization performance.
# 
# **3. Visualization Suite**:
#    - **Predicted vs Actual**: Scatter plot showing prediction accuracy with perfect prediction line
#    - **CDF/PDF of Residuals**: Cumulative and probability distributions showing error characteristics and 95% threshold
#    - **Residual Histogram**: Distribution of prediction errors with KDE overlay
# 
# **4. Model Persistence**: Save each optimized pipeline as a pickle file for production use or future analysis.

# %%
print("📊 MODEL EVALUATION & VISUALIZATION")
print("=" * 60)

# Set figure size for plots
figSize = [10, 10]

# Generate timestamp for plot filenames
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def save_plot(name, ts=timestamp):
    """Save current matplotlib plot to data folder.
    
    Parameters
    ----------
    name : str
        Base name for the plot file (without extension).
    ts : str, optional
        Timestamp string to append to filename. Defaults to current timestamp.
    
    Returns
    -------
    None
        Saves plot as PNG to data/{name}_{ts}.png
    """
    filename = f'data/{name}_{ts}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')

for est_name, est in cutoff_models:
    print(f"\n{'='*60}")
    print(f"📈 Evaluating {est_name}")
    print(f"{'='*60}")
    
    # Get the optimized model from best_models
    if est_name in best_models:
        pipeline = best_models[est_name]['best_estimator']
    else:
        # Fallback: create new pipeline with default model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', est)
        ])
    
    # Cross-validation on training data
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    cv_results = cross_val_score(pipeline, X_train, Y_train, cv=kfold, scoring='r2')
    print(f"CV R²: {cv_results.mean():.4f} (±{cv_results.std():.4f})")
    print(f"   CV Range: [{cv_results.min():.4f}, {cv_results.max():.4f}]")
    
    # Fit on full training data
    pipeline.fit(X_train, Y_train)
    
    # Predictions on validation set
    predictions = pipeline.predict(X_validation)
    
    # Calculate validation R²
    r2_validation = r2_score(Y_validation, predictions)
    print(f"✅ Validation R²: {r2_validation:.4f}")
    
    # Track validation R² in model_progression
    if est_name not in model_progression:
        model_progression[est_name] = {}
    model_progression[est_name]['validation_r2'] = r2_validation
    
    # === PLOT 1: Predicted vs Actual ===
    plt.figure(figsize=figSize)
    plt.scatter(Y_validation, predictions, alpha=0.5, edgecolors='k', s=50)
    plt.plot([Y_validation.min(), Y_validation.max()], 
             [Y_validation.min(), Y_validation.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'{est_name} - Predicted vs Actual (R² = {r2_validation:.4f})', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(f'{est_name}_predicted_vs_actual', timestamp)
    plt.show()
    
    # === PLOT 2: CDF and PDF of Absolute Residuals ===
    residuals = np.abs(Y_validation.values - predictions)
    count, bins_count = np.histogram(residuals, bins=HISTOGRAM_BINS)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    
    # Find 95% CDF threshold using exact percentile calculation
    threshold_95 = np.percentile(residuals, OPTIMIZATION_CDF_THRESHOLD * 100)
    print(f"📈 95% of residuals are below: {threshold_95:.6g}")
    
    plt.figure(figsize=figSize)
    plt.plot(bins_count[1:], pdf, color='red', linewidth=2, label='PDF', alpha=0.7)
    plt.plot(bins_count[1:], cdf, color='blue', linewidth=2, label='CDF', alpha=0.7)
    plt.axhline(y=OPTIMIZATION_CDF_THRESHOLD, color='green', linestyle='--', 
                linewidth=1.5, label=f'{OPTIMIZATION_CDF_THRESHOLD:.0%} Threshold')
    plt.axvline(x=threshold_95, color='green', linestyle='--', linewidth=1.5)
    plt.xlabel('Absolute Residual', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'{est_name} - CDF and PDF of Absolute Residuals', 
              fontsize=14, fontweight='bold')
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    save_plot(f'{est_name}_cdf_pdf_residuals', timestamp)
    plt.show()
    
    # === PLOT 3: Residual Distribution ===
    residuals_signed = Y_validation.values - predictions
    
    plt.figure(figsize=figSize)
    sns.histplot(residuals_signed, bins=HISTOGRAM_BINS, kde=True, color='steelblue', alpha=0.6)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.xlabel('Residual (Actual - Predicted)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'{est_name} - Residual Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(f'{est_name}_residual_distribution', timestamp)
    plt.show()
    
    # Save model pipeline
    file_name = f'pipeline_model_{est_name}.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(pipeline, file)
    print(f"💾 Saved model to '{file_name}'")
    
    # Log evaluation results
    logger.info(f"Model {est_name}: Validation R² = {r2_validation:.4f}, CV R² = {cv_results.mean():.4f} (±{cv_results.std():.4f})")

print("\n" + "=" * 60)
print("✅ ALL MODELS EVALUATED AND SAVED")
print("=" * 60)

# %% [markdown]
# ## Stacking Analysis
# 
# Stacking ensemble analysis will run automatically to combine multiple models for optimal performance.

# %%
# Stacking analysis is always enabled
if 'RUN_STACKING_ANALYSIS' not in globals():
    RUN_STACKING_ANALYSIS = True

print("✅ Stacking ensemble analysis enabled - proceeding with optimized model combination...")

# %% [markdown]
# ## Step 8: Stacking Ensemble
# 
# ### Methodology
# 
# Creates a stacking ensemble to combine multiple models for improved performance.
# 
# **1. Stacking Architecture**: 
#    - **Base learners**: All optimized models that passed R² threshold generate predictions
#    - **Meta-learner**: A final model learns optimal weights for combining base predictions
#    
# **2. Meta-Learner Selection**: Tests all base learners plus LinearRegression as potential meta-learners.
# 
# **3. Cross-Validation Strategy**: Uses 10-fold CV within StackingRegressor to prevent information leakage.
# 
# **4. Performance Evaluation**: Each meta-learner is evaluated on held-out validation data.
# 
# **5. Model Persistence**: All stacking pipelines are saved as pickle files for production use.

# %%
# Stacking analysis - always runs for optimal performance
print("🏗️ STACKING ENSEMBLE CONSTRUCTION (OPTIMIZED)")
print("=" * 60)

# Define column renaming function (must be defined at module level for pickling)
def rename_columns_for_stacking(X):
    """Rename DataFrame columns to generic names for stacking.
    
    Required for pickling compatibility when saving stacking pipelines.
    Converts feature names to generic 'col_0', 'col_1', etc.
    
    Parameters
    ----------
    X : pd.DataFrame or array-like
        Input data with column names to rename.
    
    Returns
    -------
    pd.DataFrame or array-like
        DataFrame with columns renamed to 'col_{i}' format if input
        has columns attribute, otherwise returns input unchanged.
    """
    if hasattr(X, 'columns'):
        return X.rename(columns={col: f'col_{i}' for i, col in enumerate(X.columns)})
    return X

# Create column renaming transformer (uses named function instead of lambda)
rename_transformer = FunctionTransformer(
    rename_columns_for_stacking,
    validate=False
)

# Extract model instances from cutoff_models for stacking
stack_learners = [(name, model) for name, model in cutoff_models]

print(f"Base learners: {[name for name, _ in stack_learners]}")
print(f"Number of base learners: {len(stack_learners)}\n")

# =============================================================================
# OPTIMIZATION: Generate base learner predictions ONCE using cross-validation
# This eliminates redundant retraining for each meta-learner
# =============================================================================

print("📊 STEP 1: Generating base learner predictions (ONE TIME ONLY)")
print("=" * 60)

# Define cross-validation strategy
cv_strategy = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

# Store out-of-fold predictions for training meta-learners
base_predictions_train = np.zeros((len(X_train), len(stack_learners)))
base_predictions_val = np.zeros((len(X_validation), len(stack_learners)))

# Generate predictions from each base learner using cross-validation
for i, (name, model) in enumerate(stack_learners):
    print(f"\n  Processing {name}...")
    
    # Create pipeline for this base learner
    base_pipeline = Pipeline([
        ('rename_columns', rename_transformer),
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Generate out-of-fold predictions for training data
    oof_preds = np.zeros(len(X_train))
    for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train = Y_train.iloc[train_idx]
        
        # Fit and predict
        base_pipeline.fit(X_fold_train, y_fold_train)
        oof_preds[val_idx] = base_pipeline.predict(X_fold_val)
    
    base_predictions_train[:, i] = oof_preds
    
    # Train on full training data and predict validation set
    base_pipeline.fit(X_train, Y_train)
    base_predictions_val[:, i] = base_pipeline.predict(X_validation)
    
    # Calculate R² for this base learner
    base_r2 = r2_score(Y_train, oof_preds)
    print(f"    Out-of-fold R²: {base_r2:.4f}")

print("\n✅ Base learner predictions generated!")
print(f"   Training predictions shape: {base_predictions_train.shape}")
print(f"   Validation predictions shape: {base_predictions_val.shape}\n")

# =============================================================================
# STEP 2: Train only meta-learners on pre-generated predictions
# This is MUCH faster than retraining base learners each time
# =============================================================================

print("🎯 STEP 2: Training meta-learners (FAST!)")
print("=" * 60)

# Store meta-learner scores
meta_learner_scores = {}

# Test all base learners + LinearRegression as meta-learners
meta_learners = base_learners + [('LinearRegression', LinearRegression())]

print(f"Testing {len(meta_learners)} potential meta-learners...\n")
print("=" * 60)

for meta_name, meta_model in meta_learners:
    print(f"\n{meta_name} as meta-learner:")
    
    # Train meta-learner on base predictions
    meta_model_instance = clone(meta_model)
    meta_model_instance.fit(base_predictions_train, Y_train)
    
    # Predictions on validation set - THIS is the real performance metric
    predictions = meta_model_instance.predict(base_predictions_val)
    validation_score = r2_score(Y_validation, predictions)
    
    # Store VALIDATION score, not training score
    meta_learner_scores[meta_name] = validation_score
    
    print(f"  Validation R²: {validation_score:.4f}")
    
    # Create full stacking pipeline for saving (for production use)
    stacking_pipeline = Pipeline([
        ('rename_columns', rename_transformer),
        ('scaler', StandardScaler()),
        ('stacked_regressor', StackingRegressor(
            estimators=stack_learners,
            final_estimator=clone(meta_model),
            cv=cv_strategy,
            n_jobs=-1
        ))
    ])
    
    # Fit the full pipeline on training data for saving
    stacking_pipeline.fit(X_train, Y_train)
    validation_score = r2_score(Y_validation, stacking_pipeline.predict(X_validation))
    
    # Track stacking R² in model_progression for each meta-learner
    stacking_model_name = f"Stacking_{meta_name}"
    if stacking_model_name not in model_progression:
        model_progression[stacking_model_name] = {}
    model_progression[stacking_model_name]['stacking_r2'] = validation_score
    model_progression[stacking_model_name]['validation_r2'] = validation_score
    model_progression[stacking_model_name]['is_stacking'] = True
    
    # === PLOT 1: Predicted vs Actual ===
    plt.figure(figsize=figSize)
    plt.scatter(Y_validation, predictions, alpha=0.5, edgecolors='k', s=50, label='Predictions')
    plt.plot([Y_validation.min(), Y_validation.max()], 
             [Y_validation.min(), Y_validation.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'Stacking ({meta_name}) - Predicted vs Actual (R² = {validation_score:.4f})', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(f'stacking_{meta_name}_predicted_vs_actual', timestamp)
    plt.show()
    
    # === PLOT 2: CDF and PDF of Absolute Residuals ===
    residuals_abs = np.abs(Y_validation.values - predictions)
    count, bins_count = np.histogram(residuals_abs, bins=HISTOGRAM_BINS)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    
    # Find 95% CDF threshold using exact percentile calculation
    threshold_95 = np.percentile(residuals_abs, OPTIMIZATION_CDF_THRESHOLD * 100)
    print(f"📈 95% of residuals are below: {threshold_95:.6g}\n")
    
    plt.figure(figsize=figSize)
    plt.plot(bins_count[1:], pdf, color='red', linewidth=2, label='PDF', alpha=0.7)
    plt.plot(bins_count[1:], cdf, color='blue', linewidth=2, label='CDF', alpha=0.7)
    plt.axhline(y=OPTIMIZATION_CDF_THRESHOLD, color='green', linestyle='--', 
                linewidth=1.5, label=f'{OPTIMIZATION_CDF_THRESHOLD:.0%} Threshold')
    plt.axvline(x=threshold_95, color='green', linestyle='--', linewidth=1.5)
    plt.xlabel('Absolute Residual', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Stacking ({meta_name}) - CDF and PDF of Absolute Residuals', 
              fontsize=14, fontweight='bold')
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    save_plot(f'stacking_{meta_name}_cdf_pdf_residuals', timestamp)
    plt.show()
    
    # === PLOT 3: Residual Distribution ===
    residuals_signed = Y_validation.values - predictions
    
    plt.figure(figsize=figSize)
    sns.histplot(residuals_signed, bins=HISTOGRAM_BINS, kde=True, color='steelblue', alpha=0.6)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.xlabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Stacking ({meta_name}) - Residual Distribution', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    save_plot(f'stacking_{meta_name}_residual_distribution', timestamp)
    plt.show()
    
    # Save stacking pipeline
    file_name = f'stacking_pipeline_{meta_name}.pkl'
    joblib.dump(stacking_pipeline, file_name)

print("\n" + "=" * 60)
print("STACKING ENSEMBLE SUMMARY")
print("=" * 60)
print("\nMeta-Learner Validation R² Scores:")
for meta_name, score in sorted(meta_learner_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {meta_name:20s}: {score:.4f}")
    logger.info(f"Stacking meta-learner {meta_name}: Validation R² = {score:.4f}")

best_meta_learner = max(meta_learner_scores, key=meta_learner_scores.get)
best_score = meta_learner_scores[best_meta_learner]

print(f"\n🏆 Best Meta-Learner: {best_meta_learner}")
print(f"   Validation R²: {best_score:.4f}")
print("=" * 60)

logger.info(f"Best stacking meta-learner: {best_meta_learner} with R² = {best_score:.4f}")


# %%
# =============================================================================
# GENERATE COMPREHENSIVE EXCEL REPORT
# =============================================================================

print("=" * 80)
print("📊 GENERATING FEATURE ANALYSIS REPORT")
print("=" * 80)

try:
    # Create timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_filename = f'Feature_Analysis_Report_{timestamp}.xlsx'
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        
        # --- TAB 1: NARRATIVE ---
        narrative_data = [
            ['REGRESSION MODELING ANALYSIS REPORT', ''],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['', ''],
            ['EXECUTIVE SUMMARY', ''],
            ['Target Variable:', dependent_var],
            ['Total Features Selected:', len(feature_statistics) if 'feature_statistics' in dir() else 'N/A'],
            ['Training Samples:', len(X_train)],
            ['Validation Samples:', len(X_validation)],
            ['Number of Models Evaluated:', len(cutoff_models)],
            ['Best Model Type:', 'Stacking Ensemble'],
            ['Best Meta-Learner:', best_meta_learner],
            ['Best R² Score:', f"{meta_learner_scores.get(best_meta_learner, 0):.4f}"],
            ['', ''],
            ['MODEL PERFORMANCE - BASE LEARNERS', ''],
        ]
        
        for name, _ in cutoff_models:
            if name in best_models:
                narrative_data.append([name, f"R² = {best_models[name]['best_score']:.4f}"])
        
        narrative_data.append(['', ''])
        narrative_data.append(['MODEL PERFORMANCE - STACKING ENSEMBLES', ''])
        
        for meta_name, score in sorted(meta_learner_scores.items(), key=lambda x: x[1], reverse=True):
            marker = " (BEST)" if meta_name == best_meta_learner else ""
            narrative_data.append([f'{meta_name}{marker}', f"R² = {score:.4f}"])
        
        narrative_df = pd.DataFrame(narrative_data, columns=['Category', 'Details'])
        narrative_df.to_excel(writer, sheet_name='Narrative', index=False)
        print("  ✅ Tab 1: Narrative")
        
        # --- TAB 2: FEATURE METRICS ---
        if 'feature_statistics' in dir() and feature_statistics:
            total_importance = sum(f.get('mean_importance', 0) for f in feature_statistics)
            feature_metrics_data = []
            
            for rank, feat in enumerate(sorted(feature_statistics, key=lambda x: x.get('mean_importance', 0), reverse=True), 1):
                fname = feat.get('feature', '')
                imp = feat.get('mean_importance', 0)
                
                # Safely get column stats
                if fname in X_train.columns:
                    col = X_train[fname]
                    feature_metrics_data.append({
                        'Rank': rank,
                        'Feature_Name': fname,
                        'Mean_Importance': imp,
                        'Importance_Pct': (imp / total_importance * 100) if total_importance > 0 else 0,
                        'Mean_Rank': feat.get('mean_rank', 0),
                        'Rank_StdDev': feat.get('std_rank', 0),
                        'Mean': col.mean(),
                        'Std_Dev': col.std(),
                        'Min': col.min(),
                        'Max': col.max()
                    })
            
            if feature_metrics_data:
                pd.DataFrame(feature_metrics_data).to_excel(writer, sheet_name='Feature Metrics', index=False)
                print(f"  ✅ Tab 2: Feature Metrics ({len(feature_metrics_data)} features)")
        else:
            pd.DataFrame({'Message': ['No feature statistics available']}).to_excel(writer, sheet_name='Feature Metrics', index=False)
            print("  ⚠️  Tab 2: Feature Metrics (empty)")
        
        # --- TAB 3: MODEL COMPARISON ---
        comparison_data = []
        
        for name, _ in cutoff_models:
            if name in best_models:
                comparison_data.append({
                    'Model_Name': name,
                    'Model_Type': 'Base Learner',
                    'R2_Score': best_models[name]['best_score']
                })
        
        for meta_name, score in meta_learner_scores.items():
            comparison_data.append({
                'Model_Name': f'Stacking_{meta_name}',
                'Model_Type': 'Stacking Ensemble',
                'R2_Score': score
            })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('R2_Score', ascending=False).reset_index(drop=True)
            comparison_df.insert(0, 'Rank', range(1, len(comparison_df) + 1))
            comparison_df.to_excel(writer, sheet_name='Model Comparison', index=False)
            print(f"  ✅ Tab 3: Model Comparison ({len(comparison_data)} models)")
        
        # --- TAB 4: IMPUTATION LOG ---
        if 'IMPUTATION_TRACKER' in dir() and IMPUTATION_TRACKER:
            impute_data = [{'Column': k, 'Method': v.get('method', 'Unknown'), 
                           'Missing_Pct': v.get('missing_pct', 0)} 
                          for k, v in IMPUTATION_TRACKER.items()]
            pd.DataFrame(impute_data).to_excel(writer, sheet_name='Imputation Log', index=False)
            print(f"  ✅ Tab 4: Imputation Log ({len(impute_data)} columns)")
        else:
            pd.DataFrame({'Message': ['No imputation tracking data']}).to_excel(writer, sheet_name='Imputation Log', index=False)
            print("  ⚠️  Tab 4: Imputation Log (empty)")
        
        # --- TAB 5: MODEL PROGRESSION ---
        if 'model_progression' in dir() and model_progression:
            progression_data = []
            for model_name, scores in model_progression.items():
                # Get all R² values, filtering out None and non-numeric
                r2_values = [v for k, v in scores.items() 
                            if k.endswith('_r2') and v is not None and isinstance(v, (int, float))]
                
                progression_data.append({
                    'Model_Name': model_name,
                    'Model_Type': 'Stacking' if scores.get('is_stacking', False) else 'Base Learner',
                    'Initial_R2': scores.get('initial_r2'),
                    'Baseline_CV_R2': scores.get('baseline_cv_r2'),
                    'Optimized_CV_R2': scores.get('optimized_cv_r2'),
                    'Validation_R2': scores.get('validation_r2'),
                    'Best_R2': max(r2_values) if r2_values else None
                })
            
            if progression_data:
                prog_df = pd.DataFrame(progression_data)
                prog_df = prog_df.sort_values('Best_R2', ascending=False, na_position='last').reset_index(drop=True)
                prog_df.insert(0, 'Rank', range(1, len(prog_df) + 1))
                prog_df.to_excel(writer, sheet_name='Model Progression', index=False)
                print(f"  ✅ Tab 5: Model Progression ({len(progression_data)} models)")
        else:
            pd.DataFrame({'Message': ['No model progression data']}).to_excel(writer, sheet_name='Model Progression', index=False)
            print("  ⚠️  Tab 5: Model Progression (empty)")
    
    print("=" * 80)
    print(f"✅ Excel report saved: {excel_filename}")
    print("=" * 80)
    
    logger.info(f"Excel report generated: {excel_filename}")
    
    # --- SAVE FEATURE LIST AS JSON ---
    import json
    
    feature_list = list(X_train.columns)
    model_metadata = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'target_variable': dependent_var,
        'feature_count': len(feature_list),
        'features': feature_list,
        'best_model': f'Stacking_{best_meta_learner}',
        'best_r2': meta_learner_scores.get(best_meta_learner, 0)
    }
    
    json_filename = f'model_features_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"✅ Feature list saved: {json_filename} ({len(feature_list)} features)")
    logger.info(f"Feature list JSON saved: {json_filename}")
    
    # Log pipeline completion
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Best model: Stacking_{best_meta_learner} (R² = {best_score:.4f})")
    logger.info(f"Features selected: {len(feature_list)}")
    logger.info("=" * 60)

except Exception as e:
    print(f"❌ Error generating Excel report: {type(e).__name__}: {e}")
    logger.error(f"Excel report generation failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()


