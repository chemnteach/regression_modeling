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

# Primary metric for model evaluation (R², MAE, MSE, RMSE)
PRIMARY_METRIC = 'R²'

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
    
    # Validate CV folds
    if CV_FOLDS < 2:
        raise ValueError("CV_FOLDS must be at least 2")
    
    return True


def print_config() -> None:
    """Print current configuration settings in a formatted display."""
    print(f"{SEPARATOR_CHAR * SEPARATOR_WIDTH}")
    print("⚙️  DATA PREPROCESSING CONFIGURATION")
    print(f"{SEPARATOR_CHAR * SEPARATOR_WIDTH}")
    print(f"\n📊 Missing Data Handling:")
    print(f"   • Columns with >{MAX_MISSING_DATA:.0%} missing → "
          f"AUTOMATICALLY DROPPED")
    print(f"   • <{LOW_MISSING_THRESHOLD:.0%} missing → "
          f"Simple imputation (median/mode)")
    print(f"   • {LOW_MISSING_THRESHOLD:.0%}-{MEDIUM_MISSING_THRESHOLD:.0%} "
          f"missing → KNN imputation")
    print(f"   • {MEDIUM_MISSING_THRESHOLD:.0%}-{HIGH_MISSING_THRESHOLD:.0%} "
          f"missing → Iterative imputation (MICE)")
    print(f"\n🧹 Automatic Cleaning (no prompts):")
    print(f"   • Date columns: {'REMOVE' if REMOVE_DATE_COLUMNS else 'KEEP'}")
    print(f"   • Duplicate rows: "
          f"{'REMOVE' if REMOVE_DUPLICATE_ROWS else 'KEEP'}")
    print(f"   • Low variance ({LOW_VARIANCE_THRESHOLD:.0%}): "
          f"{'REMOVE' if REMOVE_LOW_VARIANCE_COLS else 'KEEP'}")
    print(f"   • High cardinality ({HIGH_CARDINALITY_THRESHOLD:.0%}): REMOVE")
    print(f"\n✅ Configuration loaded successfully!")
    print(f"{SEPARATOR_CHAR * SEPARATOR_WIDTH}")


# Validate configuration on import
if __name__ != '__main__':
    validate_config()
