"""
Comprehensive Test Suite for RAPID Pipeline.

This module provides thorough pytest coverage for all components of the
RAPID (Regression Analysis Pipeline with Intelligent Data) system.

Test Categories:
    - Unit tests: Individual class/function behavior
    - Integration tests: Component interactions
    - Edge cases: Boundary conditions and error handling
    - Regression tests: Known issues that should stay fixed
    - Real data tests: Tests using realistic or actual datasets

Usage:
    pytest test_rapid_pipeline.py -v
    pytest test_rapid_pipeline.py -v -k "test_tiered"  # Run specific tests
    pytest test_rapid_pipeline.py -v -k "realistic"    # Run realistic data tests
    pytest test_rapid_pipeline.py -v --cov=feature_reduction  # With coverage

Author: RAPID Development Team
Last Updated: 2025-12-05
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.linear_model import Ridge

# Import modules under test
import config
from feature_reduction import (
    CorrelationFilter,
    FeatureImportanceSelector,
    RAPIDPipeline,
    RobustStackingRegressor,
    TieredImputer,
    _get_logger,
    get_base_learners,
    get_param_distributions,
    run_pipeline,
)


# =============================================================================
# Configuration for Real Data Tests
# =============================================================================

# Path to your actual dataset (set to None to skip real data tests)
# Update this path to point to your production data for validation
REAL_DATA_PATH = os.environ.get("RAPID_TEST_DATA_PATH", None)
REAL_DATA_TARGET = os.environ.get("RAPID_TEST_TARGET", "target")

# Skip real data tests if no data file configured
real_data_available = REAL_DATA_PATH is not None and os.path.exists(REAL_DATA_PATH or "")


# =============================================================================
# Fixtures - Reusable Test Data
# =============================================================================

@pytest.fixture
def sample_df():
    """Create a simple DataFrame for basic tests."""
    np.random.seed(42)
    n_samples = 200
    
    return pd.DataFrame({
        "feature_a": np.random.randn(n_samples),
        "feature_b": np.random.randn(n_samples) * 2,
        "feature_c": np.random.randn(n_samples) + 5,
        "target": np.random.randn(n_samples) * 10 + 100
    })


@pytest.fixture
def sample_df_with_missing():
    """Create DataFrame with missing values at different thresholds."""
    np.random.seed(42)
    n_samples = 200
    
    df = pd.DataFrame({
        "low_missing": np.random.randn(n_samples),      # Will add ~3% missing
        "medium_missing": np.random.randn(n_samples),   # Will add ~15% missing
        "high_missing": np.random.randn(n_samples),     # Will add ~35% missing
        "complete": np.random.randn(n_samples),         # No missing
        "target": np.random.randn(n_samples)
    })
    
    # Inject missing values
    low_idx = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    med_idx = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    high_idx = np.random.choice(n_samples, size=int(n_samples * 0.35), replace=False)
    
    df.loc[low_idx, "low_missing"] = np.nan
    df.loc[med_idx, "medium_missing"] = np.nan
    df.loc[high_idx, "high_missing"] = np.nan
    
    return df


@pytest.fixture
def sample_df_with_strings():
    """Create DataFrame with categorical columns of varying cardinality."""
    np.random.seed(42)
    n_samples = 200
    
    return pd.DataFrame({
        "numeric_a": np.random.randn(n_samples),
        "numeric_b": np.random.randn(n_samples),
        "low_card": np.random.choice(["A", "B", "C"], n_samples),  # 3 categories
        "med_card": np.random.choice([f"cat_{i}" for i in range(25)], n_samples),  # 25 categories
        "high_card": [f"id_{i}" for i in range(n_samples)],  # 200 unique (ID-like)
        "target": np.random.randn(n_samples)
    })


@pytest.fixture
def sample_df_with_dates():
    """Create DataFrame with date-like columns."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        "feature_a": np.random.randn(n_samples),
        "feature_b": np.random.randn(n_samples),
        "created_date": pd.date_range("2020-01-01", periods=n_samples),
        "timestamp_col": pd.date_range("2020-01-01", periods=n_samples, freq="H"),
        "dt_value": np.random.randn(n_samples),  # Name contains 'dt' but is numeric
        "date_string": ["2020-01-01"] * n_samples,  # String that parses as date
        "target": np.random.randn(n_samples)
    })


@pytest.fixture
def sample_df_with_correlations():
    """Create DataFrame with highly correlated features."""
    np.random.seed(42)
    n_samples = 200
    
    base = np.random.randn(n_samples)
    
    return pd.DataFrame({
        "base_feature": base,
        "correlated_99": base + np.random.randn(n_samples) * 0.01,  # r ≈ 0.99
        "correlated_90": base + np.random.randn(n_samples) * 0.5,   # r ≈ 0.90
        "independent": np.random.randn(n_samples),
        "target": base * 2 + np.random.randn(n_samples) * 0.5
    })


@pytest.fixture
def large_regression_df():
    """Create larger DataFrame for integration tests."""
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=10,
        noise=10,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    df["target"] = y
    
    # Add some missing values
    for col in df.columns[:5]:
        mask = np.random.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan
    
    return df


@pytest.fixture
def temp_csv_file(sample_df):
    """Create a temporary CSV file for load_data tests."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_csv_with_issues():
    """Create CSV with column name issues (spaces, mixed case)."""
    df = pd.DataFrame({
        "  Feature A  ": [1, 2, 3, 4, 5],
        "FEATURE_B": [2, 4, 6, 8, 10],
        "Feature_C": [1, 1, 1, 1, 1],
        "  Target  ": [10, 20, 30, 40, 50]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


# =============================================================================
# Realistic Dataset Fixtures - Mimics Real-World Data
# =============================================================================

@pytest.fixture
def california_housing_df():
    """
    California Housing dataset - real sklearn dataset.
    
    This is a genuine regression dataset with:
    - 20,640 samples
    - 8 features (MedInc, HouseAge, AveRooms, AveBedrms, Population, 
                  AveOccup, Latitude, Longitude)
    - Target: Median house value (in $100,000s)
    
    Good for testing with real-world data characteristics.
    """
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    df = df.rename(columns={"MedHouseVal": "target"})
    return df


@pytest.fixture
def california_housing_csv(california_housing_df):
    """California Housing as temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        california_housing_df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def realistic_manufacturing_df():
    """
    Simulated manufacturing/engineering dataset.
    
    Mimics real-world industrial data with:
    - Continuous sensor readings
    - Categorical machine/operator info
    - Date columns
    - Realistic missing patterns (sensor failures, manual entry gaps)
    - Correlated features (physics-based relationships)
    - ID columns
    - Target: Product quality metric (e.g., resistance, yield, strength)
    
    1000 samples × 25 columns
    """
    np.random.seed(42)
    n_samples = 1000
    
    # === Continuous Features (sensor readings, measurements) ===
    # Temperature affects multiple downstream features (realistic correlation)
    temperature = np.random.normal(350, 25, n_samples)  # Process temp in °F
    
    # Pressure correlates with temperature (physics relationship)
    pressure = temperature * 0.8 + np.random.normal(0, 10, n_samples)
    
    # Flow rate with some sensor noise
    flow_rate = np.random.normal(150, 20, n_samples)
    
    # Humidity (environmental)
    humidity = np.random.uniform(30, 70, n_samples)
    
    # Vibration readings from equipment
    vibration_x = np.random.exponential(2, n_samples)
    vibration_y = np.random.exponential(2.5, n_samples)
    vibration_z = np.random.exponential(1.8, n_samples)
    
    # Material properties
    material_thickness = np.random.normal(2.5, 0.1, n_samples)
    material_density = np.random.normal(7.8, 0.2, n_samples)
    
    # Time-based features
    cycle_time = np.random.normal(45, 5, n_samples)
    dwell_time = np.random.normal(12, 2, n_samples)
    
    # Highly correlated feature (should be filtered out)
    temperature_duplicate = temperature + np.random.normal(0, 0.5, n_samples)
    
    # === Categorical Features ===
    machines = np.random.choice(["Machine_A", "Machine_B", "Machine_C", "Machine_D"], n_samples)
    operators = np.random.choice([f"Op_{i:03d}" for i in range(1, 51)], n_samples)  # 50 operators
    shifts = np.random.choice(["Day", "Swing", "Night"], n_samples)
    material_batch = np.random.choice([f"Batch_{i}" for i in range(1, 201)], n_samples)  # 200 batches
    product_type = np.random.choice(["TypeA", "TypeB"], n_samples, p=[0.7, 0.3])
    
    # === ID Columns (should be dropped) ===
    record_id = [f"REC_{i:06d}" for i in range(n_samples)]
    lot_number = [f"LOT_{np.random.randint(10000, 99999)}" for _ in range(n_samples)]
    
    # === Date Columns ===
    base_date = pd.Timestamp("2023-01-01")
    production_date = [base_date + pd.Timedelta(days=int(i/3)) for i in range(n_samples)]
    timestamp = [base_date + pd.Timedelta(hours=i) for i in range(n_samples)]
    
    # === Target Variable ===
    # Realistic target based on features (simulates physics/process relationship)
    target = (
        50 +
        0.1 * temperature +
        0.05 * pressure +
        -0.02 * humidity +
        0.3 * material_thickness * 10 +
        0.2 * material_density +
        -0.5 * (vibration_x + vibration_y + vibration_z) +
        np.where(machines == "Machine_A", 2, 0) +
        np.where(machines == "Machine_D", -1.5, 0) +
        np.random.normal(0, 3, n_samples)  # Noise
    )
    
    # Build DataFrame
    df = pd.DataFrame({
        # Continuous
        "temperature": temperature,
        "pressure": pressure,
        "flow_rate": flow_rate,
        "humidity": humidity,
        "vibration_x": vibration_x,
        "vibration_y": vibration_y,
        "vibration_z": vibration_z,
        "material_thickness": material_thickness,
        "material_density": material_density,
        "cycle_time": cycle_time,
        "dwell_time": dwell_time,
        "temperature_sensor_2": temperature_duplicate,  # Correlated
        
        # Categorical
        "machine_id": machines,
        "operator_id": operators,
        "shift": shifts,
        "material_batch": material_batch,
        "product_type": product_type,
        
        # IDs (should be dropped)
        "record_id": record_id,
        "lot_number": lot_number,
        
        # Dates
        "production_date": production_date,
        "timestamp": timestamp,
        
        # Target
        "quality_metric": target
    })
    
    # === Inject Realistic Missing Patterns ===
    # Sensor failures (random missing)
    for col in ["vibration_x", "vibration_y", "vibration_z"]:
        mask = np.random.random(n_samples) < 0.03  # 3% sensor failures
        df.loc[mask, col] = np.nan
    
    # Manual entry gaps (missing in batches)
    manual_cols = ["material_thickness", "material_density"]
    batch_missing_start = np.random.choice(n_samples - 20, 5)
    for start in batch_missing_start:
        for col in manual_cols:
            df.loc[start:start+np.random.randint(5, 15), col] = np.nan
    
    # Occasional missing in categorical (data entry issues)
    mask = np.random.random(n_samples) < 0.02
    df.loc[mask, "operator_id"] = np.nan
    
    # One column with high missing (old sensor, being phased out)
    high_missing_col = np.random.normal(100, 10, n_samples)
    high_missing_col[np.random.random(n_samples) < 0.25] = np.nan  # 25% missing
    df["legacy_sensor"] = high_missing_col
    
    return df


@pytest.fixture
def realistic_manufacturing_csv(realistic_manufacturing_df):
    """Realistic manufacturing data as temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        realistic_manufacturing_df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def realistic_financial_df():
    """
    Simulated financial/credit dataset.
    
    Mimics real-world financial data with:
    - Numerical features (income, age, ratios)
    - Categorical features (employment, region)
    - Skewed distributions (typical in finance)
    - Missing data patterns
    - Target: Credit score or loan amount
    
    800 samples × 18 columns
    """
    np.random.seed(123)
    n_samples = 800
    
    # Demographics
    age = np.random.normal(42, 12, n_samples).clip(18, 80).astype(int)
    income = np.random.lognormal(10.5, 0.8, n_samples)  # Skewed distribution
    years_employed = np.random.exponential(5, n_samples).clip(0, 40)
    
    # Financial metrics
    debt_to_income = np.random.beta(2, 5, n_samples)  # Skewed toward lower values
    credit_utilization = np.random.beta(2, 3, n_samples)
    num_accounts = np.random.poisson(4, n_samples)
    num_late_payments = np.random.poisson(0.5, n_samples)
    months_since_delinquent = np.random.exponential(24, n_samples)
    
    # Categorical
    employment_type = np.random.choice(
        ["Full-Time", "Part-Time", "Self-Employed", "Retired", "Unemployed"],
        n_samples, p=[0.6, 0.15, 0.12, 0.08, 0.05]
    )
    home_ownership = np.random.choice(
        ["Own", "Mortgage", "Rent"], n_samples, p=[0.25, 0.45, 0.30]
    )
    region = np.random.choice(
        ["Northeast", "Southeast", "Midwest", "Southwest", "West"],
        n_samples
    )
    loan_purpose = np.random.choice(
        ["Debt_Consolidation", "Home_Improvement", "Major_Purchase", 
         "Medical", "Education", "Other"],
        n_samples
    )
    
    # ID column
    customer_id = [f"CUST_{i:08d}" for i in range(n_samples)]
    
    # Target (credit score influenced by features)
    target = (
        500 +
        age * 1.5 +
        np.log1p(income) * 20 +
        years_employed * 3 +
        -debt_to_income * 150 +
        -credit_utilization * 100 +
        -num_late_payments * 30 +
        np.where(employment_type == "Full-Time", 20, 0) +
        np.where(home_ownership == "Own", 25, 0) +
        np.random.normal(0, 30, n_samples)
    ).clip(300, 850)
    
    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "annual_income": income,
        "years_employed": years_employed,
        "debt_to_income_ratio": debt_to_income,
        "credit_utilization": credit_utilization,
        "num_credit_accounts": num_accounts,
        "num_late_payments_2yr": num_late_payments,
        "months_since_delinquent": months_since_delinquent,
        "employment_type": employment_type,
        "home_ownership": home_ownership,
        "region": region,
        "loan_purpose": loan_purpose,
        "credit_score": target
    })
    
    # Inject missing (some people don't report income, etc.)
    df.loc[np.random.random(n_samples) < 0.05, "annual_income"] = np.nan
    df.loc[np.random.random(n_samples) < 0.08, "years_employed"] = np.nan
    df.loc[np.random.random(n_samples) < 0.12, "months_since_delinquent"] = np.nan
    
    return df


@pytest.fixture
def realistic_financial_csv(realistic_financial_df):
    """Realistic financial data as temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        realistic_financial_df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def real_data_df():
    """
    Load actual production dataset if available.
    
    Set environment variables to use:
        RAPID_TEST_DATA_PATH=/path/to/your/data.csv
        RAPID_TEST_TARGET=your_target_column
    
    Or modify REAL_DATA_PATH and REAL_DATA_TARGET at top of file.
    """
    if not real_data_available:
        pytest.skip("Real data not configured. Set RAPID_TEST_DATA_PATH env var.")
    
    df = pd.read_csv(REAL_DATA_PATH)
    return df


@pytest.fixture
def real_data_csv():
    """Path to real data CSV if available."""
    if not real_data_available:
        pytest.skip("Real data not configured. Set RAPID_TEST_DATA_PATH env var.")
    return REAL_DATA_PATH


# =============================================================================
# TieredImputer Tests
# =============================================================================

class TestTieredImputer:
    """Tests for TieredImputer class."""
    
    def test_init_default_thresholds(self):
        """Test default threshold initialization from config."""
        imputer = TieredImputer()
        
        assert imputer.low_threshold == config.LOW_MISSING_THRESHOLD
        assert imputer.medium_threshold == config.MEDIUM_MISSING_THRESHOLD
        assert imputer.high_threshold == config.HIGH_MISSING_THRESHOLD
    
    def test_init_custom_thresholds(self):
        """Test custom threshold initialization."""
        imputer = TieredImputer(
            low_threshold=0.1,
            medium_threshold=0.3,
            high_threshold=0.5
        )
        
        assert imputer.low_threshold == 0.1
        assert imputer.medium_threshold == 0.3
        assert imputer.high_threshold == 0.5
    
    def test_fit_transform_no_missing(self, sample_df):
        """Test imputer with no missing values."""
        X = sample_df.drop(columns=["target"])
        imputer = TieredImputer()
        
        X_transformed = imputer.fit_transform(X)
        
        assert X_transformed.shape == X.shape
        assert not np.isnan(X_transformed).any()
    
    def test_fit_transform_with_missing(self, sample_df_with_missing):
        """Test imputer handles all missing value tiers."""
        X = sample_df_with_missing.drop(columns=["target"])
        imputer = TieredImputer()
        
        X_transformed = imputer.fit_transform(X)
        
        assert X_transformed.shape == X.shape
        assert not np.isnan(X_transformed).any()
    
    def test_transform_without_fit_raises(self, sample_df):
        """Test that transform before fit raises error."""
        X = sample_df.drop(columns=["target"])
        imputer = TieredImputer()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            imputer.transform(X)
    
    def test_fit_transform_returns_array(self, sample_df_with_missing):
        """Test that fit_transform returns numpy array."""
        X = sample_df_with_missing.drop(columns=["target"])
        imputer = TieredImputer()
        
        result = imputer.fit_transform(X)
        
        assert isinstance(result, np.ndarray)
    
    def test_imputer_tracks_column_methods(self, sample_df_with_missing):
        """Test that imputer records which method was used per column."""
        X = sample_df_with_missing.drop(columns=["target"])
        imputer = TieredImputer()
        imputer.fit(X)
        
        assert hasattr(imputer, "imputation_info_")
        assert len(imputer.imputation_info_) > 0


# =============================================================================
# CorrelationFilter Tests
# =============================================================================

class TestCorrelationFilter:
    """Tests for CorrelationFilter class."""
    
    def test_init_default_threshold(self):
        """Test default threshold from config."""
        cf = CorrelationFilter()
        assert cf.threshold == config.FEATURE_CORRELATION_THRESHOLD
    
    def test_init_custom_threshold(self):
        """Test custom threshold initialization."""
        cf = CorrelationFilter(threshold=0.90)
        assert cf.threshold == 0.90
    
    def test_fit_removes_highly_correlated(self, sample_df_with_correlations):
        """Test that highly correlated features are identified for removal."""
        X = sample_df_with_correlations.drop(columns=["target"])
        y = sample_df_with_correlations["target"]
        
        cf = CorrelationFilter(threshold=0.95)
        cf.fit(X, y)
        
        # Should remove correlated_99 (r > 0.95)
        assert "correlated_99" not in cf.features_to_keep_
        # Should keep the others
        assert "base_feature" in cf.features_to_keep_
        assert "independent" in cf.features_to_keep_
    
    def test_fit_keeps_feature_with_higher_target_correlation(
        self, sample_df_with_correlations
    ):
        """Test that feature with higher target correlation is kept."""
        X = sample_df_with_correlations.drop(columns=["target"])
        y = sample_df_with_correlations["target"]
        
        cf = CorrelationFilter(threshold=0.95)
        cf.fit(X, y)
        
        # base_feature should be kept (higher correlation with target)
        assert "base_feature" in cf.features_to_keep_
    
    def test_transform_reduces_features(self, sample_df_with_correlations):
        """Test that transform returns reduced feature set."""
        X = sample_df_with_correlations.drop(columns=["target"])
        y = sample_df_with_correlations["target"]
        
        cf = CorrelationFilter(threshold=0.95)
        cf.fit(X, y)
        X_transformed = cf.transform(X)
        
        assert X_transformed.shape[1] < X.shape[1]
    
    def test_transform_without_fit_raises(self, sample_df):
        """Test that transform before fit raises error."""
        X = sample_df.drop(columns=["target"])
        cf = CorrelationFilter()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            cf.transform(X)
    
    def test_no_correlation_keeps_all(self, sample_df):
        """Test that uncorrelated features are all kept."""
        X = sample_df.drop(columns=["target"])
        y = sample_df["target"]
        
        cf = CorrelationFilter(threshold=0.95)
        cf.fit(X, y)
        
        # All features should be kept
        assert len(cf.features_to_keep_) == X.shape[1]


# =============================================================================
# FeatureImportanceSelector Tests
# =============================================================================

class TestFeatureImportanceSelector:
    """Tests for FeatureImportanceSelector class."""
    
    def test_init_default_params(self):
        """Test default parameter initialization."""
        selector = FeatureImportanceSelector()
        
        assert selector.cumulative_threshold == config.OPTIMIZATION_CDF_THRESHOLD
        assert selector.min_features == config.MIN_FEATURES_TO_SELECT
    
    def test_init_custom_params(self):
        """Test custom parameter initialization."""
        selector = FeatureImportanceSelector(
            cumulative_threshold=0.90,
            min_features=5
        )
        
        assert selector.cumulative_threshold == 0.90
        assert selector.min_features == 5
    
    def test_fit_selects_features(self, large_regression_df):
        """Test that fit selects a subset of features."""
        X = large_regression_df.drop(columns=["target"]).dropna()
        y = large_regression_df.loc[X.index, "target"]
        
        selector = FeatureImportanceSelector(
            cumulative_threshold=0.95,
            min_features=5,
            random_state=42
        )
        selector.fit(X, y)
        
        assert len(selector.selected_features_) >= 5
        assert len(selector.selected_features_) <= X.shape[1]
    
    def test_fit_respects_min_features(self, sample_df):
        """Test that min_features floor is respected."""
        X = sample_df.drop(columns=["target"])
        y = sample_df["target"]
        
        selector = FeatureImportanceSelector(
            cumulative_threshold=0.50,  # Low threshold
            min_features=3,  # But require at least 3
            random_state=42
        )
        selector.fit(X, y)
        
        assert len(selector.selected_features_) >= 3
    
    def test_fit_stores_importances(self, large_regression_df):
        """Test that feature importances are stored."""
        X = large_regression_df.drop(columns=["target"]).dropna()
        y = large_regression_df.loc[X.index, "target"]
        
        selector = FeatureImportanceSelector(random_state=42)
        selector.fit(X, y)
        
        assert hasattr(selector, "feature_importances_")
        assert hasattr(selector, "feature_ranks_")
        assert hasattr(selector, "feature_std_ranks_")
        assert len(selector.feature_importances_) > 0
    
    def test_transform_reduces_features(self, large_regression_df):
        """Test that transform returns selected features only."""
        X = large_regression_df.drop(columns=["target"]).dropna()
        y = large_regression_df.loc[X.index, "target"]
        
        selector = FeatureImportanceSelector(
            cumulative_threshold=0.80,
            random_state=42
        )
        selector.fit(X, y)
        X_transformed = selector.transform(X)
        
        assert X_transformed.shape[1] == len(selector.selected_features_)
    
    def test_transform_without_fit_raises(self, sample_df):
        """Test that transform before fit raises error."""
        X = sample_df.drop(columns=["target"])
        selector = FeatureImportanceSelector()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            selector.transform(X)
    
    def test_get_feature_names_out(self, large_regression_df):
        """Test get_feature_names_out returns correct features."""
        X = large_regression_df.drop(columns=["target"]).dropna()
        y = large_regression_df.loc[X.index, "target"]
        
        selector = FeatureImportanceSelector(random_state=42)
        selector.fit(X, y)
        
        names = selector.get_feature_names_out()
        
        assert isinstance(names, np.ndarray)
        assert len(names) == len(selector.selected_features_)


# =============================================================================
# RobustStackingRegressor Tests
# =============================================================================

class TestRobustStackingRegressor:
    """Tests for RobustStackingRegressor class."""
    
    @pytest.fixture
    def fitted_base_models(self, large_regression_df):
        """Create fitted base models for stacking tests."""
        X = large_regression_df.drop(columns=["target"]).dropna().values
        y = large_regression_df.loc[
            large_regression_df.drop(columns=["target"]).dropna().index, 
            "target"
        ].values
        
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=10, random_state=42)
        
        rf.fit(X, y)
        gb.fit(X, y)
        
        return [("RF", rf), ("GB", gb)], X, y
    
    def test_init_default_params(self):
        """Test default parameter initialization."""
        stacker = RobustStackingRegressor(base_estimators=[])
        
        assert stacker.cv == config.CV_FOLDS
        assert stacker.n_jobs == config.N_JOBS
    
    def test_fit_trains_ensemble(self, large_regression_df):
        """Test that fit trains the stacking ensemble."""
        X = large_regression_df.drop(columns=["target"]).dropna().values
        y = large_regression_df.loc[
            large_regression_df.drop(columns=["target"]).dropna().index,
            "target"
        ].values
        
        from sklearn.ensemble import RandomForestRegressor
        
        base_estimators = [
            ("RF", RandomForestRegressor(n_estimators=10, random_state=42))
        ]
        
        stacker = RobustStackingRegressor(
            base_estimators=base_estimators,
            cv=3,
            random_state=42
        )
        stacker.fit(X, y)
        
        assert stacker.fitted_meta_estimator_ is not None
        assert len(stacker.fitted_base_estimators_) == 1
    
    def test_predict_returns_array(self, large_regression_df):
        """Test that predict returns numpy array."""
        X = large_regression_df.drop(columns=["target"]).dropna().values
        y = large_regression_df.loc[
            large_regression_df.drop(columns=["target"]).dropna().index,
            "target"
        ].values
        
        from sklearn.ensemble import RandomForestRegressor
        
        base_estimators = [
            ("RF", RandomForestRegressor(n_estimators=10, random_state=42))
        ]
        
        stacker = RobustStackingRegressor(
            base_estimators=base_estimators,
            cv=3,
            random_state=42
        )
        stacker.fit(X, y)
        predictions = stacker.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)
    
    def test_predict_without_fit_raises(self, sample_df):
        """Test that predict before fit raises error."""
        X = sample_df.drop(columns=["target"]).values
        stacker = RobustStackingRegressor(base_estimators=[])
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            stacker.predict(X)
    
    def test_score_returns_float(self, large_regression_df):
        """Test that score returns R² as float."""
        X = large_regression_df.drop(columns=["target"]).dropna().values
        y = large_regression_df.loc[
            large_regression_df.drop(columns=["target"]).dropna().index,
            "target"
        ].values
        
        from sklearn.ensemble import RandomForestRegressor
        
        base_estimators = [
            ("RF", RandomForestRegressor(n_estimators=10, random_state=42))
        ]
        
        stacker = RobustStackingRegressor(
            base_estimators=base_estimators,
            cv=3,
            random_state=42
        )
        stacker.fit(X, y)
        score = stacker.score(X, y)
        
        assert isinstance(score, float)
        assert -1 <= score <= 1  # R² can be negative for poor models
    
    def test_fit_from_oof(self, large_regression_df):
        """Test fit_from_oof method with pre-computed predictions."""
        X = large_regression_df.drop(columns=["target"]).dropna().values
        y = large_regression_df.loc[
            large_regression_df.drop(columns=["target"]).dropna().index,
            "target"
        ].values
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        # Create and fit a pipeline
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", rf)
        ])
        pipeline.fit(X, y)
        
        # Generate OOF predictions
        oof_preds = np.random.randn(len(y))  # Simplified for test
        
        base_estimators = [("RF", rf)]
        fitted_estimators = {"RF": pipeline}
        oof_predictions = {"RF": oof_preds}
        
        stacker = RobustStackingRegressor(
            base_estimators=base_estimators,
            random_state=42
        )
        stacker.fit_from_oof(X, y, oof_predictions, fitted_estimators)
        
        assert stacker.fitted_meta_estimator_ is not None


# =============================================================================
# RAPIDPipeline Tests
# =============================================================================

class TestRAPIDPipeline:
    """Tests for the main RAPIDPipeline class."""
    
    def test_init_creates_empty_pipeline(self):
        """Test that init creates pipeline with None attributes."""
        pipeline = RAPIDPipeline()
        
        assert pipeline.raw_data_ is None
        assert pipeline.target_column_ is None
        assert pipeline.best_models_ == {}
        assert pipeline.model_scores_ == {}
    
    def test_load_data_from_csv(self, temp_csv_file):
        """Test loading data from CSV file."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(temp_csv_file, "target")
        
        assert pipeline.raw_data_ is not None
        assert pipeline.target_column_ == "target"
        assert len(pipeline.raw_data_) > 0
    
    def test_load_data_normalizes_column_names(self, temp_csv_with_issues):
        """Test that column names are normalized to lowercase."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(temp_csv_with_issues, "  Target  ")
        
        # All columns should be lowercase and stripped
        assert all(col == col.lower().strip() for col in pipeline.raw_data_.columns)
        assert pipeline.target_column_ == "target"
    
    def test_load_data_missing_target_raises(self, temp_csv_file):
        """Test that missing target column raises ValueError."""
        pipeline = RAPIDPipeline()
        
        with pytest.raises(ValueError, match="not found"):
            pipeline.load_data(temp_csv_file, "nonexistent_column")
    
    def test_load_data_missing_file_raises(self):
        """Test that missing file raises FileNotFoundError."""
        pipeline = RAPIDPipeline()
        
        with pytest.raises(FileNotFoundError):
            pipeline.load_data("nonexistent_file.csv", "target")
    
    def test_preprocess_drops_high_missing(self, sample_df_with_missing):
        """Test that preprocess drops columns with high missing data."""
        # Create temp file with >40% missing column
        df = sample_df_with_missing.copy()
        df["very_high_missing"] = np.nan  # 100% missing
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            
        try:
            pipeline = RAPIDPipeline()
            pipeline.load_data(f.name, "target")
            pipeline.preprocess()
            
            assert "very_high_missing" not in pipeline.raw_data_.columns
        finally:
            os.unlink(f.name)
    
    def test_preprocess_encodes_categoricals(self, sample_df_with_strings):
        """Test that categorical columns are encoded."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_df_with_strings.to_csv(f.name, index=False)
        
        try:
            pipeline = RAPIDPipeline()
            pipeline.load_data(f.name, "target")
            pipeline.preprocess(encode_categoricals=True)
            
            # low_card should be one-hot encoded (creates multiple columns)
            # Original column should be gone
            assert "low_card" not in pipeline.raw_data_.columns
            # Check for one-hot encoded columns
            one_hot_cols = [c for c in pipeline.raw_data_.columns if c.startswith("low_card_")]
            assert len(one_hot_cols) > 0
            
            # high_card (ID-like) should be dropped
            assert "high_card" not in pipeline.raw_data_.columns
        finally:
            os.unlink(f.name)
    
    def test_preprocess_detects_date_columns(self, sample_df_with_dates):
        """Test that date columns are detected and dropped."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_df_with_dates.to_csv(f.name, index=False)
        
        try:
            pipeline = RAPIDPipeline()
            pipeline.load_data(f.name, "target")
            pipeline.preprocess()
            
            # Date columns should be removed
            assert "created_date" not in pipeline.raw_data_.columns
            assert "timestamp_col" not in pipeline.raw_data_.columns
        finally:
            os.unlink(f.name)
    
    def test_split_data_creates_train_test(self, temp_csv_file):
        """Test that split_data creates train and test sets."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(temp_csv_file, "target")
        pipeline.preprocess()
        pipeline.split_data()
        
        assert pipeline.X_train_ is not None
        assert pipeline.X_test_ is not None
        assert pipeline.y_train_ is not None
        assert pipeline.y_test_ is not None
        
        # Check split ratio
        total = len(pipeline.X_train_) + len(pipeline.X_test_)
        test_ratio = len(pipeline.X_test_) / total
        assert abs(test_ratio - config.TEST_SIZE) < 0.05
    
    def test_transform_new_data_handles_extra_columns(self, large_regression_df):
        """Test that transform_new_data ignores extra columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_regression_df.to_csv(f.name, index=False)
        
        try:
            pipeline = RAPIDPipeline()
            pipeline.load_data(f.name, "target")
            pipeline.preprocess()
            pipeline.split_data()
            pipeline.fit_feature_selection()
            
            # Create new data with extra columns
            new_data = large_regression_df.drop(columns=["target"]).copy()
            new_data["extra_column"] = np.random.randn(len(new_data))
            
            # Should not raise, should ignore extra column
            X_transformed = pipeline.transform_new_data(new_data)
            assert X_transformed is not None
        finally:
            os.unlink(f.name)
    
    def test_transform_new_data_missing_column_raises(self, large_regression_df):
        """Test that transform_new_data raises on missing required columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_regression_df.to_csv(f.name, index=False)
        
        try:
            pipeline = RAPIDPipeline()
            pipeline.load_data(f.name, "target")
            pipeline.preprocess()
            pipeline.split_data()
            pipeline.fit_feature_selection()
            
            # Create new data missing required columns
            new_data = pd.DataFrame({"wrong_column": [1, 2, 3]})
            
            with pytest.raises(ValueError, match="missing required"):
                pipeline.transform_new_data(new_data)
        finally:
            os.unlink(f.name)
    
    def test_method_chaining(self, temp_csv_file):
        """Test that methods return self for chaining."""
        pipeline = RAPIDPipeline()
        
        result = pipeline.load_data(temp_csv_file, "target")
        assert result is pipeline
        
        result = pipeline.preprocess()
        assert result is pipeline
        
        result = pipeline.split_data()
        assert result is pipeline
    
    def test_get_feature_importance_report(self, large_regression_df):
        """Test get_feature_importance_report returns DataFrame."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_regression_df.to_csv(f.name, index=False)
        
        try:
            pipeline = RAPIDPipeline()
            pipeline.load_data(f.name, "target")
            pipeline.preprocess()
            pipeline.split_data()
            pipeline.fit_feature_selection()
            
            report = pipeline.get_feature_importance_report()
            
            assert isinstance(report, pd.DataFrame)
            assert "feature" in report.columns
            assert "importance" in report.columns
            assert "mean_rank" in report.columns
        finally:
            os.unlink(f.name)


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_get_base_learners_returns_list(self):
        """Test that get_base_learners returns list of tuples."""
        learners = get_base_learners()
        
        assert isinstance(learners, list)
        assert len(learners) > 0
        assert all(isinstance(item, tuple) for item in learners)
        assert all(len(item) == 2 for item in learners)
    
    def test_get_base_learners_all_have_fit_predict(self):
        """Test that all base learners have fit and predict methods."""
        learners = get_base_learners()
        
        for name, model in learners:
            assert hasattr(model, "fit"), f"{name} missing fit method"
            assert hasattr(model, "predict"), f"{name} missing predict method"
    
    def test_get_param_distributions_returns_dict(self):
        """Test that get_param_distributions returns dict."""
        params = get_param_distributions()
        
        assert isinstance(params, dict)
        assert len(params) > 0
    
    def test_get_param_distributions_keys_match_learners(self):
        """Test that param distribution keys match base learner names."""
        learners = get_base_learners()
        params = get_param_distributions()
        
        learner_names = {name for name, _ in learners}
        param_names = set(params.keys())
        
        # All param names should be in learner names
        assert param_names.issubset(learner_names)


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Tests for configuration values."""
    
    def test_thresholds_are_valid_ranges(self):
        """Test that threshold values are in valid ranges."""
        assert 0 < config.LOW_MISSING_THRESHOLD < 1
        assert 0 < config.MEDIUM_MISSING_THRESHOLD < 1
        assert 0 < config.HIGH_MISSING_THRESHOLD <= 1
        assert 0 < config.MAX_MISSING_DATA <= 1
        
        # Thresholds should be in order
        assert config.LOW_MISSING_THRESHOLD < config.MEDIUM_MISSING_THRESHOLD
        assert config.MEDIUM_MISSING_THRESHOLD <= config.HIGH_MISSING_THRESHOLD
    
    def test_encoding_thresholds_valid(self):
        """Test encoding thresholds are valid."""
        assert config.ONE_HOT_ENCODING_MAX_CATEGORIES > 0
        assert config.LABEL_ENCODING_MAX_CATEGORIES > 0
        assert config.ONE_HOT_ENCODING_MAX_CATEGORIES < config.LABEL_ENCODING_MAX_CATEGORIES
    
    def test_cv_folds_valid(self):
        """Test CV folds is valid."""
        assert config.CV_FOLDS >= 2
        assert config.CV_FOLDS <= 20
    
    def test_test_size_valid(self):
        """Test test size is valid."""
        assert 0 < config.TEST_SIZE < 1
    
    def test_feature_importance_models_valid(self):
        """Test that configured feature importance models are valid."""
        valid_models = {'RF', 'XGB', 'GBT', 'CB', 'LGB', 'ETR'}
        
        for model in config.FEATURE_IMPORTANCE_MODELS:
            assert model in valid_models, f"Invalid model: {model}"


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_feature(self):
        """Test handling of single feature dataset."""
        df = pd.DataFrame({
            "only_feature": np.random.randn(100),
            "target": np.random.randn(100)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
        
        try:
            pipeline = RAPIDPipeline()
            pipeline.load_data(f.name, "target")
            pipeline.preprocess()
            pipeline.split_data()
            
            # Should not crash
            assert pipeline.X_train_ is not None
            assert pipeline.X_train_.shape[1] == 1
        finally:
            os.unlink(f.name)
    
    def test_small_dataset(self):
        """Test handling of very small dataset."""
        df = pd.DataFrame({
            "feature_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_b": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "target": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
        
        try:
            pipeline = RAPIDPipeline()
            pipeline.load_data(f.name, "target")
            pipeline.preprocess()
            pipeline.split_data()
            
            assert pipeline.X_train_ is not None
        finally:
            os.unlink(f.name)
    
    def test_all_missing_column(self):
        """Test handling of column with all missing values."""
        df = pd.DataFrame({
            "feature_a": [1, 2, 3, 4, 5],
            "all_missing": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "target": [10, 20, 30, 40, 50]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
        
        try:
            pipeline = RAPIDPipeline()
            pipeline.load_data(f.name, "target")
            pipeline.preprocess()
            
            # Column should be dropped
            assert "all_missing" not in pipeline.raw_data_.columns
        finally:
            os.unlink(f.name)
    
    def test_constant_feature(self):
        """Test handling of constant feature."""
        df = pd.DataFrame({
            "varying": np.random.randn(100),
            "constant": [42] * 100,
            "target": np.random.randn(100)
        })
        
        X = df.drop(columns=["target"])
        y = df["target"]
        
        # CorrelationFilter should handle this
        cf = CorrelationFilter()
        cf.fit(X, y)
        
        # Should not crash
        assert cf.features_to_keep_ is not None
    
    def test_identical_features(self):
        """Test handling of identical features."""
        values = np.random.randn(100)
        df = pd.DataFrame({
            "feature_1": values,
            "feature_2": values.copy(),  # Identical
            "target": np.random.randn(100)
        })
        
        X = df.drop(columns=["target"])
        y = df["target"]
        
        cf = CorrelationFilter(threshold=0.99)
        cf.fit(X, y)
        
        # One should be removed
        assert len(cf.features_to_keep_) == 1
    
    def test_negative_target_values(self):
        """Test handling of negative target values."""
        df = pd.DataFrame({
            "feature_a": np.random.randn(100),
            "target": np.random.randn(100) - 100  # All negative
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
        
        try:
            pipeline = RAPIDPipeline()
            pipeline.load_data(f.name, "target")
            pipeline.preprocess()
            pipeline.split_data()
            
            assert pipeline.y_train_.min() < 0
        finally:
            os.unlink(f.name)
    
    def test_special_characters_in_column_names(self):
        """Test handling of special characters in column names."""
        df = pd.DataFrame({
            "feature (1)": [1, 2, 3],
            "feature-2": [4, 5, 6],
            "feature.3": [7, 8, 9],
            "target": [10, 20, 30]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
        
        try:
            pipeline = RAPIDPipeline()
            pipeline.load_data(f.name, "target")
            
            # Should not crash
            assert pipeline.raw_data_ is not None
        finally:
            os.unlink(f.name)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete pipeline workflows."""
    
    @pytest.mark.slow
    def test_full_pipeline_basic(self, large_regression_df):
        """Test complete pipeline from load to evaluate."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_regression_df.to_csv(f.name, index=False)
        
        try:
            pipeline = RAPIDPipeline()
            pipeline.load_data(f.name, "target")
            pipeline.preprocess()
            pipeline.split_data()
            pipeline.fit_feature_selection()
            
            # Verify state after each step
            assert pipeline.raw_data_ is not None
            assert pipeline.X_train_ is not None
            assert pipeline.feature_selector_ is not None
            assert len(pipeline.selected_feature_names_) > 0
        finally:
            os.unlink(f.name)
    
    @pytest.mark.slow
    def test_generate_report_creates_excel(self, large_regression_df):
        """Test that generate_report creates valid Excel file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_regression_df.to_csv(f.name, index=False)
        
        try:
            pipeline = RAPIDPipeline()
            pipeline.load_data(f.name, "target")
            pipeline.preprocess()
            pipeline.split_data()
            pipeline.fit_feature_selection()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                report_path = pipeline.generate_report(output_dir=tmpdir)
                
                assert os.path.exists(report_path)
                assert report_path.endswith(".xlsx")
                
                # Verify Excel file can be read
                xl = pd.ExcelFile(report_path)
                assert "Narrative" in xl.sheet_names
                assert "Feature Metrics" in xl.sheet_names
        finally:
            os.unlink(f.name)


# =============================================================================
# Realistic Data Tests - California Housing (Real sklearn dataset)
# =============================================================================

class TestCaliforniaHousing:
    """Tests using real California Housing dataset from sklearn."""
    
    @pytest.mark.realistic
    def test_load_california_housing(self, california_housing_csv):
        """Test loading California Housing dataset."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(california_housing_csv, "target")
        
        assert pipeline.raw_data_ is not None
        assert len(pipeline.raw_data_) == 20640
        assert "MedInc" in pipeline.raw_data_.columns.str.lower().str.replace(" ", "")
    
    @pytest.mark.realistic
    def test_preprocess_california_housing(self, california_housing_csv):
        """Test preprocessing California Housing dataset."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(california_housing_csv, "target")
        pipeline.preprocess()
        
        # Should retain all features (no missing, no categoricals, no dates)
        assert pipeline.raw_data_.shape[1] >= 8
        
        # No missing values in this dataset
        assert pipeline.raw_data_.isnull().sum().sum() == 0
    
    @pytest.mark.realistic
    def test_full_pipeline_california_housing(self, california_housing_csv):
        """Test full pipeline on California Housing dataset."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(california_housing_csv, "target")
        pipeline.preprocess()
        pipeline.split_data()
        pipeline.fit_feature_selection()
        
        # Should select meaningful features
        assert len(pipeline.selected_feature_names_) >= 3
        
        # Feature importances should be available
        report = pipeline.get_feature_importance_report()
        assert len(report) > 0
        
        # MedInc (median income) should be top feature (known to be most predictive)
        top_features = report.head(3)["feature"].str.lower().tolist()
        assert any("medinc" in f or "income" in f for f in top_features)
    
    @pytest.mark.realistic
    @pytest.mark.slow
    def test_train_models_california_housing(self, california_housing_csv):
        """Test model training on California Housing dataset."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(california_housing_csv, "target")
        pipeline.preprocess()
        pipeline.split_data()
        pipeline.fit_feature_selection()
        pipeline.train_base_models()
        
        # Should have trained models
        assert len(pipeline.best_models_) > 0
        
        # At least one model should have R² > 0.5 (this is a predictable dataset)
        best_score = max(pipeline.model_scores_.values())
        assert best_score > 0.5, f"Best score {best_score} is too low for California Housing"


# =============================================================================
# Realistic Data Tests - Manufacturing Dataset
# =============================================================================

class TestManufacturingData:
    """Tests using realistic manufacturing/engineering dataset."""
    
    @pytest.mark.realistic
    def test_load_manufacturing_data(self, realistic_manufacturing_csv):
        """Test loading manufacturing dataset."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(realistic_manufacturing_csv, "quality_metric")
        
        assert pipeline.raw_data_ is not None
        assert len(pipeline.raw_data_) == 1000
    
    @pytest.mark.realistic
    def test_preprocess_handles_dates(self, realistic_manufacturing_csv):
        """Test that date columns are properly detected and removed."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(realistic_manufacturing_csv, "quality_metric")
        pipeline.preprocess()
        
        # Date columns should be removed
        cols_lower = [c.lower() for c in pipeline.raw_data_.columns]
        assert "production_date" not in cols_lower
        assert "timestamp" not in cols_lower
    
    @pytest.mark.realistic
    def test_preprocess_handles_ids(self, realistic_manufacturing_csv):
        """Test that ID columns are properly detected and removed."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(realistic_manufacturing_csv, "quality_metric")
        pipeline.preprocess(encode_categoricals=True)
        
        # High cardinality ID columns should be dropped
        cols_lower = [c.lower() for c in pipeline.raw_data_.columns]
        assert "record_id" not in cols_lower
        assert "lot_number" not in cols_lower
    
    @pytest.mark.realistic
    def test_preprocess_encodes_categoricals(self, realistic_manufacturing_csv):
        """Test categorical encoding on manufacturing data."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(realistic_manufacturing_csv, "quality_metric")
        pipeline.preprocess(encode_categoricals=True)
        
        # Low cardinality categoricals should be one-hot encoded
        # machine_id has 4 values -> one-hot
        # shift has 3 values -> one-hot
        cols = pipeline.raw_data_.columns.tolist()
        
        # Original columns should be gone
        assert "machine_id" not in cols
        assert "shift" not in cols
        
        # One-hot columns should exist
        machine_cols = [c for c in cols if c.startswith("machine_id_")]
        shift_cols = [c for c in cols if c.startswith("shift_")]
        assert len(machine_cols) >= 2  # At least 2 (drop_first=True)
        assert len(shift_cols) >= 2
    
    @pytest.mark.realistic
    def test_preprocess_handles_missing(self, realistic_manufacturing_csv):
        """Test missing data handling on manufacturing data."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(realistic_manufacturing_csv, "quality_metric")
        
        # Check that missing data exists before preprocessing
        initial_missing = pipeline.raw_data_.isnull().sum().sum()
        assert initial_missing > 0, "Test data should have missing values"
        
        pipeline.preprocess()
        
        # After imputation, should have no missing (or very few)
        final_missing = pipeline.raw_data_.isnull().sum().sum()
        assert final_missing < initial_missing
    
    @pytest.mark.realistic
    def test_correlation_filter_removes_duplicate(self, realistic_manufacturing_csv):
        """Test that highly correlated features are removed."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(realistic_manufacturing_csv, "quality_metric")
        pipeline.preprocess()
        pipeline.split_data()
        pipeline.fit_feature_selection()
        
        # temperature and temperature_sensor_2 are r≈0.99 correlated
        # One should be removed by correlation filter
        selected = [f.lower() for f in pipeline.selected_feature_names_]
        
        # Both shouldn't be present
        has_temp = "temperature" in selected
        has_temp2 = "temperature_sensor_2" in selected
        
        # At most one should be selected (correlation filter removes one)
        assert not (has_temp and has_temp2), "Both correlated features selected"
    
    @pytest.mark.realistic
    def test_feature_importance_meaningful(self, realistic_manufacturing_csv):
        """Test that feature importance identifies meaningful features."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(realistic_manufacturing_csv, "quality_metric")
        pipeline.preprocess(encode_categoricals=True)
        pipeline.split_data()
        pipeline.fit_feature_selection()
        
        report = pipeline.get_feature_importance_report()
        top_10 = report.head(10)["feature"].str.lower().tolist()
        
        # Temperature should be important (major driver in simulated target)
        temp_features = [f for f in top_10 if "temp" in f]
        assert len(temp_features) > 0, "Temperature should be important"
        
        # Machine should matter (we added machine effects to target)
        machine_features = [f for f in top_10 if "machine" in f]
        assert len(machine_features) > 0, "Machine should be important"
    
    @pytest.mark.realistic
    @pytest.mark.slow
    def test_full_pipeline_manufacturing(self, realistic_manufacturing_csv):
        """Test complete pipeline on manufacturing data."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(realistic_manufacturing_csv, "quality_metric")
        pipeline.preprocess(encode_categoricals=True)
        pipeline.split_data()
        pipeline.fit_feature_selection()
        pipeline.train_base_models()
        
        # Should have trained models
        assert len(pipeline.best_models_) > 0
        
        # Should have decent R² since target is constructed from features
        best_score = max(pipeline.model_scores_.values())
        assert best_score > 0.6, f"Best score {best_score} too low for constructed target"


# =============================================================================
# Realistic Data Tests - Financial Dataset
# =============================================================================

class TestFinancialData:
    """Tests using realistic financial/credit dataset."""
    
    @pytest.mark.realistic
    def test_load_financial_data(self, realistic_financial_csv):
        """Test loading financial dataset."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(realistic_financial_csv, "credit_score")
        
        assert pipeline.raw_data_ is not None
        assert len(pipeline.raw_data_) == 800
    
    @pytest.mark.realistic
    def test_handles_skewed_distributions(self, realistic_financial_df):
        """Test that pipeline handles skewed distributions."""
        # annual_income is log-normal (heavily skewed)
        skewness = realistic_financial_df["annual_income"].skew()
        assert abs(skewness) > 1, "Income should be skewed"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            realistic_financial_df.to_csv(f.name, index=False)
        
        try:
            pipeline = RAPIDPipeline()
            pipeline.load_data(f.name, "credit_score")
            pipeline.preprocess()
            pipeline.split_data()
            
            # Should not crash with skewed data
            assert pipeline.X_train_ is not None
        finally:
            os.unlink(f.name)
    
    @pytest.mark.realistic
    def test_categorical_encoding_financial(self, realistic_financial_csv):
        """Test categorical encoding on financial data."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(realistic_financial_csv, "credit_score")
        pipeline.preprocess(encode_categoricals=True)
        
        cols = pipeline.raw_data_.columns.tolist()
        
        # employment_type (5 categories) -> one-hot
        emp_cols = [c for c in cols if c.startswith("employment_type_")]
        assert len(emp_cols) >= 2
        
        # customer_id (high cardinality) -> dropped
        assert "customer_id" not in cols
    
    @pytest.mark.realistic
    @pytest.mark.slow
    def test_full_pipeline_financial(self, realistic_financial_csv):
        """Test complete pipeline on financial data."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(realistic_financial_csv, "credit_score")
        pipeline.preprocess(encode_categoricals=True)
        pipeline.split_data()
        pipeline.fit_feature_selection()
        pipeline.train_base_models()
        
        # Should have trained models
        assert len(pipeline.best_models_) > 0
        
        # Check feature importances make sense
        report = pipeline.get_feature_importance_report()
        top_features = report.head(5)["feature"].str.lower().tolist()
        
        # Income and debt-to-income should be important
        financial_features = [f for f in top_features if "income" in f or "debt" in f]
        assert len(financial_features) > 0, "Financial features should be important"


# =============================================================================
# Real Data Tests (Your Actual Dataset)
# =============================================================================

@pytest.mark.real_data
class TestRealData:
    """
    Tests using your actual production dataset.
    
    To enable these tests, set environment variables:
        export RAPID_TEST_DATA_PATH=/path/to/your/data.csv
        export RAPID_TEST_TARGET=your_target_column
    
    Then run:
        pytest test_rapid_pipeline.py -v -m real_data
    """
    
    def test_load_real_data(self, real_data_csv):
        """Test loading actual production data."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(real_data_csv, REAL_DATA_TARGET)
        
        assert pipeline.raw_data_ is not None
        assert len(pipeline.raw_data_) > 0
        print(f"\n✓ Loaded {len(pipeline.raw_data_)} rows × {len(pipeline.raw_data_.columns)} columns")
    
    def test_preprocess_real_data(self, real_data_csv):
        """Test preprocessing actual production data."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(real_data_csv, REAL_DATA_TARGET)
        
        initial_shape = pipeline.raw_data_.shape
        pipeline.preprocess(encode_categoricals=True)
        final_shape = pipeline.raw_data_.shape
        
        print(f"\n✓ Shape: {initial_shape} → {final_shape}")
        print(f"✓ Columns removed: {initial_shape[1] - final_shape[1]}")
        
        assert pipeline.raw_data_ is not None
    
    def test_feature_selection_real_data(self, real_data_csv):
        """Test feature selection on actual production data."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(real_data_csv, REAL_DATA_TARGET)
        pipeline.preprocess(encode_categoricals=True)
        pipeline.split_data()
        pipeline.fit_feature_selection()
        
        print(f"\n✓ Selected {len(pipeline.selected_feature_names_)} features")
        print(f"✓ Top 5 features: {pipeline.selected_feature_names_[:5]}")
        
        assert len(pipeline.selected_feature_names_) > 0
    
    @pytest.mark.slow
    def test_full_pipeline_real_data(self, real_data_csv):
        """Test complete pipeline on actual production data."""
        pipeline = RAPIDPipeline()
        pipeline.load_data(real_data_csv, REAL_DATA_TARGET)
        pipeline.preprocess(encode_categoricals=True)
        pipeline.split_data()
        pipeline.fit_feature_selection()
        pipeline.train_base_models()
        
        print(f"\n✓ Trained {len(pipeline.best_models_)} models")
        print(f"✓ Best model: {max(pipeline.model_scores_, key=pipeline.model_scores_.get)}")
        print(f"✓ Best R²: {max(pipeline.model_scores_.values()):.4f}")
        
        assert len(pipeline.best_models_) > 0


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
