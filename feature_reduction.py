"""RAPID: Regression Analysis Pipeline with Intelligent Data Preprocessing.

This module provides a production-ready machine learning pipeline for
regression analysis with automated feature selection, model training,
hyperparameter optimization, and stacking ensemble construction.

Version: 3.0
Author: RAPID Development Team
Last Updated: 2025-12-03

PEP Compliance:
    - PEP 8: Style Guide for Python Code
    - PEP 257: Docstring Conventions (NumPy style)
    - PEP 484: Type Hints
    - PEP 585: Type Hinting Generics in Standard Collections

Example
-------
>>> from rapid_pipeline import RAPIDPipeline
>>> pipeline = RAPIDPipeline()
>>> pipeline.load_data('data.csv', 'target_column')
>>> pipeline.preprocess().split_data().fit_feature_selection()
>>> pipeline.train_base_models().train_stacking_ensemble()
>>> results = pipeline.evaluate()
>>> print(results)

Notes
-----
All configuration is imported from config.py. Modify that file to
adjust thresholds, parallelization settings, and model parameters.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Import configuration from config.py
import config

warnings.filterwarnings("ignore")

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# Module-Level Logger Configuration
# =============================================================================

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    """Get or lazily initialize the module logger.

    The logger is created on first access, not at import time.
    This prevents side effects (log file creation) when the module
    is imported but not used.

    Returns
    -------
    logging.Logger
        Configured logger instance with file and console handlers.
    """
    global _logger

    if _logger is not None:
        return _logger

    logger = logging.getLogger("RAPID")
    logger.setLevel(getattr(logging, config.FILE_LOG_LEVEL))

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.CONSOLE_LOG_LEVEL))
        console_handler.setFormatter(
            logging.Formatter(config.LOG_FORMAT_CONSOLE)
        )
        logger.addHandler(console_handler)

        # File handler (create log directory if needed)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(config.LOG_DIR, f"rapid_{timestamp}.log")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(getattr(logging, config.FILE_LOG_LEVEL))
        file_handler.setFormatter(
            logging.Formatter(
                config.LOG_FORMAT_FILE,
                datefmt=config.LOG_DATE_FORMAT
            )
        )
        logger.addHandler(file_handler)

    _logger = logger
    return _logger


# =============================================================================
# Custom Transformers (sklearn API Compliant)
# =============================================================================

class TieredImputer(BaseEstimator, TransformerMixin):
    """Tiered imputation strategy based on missing data percentage.

    Applies different imputation strategies based on the percentage of
    missing values in each column:

    - < LOW_MISSING_THRESHOLD: Simple imputation (median for numeric)
    - LOW to MEDIUM threshold: KNN imputation
    - MEDIUM to HIGH threshold: Iterative imputation (MICE)
    - > HIGH threshold: Column should be dropped before this stage

    Parameters
    ----------
    low_threshold : float, default from config
        Threshold below which simple imputation is used.
    medium_threshold : float, default from config
        Threshold below which KNN imputation is used.
    high_threshold : float, default from config
        Threshold below which iterative imputation is used.
    n_neighbors : int or None, default None
        Number of neighbors for KNN imputation. If None, automatically
        calculated based on sample size using adaptive scaling.
    max_iter : int, default from config
        Maximum iterations for iterative imputation.
    random_state : int, default from config
        Random state for reproducibility.

    Attributes
    ----------
    n_neighbors_actual_ : int
        Actual number of neighbors used (after adaptive calculation).
    column_strategies_ : dict
        Mapping of column name to imputation strategy used.
    simple_imputer_ : SimpleImputer
        Fitted simple imputer for low-missing columns.
    knn_imputer_ : KNNImputer
        Fitted KNN imputer for medium-missing columns.
    iterative_imputer_ : IterativeImputer
        Fitted iterative imputer for high-missing columns.
    feature_names_ : list of str
        Column names from fitting.
    low_missing_cols_ : list of str
        Columns using simple imputation.
    medium_missing_cols_ : list of str
        Columns using KNN imputation.
    high_missing_cols_ : list of str
        Columns using iterative imputation.

    Notes
    -----
    The adaptive neighbor calculation uses:
        k = max(1, min(int(sqrt(n)), int(log10(n) * 10), n // 3))

    This formula:
    - Small datasets: sqrt and n//3 compete, keeping K conservative
    - Medium datasets: sqrt determines K
    - Large datasets: log10 * 10 caps growth, preventing excessive neighbors

    Examples
    --------
    >>> imputer = TieredImputer()
    >>> X_imputed = imputer.fit_transform(X_train)
    >>> print(f"Used {imputer.n_neighbors_actual_} neighbors for KNN")
    """

    @staticmethod
    def _calculate_adaptive_neighbors(n_samples: int) -> int:
        """Calculate optimal K for KNN imputation based on sample size.

        Uses sqrt scaling with log-based cap for large datasets.

        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset.

        Returns
        -------
        int
            Optimal number of neighbors.

        Examples
        --------
        >>> TieredImputer._calculate_adaptive_neighbors(9)
        3
        >>> TieredImputer._calculate_adaptive_neighbors(100)
        10
        >>> TieredImputer._calculate_adaptive_neighbors(10000)
        40
        """
        if n_samples <= 1:
            return 1

        sqrt_n = int(np.sqrt(n_samples))
        log_cap = int(np.log10(n_samples) * 10)
        proportional_cap = n_samples // 3

        return max(1, min(sqrt_n, log_cap, proportional_cap))

    def __init__(
        self,
        low_threshold: float | None = None,
        medium_threshold: float | None = None,
        high_threshold: float | None = None,
        n_neighbors: int | None = None,
        max_iter: int | None = None,
        random_state: int | None = None
    ) -> None:
        self.low_threshold = (
            low_threshold if low_threshold is not None
            else config.LOW_MISSING_THRESHOLD
        )
        self.medium_threshold = (
            medium_threshold if medium_threshold is not None
            else config.MEDIUM_MISSING_THRESHOLD
        )
        self.high_threshold = (
            high_threshold if high_threshold is not None
            else config.HIGH_MISSING_THRESHOLD
        )
        self.n_neighbors = n_neighbors  # None = adaptive
        self.max_iter = (
            max_iter if max_iter is not None
            else config.ITERATIVE_IMPUTER_MAX_ITER
        )
        self.random_state = (
            random_state if random_state is not None
            else config.RANDOM_STATE
        )

        # Fitted attributes
        self.n_neighbors_actual_: int = 0
        self.column_strategies_: dict[str, str] = {}
        self.simple_imputer_: SimpleImputer | None = None
        self.knn_imputer_: KNNImputer | None = None
        self.iterative_imputer_: IterativeImputer | None = None
        self.feature_names_: list[str] = []
        self.no_missing_cols_: list[str] = []
        self.low_missing_cols_: list[str] = []
        self.medium_missing_cols_: list[str] = []
        self.high_missing_cols_: list[str] = []

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None
    ) -> TieredImputer:
        """Fit the tiered imputer on training data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training data.
        y : pd.Series or np.ndarray, optional
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        self : TieredImputer
            Fitted transformer.
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self.feature_names_ = X_df.columns.tolist()
        n_samples = len(X_df)

        # Calculate adaptive neighbors if not specified
        if self.n_neighbors is None:
            self.n_neighbors_actual_ = self._calculate_adaptive_neighbors(
                n_samples
            )
        else:
            self.n_neighbors_actual_ = self.n_neighbors

        # Calculate missing percentage for each column
        missing_pct = X_df.isnull().mean()

        # Categorize columns by missing percentage
        self.no_missing_cols_ = []
        self.low_missing_cols_ = []
        self.medium_missing_cols_ = []
        self.high_missing_cols_ = []

        for col in self.feature_names_:
            pct = missing_pct[col]
            if pct == 0:
                self.no_missing_cols_.append(col)
                self.column_strategies_[col] = "none"
            elif pct < self.low_threshold:
                self.low_missing_cols_.append(col)
                self.column_strategies_[col] = "simple"
            elif pct < self.medium_threshold:
                self.medium_missing_cols_.append(col)
                self.column_strategies_[col] = "knn"
            else:
                self.high_missing_cols_.append(col)
                self.column_strategies_[col] = "iterative"

        # Fit imputers for each tier
        if self.low_missing_cols_:
            self.simple_imputer_ = SimpleImputer(strategy="median")
            self.simple_imputer_.fit(X_df[self.low_missing_cols_])

        if self.medium_missing_cols_:
            self.knn_imputer_ = KNNImputer(
                n_neighbors=self.n_neighbors_actual_
            )
            self.knn_imputer_.fit(X_df[self.medium_missing_cols_])

        if self.high_missing_cols_:
            self.iterative_imputer_ = IterativeImputer(
                max_iter=self.max_iter,
                random_state=self.random_state,
                skip_complete=True
            )
            self.iterative_imputer_.fit(X_df[self.high_missing_cols_])

        _get_logger().info(
            f"TieredImputer: {len(self.no_missing_cols_)} complete, "
            f"{len(self.low_missing_cols_)} simple, "
            f"{len(self.medium_missing_cols_)} KNN (k={self.n_neighbors_actual_}), "
            f"{len(self.high_missing_cols_)} iterative"
        )

        return self
        # FIX #1: Removed duplicate "return self" that was here

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform data by applying tiered imputation.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Data to transform.

        Returns
        -------
        np.ndarray
            Transformed data with no missing values.

        Raises
        ------
        ValueError
            If NaN values remain after imputation.
        RuntimeError
            If transformer has not been fitted.
        """
        if not self.feature_names_:
            raise RuntimeError("TieredImputer must be fitted before transform.")

        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # Ensure column order matches training
        X_df = X_df[self.feature_names_]

        # Apply imputation by tier
        result = X_df.copy()

        if self.low_missing_cols_ and self.simple_imputer_ is not None:
            result[self.low_missing_cols_] = self.simple_imputer_.transform(
                X_df[self.low_missing_cols_]
            )

        if self.medium_missing_cols_ and self.knn_imputer_ is not None:
            result[self.medium_missing_cols_] = self.knn_imputer_.transform(
                X_df[self.medium_missing_cols_]
            )

        if self.high_missing_cols_ and self.iterative_imputer_ is not None:
            result[self.high_missing_cols_] = self.iterative_imputer_.transform(
                X_df[self.high_missing_cols_]
            )

        result_array = result.values

        # Validate no NaN values remain
        if np.any(np.isnan(result_array)):
            nan_cols = result.columns[result.isnull().any()].tolist()
            raise ValueError(
                f"TieredImputer: NaN values remain after imputation in "
                f"columns: {nan_cols[:5]}{'...' if len(nan_cols) > 5 else ''}"
            )

        return result_array

    def get_imputation_report(self) -> pd.DataFrame:
        """Generate a report of imputation strategies by column.

        Returns
        -------
        pd.DataFrame
            Report with columns: column, strategy, missing_tier.
        """
        data = []
        for col, strategy in self.column_strategies_.items():
            data.append({
                "column": col,
                "strategy": strategy,
                "missing_tier": (
                    "none" if strategy == "none"
                    else f"<{self.low_threshold:.0%}" if strategy == "simple"
                    else f"{self.low_threshold:.0%}-{self.medium_threshold:.0%}"
                    if strategy == "knn"
                    else f"{self.medium_threshold:.0%}-{self.high_threshold:.0%}"
                )
            })
        return pd.DataFrame(data)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove highly correlated features to reduce multicollinearity.

    For each pair of features with correlation >= threshold, this
    transformer keeps the feature with higher correlation to the
    target variable and removes the other.

    Parameters
    ----------
    threshold : float, default=0.95
        Correlation threshold. Feature pairs with abs(correlation) >= threshold
        are considered redundant.

    Attributes
    ----------
    features_to_keep_ : list of str
        Names of features retained after filtering.
    features_dropped_ : list of str
        Names of features removed due to high correlation.

    Notes
    -----
    This transformer MUST be fitted on training data only to prevent
    data leakage. The target variable `y` is required for fitting.

    Examples
    --------
    >>> corr_filter = CorrelationFilter(threshold=0.95)
    >>> corr_filter.fit(X_train, y_train)
    >>> X_filtered = corr_filter.transform(X_train)
    """

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = (
            threshold if threshold is not None
            else config.FEATURE_CORRELATION_THRESHOLD
        )
        self.features_to_keep_: list[str] = []
        self.features_dropped_: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None
    ) -> CorrelationFilter:
        """Fit the correlation filter on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series or np.ndarray
            Target variable. Required for determining which correlated
            feature to keep.

        Returns
        -------
        self : CorrelationFilter
            Fitted transformer.

        Raises
        ------
        ValueError
            If y is None.
        """
        if y is None:
            raise ValueError(
                "CorrelationFilter requires target variable y for fitting."
            )

        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y_arr = np.asarray(y).ravel()

        # Compute correlation matrix
        corr_matrix = X_df.corr().abs()

        # Compute correlation with target for each feature
        target_corr = X_df.apply(
            lambda col: np.abs(np.corrcoef(col.values, y_arr)[0, 1])
            if col.notna().sum() > 1 else 0.0
        )

        # Identify features to drop
        to_drop: set[str] = set()
        cols = corr_matrix.columns.tolist()

        for i in range(len(cols)):
            if cols[i] in to_drop:
                continue

            for j in range(i + 1, len(cols)):
                if cols[j] in to_drop:
                    continue

                if corr_matrix.iloc[i, j] >= self.threshold:
                    # Keep feature with higher target correlation
                    if target_corr[cols[i]] >= target_corr[cols[j]]:
                        to_drop.add(cols[j])
                    else:
                        to_drop.add(cols[i])
                        break

        self.features_dropped_ = list(to_drop)
        self.features_to_keep_ = [c for c in cols if c not in to_drop]

        _get_logger().info(
            f"CorrelationFilter: Keeping {len(self.features_to_keep_)} features, "
            f"dropping {len(self.features_dropped_)}"
        )

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Transform data by removing correlated features.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Data to transform.

        Returns
        -------
        pd.DataFrame
            Transformed data with correlated features removed.
        """
        if not self.features_to_keep_:
            raise RuntimeError(
                "CorrelationFilter must be fitted before transform."
            )

        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # Handle potential column mismatches
        available = [f for f in self.features_to_keep_ if f in X_df.columns]

        return X_df[available]

    def get_feature_names_out(
        self,
        input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Get output feature names.

        Parameters
        ----------
        input_features : sequence of str, optional
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        np.ndarray
            Array of output feature names.
        """
        return np.array(self.features_to_keep_)


class FeatureImportanceSelector(BaseEstimator, TransformerMixin):
    """Select features based on ensemble feature importance.

    This transformer trains multiple tree-based models and aggregates
    their feature importance scores to select the most predictive
    features.

    Parameters
    ----------
    cumulative_threshold : float, default=0.95
        Select features until cumulative importance reaches this threshold.
    min_features : int, default=10
        Minimum number of features to select regardless of threshold.
    n_jobs : int, default from config
        Number of parallel jobs for model training.
    random_state : int, default from config
        Random state for reproducibility.

    Attributes
    ----------
    selected_features_ : list of str
        Names of selected features.
    feature_importances_ : dict
        Feature name to mean importance mapping.
    feature_ranks_ : dict
        Feature name to mean rank mapping.
    feature_std_ranks_ : dict
        Feature name to rank std deviation mapping (model consensus).

    Notes
    -----
    CRITICAL: This transformer must be fitted on training data only.
    Feature importance computed on the full dataset causes data leakage.

    Examples
    --------
    >>> selector = FeatureImportanceSelector(cumulative_threshold=0.95)
    >>> selector.fit(X_train, y_train)
    >>> X_selected = selector.transform(X_train)
    """

    def __init__(
        self,
        cumulative_threshold: float | None = None,
        min_features: int = 10,
        n_jobs: int | None = None,
        random_state: int | None = None
    ) -> None:
        self.cumulative_threshold = (
            cumulative_threshold if cumulative_threshold is not None
            else config.OPTIMIZATION_CDF_THRESHOLD
        )
        self.min_features = min_features
        self.n_jobs = n_jobs if n_jobs is not None else config.N_JOBS
        self.random_state = (
            random_state if random_state is not None else config.RANDOM_STATE
        )

        self.selected_features_: list[str] = []
        self.feature_importances_: dict[str, float] = {}
        self.feature_ranks_: dict[str, float] = {}
        self.feature_std_ranks_: dict[str, float] = {}

    def _get_importance_models(self) -> list[tuple[str, BaseEstimator]]:
        """Get ensemble of tree-based models for importance calculation.

        Models are selected based on config.FEATURE_IMPORTANCE_MODELS.
        To add/remove models, modify that config list.

        Returns
        -------
        list of (str, estimator) tuples
            Model name and instance pairs.

        Notes
        -----
        Available models: RF, XGB, GBT, CB, LGB, ETR
        """
        n_estimators = config.FEATURE_IMPORTANCE_N_ESTIMATORS

        # Registry of available models with their instantiation logic
        available_models: dict[str, callable] = {
            "RF": lambda: RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            ),
            "XGB": lambda: XGBRegressor(
                n_estimators=n_estimators,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbosity=0
            ),
            "GBT": lambda: GradientBoostingRegressor(
                n_estimators=n_estimators,
                random_state=self.random_state
            ),
            "CB": lambda: CatBoostRegressor(
                iterations=n_estimators,
                random_state=self.random_state,
                verbose=0,
                thread_count=self.n_jobs if self.n_jobs > 0 else -1
            ),
            "LGB": lambda: LGBMRegressor(
                n_estimators=n_estimators,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbosity=-1
            ),
            "ETR": lambda: ExtraTreesRegressor(
                n_estimators=n_estimators,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            ),
        }

        # Build model list from config, warn on unknown models
        models = []
        for name in config.FEATURE_IMPORTANCE_MODELS:
            if name in available_models:
                models.append((name, available_models[name]()))
            else:
                _get_logger().warning(
                    f"Unknown feature importance model '{name}' in config. "
                    f"Available: {list(available_models.keys())}"
                )

        if not models:
            raise ValueError(
                "No valid models in config.FEATURE_IMPORTANCE_MODELS. "
                f"Available: {list(available_models.keys())}"
            )

        return models

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None
    ) -> FeatureImportanceSelector:
        """Fit the feature selector on training data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training features.
        y : pd.Series or np.ndarray
            Target variable.

        Returns
        -------
        self : FeatureImportanceSelector
            Fitted transformer.

        Raises
        ------
        ValueError
            If y is None.
        """
        if y is None:
            raise ValueError(
                "FeatureImportanceSelector requires target y for fitting."
            )

        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        features = X_df.columns.tolist()
        num_features = len(features)

        # Aggregate importance across models
        feature_ranks: dict[str, list[int]] = {f: [] for f in features}
        feature_importances: dict[str, list[float]] = {f: [] for f in features}

        for name, model in self._get_importance_models():
            _get_logger().debug(f"Training {name} for feature importance...")

            # Scale data for consistent importance
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_df)

            model.fit(X_scaled, y)

            importances = model.feature_importances_
            normalized = importances / np.sum(importances)

            # Compute ranks (1 = most important)
            ranks = np.argsort(np.argsort(-normalized)) + 1

            for i, feat in enumerate(features):
                feature_ranks[feat].append(ranks[i])
                feature_importances[feat].append(normalized[i])

        # Aggregate statistics
        feature_stats = []
        for feat in features:
            feature_stats.append({
                "feature": feat,
                "mean_importance": float(np.mean(feature_importances[feat])),
                "mean_rank": float(np.mean(feature_ranks[feat])),
                "std_rank": float(np.std(feature_ranks[feat]))
            })

        # Sort by importance (descending)
        feature_stats = sorted(
            feature_stats,
            key=lambda x: x["mean_importance"],
            reverse=True
        )

        # Select features by cumulative importance
        total_importance = sum(f["mean_importance"] for f in feature_stats)
        cumulative = 0.0
        selected: list[str] = []

        for stat in feature_stats:
            cumulative += stat["mean_importance"]
            selected.append(stat["feature"])

            if cumulative / total_importance >= self.cumulative_threshold:
                break

        # Ensure minimum features
        min_count = max(
            self.min_features,
            int(np.ceil(num_features * config.MIN_FEATURE_RETENTION_RATIO))
        )
        if len(selected) < min_count:
            selected = [s["feature"] for s in feature_stats[:min_count]]

        self.selected_features_ = selected
        self.feature_importances_ = {
            s["feature"]: s["mean_importance"] for s in feature_stats
        }
        self.feature_ranks_ = {
            s["feature"]: s["mean_rank"] for s in feature_stats
        }
        self.feature_std_ranks_ = {
            s["feature"]: s["std_rank"] for s in feature_stats
        }

        _get_logger().info(
            f"FeatureImportanceSelector: Selected {len(selected)}/{num_features} "
            f"features (cumulative importance threshold: {self.cumulative_threshold})"
        )

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Transform data by selecting important features.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Data to transform.

        Returns
        -------
        pd.DataFrame
            Transformed data with only selected features.
        """
        if not self.selected_features_:
            raise RuntimeError(
                "FeatureImportanceSelector must be fitted before transform."
            )

        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        return X_df[self.selected_features_]

    def get_feature_names_out(
        self,
        input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Get output feature names.

        Parameters
        ----------
        input_features : sequence of str, optional
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        np.ndarray
            Array of selected feature names.
        """
        return np.array(self.selected_features_)


# =============================================================================
# Stacking Ensemble (Statistically Valid Implementation)
# =============================================================================

class RobustStackingRegressor:
    """Stacking ensemble using out-of-fold predictions to prevent leakage.

    This implementation uses cross_val_predict to generate out-of-fold
    predictions for the meta-learner, ensuring the meta-learner never
    sees predictions made on data the base models were trained on.

    Parameters
    ----------
    base_estimators : list of (str, estimator) tuples
        Base learner models.
    meta_estimator : estimator, default=Ridge(alpha=1.0)
        Meta-learner model.
    cv : int, default from config
        Number of cross-validation folds.
    n_jobs : int, default from config
        Number of parallel jobs.
    random_state : int, default from config
        Random state for reproducibility.
    passthrough : bool, default=False
        If True, include original features alongside base predictions.

    Attributes
    ----------
    fitted_base_estimators_ : list of (str, estimator) tuples
        Fitted base learner models.
    fitted_meta_estimator_ : estimator
        Fitted meta-learner model.
    base_scores_ : dict
        Cross-validation R² scores for each base learner.
    scaler_ : StandardScaler
        Fitted scaler for input features.

    Notes
    -----
    The mathematical justification for using out-of-fold predictions:

    INCORRECT (causes leakage):
        meta_features = [model.predict(X_train) for model in base_models]
        # Base models were trained on X_train, so predictions are overfit

    CORRECT (no leakage):
        meta_features = [cross_val_predict(model, X_train, y_train, cv=5)
                         for model in base_models]
        # Each prediction was made by a model that didn't see that sample
    """

    def __init__(
        self,
        base_estimators: list[tuple[str, BaseEstimator]],
        meta_estimator: BaseEstimator | None = None,
        cv: int | None = None,
        n_jobs: int | None = None,
        random_state: int | None = None,
        passthrough: bool = False
    ) -> None:
        self.base_estimators = base_estimators
        self.meta_estimator = (
            meta_estimator if meta_estimator is not None
            else Ridge(alpha=1.0)
        )
        self.cv = cv if cv is not None else config.CV_FOLDS
        self.n_jobs = n_jobs if n_jobs is not None else config.N_JOBS
        self.random_state = (
            random_state if random_state is not None else config.RANDOM_STATE
        )
        self.passthrough = passthrough

        self.fitted_base_estimators_: list[tuple[str, BaseEstimator]] = []
        self.fitted_meta_estimator_: BaseEstimator | None = None
        self.base_scores_: dict[str, float] = {}
        self.scaler_: StandardScaler | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RobustStackingRegressor:
        """Fit the stacking ensemble.

        Parameters
        ----------
        X : np.ndarray
            Training features (should be pre-scaled or raw).
        y : np.ndarray
            Training target.

        Returns
        -------
        self : RobustStackingRegressor
            Fitted ensemble.
        """
        n_samples = X.shape[0]
        n_base = len(self.base_estimators)

        # Scale input features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Generate out-of-fold predictions
        meta_features = np.zeros((n_samples, n_base))

        cv_splitter = KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

        for i, (name, estimator) in enumerate(self.base_estimators):
            _get_logger().debug(f"Generating OOF predictions for {name}...")

            est = clone(estimator)

            # Out-of-fold predictions
            oof_preds = cross_val_predict(
                est, X_scaled, y, cv=cv_splitter, n_jobs=self.n_jobs
            )
            meta_features[:, i] = oof_preds

            # Track CV score
            cv_scores = cross_val_score(
                clone(estimator), X_scaled, y,
                cv=cv_splitter, scoring="r2", n_jobs=self.n_jobs
            )
            self.base_scores_[name] = float(np.mean(cv_scores))

        # Optionally include original features
        if self.passthrough:
            meta_features = np.hstack([meta_features, X_scaled])

        # Fit meta-learner on OOF predictions
        self.fitted_meta_estimator_ = clone(self.meta_estimator)
        self.fitted_meta_estimator_.fit(meta_features, y)

        # Refit base estimators on full training data for inference
        self.fitted_base_estimators_ = []
        for name, estimator in self.base_estimators:
            fitted_est = clone(estimator)
            fitted_est.fit(X_scaled, y)
            self.fitted_base_estimators_.append((name, fitted_est))

        return self

    def fit_from_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
        oof_predictions: dict[str, np.ndarray],
        fitted_estimators: dict[str, BaseEstimator]
    ) -> RobustStackingRegressor:
        """Fit stacking using pre-computed OOF predictions.

        This method avoids redundant re-training of base models by using
        out-of-fold predictions that were already generated during base
        model training.

        Parameters
        ----------
        X : np.ndarray
            Training features (for scaler fitting and passthrough).
        y : np.ndarray
            Training target.
        oof_predictions : dict
            Mapping of model name to OOF prediction array.
            Keys should match the names in base_estimators.
        fitted_estimators : dict
            Mapping of model name to fitted estimator (Pipeline).
            Used for inference on new data.

        Returns
        -------
        self : RobustStackingRegressor
            Fitted ensemble.

        Notes
        -----
        This is more efficient than fit() when base models have already
        been trained and OOF predictions are available.
        """
        # Scale input features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Build meta-features from pre-computed OOF predictions
        meta_features_list = []
        for name, _ in self.base_estimators:
            if name not in oof_predictions:
                raise ValueError(
                    f"Missing OOF predictions for '{name}'. "
                    f"Available: {list(oof_predictions.keys())}"
                )
            meta_features_list.append(oof_predictions[name])

        meta_features = np.column_stack(meta_features_list)

        # Optionally include original features
        if self.passthrough:
            meta_features = np.hstack([meta_features, X_scaled])

        # Fit meta-learner on OOF predictions
        self.fitted_meta_estimator_ = clone(self.meta_estimator)
        self.fitted_meta_estimator_.fit(meta_features, y)

        # Store pre-fitted base estimators for inference
        self.fitted_base_estimators_ = []
        for name, _ in self.base_estimators:
            if name not in fitted_estimators:
                raise ValueError(
                    f"Missing fitted estimator for '{name}'. "
                    f"Available: {list(fitted_estimators.keys())}"
                )
            # Extract the model from the pipeline
            pipeline = fitted_estimators[name]
            if hasattr(pipeline, 'named_steps'):
                # It's a Pipeline, extract the model
                model = pipeline.named_steps.get('model', pipeline)
                self.fitted_base_estimators_.append((name, model))
            else:
                self.fitted_base_estimators_.append((name, pipeline))

        # Store base scores (from OOF R² if available)
        for name, oof in oof_predictions.items():
            self.base_scores_[name] = float(r2_score(y, oof))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the fitted ensemble.

        Parameters
        ----------
        X : np.ndarray
            Features to predict on.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if self.scaler_ is None or self.fitted_meta_estimator_ is None:
            raise RuntimeError(
                "RobustStackingRegressor must be fitted before predict."
            )

        X_scaled = self.scaler_.transform(X)

        # Generate base predictions
        meta_features = np.column_stack([
            est.predict(X_scaled)
            for _, est in self.fitted_base_estimators_
        ])

        if self.passthrough:
            meta_features = np.hstack([meta_features, X_scaled])

        return self.fitted_meta_estimator_.predict(meta_features)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score on given data.

        Parameters
        ----------
        X : np.ndarray
            Features.
        y : np.ndarray
            True target values.

        Returns
        -------
        float
            R² score.
        """
        return float(r2_score(y, self.predict(X)))


# =============================================================================
# Model Factory Functions
# =============================================================================

def get_base_learners() -> list[tuple[str, BaseEstimator]]:
    """Create base learner models with configuration from config.py.

    Returns
    -------
    list of (str, estimator) tuples
        Model name and instance pairs.
    """
    return [
        ("CB", CatBoostRegressor(
            verbose=0,
            random_state=config.RANDOM_STATE,
            thread_count=config.N_JOBS if config.N_JOBS > 0 else -1,
            task_type=config.CATBOOST_TASK_TYPE
        )),
        ("LGB", LGBMRegressor(
            verbose=-1,
            n_jobs=config.N_JOBS,
            random_state=config.RANDOM_STATE,
            device=config.LIGHTGBM_DEVICE
        )),
        ("XGB", XGBRegressor(
            n_jobs=config.N_JOBS,
            random_state=config.RANDOM_STATE,
            verbosity=0,
            tree_method=config.XGBOOST_TREE_METHOD
        )),
        ("RF", RandomForestRegressor(
            n_jobs=config.N_JOBS,
            random_state=config.RANDOM_STATE
        )),
        ("GBT", GradientBoostingRegressor(
            random_state=config.RANDOM_STATE
        )),
        ("KNN", KNeighborsRegressor(
            n_jobs=config.N_JOBS
        )),
        ("ETR", ExtraTreesRegressor(
            n_jobs=config.N_JOBS,
            random_state=config.RANDOM_STATE
        )),
        ("Bag", BaggingRegressor(
            n_jobs=config.N_JOBS,
            random_state=config.RANDOM_STATE
        )),
    ]


def get_param_distributions() -> dict[str, dict[str, Any]]:
    """Get hyperparameter distributions for RandomizedSearchCV.

    Returns
    -------
    dict
        Mapping of model name to parameter distributions.

    Notes
    -----
    Parameter names use 'model__' prefix for Pipeline compatibility.
    """
    return {
        "KNN": {
            "model__n_neighbors": randint(3, 30),
            "model__weights": ["uniform", "distance"],
            "model__p": randint(1, 3)
        },
        "XGB": {
            "model__learning_rate": uniform(0.01, 0.29),
            "model__n_estimators": randint(100, 500),
            "model__max_depth": randint(3, 10),
            "model__min_child_weight": randint(1, 10),
            "model__subsample": uniform(0.5, 0.5),
            "model__colsample_bytree": uniform(0.5, 0.5),
        },
        "RF": {
            "model__n_estimators": randint(100, 500),
            "model__max_depth": [None, 5, 10, 15, 20],
            "model__min_samples_split": randint(2, 21),
            "model__min_samples_leaf": randint(1, 21),
            "model__max_features": ["sqrt", "log2", None],
        },
        "GBT": {
            "model__n_estimators": randint(100, 300),
            "model__learning_rate": uniform(0.01, 0.29),
            "model__max_depth": randint(3, 10),
            "model__min_samples_split": randint(2, 21),
            "model__subsample": uniform(0.5, 0.5),
        },
        "ETR": {
            "model__n_estimators": randint(100, 300),
            "model__max_depth": [None, 5, 10, 15],
            "model__min_samples_split": randint(2, 21),
            "model__min_samples_leaf": randint(1, 21),
        },
        "Bag": {
            "model__n_estimators": randint(10, 100),
            "model__max_samples": uniform(0.5, 0.5),
            "model__max_features": uniform(0.5, 0.5),
        },
        "LGB": {
            "model__n_estimators": randint(50, 500),
            "model__learning_rate": uniform(0.01, 0.29),
            "model__num_leaves": randint(20, 150),
            "model__max_depth": randint(3, 15),
            "model__subsample": uniform(0.5, 0.5),
            "model__colsample_bytree": uniform(0.5, 0.5),
        },
        "CB": {
            "model__iterations": randint(50, 500),
            "model__depth": randint(3, 10),
            "model__l2_leaf_reg": uniform(1, 9),
            "model__learning_rate": uniform(0.01, 0.29),
        },
    }


# =============================================================================
# Main Pipeline Orchestrator
# =============================================================================

class RAPIDPipeline:
    """Main orchestrator for the RAPID machine learning pipeline.

    This class manages the complete workflow from data loading through
    model evaluation, ensuring proper data separation at each step to
    prevent leakage.

    The pipeline uses a two-stage imputation strategy:

    1. **Feature Selection Stage**: Imputes all columns to enable
       correlation filtering and feature importance calculation.
    2. **Production Stage**: Refits imputer on selected features only,
       so inference requires only those features.

    Attributes
    ----------
    raw_data_ : pd.DataFrame
        Loaded and preprocessed data.
    target_column_ : str
        Name of the target variable column.
    feature_columns_ : list of str
        Names of all original feature columns.
    X_train_, X_test_ : pd.DataFrame
        Train/test feature splits.
    y_train_, y_test_ : pd.Series
        Train/test target splits.
    feature_selection_imputer_ : TieredImputer
        Imputer fitted on all columns (used during feature selection).
    production_imputer_ : TieredImputer
        Imputer fitted on selected columns only (used for inference).
    correlation_filter_ : CorrelationFilter
        Fitted correlation filter.
    feature_selector_ : FeatureImportanceSelector
        Fitted feature selector.
    selected_feature_names_ : list of str
        Final list of selected features required for inference.
    best_models_ : dict
        Trained and optimized models.
    stacking_model_ : RobustStackingRegressor
        Trained stacking ensemble.
    model_scores_ : dict
        Evaluation metrics for all models.

    Examples
    --------
    >>> # Training
    >>> pipeline = RAPIDPipeline()
    >>> pipeline.load_data('data.csv', 'target')
    >>> pipeline.preprocess()
    >>> pipeline.split_data()
    >>> pipeline.fit_feature_selection()
    >>> pipeline.train_base_models()
    >>> pipeline.train_stacking_ensemble()
    >>> results = pipeline.evaluate()
    >>>
    >>> # Inference (only needs selected features)
    >>> new_data = pd.read_csv('new_data.csv')
    >>> X_new = pipeline.transform_new_data(new_data)
    >>> predictions = pipeline.best_models_['CB']['estimator'].predict(X_new)
    """

    def __init__(self) -> None:
        """Initialize the RAPID pipeline."""
        self.raw_data_: pd.DataFrame | None = None
        self.target_column_: str | None = None
        self.feature_columns_: list[str] = []

        self.X_train_: pd.DataFrame | None = None
        self.X_test_: pd.DataFrame | None = None
        self.y_train_: pd.Series | None = None
        self.y_test_: pd.Series | None = None

        # Two-stage imputation:
        # Stage 1: Used during feature selection (all columns)
        # Stage 2: Used for production (selected columns only)
        self.feature_selection_imputer_: TieredImputer | None = None
        self.production_imputer_: TieredImputer | None = None

        self.correlation_filter_: CorrelationFilter | None = None
        self.feature_selector_: FeatureImportanceSelector | None = None

        # Final selected feature names (for production input validation)
        self.selected_feature_names_: list[str] = []

        self.best_models_: dict[str, dict[str, Any]] = {}
        self.oof_predictions_: dict[str, np.ndarray] = {}  # OOF for stacking
        self.stacking_model_: RobustStackingRegressor | None = None
        self.model_scores_: dict[str, dict[str, Any]] = {}

    def load_data(
        self,
        filepath: str,
        target_column: str,
        encoding: str = "utf-8"
    ) -> RAPIDPipeline:
        """Load data from CSV file.

        Parameters
        ----------
        filepath : str
            Path to CSV file.
        target_column : str
            Name of the target variable column.
        encoding : str, default='utf-8'
            File encoding.

        Returns
        -------
        self : RAPIDPipeline
            Returns self for method chaining.

        Raises
        ------
        FileNotFoundError
            If filepath does not exist.
        ValueError
            If target_column is not in the data.

        Notes
        -----
        Column names are normalized to lowercase with whitespace stripped
        for consistency.
        """
        _get_logger().info(f"Loading data from {filepath}")

        try:
            self.raw_data_ = pd.read_csv(filepath, encoding=encoding)
        except UnicodeDecodeError:
            _get_logger().warning(
                "UTF-8 decode failed, trying latin-1 encoding"
            )
            self.raw_data_ = pd.read_csv(filepath, encoding="latin-1")

        # Normalize column names: lowercase and strip whitespace
        original_columns = list(self.raw_data_.columns)
        self.raw_data_.columns = (
            self.raw_data_.columns.str.strip().str.lower()
        )
        
        # Also normalize target_column for matching
        target_column_normalized = target_column.strip().lower()
        
        # Log any column name changes
        new_columns = list(self.raw_data_.columns)
        changed = [
            (orig, new) for orig, new in zip(original_columns, new_columns)
            if orig != new
        ]
        if changed:
            _get_logger().debug(
                f"Normalized {len(changed)} column names to lowercase"
            )

        if target_column_normalized not in self.raw_data_.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in data. "
                f"Available columns: {list(self.raw_data_.columns[:10])}..."
            )

        self.target_column_ = target_column_normalized

        _get_logger().info(
            f"Data loaded: {self.raw_data_.shape[0]:,} rows × "
            f"{self.raw_data_.shape[1]} columns"
        )
        _get_logger().info(f"Target column: {self.target_column_}")

        return self

    def preprocess(
        self,
        drop_high_missing_threshold: float | None = None,
        encode_categoricals: bool = True
    ) -> RAPIDPipeline:
        """Preprocess data by handling missing values and problematic columns.

        Parameters
        ----------
        drop_high_missing_threshold : float, optional
            Threshold for dropping columns with excessive missing data.
            Defaults to config.MAX_MISSING_DATA.
        encode_categoricals : bool, default=True
            Whether to encode categorical columns using tiered strategy:
            - ≤ ONE_HOT_ENCODING_MAX_CATEGORIES: one-hot encoding
            - ≤ LABEL_ENCODING_MAX_CATEGORIES: label encoding
            - > LABEL_ENCODING_MAX_CATEGORIES: drop (likely ID column)

        Returns
        -------
        self : RAPIDPipeline
            Returns self for method chaining.

        Notes
        -----
        Date-like columns are detected by both dtype and column name patterns
        (configured in config.DATE_PATTERNS) and automatically dropped.
        """
        _get_logger().info("Starting preprocessing...")

        threshold = (
            drop_high_missing_threshold
            if drop_high_missing_threshold is not None
            else config.MAX_MISSING_DATA
        )

        df = self.raw_data_.copy()

        # Drop rows with missing target
        initial_rows = len(df)
        df = df.dropna(subset=[self.target_column_])
        dropped_rows = initial_rows - len(df)

        if dropped_rows > 0:
            _get_logger().info(
                f"Dropped {dropped_rows:,} rows with missing target "
                f"({dropped_rows / initial_rows:.1%})"
            )

        # Drop high-missing columns
        missing_pct = df.isnull().mean()
        high_missing = missing_pct[missing_pct > threshold].index.tolist()
        high_missing = [c for c in high_missing if c != self.target_column_]

        if high_missing:
            _get_logger().info(
                f"Dropping {len(high_missing)} columns with "
                f">{threshold:.0%} missing data"
            )
            df = df.drop(columns=high_missing)

        # Detect and drop date-like columns
        date_cols = self._detect_date_columns(df)
        if date_cols:
            _get_logger().info(
                f"Dropping {len(date_cols)} date-like columns: {date_cols}"
            )
            df = df.drop(columns=date_cols)

        # Handle string/object columns
        string_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if self.target_column_ in string_cols:
            string_cols.remove(self.target_column_)

        if string_cols:
            if encode_categoricals:
                df = self._encode_categorical_columns(df, string_cols)
            else:
                _get_logger().info(
                    f"Dropping {len(string_cols)} string columns"
                )
                df = df.drop(columns=string_cols)

        # Store feature columns
        self.feature_columns_ = [
            c for c in df.columns if c != self.target_column_
        ]
        self.raw_data_ = df

        _get_logger().info(
            f"After preprocessing: {df.shape[0]:,} rows × "
            f"{len(self.feature_columns_)} features"
        )

        return self

    def _detect_date_columns(self, df: pd.DataFrame) -> list[str]:
        """Detect date-like columns by dtype and name patterns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze.

        Returns
        -------
        list of str
            Column names identified as date-like.
        """
        date_cols = []

        for col in df.columns:
            if col == self.target_column_:
                continue

            # Check if already datetime dtype
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
                continue

            # Check column name against patterns
            col_lower = col.lower()
            if any(
                pattern in col_lower
                for pattern in config.DATE_PATTERNS
            ):
                date_cols.append(col)
                continue

            # For object columns, try to parse as date
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    try:
                        pd.to_datetime(sample, format='mixed')
                        # If successful, it's likely a date column
                        date_cols.append(col)
                        _get_logger().debug(
                            f"Column '{col}' detected as date by content"
                        )
                    except (ValueError, TypeError):
                        pass

        return date_cols

    def _encode_categorical_columns(
        self,
        df: pd.DataFrame,
        string_cols: list[str]
    ) -> pd.DataFrame:
        """Encode categorical columns using tiered strategy.

        Strategy based on cardinality:
        - ≤ ONE_HOT_ENCODING_MAX_CATEGORIES: one-hot encoding
        - ≤ LABEL_ENCODING_MAX_CATEGORIES: label encoding
        - > LABEL_ENCODING_MAX_CATEGORIES: drop (likely ID)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to process.
        string_cols : list of str
            String column names to encode.

        Returns
        -------
        pd.DataFrame
            DataFrame with encoded columns.
        """
        one_hot_threshold = config.ONE_HOT_ENCODING_MAX_CATEGORIES
        label_threshold = config.LABEL_ENCODING_MAX_CATEGORIES

        one_hot_cols = []
        label_encode_cols = []
        drop_cols = []

        for col in string_cols:
            n_unique = df[col].nunique()

            if n_unique <= one_hot_threshold:
                one_hot_cols.append(col)
            elif n_unique <= label_threshold:
                label_encode_cols.append(col)
            else:
                drop_cols.append(col)

        # Log encoding decisions
        if one_hot_cols:
            _get_logger().info(
                f"One-hot encoding {len(one_hot_cols)} columns "
                f"(≤{one_hot_threshold} categories): {one_hot_cols}"
            )
        if label_encode_cols:
            _get_logger().info(
                f"Label encoding {len(label_encode_cols)} columns "
                f"({one_hot_threshold+1}-{label_threshold} categories): "
                f"{label_encode_cols}"
            )
        if drop_cols:
            _get_logger().info(
                f"Dropping {len(drop_cols)} high-cardinality columns "
                f"(>{label_threshold} categories, likely IDs): {drop_cols}"
            )

        # Apply one-hot encoding
        if one_hot_cols:
            df = pd.get_dummies(
                df,
                columns=one_hot_cols,
                prefix=one_hot_cols,
                drop_first=True,  # Avoid multicollinearity
                dtype=float
            )

        # Apply label encoding
        for col in label_encode_cols:
            # Create mapping preserving NaN
            unique_vals = df[col].dropna().unique()
            mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
            df[col] = df[col].map(mapping)
            # Column is now numeric (with NaN preserved)

        # Drop high-cardinality columns
        if drop_cols:
            df = df.drop(columns=drop_cols)

        return df

    def split_data(self) -> RAPIDPipeline:
        """Split data into training and test sets.

        Returns
        -------
        self : RAPIDPipeline
            Returns self for method chaining.

        Notes
        -----
        Uses config.TEST_SIZE and config.RANDOM_STATE for splitting.
        """
        _get_logger().info("Splitting data into train/test sets...")

        X = self.raw_data_[self.feature_columns_]
        y = self.raw_data_[self.target_column_]

        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = (
            train_test_split(
                X, y,
                test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE
            )
        )

        _get_logger().info(
            f"Training samples: {len(self.X_train_):,} "
            f"({1 - config.TEST_SIZE:.0%})"
        )
        _get_logger().info(
            f"Test samples: {len(self.X_test_):,} "
            f"({config.TEST_SIZE:.0%})"
        )

        return self

    def fit_feature_selection(self) -> RAPIDPipeline:
        """Fit feature selection transformers on training data.

        Returns
        -------
        self : RAPIDPipeline
            Returns self for method chaining.

        Notes
        -----
        CRITICAL: All feature selection is fitted on training data only
        to prevent data leakage.

        This method uses a two-stage imputation strategy:
        1. Stage 1 imputer: Fits on all columns to enable feature selection
        2. Stage 2 imputer: Refits on selected columns only for production

        At inference time, only Stage 2 imputer is used, requiring only
        the selected features as input.
        """
        _get_logger().info("Fitting feature selection (TRAIN DATA ONLY)...")

        # =====================================================================
        # STAGE 1: Feature Selection (uses all columns)
        # =====================================================================

        # Step 1a: Impute missing values on all columns
        self.feature_selection_imputer_ = TieredImputer()
        X_train_imputed = self.feature_selection_imputer_.fit_transform(
            self.X_train_
        )
        X_train_imputed = pd.DataFrame(
            X_train_imputed,
            columns=self.feature_columns_,
            index=self.X_train_.index
        )

        # Step 1b: Remove highly correlated features
        self.correlation_filter_ = CorrelationFilter(
            threshold=config.FEATURE_CORRELATION_THRESHOLD
        )
        self.correlation_filter_.fit(X_train_imputed, self.y_train_)
        X_train_filtered = self.correlation_filter_.transform(X_train_imputed)

        _get_logger().info(
            f"After correlation filter: "
            f"{len(self.correlation_filter_.features_to_keep_)} features"
        )

        # Step 1c: Feature importance selection
        self.feature_selector_ = FeatureImportanceSelector(
            cumulative_threshold=config.OPTIMIZATION_CDF_THRESHOLD,
            min_features=config.MIN_FEATURES_TO_SELECT,
            n_jobs=config.N_JOBS,
            random_state=config.RANDOM_STATE
        )
        self.feature_selector_.fit(X_train_filtered, self.y_train_)

        # Store final selected feature names
        self.selected_feature_names_ = self.feature_selector_.selected_features_

        _get_logger().info(
            f"After importance selection: "
            f"{len(self.selected_feature_names_)} features"
        )

        # =====================================================================
        # STAGE 2: Production Imputer (selected columns only)
        # =====================================================================

        # Extract only the selected features from raw training data
        X_train_selected_raw = self.X_train_[self.selected_feature_names_]

        # Fit production imputer on selected features only
        self.production_imputer_ = TieredImputer()
        self.production_imputer_.fit(X_train_selected_raw)

        _get_logger().info(
            "Production imputer fitted on selected features only"
        )

        return self

    def _get_transformed_data(
        self,
        X: pd.DataFrame,
        fit_imputer: bool = False
    ) -> np.ndarray:
        """Apply production transformations to data.

        This method uses the production imputer which operates only on
        selected features. Input data should contain at least the
        selected features.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform. Must contain all selected features.
        fit_imputer : bool, default=False
            If True, refits the production imputer. Should only be True
            when called on training data during model training setup.

        Returns
        -------
        np.ndarray
            Transformed feature array ready for model input.
        """
        if not self.selected_feature_names_:
            raise RuntimeError(
                "Feature selection must be completed before transformation. "
                "Call fit_feature_selection() first."
            )

        # Extract only selected features
        X_selected = X[self.selected_feature_names_]

        # Impute using production imputer (selected features only)
        if fit_imputer:
            X_imputed = self.production_imputer_.fit_transform(X_selected)
        else:
            X_imputed = self.production_imputer_.transform(X_selected)

        return X_imputed

    def transform_new_data(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data for prediction using the production pipeline.

        This is the method to use at inference time. It requires only
        the selected features as input, not all original columns.

        Parameters
        ----------
        X : pd.DataFrame
            New data to transform. Must contain columns matching
            `selected_feature_names_`. Extra columns are ignored.

        Returns
        -------
        np.ndarray
            Transformed feature array ready for model prediction.

        Raises
        ------
        ValueError
            If required features are missing from input.

        Examples
        --------
        >>> # After pipeline is trained
        >>> new_data = pd.read_csv('new_observations.csv')
        >>> X_transformed = pipeline.transform_new_data(new_data)
        >>> predictions = pipeline.best_models_['CB']['estimator'].predict(
        ...     X_transformed
        ... )
        """
        if not self.selected_feature_names_:
            raise RuntimeError(
                "Pipeline must be fitted before transforming new data."
            )

        # Normalize column names to match training
        X = X.copy()
        X.columns = X.columns.str.strip().str.lower()

        required_features = set(self.selected_feature_names_)
        provided_features = set(X.columns)

        # Check for missing required features
        missing_features = required_features - provided_features
        if missing_features:
            raise ValueError(
                f"Input data missing required features: {missing_features}"
            )

        # Warn about extra columns (will be ignored)
        extra_features = provided_features - required_features
        if extra_features:
            _get_logger().warning(
                f"Ignoring {len(extra_features)} extra columns not in model: "
                f"{sorted(extra_features)[:5]}{'...' if len(extra_features) > 5 else ''}"
            )

        # Extract only required features in correct order
        X_selected = X[self.selected_feature_names_]
        X_imputed = self.production_imputer_.transform(X_selected)

        return X_imputed

    def train_base_models(self) -> RAPIDPipeline:
        """Train and optimize base learner models.

        Also generates out-of-fold predictions for efficient stacking
        ensemble training (avoids redundant re-training).

        Returns
        -------
        self : RAPIDPipeline
            Returns self for method chaining.
        """
        _get_logger().info("Training base learner models...")

        X_train_transformed = self._get_transformed_data(
            self.X_train_, fit_imputer=True
        )
        y_train = self.y_train_.values

        base_learners = get_base_learners()
        param_distributions = get_param_distributions()

        # Helper for n_jobs with None protection
        n_jobs = config.N_JOBS if config.N_JOBS is not None else -1

        # CV splitter for OOF predictions (consistent across all models)
        cv_splitter = KFold(
            n_splits=config.CV_FOLDS,
            shuffle=True,
            random_state=config.RANDOM_STATE
        )

        for name, model in base_learners:
            _get_logger().info(f"Training {name}...")

            # Create pipeline with scaler
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])

            # Baseline CV score
            cv_scores = cross_val_score(
                pipeline, X_train_transformed, y_train,
                cv=cv_splitter,
                scoring="r2",
                n_jobs=n_jobs
            )
            baseline_score = float(np.mean(cv_scores))

            _get_logger().info(f"  {name} baseline CV R²: {baseline_score:.4f}")

            # Skip if below threshold
            if baseline_score < config.CUTOFF_R2:
                _get_logger().info(f"  {name} below threshold, skipping")
                continue

            # Check if hyperparameter tuning is enabled for this model
            tuning_enabled = config.HYPERPARAM_TUNING_ENABLED.get(name, True)

            # Hyperparameter optimization
            if name in param_distributions and tuning_enabled:
                search = RandomizedSearchCV(
                    pipeline,
                    param_distributions[name],
                    n_iter=config.HYPERPARAM_SEARCH_ITER,
                    cv=cv_splitter,
                    scoring="r2",
                    n_jobs=n_jobs,
                    random_state=config.RANDOM_STATE,
                    error_score="raise"
                )

                try:
                    search.fit(X_train_transformed, y_train)
                    optimized_score = float(search.best_score_)

                    if optimized_score >= baseline_score:
                        final_estimator = search.best_estimator_
                        final_score = optimized_score
                        optimized = True
                        _get_logger().info(
                            f"  {name} optimized CV R²: {optimized_score:.4f} "
                            f"(+{optimized_score - baseline_score:.4f})"
                        )
                    else:
                        pipeline.fit(X_train_transformed, y_train)
                        final_estimator = pipeline
                        final_score = baseline_score
                        optimized = False
                        _get_logger().info(
                            f"  {name} keeping baseline (optimization degraded)"
                        )

                except Exception as e:
                    _get_logger().warning(f"  {name} optimization failed: {e}")
                    pipeline.fit(X_train_transformed, y_train)
                    final_estimator = pipeline
                    final_score = baseline_score
                    optimized = False
            else:
                if not tuning_enabled:
                    _get_logger().info(
                        f"  {name} hyperparameter tuning disabled in config"
                    )
                pipeline.fit(X_train_transformed, y_train)
                final_estimator = pipeline
                final_score = baseline_score
                optimized = False

            # Store best model
            self.best_models_[name] = {
                "estimator": final_estimator,
                "cv_score": final_score,
                "params": getattr(search, 'best_params_', {}) if 'search' in dir() and optimized else {},
                "optimized": optimized
            }

            # Generate OOF predictions using the final estimator configuration
            # This avoids re-training during stacking
            _get_logger().debug(f"  {name} generating OOF predictions...")
            oof_predictions = cross_val_predict(
                clone(final_estimator),
                X_train_transformed,
                y_train,
                cv=cv_splitter,
                n_jobs=config.N_JOBS
            )
            self.oof_predictions_[name] = oof_predictions

        _get_logger().info(
            f"Trained {len(self.best_models_)} models passing threshold"
        )

        return self

    def train_stacking_ensemble(self) -> RAPIDPipeline:
        """Train stacking ensemble using pre-computed OOF predictions.

        Returns
        -------
        self : RAPIDPipeline
            Returns self for method chaining.

        Notes
        -----
        Uses out-of-fold predictions generated during train_base_models()
        to prevent data leakage and avoid redundant re-training.
        """
        if not config.RUN_STACKING_ANALYSIS:
            _get_logger().info("Stacking analysis disabled in config")
            return self

        _get_logger().info("Training stacking ensemble (using pre-computed OOF)...")

        if len(self.best_models_) < 2:
            _get_logger().warning("Not enough models for stacking (need >= 2)")
            return self

        if len(self.oof_predictions_) < 2:
            _get_logger().warning(
                "Not enough OOF predictions for stacking. "
                "Ensure train_base_models() was called first."
            )
            return self

        X_train_transformed = self._get_transformed_data(
            self.X_train_, fit_imputer=False
        )
        y_train = self.y_train_.values

        # Build base_estimators list from best_models (for reference)
        base_estimators = []
        fitted_estimators = {}
        for name, info in self.best_models_.items():
            pipeline = info["estimator"]
            model = clone(pipeline.named_steps["model"])
            base_estimators.append((name, model))
            fitted_estimators[name] = pipeline

        # Test different meta-learners
        meta_learners = [
            ("Ridge", Ridge(alpha=1.0)),
            ("Linear", LinearRegression()),
            ("RF", RandomForestRegressor(
                n_estimators=50, random_state=config.RANDOM_STATE
            )),
        ]

        best_score = -np.inf
        best_stacking = None
        best_meta_name = None

        X_test_transformed = self._get_transformed_data(
            self.X_test_, fit_imputer=False
        )

        for meta_name, meta_learner in meta_learners:
            _get_logger().info(f"  Testing meta-learner: {meta_name}")

            stacking = RobustStackingRegressor(
                base_estimators=base_estimators,
                meta_estimator=meta_learner,
                cv=config.CV_FOLDS,
                n_jobs=config.N_JOBS,
                random_state=config.RANDOM_STATE,
                passthrough=False
            )

            # Use pre-computed OOF predictions instead of re-training
            stacking.fit_from_oof(
                X_train_transformed,
                y_train,
                oof_predictions=self.oof_predictions_,
                fitted_estimators=fitted_estimators
            )

            test_score = stacking.score(
                X_test_transformed, self.y_test_.values
            )

            _get_logger().info(f"    {meta_name} test R²: {test_score:.4f}")

            if test_score > best_score:
                best_score = test_score
                best_stacking = stacking
                best_meta_name = meta_name

        self.stacking_model_ = best_stacking
        self.model_scores_["Stacking"] = {
            "test_r2": best_score,
            "meta_learner": best_meta_name
        }

        _get_logger().info(
            f"Best stacking: {best_meta_name} (R²={best_score:.4f})"
        )

        return self

    def evaluate(self) -> dict[str, dict[str, Any]]:
        """Evaluate all models on the held-out test set.

        Returns
        -------
        dict
            Model evaluation metrics including R², RMSE, and MAE.
        """
        _get_logger().info("Evaluating models on test set...")

        X_test_transformed = self._get_transformed_data(
            self.X_test_, fit_imputer=False
        )
        y_test = self.y_test_.values

        results: dict[str, dict[str, Any]] = {}

        for name, info in self.best_models_.items():
            pipeline = info["estimator"]
            y_pred = pipeline.predict(X_test_transformed)

            results[name] = {
                "cv_r2": info["cv_score"],
                "test_r2": float(r2_score(y_test, y_pred)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "test_mae": float(mean_absolute_error(y_test, y_pred)),
                "optimized": info["optimized"]
            }

            _get_logger().info(
                f"  {name}: CV R²={results[name]['cv_r2']:.4f}, "
                f"Test R²={results[name]['test_r2']:.4f}"
            )

        if self.stacking_model_ is not None:
            results["Stacking"] = self.model_scores_.get("Stacking", {})

        self.model_scores_ = results
        return results

    def get_feature_importance_report(self) -> pd.DataFrame:
        """Generate feature importance report.

        Returns
        -------
        pd.DataFrame
            Feature importance report with columns:
            - feature: Feature name
            - importance: Mean importance score
            - mean_rank: Mean rank across models (lower = more important)
            - std_rank: Std dev of ranks (lower = more model consensus)
            - cumulative_importance: Cumulative importance fraction
        """
        if self.feature_selector_ is None:
            raise RuntimeError("Feature selection not fitted yet")

        data = []
        for feat in self.feature_selector_.selected_features_:
            data.append({
                "feature": feat,
                "importance": self.feature_selector_.feature_importances_.get(
                    feat, 0.0
                ),
                "mean_rank": self.feature_selector_.feature_ranks_.get(
                    feat, 0.0
                ),
                "std_rank": self.feature_selector_.feature_std_ranks_.get(
                    feat, 0.0
                )
            })

        df = pd.DataFrame(data)
        df = df.sort_values("importance", ascending=False)
        df["cumulative_importance"] = (
            df["importance"].cumsum() / df["importance"].sum()
        )

        return df

    def plot_feature_importance(
        self,
        top_n: int | None = None,
        figsize: tuple[float, float] | None = None,
        use_sqrt_scale: bool | None = None,
        save: bool = True,
        output_dir: str | None = None,
        show: bool = True
    ) -> plt.Figure:
        """Plot horizontal bar chart of aggregated feature importances.

        This plots feature importances aggregated across all four models
        used in feature selection (RF, XGB, GBT, CB). For individual
        model importances, use plot_feature_importance_by_model().

        Uses square root scaling by default to prevent small values from
        disappearing when there's high magnitude disparity between features.

        Parameters
        ----------
        top_n : int, optional
            Number of top features to display.
            Defaults to config.FEATURE_IMPORTANCE_TOP_N.
        figsize : tuple of float, optional
            Figure size in inches (width, height).
            Defaults to config.FEATURE_IMPORTANCE_FIGSIZE.
        use_sqrt_scale : bool, optional
            If True, applies square root transformation to importance values
            for better visualization of small values. Actual values shown
            in annotations. Defaults to config.FEATURE_IMPORTANCE_USE_SQRT_SCALE.
        save : bool, default=True
            If True, saves with timestamped filename to output_dir.
        output_dir : str, optional
            Directory for saving. Defaults to config.FIGURES_DIR.
            Directory is created if it does not exist.
        show : bool, default=True
            If True, displays the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.

        Examples
        --------
        >>> # Default: saves to figures/ and displays
        >>> pipeline.plot_feature_importance()
        >>>
        >>> # Display only, no save
        >>> pipeline.plot_feature_importance(save=False)
        >>>
        >>> # Custom settings
        >>> pipeline.plot_feature_importance(top_n=15, output_dir='my_figures/')
        """
        if self.feature_selector_ is None:
            raise RuntimeError("Feature selection not fitted yet")

        # Apply config defaults
        top_n = top_n if top_n is not None else config.FEATURE_IMPORTANCE_TOP_N
        figsize = figsize if figsize is not None else config.FEATURE_IMPORTANCE_FIGSIZE
        use_sqrt_scale = (
            use_sqrt_scale if use_sqrt_scale is not None
            else config.FEATURE_IMPORTANCE_USE_SQRT_SCALE
        )

        # Get feature importance data
        report = self.get_feature_importance_report()
        report = report.head(top_n)

        # Reverse order so highest importance is at top
        features = report["feature"].values[::-1]
        importances = report["importance"].values[::-1]
        cumulative = report["cumulative_importance"].values[::-1]

        # Apply sqrt transformation for display if requested
        if use_sqrt_scale:
            display_values = np.sqrt(importances)
            scale_label = "√Importance (square root scale)"
        else:
            display_values = importances
            scale_label = "Importance"

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create horizontal bar chart
        bars = ax.barh(
            range(len(features)),
            display_values,
            color=plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        )

        # Add value annotations (actual importance, not sqrt)
        for i, (bar, imp, cum) in enumerate(zip(bars, importances, cumulative)):
            # Annotation shows actual importance and cumulative %
            ax.annotate(
                f"{imp:.4f} ({cum:.1%})",
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8
            )

        # Configure axes
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel(scale_label, fontsize=10)
        ax.set_title(
            f"Top {len(features)} Feature Importances (Ensemble Aggregated)\n"
            f"Models: RF, XGB, GBT, CB",
            fontsize=12
        )

        # Add grid for readability
        ax.xaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        # Tight layout
        plt.tight_layout()

        # Save with timestamped filename
        if save:
            output_dir = output_dir or config.FIGURES_DIR
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                output_dir, f"feature_importance_ensemble_{timestamp}.png"
            )
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            _get_logger().info(f"Feature importance plot saved to {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig
        # FIX #2: Removed duplicate save block that was here

    def plot_feature_importance_by_model(
        self,
        model_name: str,
        top_n: int | None = None,
        figsize: tuple[float, float] | None = None,
        use_sqrt_scale: bool | None = None,
        save: bool = True,
        output_dir: str | None = None,
        show: bool = True
    ) -> plt.Figure:
        """Plot horizontal bar chart of feature importances for a single model.

        Extracts feature importances from a specific trained model.
        Only works with tree-based models that have feature_importances_.

        Parameters
        ----------
        model_name : str
            Name of the model to plot importances for (e.g., 'CB', 'RF', 'XGB').
            Must be a trained model in best_models_.
        top_n : int, optional
            Number of top features to display.
            Defaults to config.FEATURE_IMPORTANCE_TOP_N.
        figsize : tuple of float, optional
            Figure size in inches (width, height).
            Defaults to config.FEATURE_IMPORTANCE_FIGSIZE.
        use_sqrt_scale : bool, optional
            If True, applies square root transformation to importance values.
            Defaults to config.FEATURE_IMPORTANCE_USE_SQRT_SCALE.
        save : bool, default=True
            If True, saves with timestamped filename to output_dir.
        output_dir : str, optional
            Directory for saving. Defaults to config.FIGURES_DIR.
        show : bool, default=True
            If True, displays the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.

        Examples
        --------
        >>> pipeline.plot_feature_importance_by_model('CB')
        >>> pipeline.plot_feature_importance_by_model('RF', top_n=10)
        """
        if not self.best_models_:
            raise RuntimeError("Models not trained yet")

        if model_name not in self.best_models_:
            available = list(self.best_models_.keys())
            raise ValueError(
                f"Model '{model_name}' not found. Available: {available}"
            )

        # Apply config defaults
        top_n = top_n if top_n is not None else config.FEATURE_IMPORTANCE_TOP_N
        figsize = figsize if figsize is not None else config.FEATURE_IMPORTANCE_FIGSIZE
        use_sqrt_scale = (
            use_sqrt_scale if use_sqrt_scale is not None
            else config.FEATURE_IMPORTANCE_USE_SQRT_SCALE
        )

        # Extract model from pipeline
        pipeline = self.best_models_[model_name]["estimator"]
        model = pipeline.named_steps["model"]

        if not hasattr(model, "feature_importances_"):
            raise ValueError(
                f"Model '{model_name}' does not have feature_importances_. "
                f"Only tree-based models are supported."
            )

        # Get feature names and importances
        feature_names = self.selected_feature_names_
        importances = model.feature_importances_

        # Normalize importances
        importances = importances / np.sum(importances)

        # Create DataFrame for sorting
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        # Calculate cumulative importance
        importance_df["cumulative"] = (
            importance_df["importance"].cumsum()
            / importance_df["importance"].sum()
        )

        # Take top N
        importance_df = importance_df.head(top_n)

        # Reverse for plotting (highest at top)
        features = importance_df["feature"].values[::-1]
        importances = importance_df["importance"].values[::-1]
        cumulative = importance_df["cumulative"].values[::-1]

        # Apply sqrt transformation for display if requested
        if use_sqrt_scale:
            display_values = np.sqrt(importances)
            scale_label = "√Importance (square root scale)"
        else:
            display_values = importances
            scale_label = "Importance"

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create horizontal bar chart
        bars = ax.barh(
            range(len(features)),
            display_values,
            color=plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        )

        # Add value annotations
        for i, (bar, imp, cum) in enumerate(zip(bars, importances, cumulative)):
            ax.annotate(
                f"{imp:.4f} ({cum:.1%})",
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8
            )

        # Configure axes
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel(scale_label, fontsize=10)
        ax.set_title(
            f"Top {len(features)} Feature Importances ({model_name})",
            fontsize=12
        )

        # Add grid for readability
        ax.xaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout()

        # Save with timestamped filename including model name
        if save:
            output_dir = output_dir or config.FIGURES_DIR
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                output_dir, f"feature_importance_{model_name}_{timestamp}.png"
            )
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            _get_logger().info(f"Feature importance plot saved to {save_path}")

        if show:
            plt.show()

        return fig
        # FIX #3: Removed duplicate save block that was here

    def plot_predicted_vs_actual(
        self,
        model_name: str | None = None,
        figsize: tuple[float, float] | None = None,
        save: bool = True,
        output_dir: str | None = None,
        show: bool = True
    ) -> plt.Figure:
        """Plot predicted vs actual values scatter plot.

        Parameters
        ----------
        model_name : str, optional
            Name of model to plot. If None, uses best performing model.
        figsize : tuple of float, optional
            Figure size in inches. Defaults to config.FEATURE_IMPORTANCE_FIGSIZE.
        save : bool, default=True
            If True, saves with timestamped filename to output_dir.
        output_dir : str, optional
            Directory for saving. Defaults to config.FIGURES_DIR.
        show : bool, default=True
            If True, displays the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        if not self.best_models_:
            raise RuntimeError("Models not trained yet")

        figsize = figsize if figsize is not None else config.FEATURE_IMPORTANCE_FIGSIZE

        # Select model
        if model_name is None:
            # Use best performing model by test R²
            model_name = max(
                self.model_scores_.keys(),
                key=lambda k: self.model_scores_.get(k, {}).get("test_r2", 0)
                if k != "Stacking" else 0
            )

        if model_name not in self.best_models_:
            raise ValueError(f"Model '{model_name}' not found in trained models")

        # Get predictions
        X_test_transformed = self._get_transformed_data(
            self.X_test_, fit_imputer=False
        )
        y_test = self.y_test_.values
        y_pred = self.best_models_[model_name]["estimator"].predict(
            X_test_transformed
        )

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="none", s=20)

        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot(
            [min_val, max_val], [min_val, max_val],
            "r--", linewidth=2, label="Perfect Prediction"
        )

        # Labels and title
        ax.set_xlabel("Actual Values", fontsize=10)
        ax.set_ylabel("Predicted Values", fontsize=10)
        ax.set_title(
            f"{model_name}: Predicted vs Actual\n"
            f"R² = {r2:.4f}, RMSE = {rmse:.4f}",
            fontsize=12
        )
        ax.legend(loc="upper left")

        # Equal aspect ratio
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()

        # Save with timestamped filename
        if save:
            output_dir = output_dir or config.FIGURES_DIR
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                output_dir, f"predicted_vs_actual_{model_name}_{timestamp}.png"
            )
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            _get_logger().info(f"Predicted vs actual plot saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_residuals(
        self,
        model_name: str | None = None,
        figsize: tuple[float, float] | None = None,
        bins: int | None = None,
        save: bool = True,
        output_dir: str | None = None,
        show: bool = True
    ) -> plt.Figure:
        """Plot residual distribution histogram.

        Parameters
        ----------
        model_name : str, optional
            Name of model to plot. If None, uses best performing model.
        figsize : tuple of float, optional
            Figure size in inches. Defaults to config.FEATURE_IMPORTANCE_FIGSIZE.
        bins : int, optional
            Number of histogram bins. Defaults to config.HISTOGRAM_BINS.
        save : bool, default=True
            If True, saves with timestamped filename to output_dir.
        output_dir : str, optional
            Directory for saving. Defaults to config.FIGURES_DIR.
        show : bool, default=True
            If True, displays the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        if not self.best_models_:
            raise RuntimeError("Models not trained yet")

        figsize = figsize if figsize is not None else config.FEATURE_IMPORTANCE_FIGSIZE
        bins = bins if bins is not None else config.HISTOGRAM_BINS

        # Select model
        if model_name is None:
            model_name = max(
                self.model_scores_.keys(),
                key=lambda k: self.model_scores_.get(k, {}).get("test_r2", 0)
                if k != "Stacking" else 0
            )

        if model_name not in self.best_models_:
            raise ValueError(f"Model '{model_name}' not found in trained models")

        # Get predictions and residuals
        X_test_transformed = self._get_transformed_data(
            self.X_test_, fit_imputer=False
        )
        y_test = self.y_test_.values
        y_pred = self.best_models_[model_name]["estimator"].predict(
            X_test_transformed
        )
        residuals = y_test - y_pred

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Histogram
        ax.hist(residuals, bins=bins, edgecolor="black", alpha=0.7)

        # Statistics
        mean_resid = np.mean(residuals)
        std_resid = np.std(residuals)

        # Add vertical line at zero
        ax.axvline(x=0, color="r", linestyle="--", linewidth=2, label="Zero")
        ax.axvline(
            x=mean_resid, color="g", linestyle="-", linewidth=2,
            label=f"Mean: {mean_resid:.4f}"
        )

        # Labels and title
        ax.set_xlabel("Residual (Actual - Predicted)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(
            f"{model_name}: Residual Distribution\n"
            f"Mean = {mean_resid:.4f}, Std = {std_resid:.4f}",
            fontsize=12
        )
        ax.legend(loc="upper right")

        plt.tight_layout()

        # Save with timestamped filename
        if save:
            output_dir = output_dir or config.FIGURES_DIR
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                output_dir, f"residuals_{model_name}_{timestamp}.png"
            )
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            _get_logger().info(f"Residuals plot saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_model_comparison(
        self,
        metric: str = "test_r2",
        figsize: tuple[float, float] | None = None,
        save: bool = True,
        output_dir: str | None = None,
        show: bool = True
    ) -> plt.Figure:
        """Plot bar chart comparing model performance.

        Parameters
        ----------
        metric : str, default='test_r2'
            Metric to compare. One of 'test_r2', 'cv_r2', 'test_rmse', 'test_mae'.
        figsize : tuple of float, optional
            Figure size in inches. Defaults to config.FEATURE_IMPORTANCE_FIGSIZE.
        save : bool, default=True
            If True, saves with timestamped filename to output_dir.
        output_dir : str, optional
            Directory for saving. Defaults to config.FIGURES_DIR.
        show : bool, default=True
            If True, displays the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        if not self.model_scores_:
            raise RuntimeError("Models not evaluated yet")

        figsize = figsize if figsize is not None else config.FEATURE_IMPORTANCE_FIGSIZE

        # Extract metric values
        models = []
        values = []
        for name, scores in self.model_scores_.items():
            if metric in scores:
                models.append(name)
                values.append(scores[metric])

        if not models:
            raise ValueError(f"Metric '{metric}' not found in model scores")

        # Sort by value
        sorted_pairs = sorted(zip(values, models), reverse=True)
        values, models = zip(*sorted_pairs)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Bar chart
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        bars = ax.bar(models, values, color=colors, edgecolor="black")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9
            )

        # Labels and title
        metric_labels = {
            "test_r2": "Test R²",
            "cv_r2": "CV R²",
            "test_rmse": "Test RMSE",
            "test_mae": "Test MAE"
        }
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=10)
        ax.set_title(f"Model Comparison: {metric_labels.get(metric, metric)}", fontsize=12)

        # Rotate x labels if many models
        if len(models) > 5:
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        # Save with timestamped filename
        if save:
            output_dir = output_dir or config.FIGURES_DIR
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                output_dir, f"model_comparison_{metric}_{timestamp}.png"
            )
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            _get_logger().info(f"Model comparison plot saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_all(
        self,
        save: bool = True,
        output_dir: str | None = None,
        show: bool = True,
        include_per_model_importance: bool = True
    ) -> dict[str, plt.Figure]:
        """Generate all standard visualizations.

        Creates feature importance (ensemble and per-model), predicted vs actual,
        residuals, and model comparison plots.

        Parameters
        ----------
        save : bool, default=True
            If True, saves all plots with timestamped filenames.
        output_dir : str, optional
            Directory for saving. Defaults to config.FIGURES_DIR.
        show : bool, default=True
            If True, displays all plots.
        include_per_model_importance : bool, default=True
            If True, generates feature importance plots for each individual
            tree-based model in addition to the ensemble plot.

        Returns
        -------
        dict of str to Figure
            Dictionary mapping plot names to figure objects.
        """
        figures = {}

        # Ensemble feature importance (from feature selection phase)
        figures["feature_importance_ensemble"] = self.plot_feature_importance(
            save=save, output_dir=output_dir, show=show
        )

        if self.best_models_:
            # Per-model feature importance
            if include_per_model_importance:
                for model_name in self.best_models_:
                    # Only tree-based models have feature_importances_
                    pipeline = self.best_models_[model_name]["estimator"]
                    model = pipeline.named_steps["model"]
                    if hasattr(model, "feature_importances_"):
                        key = f"feature_importance_{model_name}"
                        figures[key] = self.plot_feature_importance_by_model(
                            model_name=model_name,
                            save=save, output_dir=output_dir, show=show
                        )

            figures["predicted_vs_actual"] = self.plot_predicted_vs_actual(
                save=save, output_dir=output_dir, show=show
            )
            figures["residuals"] = self.plot_residuals(
                save=save, output_dir=output_dir, show=show
            )
            figures["model_comparison"] = self.plot_model_comparison(
                save=save, output_dir=output_dir, show=show
            )

        return figures

    def generate_report(
        self,
        output_dir: str | None = None,
        filename: str | None = None
    ) -> str:
        """Generate comprehensive multi-tab Excel report.

        Creates a professional Excel workbook with the following tabs:
        1. Narrative - Executive summary and key findings
        2. Feature Metrics - Feature importance scores and statistics
        3. Model Comparison - All models ranked by performance
        4. Imputation Log - Imputation methods used per column
        5. Model Progression - Initial → Baseline → Optimized R² progression

        Parameters
        ----------
        output_dir : str, optional
            Output directory. Defaults to config.DATA_DIR.
        filename : str, optional
            Output filename. Defaults to 'Feature_Analysis_Report_{timestamp}.xlsx'.

        Returns
        -------
        str
            Path to the generated Excel file.

        Examples
        --------
        >>> pipeline.generate_report()
        >>> pipeline.generate_report(output_dir='reports/', filename='my_report.xlsx')
        """
        output_dir = output_dir or config.DATA_DIR
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename is None:
            filename = f"Feature_Analysis_Report_{timestamp}.xlsx"

        filepath = os.path.join(output_dir, filename)

        _get_logger().info(f"Generating Excel report: {filepath}")

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # =================================================================
            # TAB 1: NARRATIVE
            # =================================================================
            narrative_data = self._build_narrative_tab()
            pd.DataFrame(narrative_data).to_excel(
                writer, sheet_name="Narrative", index=False, header=False
            )

            # =================================================================
            # TAB 2: FEATURE METRICS
            # =================================================================
            if self.feature_selector_ is not None:
                feature_df = self.get_feature_importance_report()
                feature_df.to_excel(
                    writer, sheet_name="Feature Metrics", index=False
                )

            # =================================================================
            # TAB 3: MODEL COMPARISON
            # =================================================================
            if self.model_scores_:
                comparison_df = self._build_model_comparison_tab()
                comparison_df.to_excel(
                    writer, sheet_name="Model Comparison", index=False
                )

            # =================================================================
            # TAB 4: IMPUTATION LOG
            # =================================================================
            if self.production_imputer_ is not None:
                imputation_df = self._build_imputation_log_tab()
                imputation_df.to_excel(
                    writer, sheet_name="Imputation Log", index=False
                )

            # =================================================================
            # TAB 5: MODEL PROGRESSION
            # =================================================================
            if self.best_models_:
                progression_df = self._build_model_progression_tab()
                progression_df.to_excel(
                    writer, sheet_name="Model Progression", index=False
                )

        _get_logger().info(f"Excel report generated: {filepath}")
        return filepath

    def _build_narrative_tab(self) -> list[list[str]]:
        """Build narrative tab content."""
        # Find best model
        best_model_name = None
        best_r2 = -np.inf
        for name, scores in self.model_scores_.items():
            r2 = scores.get("test_r2", 0)
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name

        # Count features
        n_original = len(self.feature_columns_) if self.feature_columns_ else 0
        n_selected = len(self.selected_feature_names_) if self.selected_feature_names_ else 0

        narrative = [
            ["RAPID - REGRESSION ANALYSIS REPORT"],
            [""],
            ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            [""],
            ["=" * 50],
            ["EXECUTIVE SUMMARY"],
            ["=" * 50],
            [""],
            ["Target Variable:", self.target_column_ or "N/A"],
            [""],
            ["Data Summary:"],
            ["  Original Features:", str(n_original)],
            ["  Selected Features:", str(n_selected)],
            ["  Feature Reduction:", f"{(1 - n_selected/n_original)*100:.1f}%" if n_original > 0 else "N/A"],
            [""],
            ["Training Samples:", f"{len(self.X_train_):,}" if self.X_train_ is not None else "N/A"],
            ["Test Samples:", f"{len(self.X_test_):,}" if self.X_test_ is not None else "N/A"],
            [""],
            ["=" * 50],
            ["MODEL PERFORMANCE"],
            ["=" * 50],
            [""],
            ["Best Model:", best_model_name or "N/A"],
            ["Best Test R²:", f"{best_r2:.4f}" if best_r2 > -np.inf else "N/A"],
            [""],
            ["Models Trained:", str(len(self.best_models_))],
            ["Stacking Enabled:", "Yes" if self.stacking_model_ is not None else "No"],
            [""],
        ]

        # Add individual model scores
        if self.model_scores_:
            narrative.append(["Model Performance Summary:"])
            for name, scores in sorted(
                self.model_scores_.items(),
                key=lambda x: x[1].get("test_r2", 0),
                reverse=True
            ):
                r2 = scores.get("test_r2", 0)
                narrative.append([f"  {name}:", f"R² = {r2:.4f}"])

        narrative.extend([
            [""],
            ["=" * 50],
            ["CONFIGURATION"],
            ["=" * 50],
            [""],
            ["CV Folds:", str(config.CV_FOLDS)],
            ["Test Size:", f"{config.TEST_SIZE:.0%}"],
            ["R² Cutoff:", f"{config.CUTOFF_R2:.2f}"],
            ["Correlation Threshold:", f"{config.FEATURE_CORRELATION_THRESHOLD:.2f}"],
            ["Cumulative Importance:", f"{config.OPTIMIZATION_CDF_THRESHOLD:.0%}"],
        ])

        return narrative

    def _build_model_comparison_tab(self) -> pd.DataFrame:
        """Build model comparison DataFrame."""
        data = []
        for name, scores in self.model_scores_.items():
            row = {
                "Model": name,
                "Test_R2": scores.get("test_r2"),
                "CV_R2": self.best_models_.get(name, {}).get("cv_score"),
                "Test_RMSE": scores.get("test_rmse"),
                "Test_MAE": scores.get("test_mae"),
                "Optimized": self.best_models_.get(name, {}).get("optimized", False),
            }
            data.append(row)

        df = pd.DataFrame(data)
        df = df.sort_values("Test_R2", ascending=False, na_position="last")
        df.insert(0, "Rank", range(1, len(df) + 1))
        return df

    def _build_imputation_log_tab(self) -> pd.DataFrame:
        """Build imputation log DataFrame."""
        data = []

        if hasattr(self.production_imputer_, "imputation_info_"):
            for col, info in self.production_imputer_.imputation_info_.items():
                data.append({
                    "Column": col,
                    "Missing_Pct": info.get("missing_pct", 0),
                    "Method": info.get("method", "Unknown"),
                    "Fill_Value": str(info.get("fill_value", ""))[:50]
                })
        else:
            # Fallback: show columns that had missing data
            if self.X_train_ is not None:
                missing = self.X_train_.isnull().mean()
                for col, pct in missing[missing > 0].items():
                    data.append({
                        "Column": col,
                        "Missing_Pct": pct,
                        "Method": "Tiered (auto)",
                        "Fill_Value": ""
                    })

        if not data:
            data = [{"Column": "No imputation required", "Missing_Pct": 0,
                     "Method": "N/A", "Fill_Value": ""}]

        return pd.DataFrame(data)

    def _build_model_progression_tab(self) -> pd.DataFrame:
        """Build model progression DataFrame."""
        data = []

        for name, info in self.best_models_.items():
            cv_score = info.get("cv_score", 0)
            test_score = self.model_scores_.get(name, {}).get("test_r2", 0)
            optimized = info.get("optimized", False)

            data.append({
                "Model": name,
                "Baseline_CV_R2": cv_score if not optimized else None,
                "Optimized_CV_R2": cv_score if optimized else None,
                "Test_R2": test_score,
                "Optimized": optimized,
                "Delta": test_score - cv_score if cv_score else None
            })

        # Add stacking if present
        if "Stacking" in self.model_scores_:
            stack_score = self.model_scores_["Stacking"].get("test_r2", 0)
            meta_learner = self.model_scores_["Stacking"].get("meta_learner", "Unknown")
            data.append({
                "Model": f"Stacking ({meta_learner})",
                "Baseline_CV_R2": None,
                "Optimized_CV_R2": None,
                "Test_R2": stack_score,
                "Optimized": True,
                "Delta": None
            })

        df = pd.DataFrame(data)
        df = df.sort_values("Test_R2", ascending=False, na_position="last")
        df.insert(0, "Rank", range(1, len(df) + 1))
        return df

    def save_results(self, output_dir: str | None = None) -> None:
        """Save models, reports, and metrics to disk.

        Parameters
        ----------
        output_dir : str, optional
            Output directory. Defaults to config.DATA_DIR.
        """
        output_dir = output_dir or config.DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model scores
        scores_path = os.path.join(
            output_dir, f"model_scores_{timestamp}.json"
        )
        with open(scores_path, "w") as f:
            json.dump(self.model_scores_, f, indent=2)

        # Save feature importance
        feature_report = self.get_feature_importance_report()
        feature_path = os.path.join(
            output_dir, f"feature_importance_{timestamp}.csv"
        )
        feature_report.to_csv(feature_path, index=False)

        _get_logger().info(f"Results saved to {output_dir}")


# =============================================================================
# Convenience Function
# =============================================================================

def run_pipeline(
    filepath: str,
    target_column: str,
    save_results: bool = True
) -> RAPIDPipeline:
    """Run the complete RAPID pipeline.

    Parameters
    ----------
    filepath : str
        Path to CSV data file.
    target_column : str
        Name of target variable column.
    save_results : bool, default=True
        Whether to save results to disk.

    Returns
    -------
    RAPIDPipeline
        Fitted pipeline with results.

    Examples
    --------
    >>> pipeline = run_pipeline('data.csv', 'target')
    >>> print(pipeline.model_scores_)
    """
    pipeline = RAPIDPipeline()

    pipeline.load_data(filepath, target_column)
    pipeline.preprocess()
    pipeline.split_data()
    pipeline.fit_feature_selection()
    pipeline.train_base_models()
    pipeline.train_stacking_ensemble()
    pipeline.evaluate()

    if save_results:
        pipeline.save_results()

    return pipeline


# =============================================================================
# SHAP Analysis Module (Post-Training Interpretation)
# =============================================================================

class SHAPAnalyzer:
    """SHAP-based model interpretation for trained pipelines.

    This class provides post-hoc explanation of model predictions using
    SHAP (SHapley Additive exPlanations) values. Use this AFTER training
    to understand why the model makes specific predictions.

    Parameters
    ----------
    pipeline : RAPIDPipeline
        A fitted RAPIDPipeline instance.
    model_name : str, optional
        Name of model to analyze. If None, uses the best performing model.

    Attributes
    ----------
    explainer_ : shap.Explainer
        Fitted SHAP explainer object.
    shap_values_ : np.ndarray
        SHAP values for the analyzed dataset.
    feature_names_ : list of str
        Feature names corresponding to SHAP values.

    Notes
    -----
    SHAP analysis is computationally expensive. For large datasets,
    consider using a representative sample (e.g., 1000 rows).

    This module requires the `shap` package:
        pip install shap

    Examples
    --------
    >>> from rapid_pipeline import RAPIDPipeline, SHAPAnalyzer
    >>> pipeline = RAPIDPipeline()
    >>> pipeline.load_data('data.csv', 'target')
    >>> pipeline.preprocess().split_data().fit_feature_selection()
    >>> pipeline.train_base_models().train_stacking_ensemble()
    >>>
    >>> # Post-training SHAP analysis
    >>> analyzer = SHAPAnalyzer(pipeline)
    >>> analyzer.fit(pipeline.X_test_)
    >>> analyzer.plot_summary()
    >>> analyzer.plot_dependence('top_feature')
    >>> analyzer.explain_sample(0)  # Explain first test sample
    """

    def __init__(
        self,
        pipeline: RAPIDPipeline,
        model_name: str | None = None
    ) -> None:
        self.pipeline = pipeline
        self.model_name = model_name

        self.explainer_: Any = None
        self.shap_values_: np.ndarray | None = None
        self.feature_names_: list[str] = []
        self._X_analyzed: pd.DataFrame | None = None

        # Validate pipeline is fitted
        if pipeline.best_model_ is None:
            raise RuntimeError(
                "Pipeline must be fully trained before SHAP analysis. "
                "Call train_base_models() and train_stacking_ensemble() first."
            )

    def _get_model(self) -> tuple[str, Any]:
        """Get the model to analyze.

        Returns
        -------
        tuple of (str, estimator)
            Model name and fitted model instance.
        """
        if self.model_name is not None:
            # User specified a model
            if self.model_name in self.pipeline.trained_models_:
                return self.model_name, self.pipeline.trained_models_[self.model_name]
            elif self.model_name in self.pipeline.stacking_models_:
                return self.model_name, self.pipeline.stacking_models_[self.model_name]
            else:
                available = (
                    list(self.pipeline.trained_models_.keys()) +
                    list(self.pipeline.stacking_models_.keys())
                )
                raise ValueError(
                    f"Model '{self.model_name}' not found. "
                    f"Available models: {available}"
                )
        else:
            # Use best model
            return self.pipeline.best_model_name_, self.pipeline.best_model_

    def _import_shap(self) -> Any:
        """Import SHAP with helpful error message if not installed.

        Returns
        -------
        module
            The shap module.

        Raises
        ------
        ImportError
            If shap is not installed.
        """
        try:
            import shap
            return shap
        except ImportError:
            raise ImportError(
                "SHAP analysis requires the 'shap' package.\n"
                "Install it with: pip install shap\n"
                "Note: SHAP is optional and not required for the main pipeline."
            )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        max_samples: int = 1000,
        background_samples: int = 100
    ) -> SHAPAnalyzer:
        """Compute SHAP values for the given data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Data to explain. Should be preprocessed (same format as training).
        max_samples : int, default=1000
            Maximum samples to analyze. Randomly samples if X is larger.
        background_samples : int, default=100
            Number of background samples for SHAP explainer.

        Returns
        -------
        self : SHAPAnalyzer
            Fitted analyzer with computed SHAP values.
        """
        shap = self._import_shap()

        model_name, model = self._get_model()
        _get_logger().info(f"Computing SHAP values for model: {model_name}")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(
                X,
                columns=self.pipeline.feature_selector_.selected_features_
            )
        else:
            X_df = X.copy()

        self.feature_names_ = X_df.columns.tolist()

        # Sample if dataset is large
        if len(X_df) > max_samples:
            _get_logger().info(
                f"Sampling {max_samples} rows from {len(X_df)} for SHAP analysis"
            )
            X_df = X_df.sample(n=max_samples, random_state=config.RANDOM_STATE)

        self._X_analyzed = X_df

        # Create background dataset for explainer
        background = X_df.sample(
            n=min(background_samples, len(X_df)),
            random_state=config.RANDOM_STATE
        )

        # Create appropriate explainer based on model type
        model_type = type(model).__name__

        if model_type in ['XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor']:
            # Tree-based models use TreeExplainer (fast)
            self.explainer_ = shap.TreeExplainer(model)
            _get_logger().debug(f"Using TreeExplainer for {model_type}")
        elif model_type in ['RandomForestRegressor', 'ExtraTreesRegressor',
                           'GradientBoostingRegressor']:
            # Sklearn tree ensembles
            self.explainer_ = shap.TreeExplainer(model)
            _get_logger().debug(f"Using TreeExplainer for {model_type}")
        else:
            # Fallback to KernelExplainer (slower but universal)
            self.explainer_ = shap.KernelExplainer(model.predict, background)
            _get_logger().debug(f"Using KernelExplainer for {model_type}")

        # Compute SHAP values
        _get_logger().info("Computing SHAP values (this may take a moment)...")
        self.shap_values_ = self.explainer_.shap_values(X_df)

        _get_logger().info(
            f"SHAP analysis complete: {X_df.shape[0]} samples, "
            f"{X_df.shape[1]} features"
        )

        return self

    def plot_summary(
        self,
        max_display: int = 20,
        plot_type: str = "dot",
        save: bool = True,
        output_dir: str | None = None,
        show: bool = True
    ) -> plt.Figure:
        """Generate SHAP summary plot.

        Parameters
        ----------
        max_display : int, default=20
            Maximum number of features to display.
        plot_type : str, default="dot"
            Type of plot: "dot", "bar", or "violin".
        save : bool, default=True
            Whether to save the figure.
        output_dir : str, optional
            Directory to save figure. Defaults to config.FIGURES_DIR.
        show : bool, default=True
            Whether to display the figure.

        Returns
        -------
        plt.Figure
            The matplotlib figure.
        """
        if self.shap_values_ is None:
            raise RuntimeError("Must call fit() before plotting.")

        shap = self._import_shap()

        fig = plt.figure(figsize=(10, 8))

        shap.summary_plot(
            self.shap_values_,
            self._X_analyzed,
            feature_names=self.feature_names_,
            max_display=max_display,
            plot_type=plot_type,
            show=False
        )

        plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            output_dir = output_dir or config.FIGURES_DIR
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(output_dir, f"shap_summary_{timestamp}.png")
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            _get_logger().info(f"Saved SHAP summary plot: {filepath}")

        if show:
            plt.show()

        return fig

    def plot_dependence(
        self,
        feature: str,
        interaction_feature: str | None = "auto",
        save: bool = True,
        output_dir: str | None = None,
        show: bool = True
    ) -> plt.Figure:
        """Generate SHAP dependence plot for a feature.

        Shows how a feature's value affects the model prediction,
        optionally colored by an interaction feature.

        Parameters
        ----------
        feature : str
            Feature name to plot.
        interaction_feature : str or "auto", optional
            Feature to use for coloring. "auto" selects automatically.
        save : bool, default=True
            Whether to save the figure.
        output_dir : str, optional
            Directory to save figure.
        show : bool, default=True
            Whether to display the figure.

        Returns
        -------
        plt.Figure
            The matplotlib figure.
        """
        if self.shap_values_ is None:
            raise RuntimeError("Must call fit() before plotting.")

        if feature not in self.feature_names_:
            raise ValueError(
                f"Feature '{feature}' not found. "
                f"Available: {self.feature_names_[:10]}..."
            )

        shap = self._import_shap()

        fig = plt.figure(figsize=(10, 6))

        shap.dependence_plot(
            feature,
            self.shap_values_,
            self._X_analyzed,
            feature_names=self.feature_names_,
            interaction_index=interaction_feature,
            show=False
        )

        plt.title(f"SHAP Dependence: {feature}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            output_dir = output_dir or config.FIGURES_DIR
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = feature.replace("/", "_").replace(" ", "_")
            filepath = os.path.join(
                output_dir,
                f"shap_dependence_{safe_name}_{timestamp}.png"
            )
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            _get_logger().info(f"Saved SHAP dependence plot: {filepath}")

        if show:
            plt.show()

        return fig

    def explain_sample(
        self,
        sample_index: int,
        max_display: int = 10
    ) -> pd.DataFrame:
        """Explain a single prediction.

        Parameters
        ----------
        sample_index : int
            Index of sample in the analyzed data.
        max_display : int, default=10
            Maximum number of features to show.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature contributions sorted by absolute impact.
        """
        if self.shap_values_ is None:
            raise RuntimeError("Must call fit() before explaining.")

        if sample_index >= len(self.shap_values_):
            raise IndexError(
                f"Sample index {sample_index} out of range "
                f"(analyzed {len(self.shap_values_)} samples)"
            )

        # Get SHAP values for this sample
        sample_shap = self.shap_values_[sample_index]
        sample_data = self._X_analyzed.iloc[sample_index]

        # Create explanation DataFrame
        explanation = pd.DataFrame({
            "feature": self.feature_names_,
            "value": sample_data.values,
            "shap_value": sample_shap,
            "abs_shap": np.abs(sample_shap)
        })

        # Sort by absolute impact
        explanation = explanation.sort_values("abs_shap", ascending=False)
        explanation = explanation.head(max_display)

        # Add direction column
        explanation["direction"] = explanation["shap_value"].apply(
            lambda x: "↑ increases" if x > 0 else "↓ decreases"
        )

        # Get prediction info
        model_name, model = self._get_model()
        base_value = self.explainer_.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]

        predicted = base_value + sample_shap.sum()

        print(f"\n{'='*60}")
        print(f"SHAP Explanation for Sample {sample_index}")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Base value (average prediction): {base_value:.4f}")
        print(f"Predicted value: {predicted:.4f}")
        print(f"Sum of SHAP values: {sample_shap.sum():.4f}")
        print(f"\nTop {max_display} Contributing Features:")
        print("-" * 60)

        for _, row in explanation.iterrows():
            direction = "+" if row["shap_value"] > 0 else ""
            print(
                f"  {row['feature']:30s} = {row['value']:10.4f} -> "
                f"{direction}{row['shap_value']:.4f} ({row['direction']})"
            )

        print("=" * 60)

        return explanation[["feature", "value", "shap_value", "direction"]]

    def get_global_importance(self) -> pd.DataFrame:
        """Get global feature importance from SHAP values.

        Returns
        -------
        pd.DataFrame
            DataFrame with mean absolute SHAP values per feature.
        """
        if self.shap_values_ is None:
            raise RuntimeError("Must call fit() before getting importance.")

        mean_abs_shap = np.abs(self.shap_values_).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": self.feature_names_,
            "mean_abs_shap": mean_abs_shap
        })

        importance_df = importance_df.sort_values(
            "mean_abs_shap",
            ascending=False
        )
        importance_df["cumulative_importance"] = (
            importance_df["mean_abs_shap"].cumsum() /
            importance_df["mean_abs_shap"].sum()
        )

        return importance_df


# =============================================================================
# Module Entry Point
# =============================================================================

if __name__ == "__main__":
    print("RAPID Pipeline v3.0")
    print(config.SEPARATOR_CHAR * config.SEPARATOR_WIDTH)
    config.print_config()
    print("\nUsage:")
    print("  from rapid_pipeline import run_pipeline")
    print("  pipeline = run_pipeline('data.csv', 'target_column')")
    print("  print(pipeline.model_scores_)")
