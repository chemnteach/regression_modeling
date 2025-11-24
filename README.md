# Regression Modeling with Automated Feature Reduction

A comprehensive machine learning pipeline for regression analysis featuring automated data preprocessing, intelligent feature selection, and ensemble model comparison with stacking capabilities.

## 🎯 Overview

This project provides an end-to-end automated workflow for regression modeling that handles:
- **Intelligent Data Preprocessing**: Automated missing data imputation, outlier detection, and quality checks
- **Feature Selection**: Multiple reduction strategies (correlation-based, variance-based, tree-based importance)
- **Model Comparison**: Evaluates 10+ regression algorithms with hyperparameter optimization
- **Ensemble Methods**: Stacking regressors with multiple meta-learners for optimal performance
- **Export Capabilities**: Generates comprehensive Excel reports and publication-ready PNG plots

### Key Features
✅ **"Hit and Walk Away" Automation** - Set target variable once and run entire pipeline  
✅ **Configurable Constants** - 30+ parameters for complete workflow customization  
✅ **Advanced Imputation** - Threshold-based strategies (simple → KNN → iterative)  
✅ **Robust Validation** - K-Fold cross-validation with stratified sampling  
✅ **Professional Reporting** - Multi-tab Excel exports with narrative insights  
✅ **Presentation-Ready Outputs** - High-DPI PNG plots for immediate use  

---

## 📋 Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Supported Models](#supported-models)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🔧 Requirements

### System Requirements
- **Python Version**: 3.10+ (Tested on 3.12.0)
- **Operating System**: Windows, macOS, or Linux
- **RAM**: 8GB minimum (16GB recommended for large datasets)
- **Storage**: 500MB for dependencies + space for your data

### Python Version Note
While developed with Python 3.12.0, this project should work with Python 3.10 or higher. If you encounter compatibility issues, we recommend using Python 3.12.x for best results.

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/chemnteach/regression_modeling.git
cd regression_modeling
```

### 2. Create Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: If you're behind a corporate proxy (like Intel's), use:
```bash
pip install --proxy http://proxy-dmz.intel.com:912 -r requirements.txt
```

### 4. Verify Installation
Open Jupyter Notebook and ensure all imports work:
```bash
jupyter notebook "Feature Reduction.ipynb"
```

---

## ⚡ Quick Start

### Option 1: Automated Execution (Recommended)
1. Open `Feature Reduction.ipynb` in Jupyter/VS Code
2. Set `AUTO_MODE = True` in the Constants cell
3. Set `DEFAULT_TARGET_VARIABLE = "your_target_column_name"`
4. Run all cells (Kernel → Run All)
5. Find outputs in the project directory:
   - `feature_importance_scores.csv` - Feature rankings
   - `Feature_Analysis_Report_YYYYMMDD_HHMMSS.xlsx` - Comprehensive report
   - `best_model_*.png` - Model performance plots

### Option 2: Interactive Execution
1. Open `Feature Reduction.ipynb`
2. Keep `AUTO_MODE = False` (default)
3. Execute cells sequentially
4. Use interactive file dialog to select CSV
5. Choose target variable from dropdown menu
6. Review outputs at each stage

### Option 3: Standalone Script (Future Enhancement)
```bash
python main.py --data your_data.csv --target column_name
```
*(Note: main.py currently serves as project entry point; full CLI support planned)*

---

## 📁 Project Structure

```
regression_modeling/
│
├── Feature Reduction.ipynb     # Main analysis notebook (3,100+ lines)
├── main.py                      # Project entry point / utilities
├── feature_reduction.py         # Core preprocessing functions
├── EXPORT_TO_EXCEL.py          # Standalone Excel report generator
│
├── requirements.txt             # Python package dependencies
├── old_requirements.txt         # Legacy dependency list
├── README.md                    # This file
│
├── GMZ_Resistance_Data.csv     # Example dataset
├── Vmin Kitchen Sink.ipynb     # Additional analysis notebook
│
└── venv/                        # Virtual environment (created during setup)
```

### Key Files Explained

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `Feature Reduction.ipynb` | Complete ML pipeline with documentation | 3,134 | Primary |
| `EXPORT_TO_EXCEL.py` | Generate 3-tab Excel reports | 145 | Active |
| `feature_reduction.py` | Reusable preprocessing functions | - | Support |
| `main.py` | Project scaffolding / future CLI | - | Support |
| `requirements.txt` | All package dependencies | - | Active |

---

## 📖 Usage Guide

### Workflow Overview

The notebook follows a 7-stage pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SETUP ENVIRONMENT                                        │
│    → Install packages, import libraries, set constants      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. LOAD DATA                                                │
│    → Interactive file selection or automatic CSV loading    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. INITIAL QUALITY CHECK                                    │
│    → Remove columns with >50% missing data                  │
│    → Remove date columns, duplicates, high cardinality      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. STRATEGIC IMPUTATION                                     │
│    → <5% missing: Median/mode imputation                    │
│    → 5-20% missing: KNN imputation                          │
│    → 20-40% missing: Iterative imputation                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. FEATURE SELECTION                                        │
│    → Correlation analysis, variance thresholds              │
│    → Tree-based feature importance                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. MODEL TRAINING & COMPARISON                              │
│    → Train 10+ base models with cross-validation           │
│    → Hyperparameter optimization via RandomizedSearchCV     │
│    → Evaluate using R² scoring                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. ENSEMBLE STACKING                                        │
│    → Combine top 5 base models                              │
│    → Test 3 meta-learners (Ridge, LinearRegression, SVR)   │
│    → Select best performing ensemble                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. EXPORT RESULTS                                           │
│    → CSV: Feature importance scores                         │
│    → Excel: 3-tab comprehensive report                      │
│    → PNG: High-resolution model plots                       │
└─────────────────────────────────────────────────────────────┘
```

### Automation Mode

Set these constants in the notebook's Constants cell:

```python
# Enable full automation
AUTO_MODE = True

# Specify your target variable (must match column name exactly)
DEFAULT_TARGET_VARIABLE = "resistance_value"  # Change to your target
```

With automation enabled:
- No file dialogs - automatically loads first CSV found
- No manual target selection - uses `DEFAULT_TARGET_VARIABLE`
- Runs entire pipeline without interruption
- Validates target variable existence before processing

---

## ⚙️ Configuration

### Key Constants (Located in Constants Cell)

#### Missing Data Handling
```python
MAX_MISSING_DATA = 0.5            # Remove columns with >50% missing
LOW_MISSING_THRESHOLD = 0.05      # Use simple imputation if <5% missing
MEDIUM_MISSING_THRESHOLD = 0.20   # Use KNN if 5-20% missing
HIGH_MISSING_THRESHOLD = 0.40     # Use iterative if 20-40% missing
```

#### Data Quality Filters
```python
REMOVE_DATE_COLUMNS = True        # Drop datetime columns
HIGH_CARDINALITY_THRESHOLD = 0.8  # Remove likely ID columns
REMOVE_DUPLICATE_ROWS = True      # Drop exact duplicate rows
LOW_VARIANCE_THRESHOLD = 0.99     # Remove near-constant columns
```

#### Model Training
```python
TRAINING_DATA_SPLIT = 0.8         # 80% train, 20% validation
CROSS_VALIDATION_FOLDS = 5        # K-Fold CV splits
RANDOM_STATE = 42                 # Reproducibility seed
HYPERPARAMETER_ITERATIONS = 20    # RandomizedSearchCV iterations
```

#### Feature Selection
```python
FEATURE_IMPORTANCE_THRESHOLD = 0.001  # Minimum importance to keep
CORRELATION_THRESHOLD = 0.95          # Remove highly correlated features
VARIANCE_THRESHOLD = 0.01             # Minimum variance required
```

#### Visualization
```python
HISTOGRAM_BINS = 30               # Residual histogram bins
figSize = [10, 10]                # Default plot dimensions
```

### All 30+ Configurable Parameters
See the **Constants** cell in `Feature Reduction.ipynb` for the complete list with detailed comments.

---

## 📊 Output Files

### 1. Feature Importance CSV
**Filename**: `feature_importance_scores.csv`

Contains ranked features with comprehensive statistics:
- Feature name
- Importance score (0-1)
- Mean, median, std, min, max
- Missing data percentage
- Unique value count
- Data type

**Use Case**: Quick reference for feature engineering decisions

---

### 2. Excel Comprehensive Report
**Filename**: `Feature_Analysis_Report_YYYYMMDD_HHMMSS.xlsx`

Three-tab workbook generated by `EXPORT_TO_EXCEL.py`:

#### Tab 1: Narrative
- Executive summary
- Methodology overview
- Model performance highlights
- Key insights and patterns
- Data quality assessment
- Actionable recommendations

#### Tab 2: Feature Metrics
- All features ranked by importance
- Complete statistical profiles
- Missing data analysis
- Data type classification

#### Tab 3: Model Comparison
- All models ranked by R² score
- Base learners + stacking ensembles
- Performance deltas
- Model selection rationale

**Use Case**: Client presentations, stakeholder reports, documentation

---

### 3. Model Performance Plots (PNG)
**Filenames**: `best_model_<ModelName>_<plot_type>.png`

Three high-resolution (300 DPI) plots:

1. **Predicted vs Actual**: Scatter plot with perfect prediction line
2. **Residuals CDF/PDF**: Distribution analysis with KDE overlay
3. **Residuals Distribution**: Histogram with zero-error reference

**Use Case**: Presentations, publications, model validation reports

---

## 🤖 Supported Models

### Base Learners (10 Algorithms)

| Category | Models |
|----------|--------|
| **Linear** | Linear Regression, Ridge, ElasticNet |
| **Tree-Based** | Random Forest, Extra Trees, Gradient Boosting |
| **Boosting** | XGBoost, LightGBM, CatBoost |
| **Instance-Based** | KNN Regressor |
| **Kernel** | Support Vector Regressor (SVR) |
| **Ensemble** | Bagging Regressor |

### Meta-Learners (Stacking)

- Ridge Regression (default best performer)
- Linear Regression
- Support Vector Regressor (RBF kernel)

### Model Selection Criteria
- **Primary Metric**: R² score (coefficient of determination)
- **Validation**: 5-fold cross-validation
- **Optimization**: RandomizedSearchCV with 20 iterations
- **Selection**: Top 5 base models → Stacking → Best meta-learner

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: "ModuleNotFoundError: No module named 'openpyxl'"
**Solution**: Install Excel export support:
```bash
pip install openpyxl
```
**Note**: This may restart your notebook kernel - re-run all cells afterward.

---

#### Issue: "Target variable not found in dataset"
**Cause**: `DEFAULT_TARGET_VARIABLE` doesn't match any column name

**Solution**: 
1. Open your CSV and verify exact column name (case-sensitive)
2. Update constant: `DEFAULT_TARGET_VARIABLE = "exact_column_name"`
3. Or set `AUTO_MODE = False` to use interactive selection

---

#### Issue: Excel export fails with "name 'feature_statistics' is not defined"
**Cause**: Kernel was restarted and variables lost

**Solution**:
1. Re-run entire notebook (Kernel → Restart & Run All)
2. After completion, run: `%run EXPORT_TO_EXCEL.py`

---

#### Issue: Plots not saving as PNG files
**Cause**: Need to execute plot-saving cell

**Solution**:
1. Ensure notebook has run completely
2. Execute the final cell that contains plot-saving code
3. Check working directory for `best_model_*.png` files

---

#### Issue: VS Code notebook cells not saving edits
**Cause**: Known VS Code caching issue with notebook cells

**Workaround**:
1. Use standalone scripts (`EXPORT_TO_EXCEL.py`) instead
2. Restart VS Code
3. Clear VS Code cache: `%APPDATA%\Code\Cache`

---

#### Issue: "Behind corporate proxy" - pip install fails
**Solution**: Use proxy flag:
```bash
pip install --proxy http://your-proxy:port -r requirements.txt
```
For Intel networks: `http://proxy-dmz.intel.com:912`

---

### Performance Optimization

**For Large Datasets (>1M rows)**:
- Reduce `CROSS_VALIDATION_FOLDS` from 5 to 3
- Decrease `HYPERPARAMETER_ITERATIONS` from 20 to 10
- Consider removing computationally expensive models (XGBoost, LightGBM, CatBoost)

**For High-Dimensional Data (>100 features)**:
- Tighten `CORRELATION_THRESHOLD` to 0.90
- Increase `FEATURE_IMPORTANCE_THRESHOLD` to 0.005
- Enable aggressive variance filtering

**Memory Issues**:
- Use `TRAINING_DATA_SPLIT = 0.7` for smaller training set
- Remove ensemble models and use best base learner only
- Process data in chunks if possible

---

## 🤝 Contributing

### Reporting Issues
Submit bug reports or feature requests via [GitHub Issues](https://github.com/chemnteach/regression_modeling/issues).

Include:
- Python version
- Full error traceback
- Minimal reproducible example
- Dataset characteristics (rows, columns, missing data %)

### Pull Requests
1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature-name`
3. Follow PEP 8 style guidelines
4. Add docstrings (Google/NumPy style)
5. Test with sample data
6. Submit PR with clear description

---

## 📜 License

This project is provided as-is for educational and commercial use. See repository for specific license terms.

---

## 📧 Contact

**Author**: Chris Brown  
**Organization**: Intel Corporation  
**Repository**: [github.com/chemnteach/regression_modeling](https://github.com/chemnteach/regression_modeling)

---

## 🙏 Acknowledgments

Built with:
- scikit-learn ecosystem
- XGBoost, LightGBM, CatBoost
- pandas, NumPy, seaborn
- Jupyter/VS Code notebook environment

---

## 📚 Additional Resources

### Learning Materials
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Ensemble Methods Guide](https://scikit-learn.org/stable/modules/ensemble.html)
- [Feature Selection Strategies](https://scikit-learn.org/stable/modules/feature_selection.html)

### Related Projects
- `Vmin Kitchen Sink.ipynb` - Extended analysis examples
- `feature_reduction.py` - Reusable function library

---

**Last Updated**: November 24, 2025  
**Version**: 1.0  
**Status**: Production Ready ✅ 
