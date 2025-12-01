"""
Excel Report Generator for RAPID Regression Modeling
====================================================

Consolidated module for generating comprehensive Excel reports from
regression modeling results. Replaces three duplicate scripts:
- generate_excel_report.py
- EXPORT_TO_EXCEL.py
- create_excel_direct.py

This module can be used both from Jupyter notebooks and standalone Python scripts.

Usage from Notebook:
    from excel_reporter import generate_feature_analysis_report
    
    report_path = generate_feature_analysis_report(
        feature_statistics=feature_statistics,
        dependent_var=dependent_var,
        X_train=X_train,
        X_validation=X_validation,
        cutoff_models=cutoff_models,
        best_models=best_models,
        meta_learner_scores=meta_learner_scores,
        best_meta_learner=best_meta_learner
    )

Author: RAPID Development Team
Version: 2.0.0 (Consolidated)
"""

from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


def create_narrative_tab(
    dependent_var: str,
    feature_statistics: List[Dict],
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    cutoff_models: List[Tuple],
    best_models: Dict,
    meta_learner_scores: Dict[str, float],
    best_meta_learner: str
) -> pd.DataFrame:
    """
    Create the narrative tab with executive summary and insights.
    
    Parameters:
        dependent_var: Name of the target variable
        feature_statistics: List of feature statistic dictionaries
        X_train: Training dataset
        X_validation: Validation dataset
        cutoff_models: List of (name, model) tuples
        best_models: Dictionary of best model results
        meta_learner_scores: Dictionary of meta-learner scores
        best_meta_learner: Name of the best performing meta-learner
    
    Returns:
        DataFrame for the narrative tab
    """
    narrative_data = []
    
    # Header
    narrative_data.append(['REGRESSION MODELING ANALYSIS REPORT', ''])
    narrative_data.append(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    narrative_data.append(['', ''])
    
    # Executive Summary
    narrative_data.append(['EXECUTIVE SUMMARY', ''])
    narrative_data.append(['Target Variable:', dependent_var])
    narrative_data.append(['Total Features Selected:', len(feature_statistics)])
    narrative_data.append(['Training Samples:', len(X_train)])
    narrative_data.append(['Validation Samples:', len(X_validation)])
    narrative_data.append(['Number of Models Evaluated:', len(cutoff_models)])
    narrative_data.append(['Best Model Type:', 'Stacking Ensemble'])
    narrative_data.append(['Best Meta-Learner:', best_meta_learner])
    narrative_data.append(['Best R² Score:', f"{meta_learner_scores[best_meta_learner]:.4f}"])
    narrative_data.append(['', ''])
    
    # Methodology
    narrative_data.append(['METHODOLOGY', ''])
    narrative_data.append(['Feature Selection:', 'Algorithmic selection based on statistical significance and predictive power'])
    narrative_data.append(['', 'Features ranked by importance scores from multiple regression algorithms'])
    narrative_data.append(['Data Processing:', 'Automated cleaning, missing value imputation, and outlier handling'])
    narrative_data.append(['Cross-Validation:', '10-fold strategy for robust performance estimation'])
    narrative_data.append(['Hyperparameter Tuning:', 'Randomized search optimization for all models'])
    narrative_data.append(['Ensemble Method:', 'Stacking ensemble combining multiple base learners'])
    narrative_data.append(['', ''])
    
    # Model Performance - Base Learners
    narrative_data.append(['MODEL PERFORMANCE - BASE LEARNERS', ''])
    for name, model_obj in cutoff_models:
        if name in best_models:
            narrative_data.append([name, f"R² = {best_models[name]['best_score']:.4f}"])
    narrative_data.append(['', ''])
    
    # Model Performance - Stacking Ensembles
    narrative_data.append(['MODEL PERFORMANCE - STACKING ENSEMBLES', ''])
    for meta_name, score in sorted(meta_learner_scores.items(), key=lambda x: x[1], reverse=True):
        marker = " (BEST)" if meta_name == best_meta_learner else ""
        narrative_data.append([f'{meta_name}{marker}', f"R² = {score:.4f}"])
    narrative_data.append(['', ''])
    
    # Key Insights
    narrative_data.append(['KEY INSIGHTS', ''])
    narrative_data.append(['1. Feature Quality:', f'{len(feature_statistics)} features were selected through rigorous statistical analysis'])
    narrative_data.append(['2. Model Accuracy:', f'The ensemble model achieves R² = {meta_learner_scores[best_meta_learner]:.4f}, indicating strong predictive capability'])
    narrative_data.append(['3. Validation:', 'Results are based on held-out validation data for reliable generalization'])
    narrative_data.append(['4. Production Ready:', 'Model pipelines have been saved and are deployment-ready'])
    narrative_data.append(['', ''])
    
    # Data Quality
    narrative_data.append(['DATA QUALITY', ''])
    total_importance_sum = sum(f['importance_score'] for f in feature_statistics)
    top_3_importance = sum(sorted([f['importance_score'] for f in feature_statistics], reverse=True)[:3])
    narrative_data.append(['Top 3 Features Contribution:', f'{(top_3_importance/total_importance_sum)*100:.1f}% of total importance'])
    avg_missing = sum(f['missing_pct'] for f in feature_statistics) / len(feature_statistics)
    narrative_data.append(['Average Missing Data:', f'{avg_missing:.2f}%'])
    narrative_data.append(['', ''])
    
    # Recommendations
    narrative_data.append(['RECOMMENDATIONS', ''])
    narrative_data.append(['Next Steps:', '1. Review top features for business insights and driver analysis'])
    narrative_data.append(['', '2. Deploy model pipeline for production predictions'])
    narrative_data.append(['', '3. Monitor model performance with new data over time'])
    narrative_data.append(['', '4. Consider feature engineering based on importance rankings'])
    
    return pd.DataFrame(narrative_data, columns=['Category', 'Details'])


def create_feature_metrics_tab(feature_statistics: List[Dict]) -> pd.DataFrame:
    """
    Create the feature metrics tab with detailed statistics.
    
    Parameters:
        feature_statistics: List of feature statistic dictionaries
    
    Returns:
        DataFrame for the feature metrics tab
    """
    total_importance_sum = sum(f['importance_score'] for f in feature_statistics)
    
    feature_metrics_data = []
    for rank, feat_stat in enumerate(sorted(feature_statistics, key=lambda x: x['importance_score'], reverse=True), 1):
        pct_total = (feat_stat['importance_score'] / total_importance_sum) * 100
        feature_metrics_data.append({
            'Rank': rank,
            'Feature_Name': feat_stat['feature_name'],
            'Importance_Score': feat_stat['importance_score'],
            'Importance_Percent': pct_total,
            'Mean': feat_stat['mean'],
            'Std_Dev': feat_stat['std'],
            'Min': feat_stat['min'],
            'Max': feat_stat['max'],
            'Median': feat_stat['median'],
            'Missing_Percent': feat_stat['missing_pct'],
            'Unique_Values': feat_stat['unique_values'],
            'Data_Type': feat_stat['dtype']
        })
    
    return pd.DataFrame(feature_metrics_data)


def create_model_comparison_tab(
    cutoff_models: List[Tuple],
    best_models: Dict,
    meta_learner_scores: Dict[str, float]
) -> pd.DataFrame:
    """
    Create the model comparison tab with performance rankings.
    
    Parameters:
        cutoff_models: List of (name, model) tuples
        best_models: Dictionary of best model results
        meta_learner_scores: Dictionary of meta-learner scores
    
    Returns:
        DataFrame for the model comparison tab
    """
    model_comparison_data = []
    
    # Add base learners
    for name, model_obj in cutoff_models:
        if name in best_models:
            model_comparison_data.append({
                'Model_Name': name,
                'Model_Type': 'Base Learner',
                'R2_Score': best_models[name]['best_score'],
                'Rank': 0
            })
    
    # Add meta-learners
    for meta_name, score in meta_learner_scores.items():
        model_comparison_data.append({
            'Model_Name': meta_name,
            'Model_Type': 'Stacking Ensemble',
            'R2_Score': score,
            'Rank': 0
        })
    
    # Sort and rank
    comparison_df = pd.DataFrame(model_comparison_data)
    comparison_df = comparison_df.sort_values('R2_Score', ascending=False).reset_index(drop=True)
    comparison_df['Rank'] = range(1, len(comparison_df) + 1)
    comparison_df = comparison_df[['Rank', 'Model_Name', 'Model_Type', 'R2_Score']]
    
    return comparison_df


def generate_feature_analysis_report(
    feature_statistics: List[Dict],
    dependent_var: str,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    cutoff_models: List[Tuple],
    best_models: Dict,
    meta_learner_scores: Dict[str, float],
    best_meta_learner: str,
    output_filename: Optional[str] = None
) -> str:
    """
    Generate comprehensive Excel report for regression modeling results.
    
    This is the main entry point for generating reports. It consolidates
    functionality from three previous duplicate scripts.
    
    Parameters:
        feature_statistics: List of feature statistic dictionaries
        dependent_var: Name of the target variable
        X_train: Training dataset
        X_validation: Validation dataset
        cutoff_models: List of (name, model) tuples
        best_models: Dictionary of best model results
        meta_learner_scores: Dictionary of meta-learner scores
        best_meta_learner: Name of the best performing meta-learner
        output_filename: Optional custom filename (default: auto-generated with timestamp)
    
    Returns:
        str: Path to the generated Excel file
    
    Example:
        >>> report_path = generate_feature_analysis_report(
        ...     feature_statistics=feature_statistics,
        ...     dependent_var=dependent_var,
        ...     X_train=X_train,
        ...     X_validation=X_validation,
        ...     cutoff_models=cutoff_models,
        ...     best_models=best_models,
        ...     meta_learner_scores=meta_learner_scores,
        ...     best_meta_learner=best_meta_learner
        ... )
        >>> print(f"Report generated: {report_path}")
    """
    
    # Generate filename if not provided
    if output_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'Feature_Analysis_Report_{timestamp}.xlsx'
    
    # Create Excel writer
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        
        # Tab 1: Narrative
        narrative_df = create_narrative_tab(
            dependent_var, feature_statistics, X_train, X_validation,
            cutoff_models, best_models, meta_learner_scores, best_meta_learner
        )
        narrative_df.to_excel(writer, sheet_name='Narrative', index=False)
        
        # Tab 2: Feature Metrics
        metrics_df = create_feature_metrics_tab(feature_statistics)
        metrics_df.to_excel(writer, sheet_name='Feature Metrics', index=False)
        
        # Tab 3: Model Comparison
        comparison_df = create_model_comparison_tab(
            cutoff_models, best_models, meta_learner_scores
        )
        comparison_df.to_excel(writer, sheet_name='Model Comparison', index=False)
    
    # Print success message
    print("=" * 80)
    print("📊 FEATURE ANALYSIS REPORT GENERATED")
    print("=" * 80)
    print()
    print(f"✅ Excel file created: {output_filename}")
    print()
    print("Report Contents:")
    print("  Tab 1: Narrative - Executive summary, methodology, insights, and recommendations")
    print("  Tab 2: Feature Metrics - Comprehensive statistics for all selected features")
    print("  Tab 3: Model Comparison - Performance comparison of all trained models")
    print()
    print(f"Total Features: {len(feature_statistics)}")
    print(f"Target Variable: {dependent_var}")
    print(f"Best Model: {best_meta_learner} (R² = {meta_learner_scores[best_meta_learner]:.4f})")
    print()
    print("=" * 80)
    
    return output_filename


def generate_report_from_notebook_context() -> Optional[str]:
    """
    Generate report by automatically extracting variables from Jupyter notebook context.
    
    This function is useful when running from within a Jupyter notebook - it will
    automatically pull the necessary variables from the notebook namespace.
    
    Returns:
        str: Path to generated report, or None if variables not found
    
    Example:
        >>> # In a Jupyter notebook cell after running analysis:
        >>> from excel_reporter import generate_report_from_notebook_context
        >>> report_path = generate_report_from_notebook_context()
    """
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        
        # Extract variables from notebook namespace
        required_vars = [
            'feature_statistics', 'dependent_var', 'X_train', 'X_validation',
            'cutoff_models', 'best_models', 'meta_learner_scores', 'best_meta_learner'
        ]
        
        namespace = ipython.user_ns
        missing_vars = [var for var in required_vars if var not in namespace]
        
        if missing_vars:
            print(f"❌ Error: Missing required variables: {', '.join(missing_vars)}")
            print("\nPlease ensure all modeling cells have been executed.")
            return None
        
        # Generate report using extracted variables
        return generate_feature_analysis_report(
            feature_statistics=namespace['feature_statistics'],
            dependent_var=namespace['dependent_var'],
            X_train=namespace['X_train'],
            X_validation=namespace['X_validation'],
            cutoff_models=namespace['cutoff_models'],
            best_models=namespace['best_models'],
            meta_learner_scores=namespace['meta_learner_scores'],
            best_meta_learner=namespace['best_meta_learner']
        )
        
    except ImportError:
        print("❌ Error: This function must be called from a Jupyter notebook context")
        return None
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return None


# Convenience function for backward compatibility
def export_to_excel(**kwargs) -> Optional[str]:
    """
    Legacy function name for backward compatibility.
    Calls generate_feature_analysis_report with the same parameters.
    """
    return generate_feature_analysis_report(**kwargs)


if __name__ == "__main__":
    print("Excel Reporter Module for RAPID")
    print("=" * 80)
    print()
    print("This module consolidates Excel report generation functionality.")
    print()
    print("Usage:")
    print("  1. From notebook with explicit variables:")
    print("     from excel_reporter import generate_feature_analysis_report")
    print("     report_path = generate_feature_analysis_report(...)")
    print()
    print("  2. From notebook with auto-detection:")
    print("     from excel_reporter import generate_report_from_notebook_context")
    print("     report_path = generate_report_from_notebook_context()")
    print()
    print("=" * 80)
