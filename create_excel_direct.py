"""
Direct script to create Excel report from notebook data.
Run this after the notebook has been executed with all variables in memory.
"""
print("Creating Excel report from notebook data...")
print("Please copy and paste this code into a NEW cell in your notebook and run it:")
print()
print("=" * 80)
print()
print("""
from datetime import datetime
import pandas as pd

# Create timestamp for filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
excel_filename = f'Feature_Analysis_Report_{timestamp}.xlsx'

# Create Excel writer
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    
    # TAB 1: NARRATIVE
    narrative_data = []
    narrative_data.append(['REGRESSION MODELING ANALYSIS REPORT', ''])
    narrative_data.append(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    narrative_data.append(['', ''])
    
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
    
    narrative_data.append(['METHODOLOGY', ''])
    narrative_data.append(['Feature Selection:', 'Algorithmic selection based on statistical significance and predictive power'])
    narrative_data.append(['', 'Features ranked by importance scores from multiple regression algorithms'])
    narrative_data.append(['Data Processing:', 'Automated cleaning, missing value imputation, and outlier handling'])
    narrative_data.append(['Cross-Validation:', '10-fold strategy for robust performance estimation'])
    narrative_data.append(['Hyperparameter Tuning:', 'Randomized search optimization for all models'])
    narrative_data.append(['Ensemble Method:', 'Stacking ensemble combining multiple base learners'])
    narrative_data.append(['', ''])
    
    narrative_data.append(['MODEL PERFORMANCE - BASE LEARNERS', ''])
    for name, model_obj in cutoff_models:
        if name in best_models:
            narrative_data.append([name, f"R² = {best_models[name]['best_score']:.4f}"])
    narrative_data.append(['', ''])
    
    narrative_data.append(['MODEL PERFORMANCE - STACKING ENSEMBLES', ''])
    for meta_name, score in sorted(meta_learner_scores.items(), key=lambda x: x[1], reverse=True):
        marker = " (BEST)" if meta_name == best_meta_learner else ""
        narrative_data.append([f'{meta_name}{marker}', f"R² = {score:.4f}"])
    narrative_data.append(['', ''])
    
    narrative_data.append(['KEY INSIGHTS', ''])
    narrative_data.append(['1. Feature Quality:', f'{len(feature_statistics)} features were selected through rigorous statistical analysis'])
    narrative_data.append(['2. Model Accuracy:', f'The ensemble model achieves R² = {meta_learner_scores[best_meta_learner]:.4f}, indicating strong predictive capability'])
    narrative_data.append(['3. Validation:', 'Results are based on held-out validation data for reliable generalization'])
    narrative_data.append(['4. Production Ready:', 'Model pipelines have been saved and are deployment-ready'])
    narrative_data.append(['', ''])
    
    narrative_data.append(['DATA QUALITY', ''])
    total_importance_sum = sum(f['importance_score'] for f in feature_statistics)
    top_3_importance = sum(sorted([f['importance_score'] for f in feature_statistics], reverse=True)[:3])
    narrative_data.append(['Top 3 Features Contribution:', f'{(top_3_importance/total_importance_sum)*100:.1f}% of total importance'])
    avg_missing = sum(f['missing_pct'] for f in feature_statistics) / len(feature_statistics)
    narrative_data.append(['Average Missing Data:', f'{avg_missing:.2f}%'])
    narrative_data.append(['', ''])
    
    narrative_data.append(['RECOMMENDATIONS', ''])
    narrative_data.append(['Next Steps:', '1. Review top features for business insights and driver analysis'])
    narrative_data.append(['', '2. Deploy model pipeline for production predictions'])
    narrative_data.append(['', '3. Monitor model performance with new data over time'])
    narrative_data.append(['', '4. Consider feature engineering based on importance rankings'])
    
    narrative_df = pd.DataFrame(narrative_data, columns=['Category', 'Details'])
    narrative_df.to_excel(writer, sheet_name='Narrative', index=False)
    
    # TAB 2: FEATURE METRICS
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
    
    metrics_df = pd.DataFrame(feature_metrics_data)
    metrics_df.to_excel(writer, sheet_name='Feature Metrics', index=False)
    
    # TAB 3: MODEL COMPARISON
    model_comparison_data = []
    for name, model_obj in cutoff_models:
        if name in best_models:
            model_comparison_data.append({
                'Model_Name': name,
                'Model_Type': 'Base Learner',
                'R2_Score': best_models[name]['best_score'],
                'Rank': 0
            })
    
    for meta_name, score in meta_learner_scores.items():
        model_comparison_data.append({
            'Model_Name': meta_name,
            'Model_Type': 'Stacking Ensemble',
            'R2_Score': score,
            'Rank': 0
        })
    
    comparison_df = pd.DataFrame(model_comparison_data)
    comparison_df = comparison_df.sort_values('R2_Score', ascending=False).reset_index(drop=True)
    comparison_df['Rank'] = range(1, len(comparison_df) + 1)
    comparison_df = comparison_df[['Rank', 'Model_Name', 'Model_Type', 'R2_Score']]
    comparison_df.to_excel(writer, sheet_name='Model Comparison', index=False)

print("=" * 80)
print("📊 FEATURE ANALYSIS REPORT GENERATED")
print("=" * 80)
print()
print(f"✅ Excel file created: {excel_filename}")
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
""")
print()
print("=" * 80)
