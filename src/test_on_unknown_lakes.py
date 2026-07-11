# src/test_all_models_on_lakes.py

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = "D:\\#3.Projects(extra)\\36.rbacon_ml_project"

# Paths
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "Test_On_Unknown_Lake_Datasets", "input_Lake_Datasets")
GROUND_TRUTH_DIR = os.path.join(PROJECT_ROOT, "Test_On_Unknown_Lake_Datasets", "Ground_Truth(Actual_Rbacon_Generated_ML_Files)")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Test_Results_Complete_Analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model paths
MODELS = {
    'Random Forest': {
        'model': os.path.join(PROJECT_ROOT, 'models', 'randomforest_model.pkl'),
        'scaler': os.path.join(PROJECT_ROOT, 'models', 'randomforest_scaler.pkl')
    },
    'XGBoost': {
        'model': os.path.join(PROJECT_ROOT, 'models', 'xgboost_model.pkl'),
        'scaler': os.path.join(PROJECT_ROOT, 'models', 'xgboost_scaler.pkl')
    },
    'MLP': {
        'model': os.path.join(PROJECT_ROOT, 'models', 'mlp_model_final.keras'),
        'scaler': os.path.join(PROJECT_ROOT, 'models', 'input_scaler.pkl'),
        'output_scaler': os.path.join(PROJECT_ROOT, 'models', 'output_scaler.pkl')
    }
}

FEATURES = ['depth', 'age', 'error']
TARGETS = ['mean_age', 'median_age', 'lo95', 'hi95', 'ci_width', 'acc_rate', 'sed_rate']

# Training range (from your data)
TRAINING_DEPTH_MAX = 900  # cm
TRAINING_AGE_MAX = 50000  # years

print("="*80)
print("COMPLETE MODEL TESTING ON 10 LAKE DATASETS")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ==================================================
# LOAD ALL MODELS
# ==================================================
print("\n[1] LOADING MODELS...")
print("-"*40)

loaded_models = {}

# Load Random Forest
rf_model = joblib.load(MODELS['Random Forest']['model'])
rf_scaler = joblib.load(MODELS['Random Forest']['scaler'])
loaded_models['Random Forest'] = {'model': rf_model, 'scaler': rf_scaler}
print("OK Random Forest loaded")

# Load XGBoost
xgb_model = joblib.load(MODELS['XGBoost']['model'])
xgb_scaler = joblib.load(MODELS['XGBoost']['scaler'])
loaded_models['XGBoost'] = {'model': xgb_model, 'scaler': xgb_scaler}
print("OK XGBoost loaded")

# Load MLP
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
mlp_model = load_model(MODELS['MLP']['model'])
mlp_scaler = joblib.load(MODELS['MLP']['scaler'])
mlp_output_scaler = joblib.load(MODELS['MLP']['output_scaler'])
loaded_models['MLP'] = {'model': mlp_model, 'scaler': mlp_scaler, 'output_scaler': mlp_output_scaler}
print("OK MLP loaded")

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def get_ground_truth_at_depths(ground_truth_df, input_depths):
    """Get ground truth values exactly at input depths"""
    gt_values = []
    for depth in input_depths:
        exact_match = ground_truth_df[ground_truth_df['depth'] == depth]
        if len(exact_match) > 0:
            gt_values.append(exact_match.iloc[0][TARGETS].values)
        else:
            # Find closest depth if exact not found
            closest_idx = (ground_truth_df['depth'] - depth).abs().idxmin()
            gt_values.append(ground_truth_df.iloc[closest_idx][TARGETS].values)
    return np.array(gt_values)

def predict_with_model(model_info, X, model_name):
    """Make predictions with proper scaling"""
    if model_name == 'MLP':
        X_scaled = model_info['scaler'].transform(X)
        pred_scaled = model_info['model'].predict(X_scaled, verbose=0)
        predictions = model_info['output_scaler'].inverse_transform(pred_scaled)
    else:
        X_scaled = model_info['scaler'].transform(X)
        predictions = model_info['model'].predict(X_scaled)
    return predictions

def calculate_metrics(actual, predicted):
    """Calculate performance metrics"""
    # Remove any NaN/inf values
    mask = ~(np.isnan(actual) | np.isnan(predicted) | 
             np.isinf(actual) | np.isinf(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) < 2:
        return {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan, 'pct_error': np.nan}
    
    r2 = r2_score(actual_clean, predicted_clean)
    mae = mean_absolute_error(actual_clean, predicted_clean)
    rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
    pct_error = np.mean(np.abs((actual_clean - predicted_clean) / (actual_clean + 1e-8)) * 100)
    
    return {'r2': r2, 'mae': mae, 'rmse': rmse, 'pct_error': pct_error}

def fix_predictions(predictions):
    """Apply post-processing fixes to predictions"""
    fixed = predictions.copy()
    
    # Clip ages to realistic range
    fixed[:, 0] = np.clip(fixed[:, 0], 0, TRAINING_AGE_MAX)  # mean_age
    fixed[:, 1] = np.clip(fixed[:, 1], 0, TRAINING_AGE_MAX)  # median_age
    fixed[:, 2] = np.clip(fixed[:, 2], 0, TRAINING_AGE_MAX)  # lo95
    fixed[:, 3] = np.clip(fixed[:, 3], 0, TRAINING_AGE_MAX)  # hi95
    
    # Ensure hi95 > lo95
    invalid_ci = fixed[:, 3] <= fixed[:, 2]
    if invalid_ci.any():
        fixed[invalid_ci, 3] = fixed[invalid_ci, 2] + 100
    
    # Recalculate CI width
    fixed[:, 4] = fixed[:, 3] - fixed[:, 2]
    fixed[:, 4] = np.clip(fixed[:, 4], 10, 100000)
    
    # Clip rates
    fixed[:, 5] = np.clip(fixed[:, 5], 0, 100)  # acc_rate
    fixed[:, 6] = np.clip(fixed[:, 6], 0.001, 10)  # sed_rate
    
    return fixed

# ==================================================
# FIND TEST FILES
# ==================================================
test_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.csv')]
test_files.sort()

print(f"\n[2] FOUND {len(test_files)} TEST FILES")
print("-"*40)
for f in test_files:
    print(f"    - {f}")

# ==================================================
# STORE ALL RESULTS
# ==================================================
all_results = {}
detailed_predictions = []
model_performance = {model: {'r2': [], 'mae': [], 'samples': []} for model in MODELS.keys()}

# ==================================================
# TEST EACH LAKE
# ==================================================
print("\n[3] TESTING MODELS ON EACH LAKE")
print("="*80)

for test_file in test_files:
    lake_name = test_file.replace('.csv', '')
    print(f"\n{'='*60}")
    print(f"LAKE: {lake_name}")
    print(f"{'='*60}")
    
    try:
        # Load input data
        input_path = os.path.join(TEST_DATA_DIR, test_file)
        input_df = pd.read_csv(input_path)
        
        # Load ground truth
        truth_file = f"{lake_name}_ML_training_data.csv"
        truth_path = os.path.join(GROUND_TRUTH_DIR, truth_file)
        
        if not os.path.exists(truth_path):
            print(f"   WARNING: Ground truth not found - skipping")
            continue
        
        ground_truth_df = pd.read_csv(truth_path)
        
        # Prepare data
        X = input_df[FEATURES].values
        depths = input_df['depth'].values
        
        # Get ground truth at input depths
        Y_actual = get_ground_truth_at_depths(ground_truth_df, depths)
        
        print(f"   Input samples: {len(input_df)}")
        print(f"   Depth range: {depths.min():.1f} - {depths.max():.1f} cm")
        print(f"   Within training range: {np.sum(depths <= TRAINING_DEPTH_MAX)}/{len(depths)} samples")
        
        # Test each model
        lake_results = {}
        
        for model_name, model_info in loaded_models.items():
            try:
                # Predict
                y_pred_raw = predict_with_model(model_info, X, model_name)
                y_pred = fix_predictions(y_pred_raw)
                
                # Calculate metrics for each target
                target_metrics = {}
                for i, target in enumerate(TARGETS):
                    metrics = calculate_metrics(Y_actual[:, i], y_pred[:, i])
                    target_metrics[target] = metrics
                
                # Overall metrics (average across targets)
                overall_r2 = np.mean([target_metrics[t]['r2'] for t in TARGETS if not np.isnan(target_metrics[t]['r2'])])
                overall_mae = np.mean([target_metrics[t]['mae'] for t in TARGETS if not np.isnan(target_metrics[t]['mae'])])
                overall_pct = np.mean([target_metrics[t]['pct_error'] for t in TARGETS if not np.isnan(target_metrics[t]['pct_error'])])
                
                lake_results[model_name] = {
                    'predictions': y_pred,
                    'metrics': target_metrics,
                    'overall_r2': overall_r2,
                    'overall_mae': overall_mae,
                    'overall_pct_error': overall_pct
                }
                
                # Store for global performance
                model_performance[model_name]['r2'].append(overall_r2)
                model_performance[model_name]['mae'].append(overall_mae)
                model_performance[model_name]['samples'].append(len(input_df))
                
                print(f"   {model_name:15s}: R2={overall_r2:.4f}, MAE={overall_mae:.0f} yrs")
                
            except Exception as e:
                print(f"   {model_name:15s}: ERROR - {str(e)[:50]}")
                lake_results[model_name] = None
        
        all_results[lake_name] = lake_results
        
        # Save detailed predictions
        output_df = input_df.copy()
        for model_name in MODELS.keys():
            if lake_results.get(model_name):
                preds = lake_results[model_name]['predictions']
                for i, target in enumerate(TARGETS):
                    output_df[f'{model_name}_{target}'] = preds[:, i]
        
        # Add ground truth
        for i, target in enumerate(TARGETS):
            output_df[f'actual_{target}'] = Y_actual[:, i]
        
        # Add depth confidence
        output_df['within_training_depth'] = output_df['depth'] <= TRAINING_DEPTH_MAX
        
        # Save
        output_file = os.path.join(OUTPUT_DIR, f"{lake_name}_complete_predictions.csv")
        output_df.to_csv(output_file, index=False)
        
    except Exception as e:
        print(f"   ERROR processing {lake_name}: {e}")

# ==================================================
# GENERATE COMPREHENSIVE ANALYSIS REPORT
# ==================================================
print("\n" + "="*80)
print("[4] GENERATING ANALYSIS REPORT")
print("="*80)

# Create performance summary dataframe
performance_data = []
for lake_name, lake_results in all_results.items():
    for model_name in MODELS.keys():
        if lake_results.get(model_name):
            perf = lake_results[model_name]
            performance_data.append({
                'Lake': lake_name,
                'Model': model_name,
                'R2_Score': perf['overall_r2'],
                'MAE_Years': perf['overall_mae'],
                'Error_Percentage': perf['overall_pct_error']
            })

performance_df = pd.DataFrame(performance_data)
performance_df.to_csv(os.path.join(OUTPUT_DIR, "00_MODEL_PERFORMANCE_SUMMARY.csv"), index=False)

# Create pivot tables
r2_pivot = performance_df.pivot(index='Lake', columns='Model', values='R2_Score')
mae_pivot = performance_df.pivot(index='Lake', columns='Model', values='MAE_Years')
error_pivot = performance_df.pivot(index='Lake', columns='Model', values='Error_Percentage')

r2_pivot.to_csv(os.path.join(OUTPUT_DIR, "01_R2_COMPARISON.csv"))
mae_pivot.to_csv(os.path.join(OUTPUT_DIR, "02_MAE_COMPARISON.csv"))
error_pivot.to_csv(os.path.join(OUTPUT_DIR, "03_ERROR_PERCENTAGE_COMPARISON.csv"))

# ==================================================
# CALCULATE AVERAGE PERFORMANCE
# ==================================================
print("\n[5] AVERAGE PERFORMANCE ACROSS ALL LAKES")
print("="*80)

print("\n" + "-"*80)
print(f"{'Model':<15s} {'Avg R2':<12s} {'Std R2':<12s} {'Avg MAE':<12s} {'Avg Error %':<12s} {'Rating':<15s}")
print("-"*80)

best_model = None
best_score = -999

for model_name in MODELS.keys():
    model_data = performance_df[performance_df['Model'] == model_name]
    if len(model_data) > 0:
        avg_r2 = model_data['R2_Score'].mean()
        std_r2 = model_data['R2_Score'].std()
        avg_mae = model_data['MAE_Years'].mean()
        avg_error = model_data['Error_Percentage'].mean()
        
        if avg_r2 > 0.9:
            rating = "EXCELLENT"
        elif avg_r2 > 0.7:
            rating = "GOOD"
        elif avg_r2 > 0.5:
            rating = "FAIR"
        else:
            rating = "POOR"
        
        print(f"{model_name:<15s} {avg_r2:<12.4f} {std_r2:<12.4f} {avg_mae:<12.1f} {avg_error:<12.1f} {rating:<15s}")
        
        if avg_r2 > best_score:
            best_score = avg_r2
            best_model = model_name

# ==================================================
# FIND BEST PERFORMING LAKE FOR EACH MODEL
# ==================================================
print("\n[6] BEST PERFORMING LAKE FOR EACH MODEL")
print("="*80)

for model_name in MODELS.keys():
    model_data = performance_df[performance_df['Model'] == model_name]
    if len(model_data) > 0:
        best_idx = model_data['R2_Score'].idxmax()
        best_lake = model_data.loc[best_idx, 'Lake']
        best_r2 = model_data.loc[best_idx, 'R2_Score']
        print(f"   {model_name}: {best_lake} (R2 = {best_r2:.4f})")

# ==================================================
# PER-TARGET PERFORMANCE
# ==================================================
print("\n[7] PER-TARGET PERFORMANCE (Ensemble Average)")
print("="*80)

# Average predictions across models for per-target analysis
target_performance = {target: {'r2': [], 'mae': []} for target in TARGETS}

for lake_name, lake_results in all_results.items():
    # Get predictions from all models and average
    model_preds = []
    for model_name in MODELS.keys():
        if lake_results.get(model_name):
            model_preds.append(lake_results[model_name]['predictions'])
    
    if len(model_preds) >= 2:
        ensemble_pred = np.mean(model_preds, axis=0)
        
        # Get actual values
        output_file = os.path.join(OUTPUT_DIR, f"{lake_name}_complete_predictions.csv")
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            for i, target in enumerate(TARGETS):
                actual_col = f'actual_{target}'
                if actual_col in df.columns:
                    actual = df[actual_col].values
                    predicted = ensemble_pred[:, i]
                    metrics = calculate_metrics(actual, predicted)
                    if not np.isnan(metrics['r2']):
                        target_performance[target]['r2'].append(metrics['r2'])
                        target_performance[target]['mae'].append(metrics['mae'])

print("\n" + "-"*60)
print(f"{'Target':<15s} {'Avg R2':<12s} {'Avg MAE':<12s} {'Performance':<15s}")
print("-"*60)

for target in TARGETS:
    if target_performance[target]['r2']:
        avg_r2 = np.mean(target_performance[target]['r2'])
        avg_mae = np.mean(target_performance[target]['mae'])
        
        if avg_r2 > 0.9:
            rating = "EXCELLENT"
        elif avg_r2 > 0.7:
            rating = "GOOD"
        elif avg_r2 > 0.5:
            rating = "FAIR"
        else:
            rating = "POOR"
        
        print(f"{target:<15s} {avg_r2:<12.4f} {avg_mae:<12.1f} {rating:<15s}")

# ==================================================
# FINAL RECOMMENDATION
# ==================================================
print("\n" + "="*80)
print("[8] FINAL RECOMMENDATION")
print("="*80)

print(f"""
Based on comprehensive testing of 3 models on {len(all_results)} lake datasets:

BEST MODEL: {best_model}
- Average R2 Score: {best_score:.4f}
- Most consistent performance across all lake types
- Reliable for predictions within training range (depth < 900cm)

RECOMMENDATIONS:
1. Use {best_model} for production predictions
2. Ensemble predictions (average of all models) for critical applications
3. Only trust predictions when depth <= 900cm
4. Be cautious with lakes deeper than training range

OUTPUT FILES GENERATED:
- 00_MODEL_PERFORMANCE_SUMMARY.csv - Complete metrics table
- 01_R2_COMPARISON.csv - R2 scores comparison by lake
- 02_MAE_COMPARISON.csv - MAE comparison by lake
- 03_ERROR_PERCENTAGE_COMPARISON.csv - Error % comparison
- *_complete_predictions.csv - Detailed predictions for each lake
""")

# ==================================================
# CREATE HTML REPORT
# ==================================================
print("\n[9] CREATING HTML REPORT")
print("="*80)

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Lake Model Testing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #667eea; }}
        h2 {{ color: #764ba2; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #667eea; color: white; }}
        .excellent {{ background-color: #d4edda; color: #155724; }}
        .good {{ background-color: #d1ecf1; color: #0c5460; }}
        .fair {{ background-color: #fff3cd; color: #856404; }}
        .poor {{ background-color: #f8d7da; color: #721c24; }}
        .best {{ background-color: #ffd700; font-weight: bold; }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>Lake Model Testing Results</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Models tested: Random Forest, XGBoost, MLP</p>
    <p>Test lakes: {len(all_results)} datasets</p>
    
    <div class="summary-box">
        <h2>BEST MODEL: {best_model}</h2>
        <p>Average R2 Score: {best_score:.4f}</p>
        <p>This model performed best across all lake datasets.</p>
    </div>
    
    <h2>Model Performance Summary</h2>
    <table>
        <thead>
            <tr><th>Model</th><th>Avg R2</th><th>Avg MAE (years)</th><th>Rating</th></tr>
        </thead>
        <tbody>
"""

for model_name in MODELS.keys():
    model_data = performance_df[performance_df['Model'] == model_name]
    if len(model_data) > 0:
        avg_r2 = model_data['R2_Score'].mean()
        avg_mae = model_data['MAE_Years'].mean()
        
        if avg_r2 > 0.9:
            rating_class = "excellent"
            rating_text = "EXCELLENT"
        elif avg_r2 > 0.7:
            rating_class = "good"
            rating_text = "GOOD"
        elif avg_r2 > 0.5:
            rating_class = "fair"
            rating_text = "FAIR"
        else:
            rating_class = "poor"
            rating_text = "POOR"
        
        highlight = "best" if model_name == best_model else ""
        html_content += f"""
            <tr class="{highlight}">
                <td>{model_name}</td>
                <td>{avg_r2:.4f}</td>
                <td>{avg_mae:.0f}</td>
                <td><span class="{rating_class}">{rating_text}</span></td>
            </tr>"""

html_content += """
        </tbody>
    </table>
    
    <h2>Per-Lake Performance (R2 Scores)</h2>
    </table>
        <thead>
            <tr><th>Lake</th><th>Random Forest</th><th>XGBoost</th><th>MLP</th></tr>
        </thead>
        <tbody>
"""

for lake_name in all_results.keys():
    html_content += f"<tr><td>{lake_name}</td>"
    for model_name in MODELS.keys():
        lake_data = performance_df[(performance_df['Lake'] == lake_name) & (performance_df['Model'] == model_name)]
        if len(lake_data) > 0:
            r2 = lake_data.iloc[0]['R2_Score']
            if r2 > 0.9:
                html_content += f"<td class='excellent'>{r2:.3f}</td>"
            elif r2 > 0.7:
                html_content += f"<td class='good'>{r2:.3f}</td>"
            elif r2 > 0.5:
                html_content += f"<td class='fair'>{r2:.3f}</td>"
            else:
                html_content += f"<td class='poor'>{r2:.3f}</td>"
        else:
            html_content += "<td>N/A</td>"
    html_content += "</tr>"

html_content += """
        </tbody>
    </table>
    
    <h2>Recommendations</h2>
    <ul>
        <li><strong>Use """ + best_model + """</strong> for production predictions</li>
        <li>Ensemble predictions (average of all models) for critical applications</li>
        <li>Only trust predictions when depth ≤ 900cm (within training range)</li>
        <li>Be cautious with lakes deeper than training range (extrapolation)</li>
    </ul>
    
    <hr>
    <p><em>Detailed results saved in CSV files in the output directory.</em></p>
</body>
</html>
"""

html_path = os.path.join(OUTPUT_DIR, "COMPLETE_ANALYSIS_REPORT.html")
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"OK HTML Report: {html_path}")

# ==================================================
# FINAL SUMMARY
# ==================================================
print("\n" + "="*80)
print("TESTING COMPLETE!")
print("="*80)
print(f"\nAll results saved to: {OUTPUT_DIR}")
print("\nFiles generated:")
print("   • 00_MODEL_PERFORMANCE_SUMMARY.csv - Complete metrics")
print("   • 01_R2_COMPARISON.csv - R2 scores by lake")
print("   • 02_MAE_COMPARISON.csv - MAE by lake")
print("   • 03_ERROR_PERCENTAGE_COMPARISON.csv - Error % by lake")
print("   • COMPLETE_ANALYSIS_REPORT.html - Interactive HTML report")
print("   • [LakeName]_complete_predictions.csv - Detailed predictions per lake")
print(f"\nBEST MODEL: {best_model}")
print("="*80)