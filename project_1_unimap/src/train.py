"""
IFJC4002 - Practical Artificial Intelligence with Python
Faculty of Intelligent Computing, Universiti Malaysia Perlis (UniMAP)
Case 2: Used Car Price Prediction (Regression)
Script: train.py - Full Preprocessing, Training, Evaluation, and Comprehensive Artifact Saving
"""

import os
import datetime
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def main():
    print("🚀 Starting IFJC4002 Used Car Price Prediction Training Pipeline...")
    
    # 1. Setup Directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # 2. Load Dataset
    candidate_paths = [
        'data/raw/02_Used_Car_Price.xlsx',
        '02_Used_Car_Price.xlsx',
        'downloads/02_Used_Car_Price.xlsx'
    ]
    data_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if not data_path:
        raise FileNotFoundError("❌ Could not find 02_Used_Car_Price.xlsx in data/raw/ or current directory.")

    df = pd.read_excel(data_path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 3. Descriptive Stats & Automated EDA Figure Saving
    num_cols = ['vehicle_age_years', 'mileage_km', 'engine_size_l', 'condition_rating', 'previous_owners', 'resale_price_rm']
    stats_df = df[num_cols].describe().round(2)
    stats_df.to_csv('outputs/descriptive_statistics.csv')
    print("💾 Saved outputs/descriptive_statistics.csv")

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['figure.dpi'] = 150

    # Figure 1: Resale Price Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['resale_price_rm'], kde=True, color='royalblue', bins=30)
    plt.title('Figure 1: Distribution of Used Car Resale Price (RM)', fontweight='bold')
    plt.xlabel('Resale Price (RM)')
    plt.ylabel('Vehicle Count')
    plt.tight_layout()
    plt.savefig('outputs/01_price_distribution.png', bbox_inches='tight')
    plt.close()

    # Figure 2: Resale Price by Market Segment
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='market_segment', y='resale_price_rm', data=df, palette='Set2')
    plt.title('Figure 2: Resale Price by Market Segment', fontweight='bold')
    plt.xlabel('Market Segment')
    plt.ylabel('Resale Price (RM)')
    plt.tight_layout()
    plt.savefig('outputs/02_market_segment_boxplot.png', bbox_inches='tight')
    plt.close()

    # Figure 3: Correlation Heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Figure 3: Correlation Matrix with Resale Price (RM)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/03_correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    print("💾 Saved EDA figures (01, 02, 03) to outputs/")

    # 4. Feature Selection & 80/20 Train-Test Split
    feature_cols = [
        'vehicle_age_years', 'mileage_km', 'engine_size_l',
        'fuel_type', 'transmission', 'market_segment',
        'condition_rating', 'previous_owners'
    ]
    X = df[feature_cols]
    y = df['resale_price_rm']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save partitioned sets to data/processed/
    train_save = X_train.copy()
    train_save['resale_price_rm'] = y_train
    train_save.to_csv('data/processed/train_data.csv', index=False)

    test_save = X_test.copy()
    test_save['resale_price_rm'] = y_test
    test_save.to_csv('data/processed/test_data.csv', index=False)
    print("💾 Saved train_data.csv and test_data.csv to data/processed/")

    # 5. Preprocessing Pipeline
    numeric_features = ['vehicle_age_years', 'mileage_km', 'engine_size_l', 'condition_rating', 'previous_owners']
    categorical_features = ['fuel_type', 'transmission', 'market_segment']

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ])

    # 6. Baseline Model (Predicting Mean)
    y_train_mean = y_train.mean()
    y_base_pred = np.full_like(y_test, fill_value=y_train_mean)
    mae_base = mean_absolute_error(y_test, y_base_pred)
    rmse_base = root_mean_squared_error(y_test, y_base_pred)
    r2_base = r2_score(y_test, y_base_pred)

    results = [{
        'Model': 'Baseline (Mean Predictor)',
        '5-Fold CV R²': 0.0,
        'Test MAE (RM)': round(mae_base, 2),
        'Test RMSE (RM)': round(rmse_base, 2),
        'Test R²': round(r2_base, 4)
    }]

    # 7. Candidate Models Training & 5-Fold Cross Validation
    candidate_models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    trained_pipelines = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    test_preds_dict = {'actual_resale_price_rm': y_test.values, 'baseline_pred': y_base_pred}

    for name, model in candidate_models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='r2').mean()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': name,
            '5-Fold CV R²': round(cv_r2, 4),
            'Test MAE (RM)': round(mae, 2),
            'Test RMSE (RM)': round(rmse, 2),
            'Test R²': round(r2, 4)
        })
        trained_pipelines[name] = pipe
        test_preds_dict[f'{name}_pred'] = np.round(y_pred, 2)

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv('outputs/model_comparison.csv', index=False)
    print("💾 Saved outputs/model_comparison.csv")

    # Detailed test set predictions
    test_preds_df = pd.DataFrame(test_preds_dict, index=X_test.index)
    test_preds_df = pd.concat([X_test, test_preds_df], axis=1)
    test_preds_df.to_csv('outputs/test_predictions_detailed.csv', index=False)
    print("💾 Saved outputs/test_predictions_detailed.csv")

    # 8. Select Best Model & Save Residual and Feature Importance Plots
    best_model_name = comparison_df.sort_values(by='Test R²', ascending=False).iloc[0]['Model']
    best_pipe = trained_pipelines[best_model_name]
    y_best_pred = best_pipe.predict(X_test)
    residuals = y_test - y_best_pred

    # Figure 4: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_best_pred, alpha=0.55, color='teal', edgecolors='k', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2.5, label='Ideal Line (y=x)')
    plt.title(f'Figure 4: Actual vs. Predicted Resale Price ({best_model_name})', fontweight='bold')
    plt.xlabel('Actual Price (RM)')
    plt.ylabel('Predicted Price (RM)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/04_actual_vs_predicted.png', bbox_inches='tight')
    plt.close()

    # Figure 5: Residuals Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color='crimson', bins=30)
    plt.axvline(0, color='black', linestyle='--', lw=2)
    plt.title('Figure 5: Residual Error Distribution', fontweight='bold')
    plt.xlabel('Residual Error (RM)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('outputs/05_residuals_distribution.png', bbox_inches='tight')
    plt.close()

    # Figure 6: Feature Importances
    if hasattr(best_pipe.named_steps['regressor'], 'feature_importances_'):
        encoded_cat = best_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        all_features = list(numeric_features) + list(encoded_cat)
        importances = best_pipe.named_steps['regressor'].feature_importances_
        imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values('Importance', ascending=True)

        plt.figure(figsize=(9, 6))
        plt.barh(imp_df['Feature'], imp_df['Importance'], color='steelblue')
        plt.title(f'Figure 6: Feature Importance Analysis ({best_model_name})', fontweight='bold')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('outputs/06_feature_importance.png', bbox_inches='tight')
        plt.close()
        imp_df.sort_values('Importance', ascending=False).to_csv('outputs/feature_importances.csv', index=False)
        print("💾 Saved outputs/06_feature_importance.png & feature_importances.csv")

    # 9. Export Pipeline & Metadata
    model_export_path = 'models/car_price_pipeline.joblib'
    joblib.dump(best_pipe, model_export_path)
    print(f"💾 Saved model pipeline to: {model_export_path}")

    metadata = {
        'project_title': 'Used Car Resale Price Prediction',
        'course': 'IFJC4002 Practical AI with Python (UniMAP)',
        'training_timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_dataset_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'selected_model': best_model_name,
        'target_column': 'resale_price_rm',
        'features': feature_cols,
        'evaluation_summary': comparison_df.to_dict(orient='records')
    }
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("💾 Saved models/model_metadata.json")

    # 10. Save Detailed Text Summary Report
    report_text = f"""========================================================================
IFJC4002 PRACTICAL ARTIFICIAL INTELLIGENCE - PROJECT EXECUTION SUMMARY
Case 2: Used Car Price Prediction (Regression)
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
========================================================================

1. DATASET CHARACTERISTICS:
   - Total Samples: {len(df)} records
   - Target Variable: resale_price_rm (Mean: RM {df['resale_price_rm'].mean():,.2f}, Range: RM {df['resale_price_rm'].min():,.2f} - RM {df['resale_price_rm'].max():,.2f})
   - Missing Values Handled: vehicle_age_years ({df['vehicle_age_years'].isnull().sum()}), mileage_km ({df['mileage_km'].isnull().sum()})
   - Train Split: {len(X_train)} samples (80%)
   - Test Split: {len(X_test)} samples (20%)

2. MODEL EVALUATION & COMPARISON TABLE:
{comparison_df.to_string(index=False)}

3. SELECTED FINAL MODEL:
   - Best Model: {best_model_name}
   - Exported Pipeline: {model_export_path}

4. SAVED ARTIFACTS:
   - outputs/01_price_distribution.png
   - outputs/02_market_segment_boxplot.png
   - outputs/03_correlation_heatmap.png
   - outputs/04_actual_vs_predicted.png
   - outputs/05_residuals_distribution.png
   - outputs/06_feature_importance.png
   - outputs/model_comparison.csv
   - outputs/test_predictions_detailed.csv
   - outputs/descriptive_statistics.csv
   - models/car_price_pipeline.joblib
   - models/model_metadata.json
   - data/processed/train_data.csv & test_data.csv
========================================================================
"""
    with open('outputs/training_summary_report.txt', 'w') as f:
        f.write(report_text)
    print("💾 Saved outputs/training_summary_report.txt")
    print("🎉 All artifacts successfully generated and saved!")


if __name__ == '__main__':
    main()
