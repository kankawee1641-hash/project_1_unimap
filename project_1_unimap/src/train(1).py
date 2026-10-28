"""
IFJC4002 - Practical Artificial Intelligence with Python
Case 2: Used Car Price Prediction (Regression)
Script: train.py - Data Preprocessing, Model Training, Evaluation, and Pipeline Export
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def load_data(file_path):
    """Load dataset from Excel or CSV file."""
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def build_preprocessor():
    """Build ColumnTransformer for numeric and categorical features."""
    numeric_features = [
        'vehicle_age_years',
        'mileage_km',
        'engine_size_l',
        'condition_rating',
        'previous_owners'
    ]
    categorical_features = [
        'fuel_type',
        'transmission',
        'market_segment'
    ]

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def train_and_evaluate(data_path, output_model_path="models/car_price_pipeline.joblib"):
    # 1. Load Data
    df = load_data(data_path)

    # 2. Exclude identifiers and separate features / target
    feature_cols = [
        'vehicle_age_years', 'mileage_km', 'engine_size_l',
        'fuel_type', 'transmission', 'market_segment',
        'condition_rating', 'previous_owners'
    ]
    X = df[feature_cols]
    y = df['resale_price_rm']

    # 3. Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train split: {len(X_train)} samples | Test split: {len(X_test)} samples\n")

    # 4. Baseline Evaluation (Mean Predictor)
    y_mean = y_train.mean()
    y_pred_base = np.full_like(y_test, fill_value=y_mean)
    mae_base = mean_absolute_error(y_test, y_pred_base)
    rmse_base = root_mean_squared_error(y_test, y_pred_base)
    r2_base = r2_score(y_test, y_pred_base)

    print("=" * 65)
    print("BASELINE MODEL (Training Mean Predictor)")
    print(f"Mean Price: RM {y_mean:,.2f}")
    print(f"Test MAE  : RM {mae_base:,.2f}")
    print(f"Test RMSE : RM {rmse_base:,.2f}")
    print(f"Test R²   : {r2_base:.4f}")
    print("=" * 65 + "\n")

    # 5. Define Candidate Algorithms
    preprocessor = build_preprocessor()
    candidate_models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression (alpha=1.0)': Ridge(alpha=1.0),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = []
    trained_pipelines = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in candidate_models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # 5-Fold Cross-Validation on Training set only (prevent leakage)
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='r2').mean()

        # Fit on training data and test on reserved test set
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': name,
            'CV R² (5-Fold)': cv_r2,
            'Test MAE (RM)': mae,
            'Test RMSE (RM)': rmse,
            'Test R²': r2
        })
        trained_pipelines[name] = pipe

    # Format Results Table
    results_df = pd.DataFrame(results)
    print("=" * 75)
    print("MODEL COMPARISON RESULTS")
    print(results_df.to_string(index=False))
    print("=" * 75 + "\n")

    # 6. Select and Save Best Model Pipeline
    # Usually Gradient Boosting or Random Forest achieves highest R² and lowest MAE/RMSE
    best_model_name = results_df.sort_values(by='Test R²', ascending=False).iloc[0]['Model']
    best_pipe = trained_pipelines[best_model_name]
    print(f"Selected Final Model: {best_model_name}")

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(best_pipe, output_model_path)
    print(f"Exported end-to-end pipeline to: {output_model_path}")


if __name__ == '__main__':
    # Local path for dataset
    data_file = 'data/raw/02_Used_Car_Price.xlsx'
    if not os.path.exists(data_file):
        data_file = 'downloads/02_Used_Car_Price.xlsx'
    train_and_evaluate(data_file)
