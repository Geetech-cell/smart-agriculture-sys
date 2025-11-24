"""Data ingestion and feature engineering utilities for crop yield modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    OneHotEncoder, 
    StandardScaler,
    FunctionTransformer,
    PowerTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_regression


@dataclass
class DatasetConfig:
    """Configuration for dataset handling."""
    data_path: str
    target: str
    feature_columns: List[str]
    categorical_features: List[str]
    test_size: float = 0.2
    random_state: int = 42


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Convert category dtypes to string for processing
        cat_cols = X.select_dtypes(include=['category']).columns
        if not cat_cols.empty:
            X[cat_cols] = X[cat_cols].astype(str)
        
        # Add temporal features if timestamp exists
        if 'timestamp' in X.columns:
            X['timestamp'] = pd.to_datetime(X['timestamp'])
            X['hour_of_day'] = X['timestamp'].dt.hour
            X['day_of_week'] = X['timestamp'].dt.dayofweek
            X['month'] = X['timestamp'].dt.month
        
        # Add interaction terms
        if 'avg_temp_c' in X.columns and 'avg_rainfall_mm' in X.columns:
            X['temp_rain_ratio'] = X['avg_temp_c'] / (X['avg_rainfall_mm'] + 1e-6)
            
        if 'soil_moisture_pct' in X.columns and 'fertilizer_kg_per_ha' in X.columns:
            X['moisture_fertilizer_interaction'] = X['soil_moisture_pct'] * X['fertilizer_kg_per_ha']
        
        # Add growth stage encoding if available
        if 'growth_stage' in X.columns:
            growth_stage_order = {
                'Germination': 0,
                'Vegetative': 1,
                'Tillering': 2,
                'Stem Elongation': 3,
                'Flowering': 4,
                'Pod Formation': 5,
                'Maturity': 6,
                'Harvest': 7
            }
            X['growth_stage_encoded'] = X['growth_stage'].map(growth_stage_order).fillna(-1)
        
        return X


def load_dataset(config: Union[DatasetConfig, Dict[str, Any]]) -> pd.DataFrame:
    """Load and preprocess dataset from CSV."""
    if isinstance(config, dict):
        config = DatasetConfig(**{k: v for k, v in config.items() 
                                if k in DatasetConfig.__annotations__})
    
    # Debug info
    print(f"Loading data from: {config.data_path}")
    print(f"Target column: {config.target}")
    print(f"Feature columns: {config.feature_columns}")
    print(f"Categorical features: {config.categorical_features}")
    
    # Load data
    df = pd.read_csv(config.data_path)
    
    # Ensure target is not in features
    if config.target in config.feature_columns:
        config.feature_columns = [f for f in config.feature_columns if f != config.target]
    
    # Check for required columns
    required_columns = set(config.feature_columns + [config.target] + config.categorical_features)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    
    # Apply initial preprocessing
    df = df[list(required_columns) + [config.target] if config.target not in required_columns else list(required_columns)]
    
    # Debug: Print DataFrame info
    print(f"Loaded DataFrame shape: {df.shape}")
    print("DataFrame columns:", df.columns.tolist())
    
    # Handle missing values
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Convert categorical columns to string type
    for col in config.categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df


def build_preprocessor(config: Union[DatasetConfig, Dict[str, Any]]) -> Pipeline:
    """Create preprocessing pipeline for numerical and categorical data."""
    if isinstance(config, dict):
        config = DatasetConfig(**{k: v for k, v in config.items() 
                                if k in DatasetConfig.__annotations__})
    
    # Separate numeric and categorical features
    numeric_features = [
        col for col in config.feature_columns 
        if col not in config.categorical_features
    ]
    
    # Feature engineering steps - without feature selection for now
    feature_engineering = Pipeline([
        ('feature_engineering', FeatureEngineer())
    ])

    # Numeric pipeline with transformation options
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('transformer', PowerTransformer(method='yeo-johnson', standardize=False))
    ])

    # Categorical pipeline with handling for unknown categories
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine all preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, config.categorical_features)
        ],
        remainder='drop'
    )

    # Full pipeline with feature engineering and feature selection
    full_pipeline = Pipeline([
        ('features', feature_engineering),
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_regression, k='all'))
    ])

    return full_pipeline


def split_data(
    df: pd.DataFrame, 
    config: Union[DatasetConfig, Dict[str, Any]],
    time_based_split: bool = False, 
    time_column: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataframe into train/test sets with options for time-based splitting.
    
    Args:
        df: Input DataFrame
        config: Configuration object or dictionary
        time_based_split: If True, perform time-based split
        time_column: Column to use for time-based splitting
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if isinstance(config, dict):
        config = DatasetConfig(**{k: v for k, v in config.items() 
                                if k in DatasetConfig.__annotations__})
    
    # Debug info
    print(f"Splitting data. DataFrame shape: {df.shape}")
    print(f"Features: {config.feature_columns}")
    print(f"Target: {config.target}")
    
    # Ensure target is not in features
    features = [f for f in config.feature_columns if f != config.target]
    
    # Create X and y
    X = df[features].copy()
    y = df[config.target].copy()
    
    # Debug shapes
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Handle categorical features - convert to string to avoid issues
    for col in config.categorical_features:
        if col in X.columns:
            X[col] = X[col].astype(str)
    
    if time_based_split and time_column and time_column in df.columns:
        # Time-based split (chronological order)
        df = df.sort_values(time_column)
        split_idx = int(len(df) * (1 - config.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        # Random split with stratification on the first categorical feature
        stratify_col = df[config.categorical_features[0]] if config.categorical_features else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify_col,
            shuffle=True
        )
    
    # Debug split results
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test