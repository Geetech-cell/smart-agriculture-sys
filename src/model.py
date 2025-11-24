from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
import logging
from src.data_pipeline import load_dataset
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    data_path: str
    target: str
    feature_columns: list[str]
    categorical_features: list[str]
    test_size: float = 0.2
    random_state: int = 42
    model_params: dict = None
    cross_validation: dict = None
    feature_engineering: dict = None
    data_processing: dict = None
    
    def __post_init__(self):
        # Set default values if not provided
        if self.model_params is None:
            self.model_params = {}
        if self.cross_validation is None:
            self.cross_validation = {
                'cv_folds': 3,
                'scoring': 'neg_mean_absolute_error',
                'n_iter': 10,
                'randomized_search': True
            }
        if self.feature_engineering is None:
            self.feature_engineering = {
                'polynomial_degree': 1,
                'interaction_terms': False,
                'temporal_features': True,
                'create_interactions': []
            }
        if self.data_processing is None:
            self.data_processing = {
                'outlier_threshold': 3.0,
                'scale_features': True,
                'feature_selection': True
            }

def build_preprocessor(config: DatasetConfig) -> ColumnTransformer:
    """Build preprocessing pipeline with feature engineering."""
    numeric_features = [f for f in config.feature_columns 
                      if f not in config.categorical_features]
    
    # Create transformers
    numeric_steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
    
    # Add polynomial features if enabled
    if config.feature_engineering.get('polynomial_degree', 1) > 1:
        numeric_steps.append(
            ('poly', PolynomialFeatures(
                degree=config.feature_engineering['polynomial_degree'],
                include_bias=False,
                interaction_only=False
            ))
        )
    
    numeric_transformer = Pipeline(steps=numeric_steps)
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, config.categorical_features)
        ],
        remainder='drop',
        n_jobs=-1
    )
    
    return preprocessor

def build_model(params: Dict[str, Any] = None) -> Pipeline:
    """Build model with hyperparameter tuning."""
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }
    
    return RandomForestRegressor(**params)

def train_pipeline(
    config: DatasetConfig, 
    model_params: Dict = None
) -> Tuple[Pipeline, Dict[str, float]]:
    """Train pipeline with cross-validation and return fitted model plus metrics."""
    # Load and split data
    df = load_dataset(config)
    X = df[config.feature_columns]
    y = df[config.target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True
    )
    
    # Build and train pipeline with GridSearchCV
    preprocessor = build_preprocessor(config)
    model = build_model(model_params)
    
    # Add feature selection if enabled
    feature_selector = ('feature_selector', SelectKBest(score_func=f_regression, k='all'))
    if config.data_processing.get('feature_selection', False):
        pipeline_steps = [
            ('preprocessor', preprocessor),
            feature_selector,
            ('model', model)
        ]
    else:
        pipeline_steps = [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    
    pipeline = Pipeline(steps=pipeline_steps)
    
    # Hyperparameter grid for tuning
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2', None],
        'model__bootstrap': [True, False],
        'feature_selector__k': ['all', 10, 15, 20]
    }
    
    # Get search parameters from config
    cv_folds = config.cross_validation.get('cv_folds', 5)
    n_iter = config.cross_validation.get('n_iter', 20)
    use_randomized = config.cross_validation.get('randomized_search', True)
    
    if use_randomized:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=config.cross_validation['scoring'],
            n_jobs=-1,
            random_state=config.random_state,
            verbose=1
        )
    else:
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=config.cross_validation['scoring'],
            n_jobs=-1,
            verbose=1
        )
    
    logger.info(f"Starting model training with {'randomized' if use_randomized else 'grid'} search...")
    search.fit(X_train, y_train)
    
    # Get best model
    best_pipeline = search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_pipeline.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'best_params': search.best_params_
    }
    
    logger.info(f"Training complete. Best params: {search.best_params_}")
    logger.info(f"Test MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    
    return best_pipeline, metrics

def save_model(pipeline: Pipeline, output_path: str) -> None:
    """Persist trained pipeline with error handling."""
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, output_path)
        logger.info(f"Model saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(model_path: str) -> Pipeline:
    """Load pipeline from disk with error handling."""
    try:
        pipeline = joblib.load(model_path)
        if not hasattr(pipeline, 'predict'):
            raise ValueError("Loaded object is not a valid scikit-learn pipeline")
        return pipeline
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def audit_fairness(
    df: pd.DataFrame,
    pipeline: Pipeline,
    config: DatasetConfig,
    group_feature: str,
) -> Dict[str, float]:
    """Compute MAE per group to detect disparities with error handling."""
    if not hasattr(pipeline, 'predict'):
        raise ValueError("Pipeline must be fitted before auditing fairness")
        
    if group_feature not in df.columns:
        raise ValueError(f"Group feature '{group_feature}' not found in dataset")

    results = {}
    try:
        X = df[config.feature_columns]
        y = df[config.target]
        groups = df[group_feature]
        
        # Make predictions
        y_pred = pipeline.predict(X)
        
        # Calculate MAE per group
        for group in groups.unique():
            mask = (groups == group)
            if not mask.any():
                continue
            mae = mean_absolute_error(y[mask], y_pred[mask])
            results[str(group)] = float(mae)
            
        # Calculate max MAE gap if we have at least 2 groups
        if len(results) >= 2:
            mae_values = list(results.values())
            results["max_mae_gap"] = float(max(mae_values) - min(mae_values))
            
    except Exception as e:
        logger.error(f"Error in fairness audit: {str(e)}")
        raise
        
    return results