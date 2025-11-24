"""CLI entry point for training crop-yield models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_pipeline import load_dataset
from src.model import (
    DatasetConfig,
    train_pipeline,
    save_model,
    audit_fairness,
    load_model
)

DEFAULT_CONFIG = "configs/default.json"

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train crop yield regressor.")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model",
    )
    return parser.parse_args()

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)

def main() -> None:
    """Main training function."""
    args = parse_args()
    
    try:
        # Load configuration
        config_data = load_config(args.config)
        
        # Create dataset config with all parameters
        config = DatasetConfig(
            data_path=config_data["data_path"],
            target=config_data["target"],
            feature_columns=config_data["feature_columns"],
            categorical_features=config_data.get("categorical_features", []),
            test_size=config_data.get("test_size", 0.2),
            random_state=config_data.get("random_state", 42),
            model_params=config_data.get("model_params", {}),
            cross_validation=config_data.get("cross_validation"),
            feature_engineering=config_data.get("feature_engineering"),
            data_processing=config_data.get("data_processing")
        )
        
        # Train the model
        print("\n🚀 Starting model training...")
        pipeline, metrics = train_pipeline(config, config_data.get("model_params", {}))
        
        # Save the model
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "crop_yield_model.pkl"
        save_model(pipeline, str(model_path))
        
        print("\n✅ Model training complete!")
        print(f"📊 Model performance:")
        print(f"   - Test MAE: {metrics['mae']:.4f}")
        print(f"   - Test R²: {metrics['r2']:.4f}")
        
        if 'best_params' in metrics and metrics['best_params']:
            print("\n🔧 Best hyperparameters:")
            for param, value in metrics['best_params'].items():
                print(f"   - {param}: {value}")
        
        # Audit model fairness
        try:
            print("\n🔍 Running fairness audit...")
            df = pd.read_csv(config.data_path)
            group_feature = config.categorical_features[0] if config.categorical_features else None
            
            if group_feature:
                fairness_results = audit_fairness(
                    df=df,
                    pipeline=pipeline,
                    config=config,
                    group_feature=group_feature
                )
                
                print("\n📈 Fairness Audit Results:")
                for group, mae in fairness_results.items():
                    if group != "max_mae_gap":
                        print(f"   - {group}: MAE = {mae:.4f}")
                
                if "max_mae_gap" in fairness_results:
                    print(f"\n📊 Maximum MAE gap between groups: {fairness_results['max_mae_gap']:.4f}")
            else:
                print("⚠️  No categorical features found for fairness audit.")
                
        except Exception as e:
            print(f"\n⚠️  Warning: Could not complete fairness audit: {str(e)}")
        
        print(f"\n💾 Model saved to: {model_path.absolute()}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Configuration file not found: {e.filename}")
    except json.JSONDecodeError:
        print(f"\n❌ Error: Invalid JSON format in config file")
    except KeyError as e:
        print(f"\n❌ Error: Missing required configuration: {str(e)}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()