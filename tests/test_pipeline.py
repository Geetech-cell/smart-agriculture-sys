import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import DatasetConfig, load_dataset, split_data


def test_load_dataset_columns(tmp_path):
    csv_path = tmp_path / "data.csv"
    data = pd.DataFrame(
        {
            "avg_temp_c": [25],
            "avg_rainfall_mm": [200],
            "soil_moisture_pct": [22],
            "ndvi": [0.6],
            "soil_ph": [6.4],
            "pest_pressure_idx": [0.25],
            "fertilizer_kg_per_ha": [110],
            "region": ["Test"],
            "crop": ["Maize"],
            "season": ["2021-01"],
            "yield_t_per_ha": [4.5],
        }
    )
    data.to_csv(csv_path, index=False)

    cfg = DatasetConfig(
        data_path=csv_path,
        target="yield_t_per_ha",
        feature_columns=[
            "avg_temp_c",
            "avg_rainfall_mm",
            "soil_moisture_pct",
            "ndvi",
            "soil_ph",
            "pest_pressure_idx",
            "fertilizer_kg_per_ha",
            "region",
            "crop",
            "season",
        ],
        categorical_features=["region", "crop", "season"],
    )

    df = load_dataset(cfg)
    assert not df.empty

    X_train, X_test, y_train, y_test = split_data(df, cfg)
    assert len(X_train) + len(X_test) == len(df)
    assert y_train.name == "yield_t_per_ha"

