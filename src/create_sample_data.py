# Update create_sample_data.py
import pandas as pd
import numpy as np
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Sample crop data with all required columns
crop_data = {
    "region": ["North", "South", "East", "West"] * 25,
    "crop": ["Wheat", "Corn", "Rice", "Soybean"] * 25,
    "season": ["Summer", "Winter", "Spring", "Autumn"] * 25,
    "avg_temp_c": np.random.normal(25, 5, 100),
    "avg_rainfall_mm": np.random.gamma(2, 10, 100),
    "soil_moisture_pct": np.random.uniform(10, 50, 100),
    "ndvi": np.random.uniform(0.2, 0.9, 100),
    "soil_ph": np.random.uniform(5.0, 8.0, 100),
    "pest_pressure_idx": np.random.uniform(0, 1, 100),
    "fertilizer_kg_per_ha": np.random.randint(50, 200, 100),
    "yield_t_per_ha": np.random.uniform(2.0, 8.0, 100)
}

# Ensure all required columns are present and in the correct order
required_columns = [
    "region", "crop", "season", "avg_temp_c", "avg_rainfall_mm",
    "soil_moisture_pct", "ndvi", "soil_ph", "pest_pressure_idx",
    "fertilizer_kg_per_ha", "yield_t_per_ha"
]

# Create DataFrame with specified column order
df = pd.DataFrame({col: crop_data[col] for col in required_columns})

# Save to CSV
output_path = data_dir / "sample_crop_data.csv"
df.to_csv(output_path, index=False)
print(f"Sample data created at {output_path}")

# Print the first few rows to verify
print("\nFirst 3 rows of the generated data:")
print(df.head(3))