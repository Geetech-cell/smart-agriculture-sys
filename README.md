# 🌱 Smart Agriculture System

## 📊 Model Performance
- **R² Score:** 0.972
- **Mean Absolute Error (MAE):** 0.298
- **Prediction Latency:** < 100ms

## 🎯 Project Overview
A machine learning-powered agricultural intelligence platform that helps farmers and agronomists make data-driven decisions for crop management and yield optimization.

## 🌟 Key Features

### 📈 Crop Yield Prediction
- **Smart Input Organization** - Organized expander sections for Location & Weather, Crop & Soil Conditions, and Field Management
- **Enhanced Result Display** - Beautiful gradient yield card with confidence indicators
- **Limiting Factors Analysis** - Visual bar chart identifying which factors are limiting yield potential
- **AI-Powered Predictions** - Accurate crop yield forecasting using ensemble models

### 🏥 Crop Health Analyzer
- **Growth Stage Tracking** - Stage-specific recommendations for Seedling, Vegetative, Reproductive, and Maturity phases
- **Detailed Diagnosis** - AI-powered health analysis with issue detection and remediation advice
- **Risk Assessment Matrix** - Visual heatmap showing Pest, Disease, and Climate risks
- **Health Metrics Dashboard** - Soil Quality, Pest Risk, Water Balance, and overall Health Score
- **Smart Irrigation Recommendations** - Context-aware irrigation advice based on soil moisture, weather, and temperature

### 📊 Data Explorer
- **Interactive Visualizations** - Correlation heatmaps, scatter plots, 3D charts, violin plots, and more
- **Pivot Table Analysis** - Dynamic data aggregation and analysis
- **Outlier Detection** - IQR and Z-Score methods for identifying anomalies
- **Target Correlation** - Identify features most correlated with crop yield
- **Advanced Filtering** - Multi-dimensional data filtering with reset capabilities
- **Data Quality Reports** - Comprehensive analysis of missing values, duplicates, and data integrity

### 🔔 Real-time Monitoring & Alerts
- Track key metrics and receive intelligent alerts
- Historical analysis tracking with save functionality
- Data-driven insights for better farming decisions

## 🛠️ Technical Stack
- **Backend:** Python 3.10+
- **Machine Learning:** scikit-learn, XGBoost, LightGBM
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Web Framework:** Streamlit
- **Data Processing:** pandas, NumPy

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smart-agriculture-system.git
   cd smart-agriculture-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\\venv\\Scripts\\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```
   The application will be available at `http://localhost:8501`

## 📁 Project Structure
```
smart-agriculture-system/
├── app.py                  # Main Streamlit application
├── train.py                # Model training script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── REPORT.md              # Detailed project report
├── data/                  # Sample data files
│   └── sample_crop_data.csv
├── src/                   # Source code
│   ├── __init__.py
│   ├── model.py           # ML model implementation
│   ├── predict_service.py # Prediction service
│   ├── data_pipeline.py   # Data processing utilities
│   ├── alert_store.py     # Alert management
│   ├── ui_components.py   # UI components and themes
│   ├── crop_health.py     # Crop Health Analyzer module
│   ├── mobile_ui.py       # Mobile UI components
│   └── mobile_integration.py # Mobile features
├── artifacts/             # Model artifacts
│   └── metadata.json
└── tests/                 # Test files
```

## 🎮 Using the Dashboard

### 1. Launch the Application
Open your browser and navigate to `http://localhost:8501`

### 2. Configure the Model
In the sidebar, ensure the metadata path points to `artifacts/metadata.json`

### 3. Explore Features

#### 🌱 Crop Prediction Tab
1. **Set Location & Weather** - Choose your region and adjust environmental conditions
2. **Configure Crop & Soil** - Select crop type and set soil parameters
3. **Manage Field Inputs** - Set fertilizer and pest pressure levels
4. **Predict Yield** - Click the button to see a beautiful yield card with limiting factors analysis

#### 🏥 Crop Health Tab
1. **Select Growth Stage** - Choose current crop development phase
2. **View Metrics** - Review Soil Quality, Pest Risk, Water Balance, and Health Score
3. **Read Diagnosis** - Get stage-specific recommendations and issue detection
4. **Check Risk Matrix** - Visual assessment of Pest, Disease, and Climate risks
5. **Get Irrigation Advice** - Context-aware watering recommendations
6. **Save Analysis** - Track historical health data

#### 📊 Data Explorer Tab
1. **Load Data** - Use sample data or import your own CSV
2. **View Overview** - See dataset statistics and quality metrics
3. **Create Visualizations** - Choose from 7+ chart types
4. **Analyze Pivot Tables** - Aggregate data by multiple dimensions
5. **Detect Outliers** - Identify anomalies using IQR or Z-Score methods
6. **Export Results** - Download filtered and analyzed data

## 📊 Data Sources
- **Sample Data:** Included sample dataset in `data/sample_crop_data.csv`
- **Custom Data:** Support for importing your own agricultural datasets (CSV format)
- **Weather Data:** Optional integration with Open-Meteo API (no key required)
- **IoT Sensors:** Support for real-time sensor feeds (configurable)

## 🧪 Testing

Run the test suite:
```bash
pytest
```

## 🔄 Model Training

Train a new model with custom data:
```bash
python train.py --config configs/default.json
```

## 🌍 Configuration

### Weather API
The application uses the Open-Meteo API for real-time weather data (no API key required).

### Sensor Feeds
Configure live IoT feeds via `.streamlit/secrets.toml`:
```toml
[sensor_feed]
url = "file://data/sample_sensor_feed.json"
api_key = "YOUR_TOKEN"
timeout = 10
```

### Alert Storage
Configure alert history logging:
```toml
[alert_store]
path = "data/alert_history.jsonl"
webhook_url = "https://your-webhook.com"
```

## 🌟 Recent Updates

### v2.1.0 (November 2024)
- ✨ **Enhanced UI/UX** - Organized input sections with expanders
- 📊 **Limiting Factors Analysis** - Visual identification of yield-limiting factors
- 🏥 **Crop Health Improvements** - Growth stage tracking, detailed diagnosis, risk matrix
- 📈 **Advanced Visualizations** - 3D scatter plots, violin plots, pivot tables
- 🔍 **Outlier Detection** - IQR and Z-Score analysis
- 🎯 **Target Correlation** - Feature importance visualization
- 💾 **Historical Tracking** - Save and review analysis history

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact
For questions or feedback, please open an issue on GitHub.

## 🌱 Ethical & Sustainability Considerations
- **Bias Mitigation:** Regular fairness assessments and transparent reporting
- **Environmental Impact:** Energy-efficient models and carbon-aware training
- **Inclusivity:** Multi-language support and accessibility compliance
- **Privacy:** End-to-end encryption and GDPR compliance

## 🚀 Future Roadmap
- Integration with blockchain for supply chain transparency
- Mobile app with offline capabilities
- AI-powered pest and disease detection from images
- Advanced water management system
- Carbon credit marketplace integration
- Multi-language support expansion

---

*For detailed technical information, see [REPORT.md](REPORT.md)*
