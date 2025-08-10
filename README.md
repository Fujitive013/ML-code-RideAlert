# RideAlert: Enhanced GPS Correction ML Pipeline

A machine learning pipeline for high-precision GPS correction using smartphone GNSS and IMU sensor data, targeting sub-10 meter accuracy for location-based applications.

## 🎯 Project Overview

This project implements an advanced GPS correction system that uses machine learning to improve smartphone GPS accuracy from typical 5-20 meter precision to sub-10 meter accuracy. The system combines GNSS (Global Navigation Satellite System) signal data with IMU (Inertial Measurement Unit) sensor readings to predict and apply intelligent corrections to raw GPS coordinates.

### Key Features
- **Enhanced Signal Quality Filtering**: Filters for strong GNSS signals (Cn0DbHz > 40) and high satellite elevation (> 20°)
- **Multi-Sensor Fusion**: Combines GNSS satellite data with smartphone IMU measurements
- **Ensemble Machine Learning**: Uses Gradient Boosting + Random Forest ensemble for robust predictions
- **Real-time Applicability**: Designed for deployment in mobile applications
- **Sub-10m Accuracy Target**: Achieves mean correction accuracy under 10 meters

## 📊 Dataset

The project uses the **Smartphone Decimeter Challenge 2023** dataset, which includes:
- **GNSS Data**: Satellite signal measurements, positions, signal strength
- **IMU Data**: Accelerometer, gyroscope, and magnetometer readings
- **Ground Truth**: High-precision reference positions for training and validation

### Data Structure
```
smartphone-decimeter-2023/sdc2023/
├── train/                     # Training data
│   └── [date-location]/       # Drive sessions
│       └── [phone-model]/     # Phone-specific data
│           ├── device_gnss.csv
│           ├── device_imu.csv
│           └── ground_truth.csv
└── test/                      # Test data (similar structure)
```

## 🔬 Technical Approach

### Feature Engineering
**GNSS Features:**
- `Cn0DbHz` - Signal strength (Signal-to-Noise Ratio)
- `SvElevationDegrees` - Satellite elevation angle
- `SvAzimuthDegrees` - Satellite azimuth angle
- `Svid` - Satellite ID
- `WlsPosition[XYZ]EcefMeters` - Weighted Least Squares ECEF coordinates

**IMU Features:**
- `MeasurementX/Y/Z` - Raw sensor measurements (acceleration, angular velocity)
- `BiasX/Y/Z` - Sensor bias corrections
- `IMU_MessageType` - Sensor type (UncalAccel, UncalGyro, UncalMag)

**Derived Features:**
- `SignalQuality` = Cn0DbHz × sin(SvElevationDegrees)
- `WLS_Distance` = √(X² + Y² + Z²)

### Machine Learning Models

**Ensemble Architecture:**
- **Gradient Boosting Regressor**: Primary model for pattern learning
- **Random Forest Regressor**: Secondary model for ensemble diversity
- **Weighted Ensemble**: 60% Gradient Boosting + 40% Random Forest

**Model Configuration:**
```python
# Gradient Boosting
n_estimators=200, max_depth=8, learning_rate=0.05

# Random Forest  
n_estimators=200, max_depth=20, min_samples_split=30
```

### Target Prediction
The models predict **correction offsets** (LatCorrection, LngCorrection) that are added to the raw WLS coordinates:
```
corrected_lat = wls_lat + predicted_lat_correction
corrected_lng = wls_lng + predicted_lng_correction
```

## 📁 Project Files

### Main Notebooks
- **`v6 working.ipynb`** - Main enhanced GPS correction pipeline (v6)
- **`gps_error_correction_model_comparison.ipynb`** - Model comparison and evaluation
- **`gps_rf_train_all_combined.ipynb`** - Combined training with Random Forest

### Code Files
- **`gps_hardware_adapter.py`** - Hardware interface adapter for real-time deployment

### Documentation
- **`GPS_Accuracy_Improvements.md`** - Detailed accuracy analysis and improvements

## 🚀 Usage

### Training the Model
1. **Data Preparation**: Load and merge GNSS, IMU, and ground truth data
2. **Feature Engineering**: Create derived features and apply quality filtering
3. **Model Training**: Train Gradient Boosting and Random Forest models
4. **Ensemble Creation**: Combine models with weighted averaging

### Making Predictions
```python
# Load trained models
gb_model = joblib.load('gradient_boosting_model_v6.pkl')
rf_model = joblib.load('random_forest_model_v6.pkl')
scaler = joblib.load('robust_scaler_v6.pkl')

# Prepare features and predict corrections
X_scaled = scaler.transform(features)
gb_corrections = gb_model.predict(X_scaled)
rf_corrections = rf_model.predict(X_scaled)
ensemble_corrections = 0.6 * gb_corrections + 0.4 * rf_corrections

# Apply corrections
corrected_lat = raw_lat + ensemble_corrections[:, 0]
corrected_lng = raw_lng + ensemble_corrections[:, 1]
```

## 📈 Performance Results

### Model Performance (v6)
- **Mean Correction Accuracy**: < 10 meters (target achieved)
- **95th Percentile**: < 15 meters
- **Percentage under 10m**: > 80%
- **Percentage under 5m**: > 60%

### Quality Improvements
- **Before Correction**: 5-20m typical smartphone GPS accuracy
- **After Correction**: Sub-10m mean accuracy with ML enhancement
- **Use Cases**: Navigation, location services, mapping applications

## 🛠️ Requirements

### Python Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.0.0
folium>=0.12.0
matplotlib>=3.4.0
seaborn>=0.11.0
geopy>=2.2.0
pyproj>=3.2.0
```

### Hardware Requirements
- **Training**: 8GB+ RAM recommended for full dataset processing
- **Inference**: Minimal requirements, suitable for mobile deployment

## 🔄 Model Versions

- **v6 (Current)**: Enhanced ensemble with improved signal quality filtering
- **v5**: Random Forest with feature selection
- **v4**: Basic Gradient Boosting implementation

## 📊 Visualization Features

The pipeline includes comprehensive visualization tools:
- **Interactive Maps**: Folium-based route comparison (raw vs corrected)
- **Performance Charts**: Correction distance distributions and accuracy metrics
- **Route Analysis**: Before/after GPS track overlays with correction vectors

## 🎯 Applications

### Target Use Cases
- **Autonomous Vehicles**: Enhanced positioning for self-driving cars
- **Delivery Services**: Precise location for package delivery
- **Emergency Services**: Accurate location for first responders
- **Mapping Applications**: Improved GPS tracks for crowdsourced mapping
- **IoT Devices**: Better positioning for location-aware devices

### Deployment Considerations
- **Real-time Processing**: Optimized for mobile/edge deployment
- **Low Latency**: Sub-second correction prediction
- **Battery Efficiency**: Minimal computational overhead
- **Cross-platform**: Compatible with Android/iOS sensor APIs

## 🔬 Research Contributions

This project demonstrates:
1. **Multi-sensor Fusion**: Effective combination of GNSS and IMU data
2. **Quality-based Filtering**: Signal strength and elevation filtering for accuracy
3. **Ensemble Learning**: Improved robustness through model combination
4. **Real-world Applicability**: Practical deployment considerations

## 📝 License

This project is developed for academic and research purposes. Please cite appropriately if used in academic work.

## 👥 Contributors

- **Axel Paredes** - Lead Developer and ML Engineer

## 🔗 Related Work

Based on the Smartphone Decimeter Challenge 2023 dataset and methodologies for precise GNSS positioning using consumer-grade smartphone sensors.

---

*For detailed implementation, see the notebook files. For questions or contributions, please open an issue or submit a pull request.*
