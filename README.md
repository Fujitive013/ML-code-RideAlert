# RideAlert: Enhanced GPS Correction ML Pipeline (v6)

This project implements a machine learning pipeline for high-precision GPS correction using smartphone GNSS and IMU sensor data, targeting sub-10 meter accuracy for location-based applications. The latest version (v6) introduces advanced feature engineering, strict domain filtering (SpeedMps ≤ 15), and robust ensemble modeling.

## 🎯 Project Overview

This project implements an advanced GPS correction system that uses machine learning to improve smartphone GPS accuracy from typical 5-20 meter precision to sub-10 meter accuracy. The system combines GNSS (Global Navigation Satellite System) signal data with IMU (Inertial Measurement Unit) sensor readings to predict and apply intelligent corrections to raw GPS coordinates.

-   **Offset-Based Correction**: The model predicts latitude and longitude correction offsets (`LatCorrection`, `LngCorrection`). These offsets are added to the raw Weighted Least Squares (WLS) coordinates to produce corrected GPS positions:

    -   `corrected_lat = wls_lat + predicted_lat_correction`
    -   `corrected_lng = wls_lng + predicted_lng_correction`

    **Offset Calculation:**

    -   For each sample, the offset is calculated as the difference between the ground truth position and the WLS-estimated position:
        -   `LatCorrection = LatitudeDegrees_gt - WlsLat`
        -   `LngCorrection = LongitudeDegrees_gt - WlsLng`
    -   The model is trained to predict these offsets using both raw and derived features.

    **Academic Notes:**

    -   This approach allows the model to learn systematic errors in the WLS solution and correct them based on sensor and satellite context.
    -   The offset-based method is robust to domain shifts and can generalize corrections for different environments.

-   **Derived Features Usage**:

    -   **SignalQuality:** Defined as `Cn0DbHz * sin(SvElevationDegrees)`; captures the effective satellite signal strength considering elevation. Used as a feature to help the model distinguish high-quality satellite measurements.
    -   **WLS_Distance:** Calculated as the Euclidean norm of the ECEF coordinates: `sqrt(X^2 + Y^2 + Z^2)`. Used to provide spatial context and detect outliers in position estimates.
    -   Both derived features are included in the model input and shown to improve correction accuracy, especially in noisy or low-quality signal scenarios.

-   **Academic Paper Notes:**
    -   Clearly state the offset-based correction approach and its advantages.
    -   Explain the use and impact of derived features (SignalQuality, WLS_Distance) in improving model robustness.
    -   Emphasize domain filtering (SpeedMps ≤ 15) for deployment relevance.
    -   Discuss how the model learns to correct systematic WLS errors using sensor and satellite context.
    -   Note: All categorical features are label-encoded; missing values are filled with defaults for robustness.
-   **Offset-Based Correction**: The model predicts latitude and longitude correction offsets (`LatCorrection`, `LngCorrection`). These offsets are added to the raw Weighted Least Squares (WLS) coordinates to produce corrected GPS positions:
    -   `corrected_lat = wls_lat + predicted_lat_correction`
    -   `corrected_lng = wls_lng + predicted_lng_correction`
-   **Speed Domain Filtering**: Training data is filtered to SpeedMps ≤ 15 m/s (Philippine city bus domain)
-   **Enhanced Signal Quality Filtering**: Strong GNSS signals (Cn0DbHz > 40), high satellite elevation (> 20°)
-   **Multi-Sensor Fusion**: Combines GNSS satellite data with smartphone IMU measurements
-   **Advanced Feature Engineering**: Derived features (SignalQuality, WLS_Distance) and categorical encodings
-   **Ensemble Machine Learning**: Gradient Boosting (primary) and Random Forest (optional)
-   **Sub-10m Accuracy Target**: Achieves mean correction accuracy under 10 meters

## 📊 Dataset

Uses the **Smartphone Decimeter Challenge 2023** dataset:

-   **GNSS Data**: Satellite signal measurements, positions, signal strength
-   **IMU Data**: Accelerometer, gyroscope, and magnetometer readings
-   **Ground Truth**: High-precision reference positions for training and validation

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

-   `Cn0DbHz` - Signal strength (SNR)
-   `SvElevationDegrees` - Satellite elevation angle
-   `SvAzimuthDegrees` - Satellite azimuth angle
-   `Svid` - Satellite ID
-   `WlsPosition[XYZ]EcefMeters` - Weighted Least Squares ECEF coordinates

**IMU Features:**

-   `IMU_MessageType` - Sensor type (UncalAccel, UncalGyro, UncalMag)
-   `MeasurementX/Y/Z` - Raw sensor measurements
-   `BiasX/Y/Z` - Sensor bias corrections
-   `SpeedMps` - Speed in meters per second (used for domain filtering)

**Derived Features:**

-   `SignalQuality` = Cn0DbHz × sin(SvElevationDegrees)
-   `WLS_Distance` = √(X² + Y² + Z²)

### Machine Learning Models

**Single Model (v6):**

-   Gradient Boosting Regressor (primary)
-   RobustScaler for feature scaling

**Optional Ensemble:**

-   Random Forest Regressor (secondary)
-   Weighted averaging (60% GB, 40% RF)

**Model Configuration:**

```python
# Gradient Boosting
n_estimators=200, max_depth=8, learning_rate=0.05
# Random Forest
n_estimators=200, max_depth=20, min_samples_split=30
```

### Target Prediction

Models predict **correction offsets** (LatCorrection, LngCorrection) added to raw WLS coordinates:

```
corrected_lat = wls_lat + predicted_lat_correction
corrected_lng = wls_lng + predicted_lng_correction
```

## 📁 Project Files

### Main Notebooks

-   **`v6 working.ipynb`** - Main enhanced GPS correction pipeline (v6)
-   **`gps_error_correction_model_comparison.ipynb`** - Model comparison and evaluation
-   **`gps_rf_train_all_combined.ipynb`** - Combined training with Random Forest

### Code Files

-   **`gps_hardware_adapter.py`** - Hardware interface adapter for real-time deployment

### Documentation

-   **`GPS_Accuracy_Improvements.md`** - Detailed accuracy analysis and improvements

## 🚀 Usage

### Training the Model

1. **Data Preparation**: Load and merge GNSS, IMU, and ground truth data
2. **Feature Engineering**: Create derived features, label encode categorical columns
3. **Domain Filtering**: Filter training data to SpeedMps ≤ 15 m/s (city bus domain)
4. **Quality Filtering**: Apply GNSS signal and satellite elevation filters
5. **Model Training**: Train Gradient Boosting model (optionally ensemble with Random Forest)

### Making Predictions

```python
# Load trained model and scaler
gb_model = joblib.load('gradient_boosting_model_v6.pkl')
scaler = joblib.load('robust_scaler_v6.pkl')
features = joblib.load('enhanced_features_v6.pkl')
encoders = joblib.load('enhanced_label_encoders_v6.pkl')

# Prepare features and predict corrections
X_scaled = scaler.transform(features)
gb_corrections = gb_model.predict(X_scaled)

# Apply corrections
corrected_lat = raw_lat + gb_corrections[:, 0]
corrected_lng = raw_lng + gb_corrections[:, 1]
```

## 📈 Performance Results

### Model Performance (v6)

-   **Mean Correction Accuracy**: < 10 meters (target achieved)
-   **95th Percentile**: < 15 meters
-   **Percentage under 10m**: > 80%
-   **Percentage under 5m**: > 60%

### Quality Improvements

-   **Before Correction**: 5-20m typical smartphone GPS accuracy
-   **After Correction**: Sub-10m mean accuracy with ML enhancement
-   **Use Cases**: Navigation, location services, mapping applications

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

-   **Training**: 8GB+ RAM recommended for full dataset processing
-   **Inference**: Minimal requirements, suitable for mobile deployment

## 🔄 Model Versions

-   **v6 (Current)**: Single Gradient Boosting model, strict SpeedMps filtering, enhanced feature engineering
-   **v5**: Random Forest with feature selection
-   **v4**: Basic Gradient Boosting implementation

## 📊 Visualization Features

Pipeline includes:

-   **Interactive Maps**: Folium-based route comparison (raw vs corrected)
-   **Performance Charts**: Correction distance distributions and accuracy metrics
-   **Route Analysis**: Before/after GPS track overlays with correction vectors

## 🎯 Applications

### Target Use Cases

-   **Public Transport/Bus Tracking**: Optimized for city bus domain (SpeedMps ≤ 15)
-   **Autonomous Vehicles**: Enhanced positioning for self-driving cars
-   **Delivery Services**: Precise location for package delivery
-   **Emergency Services**: Accurate location for first responders
-   **Mapping Applications**: Improved GPS tracks for crowdsourced mapping
-   **IoT Devices**: Better positioning for location-aware devices

### Deployment Considerations

-   **Real-time Processing**: Optimized for mobile/edge deployment
-   **Low Latency**: Sub-second correction prediction
-   **Battery Efficiency**: Minimal computational overhead
-   **Cross-platform**: Compatible with Android/iOS sensor APIs

## 🔬 Research Contributions

This project demonstrates:

1. **Domain Filtering**: Strict SpeedMps filtering for deployment relevance
2. **Multi-sensor Fusion**: Effective combination of GNSS and IMU data
3. **Quality-based Filtering**: Signal strength and elevation filtering for accuracy
4. **Ensemble Learning**: Improved robustness through model combination
5. **Real-world Applicability**: Practical deployment considerations

## 📝 License

This project is developed for academic and research purposes. Please cite appropriately if used in academic work.

## 👥 Contributors

-   **Axel Paredes** - Lead Developer and ML Engineer

## 🔗 Related Work

Based on the Smartphone Decimeter Challenge 2023 dataset and methodologies for precise GNSS positioning using consumer-grade smartphone sensors.

---

_For detailed implementation, see the notebook files. For questions or contributions, please open an issue or submit a pull request._
