# RideAlert: Enhanced GPS Correction ML Pipeline (v6)

This repository contains a machine learning pipeline for high-precision GPS correction using smartphone GNSS and IMU sensor data. Version v6 focuses on a single high-performing model (Gradient Boosting), strict domain filtering (SpeedMps ≤ 15), and derived features to achieve sub-10m mean accuracy.

## 🎯 Project Overview

-   **Offset-based correction:** Model predicts latitude/longitude offsets (`LatCorrection`, `LngCorrection`) added to raw WLS coordinates:
    -   `corrected_lat = wls_lat + predicted_lat_correction`
    -   `corrected_lng = wls_lng + predicted_lng_correction`
-   **Derived features:**
    -   `SignalQuality = Cn0DbHz * sin(SvElevationDegrees)`
    -   `WLS_Distance = sqrt(X^2 + Y^2 + Z^2)`
-   **Filtering:** Cn0DbHz > 40, SvElevationDegrees > 20°, and SpeedMps ≤ 15 m/s
-   **Model:** GradientBoostingRegressor wrapped in `MultiOutputRegressor` with `RobustScaler`
-   **Target:** Sub-10m mean correction error on validation/test routes

## 📦 Repository Structure

```
Capstone-RideAlert/
├─ v6 working.ipynb
├─ smartphone-decimeter-2023/
│  └─ sdc2023/
│     ├─ train/ ...
│     └─ test/  ...
│
```

## 📊 Dataset

Uses the Smartphone Decimeter Challenge 2023 dataset (GNSS, IMU, and ground-truth). Place the extracted folder at `smartphone-decimeter-2023/sdc2023/` as shown above.

### Downloading the Dataset

-   Download archive: https://drive.google.com/file/d/1rO3S6I3g2P5q4xm-P8GG9OQYpK0jxnPQ/view?usp=sharing
-   Extract so that the structure matches:
    -   `smartphone-decimeter-2023/sdc2023/train/...`
    -   `smartphone-decimeter-2023/sdc2023/test/...`

## 🔬 Technical Approach

-   **GNSS features:** `Cn0DbHz`, `SvElevationDegrees`, `SvAzimuthDegrees`, `Svid`, `WlsPosition[XYZ]EcefMeters`
-   **IMU features:** `IMU_MessageType`, `MeasurementX/Y/Z`, `BiasX/Y/Z`, `SpeedMps`
-   **Derived:** `SignalQuality`, `WLS_Distance`
-   **Scaling:** `RobustScaler`
-   **Model:** `GradientBoostingRegressor` (n_estimators=200, max_depth=8, learning_rate=0.05, subsample=0.8, min_samples_split=50, min_samples_leaf=20, max_features='sqrt') wrapped in `MultiOutputRegressor`
-   **Split:** Stratified by correction magnitude bins on validation

## 🚀 How to Run (v6)

1. Prepare data

-   Download and extract the dataset into `smartphone-decimeter-2023/sdc2023/`.

2. Install dependencies (example)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy scikit-learn joblib folium matplotlib seaborn geopy pyproj
```

3. Train and evaluate

-   Open `v6 working.ipynb` and run all cells (1–12):
    -   Cells 1–7: Data discovery, filtering, feature prep, split
    -   Cells 8–9b: Train Gradient Boosting, evaluate, and save artifacts
    -   Cells 10–12: Test on a route from `test/`, generate figures and map

## 🧾 Artifacts and Outputs

Generated during `v6 working.ipynb` execution:

-   Models and helpers (repo root):
    -   `gradient_boosting_model_v6.pkl`
    -   `robust_scaler_v6.pkl`
    -   `enhanced_label_encoders_v6.pkl`
    -   `enhanced_features_v6.pkl`
    -   `best_model_v6_config.json`
-   Evaluation/visuals (date-stamped):
    -   `single_model_gps_correction_v6_<date>.png`
    -   `single_model_gps_summary_v6_<date>.json`
    -   `single_model_gps_map_v6_<date>.html`
-   Predictions:
    -   `predictions/test_predictions_v6.csv`

## 📈 Expected Performance (v6)

-   Mean correction: < 10 m (target)
-   95th percentile: < 15 m
-   % under 10 m: > 80%
-   % under 5 m: > 60%

## 🛠️ Requirements

Python packages:

```text
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

Hardware:

-   Training: 8GB+ RAM recommended
-   Inference: lightweight

## 🔬 Research Notes

-   Offset-learning corrects systematic WLS errors using sensor/satellite context
-   Derived features (`SignalQuality`, `WLS_Distance`) improve robustness
-   Domain filtering (SpeedMps ≤ 15) aligns with deployment scenario

## 📝 License

Academic and research use. Please cite appropriately.

## 👥 Contributor

-   Axel Paredes — Lead Developer and ML Engineer

---

For details, see `v6 working.ipynb`. Questions or ideas? Open an issue or PR.
