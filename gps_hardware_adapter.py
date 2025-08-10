#!/usr/bin/env python3
"""
GPS Hardware Adapter for Enhanced GPS Correction Model v6

This script corrects GPS errors using the trained machine learning model.
The model was specifically trained to handle low-quality GPS signals and improve accuracy.

Purpose: Take your real GPS hardware data and apply ML-based corrections to improve accuracy.
"""

import numpy as np
import pandas as pd
import joblib
from pyproj import Transformer
from geopy.distance import geodesic


def lla_to_ecef(lat_deg, lon_deg, alt_m):
    """Convert GPS LLA to ECEF coordinates (WLS format expected by model)"""
    # WGS84 constants
    a = 6378137.0          # Semi-major axis (meters)
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f * f     # First eccentricity squared

    # Convert degrees to radians
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    # Calculate prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

    # Convert to ECEF (this becomes WlsPositionXEcefMeters, etc.)
    X = (N + alt_m) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + alt_m) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (N * (1 - e2) + alt_m) * np.sin(lat_rad)

    return X, Y, Z


def prepare_satellite_features(gps_data, include_low_quality=True):
    """
    Convert GPS hardware data to model-ready features

    Args:
        gps_data: Dict with latitude, longitude, altitude, satellites list
        include_low_quality: True to include all satellites (recommended for correction)

    Returns:
        List of feature dictionaries, one per satellite
    """

    # Convert GPS position to ECEF (WLS coordinates)
    ecef_x, ecef_y, ecef_z = lla_to_ecef(
        gps_data['latitude'],
        gps_data['longitude'],
        gps_data['altitude']
    )

    satellite_features = []
    high_quality_count = 0

    for sat in gps_data['satellites']:
        # Check quality but don't filter (model handles low quality)
        is_high_quality = sat['snr'] > 40 and sat['elevation'] > 20
        if is_high_quality:
            high_quality_count += 1

        # Create features for ALL satellites (let model decide what to do)
        features = {
            'Cn0DbHz': float(sat['snr']),
            'SvElevationDegrees': float(sat['elevation']),
            'SvAzimuthDegrees': float(sat['azimuth']),
            'Svid': int(sat['prn']),
            'WlsPositionXEcefMeters': ecef_x,  # Base GPS position to be corrected
            'WlsPositionYEcefMeters': ecef_y,
            'WlsPositionZEcefMeters': ecef_z
        }

        # Add derived features (same as training)
        features['SignalQuality'] = features['Cn0DbHz'] * \
            np.sin(np.radians(features['SvElevationDegrees']))
        features['WLS_Distance'] = np.sqrt(ecef_x**2 + ecef_y**2 + ecef_z**2)

        satellite_features.append(features)

    print(f"📡 Processing {len(satellite_features)} satellites")
    print(
        f"🎯 High-quality satellites: {high_quality_count}/{len(satellite_features)}")
    print(f"⚡ Model will correct errors from all available satellites")

    return satellite_features


def predict_gps_corrections(satellite_features, model_files_path='./'):
    """
    Apply the trained GPS correction model to improve GPS accuracy

    Returns:
        Average correction (lat_deg, lng_deg) or None if prediction fails
    """

    # Load trained model components
    try:
        gb_model = joblib.load(
            f'{model_files_path}gradient_boosting_model_v6.pkl')
        rf_model = joblib.load(f'{model_files_path}random_forest_model_v6.pkl')
        scaler = joblib.load(f'{model_files_path}robust_scaler_v6.pkl')
        encoders = joblib.load(
            f'{model_files_path}enhanced_label_encoders_v6.pkl')
        features = joblib.load(f'{model_files_path}enhanced_features_v6.pkl')
        print("✅ Loaded trained GPS correction model v6")
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("   Make sure the model has been trained first")
        return None

    if not satellite_features:
        print("❌ No satellite data provided")
        return None

    all_corrections = []

    for i, sat_features in enumerate(satellite_features):
        try:
            # Create feature vector in the exact order expected by model
            X_input = pd.DataFrame([sat_features])[features]

            # Apply categorical encoding (for Svid)
            for col in encoders:
                if col in X_input.columns:
                    le = encoders[col]
                    sat_value = str(X_input[col].iloc[0])

                    # Handle unknown satellites (use first known class)
                    if sat_value in le.classes_:
                        X_input[col] = le.transform([sat_value])
                    else:
                        X_input[col] = le.transform([le.classes_[0]])
                        print(
                            f"⚠️ Unknown satellite {sat_value}, mapped to {le.classes_[0]}")

            # Apply same scaling as training
            X_scaled = scaler.transform(X_input)

            # Make ensemble prediction (60% GB + 40% RF)
            gb_pred = gb_model.predict(X_scaled)
            rf_pred = rf_model.predict(X_scaled)
            ensemble_correction = 0.6 * gb_pred + 0.4 * rf_pred

            # [lat_correction, lng_correction]
            all_corrections.append(ensemble_correction[0])

        except Exception as e:
            print(f"⚠️ Skipping satellite {i+1}: {e}")
            continue

    if not all_corrections:
        print("❌ No valid corrections generated")
        return None

    # Average corrections across all satellites (robust ensemble approach)
    final_correction = np.mean(all_corrections, axis=0)

    print(f"🔧 Generated corrections from {len(all_corrections)} satellites")
    return final_correction


def apply_gps_correction(original_lat, original_lng, correction):
    """Apply the ML-predicted correction to GPS coordinates"""
    corrected_lat = original_lat + correction[0]
    corrected_lng = original_lng + correction[1]
    return corrected_lat, corrected_lng


def correct_gps_position(gps_data, model_path='./'):
    """
    Main function: Take raw GPS data and return ML-corrected position

    Args:
        gps_data: Dict with your GPS hardware output
        model_path: Path to trained model files

    Returns:
        Dict with original and corrected coordinates, plus correction info
    """

    print("🛰️ GPS Error Correction System v6")
    print("=" * 50)
    print(
        f"📍 Original GPS: ({gps_data['latitude']:.6f}, {gps_data['longitude']:.6f})")

    # Prepare satellite features
    satellite_features = prepare_satellite_features(gps_data)

    # Get ML corrections
    correction = predict_gps_corrections(satellite_features, model_path)

    if correction is None:
        print("❌ GPS correction failed")
        return {
            'success': False,
            'original_lat': gps_data['latitude'],
            'original_lng': gps_data['longitude'],
            'corrected_lat': gps_data['latitude'],
            'corrected_lng': gps_data['longitude'],
            'correction_distance_m': 0,
            'correction_applied': False
        }

    # Apply correction
    corrected_lat, corrected_lng = apply_gps_correction(
        gps_data['latitude'],
        gps_data['longitude'],
        correction
    )

    # Calculate correction distance
    correction_distance = geodesic(
        (gps_data['latitude'], gps_data['longitude']),
        (corrected_lat, corrected_lng)
    ).meters

    # Assess correction quality
    if correction_distance < 1:
        quality = "🔍 Minimal correction (< 1m)"
    elif correction_distance < 5:
        quality = "🏆 Excellent correction (< 5m)"
    elif correction_distance < 10:
        quality = "✅ Good correction (< 10m)"
    elif correction_distance < 20:
        quality = "⚡ Significant correction (< 20m)"
    else:
        quality = "🔧 Large correction (> 20m)"

    print(f"🎯 Corrected GPS: ({corrected_lat:.6f}, {corrected_lng:.6f})")
    print(f"📏 Correction: {correction[0]:.8f}° lat, {correction[1]:.8f}° lng")
    print(f"📐 Distance: {correction_distance:.2f} meters")
    print(f"📊 {quality}")

    return {
        'success': True,
        'original_lat': gps_data['latitude'],
        'original_lng': gps_data['longitude'],
        'corrected_lat': corrected_lat,
        'corrected_lng': corrected_lng,
        'correction_lat_deg': correction[0],
        'correction_lng_deg': correction[1],
        'correction_distance_m': correction_distance,
        'correction_applied': True,
        'satellite_count': len(gps_data['satellites']),
        'quality_assessment': quality
    }


# Example usage and testing
if __name__ == "__main__":

    # Your actual GPS hardware data
    real_gps_data = {
        'latitude': 8.468738,
        'longitude': 124.649773,
        'altitude': 42.0,
        'satellites': [
            {'prn': 1, 'elevation': 26, 'azimuth': 23, 'snr': 45},    # High quality
            # Low elevation, weak signal
            {'prn': 2, 'elevation': 4, 'azimuth': 38, 'snr': 27},
            {'prn': 3, 'elevation': 49, 'azimuth': 66, 'snr': 48},    # High quality
            {'prn': 6, 'elevation': 29, 'azimuth': 253, 'snr': 29},   # Weak signal
            {'prn': 7, 'elevation': 34, 'azimuth': 199, 'snr': 38},   # Borderline
            {'prn': 8, 'elevation': 12, 'azimuth': 80,
                'snr': 34},    # Low elevation, weak signal
        ]
    }

    # Apply GPS correction (this is what you'll use in production)
    result = correct_gps_position(real_gps_data)

    if result['success']:
        print("\n✅ GPS Error Correction Complete!")
        print(
            f"   Improvement: {result['correction_distance_m']:.2f}m correction applied")
        print(
            f"   Use corrected coordinates: ({result['corrected_lat']:.6f}, {result['corrected_lng']:.6f})")
    else:
        print("\n❌ GPS correction failed - using original coordinates")
