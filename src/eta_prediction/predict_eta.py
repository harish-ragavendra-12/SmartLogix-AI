# ============================================================
# SMARTLOGIX AI - ETA PREDICTION
# ============================================================

import joblib
import pandas as pd

from src.config.config import MODELS_DIR


# ============================================================
# LOAD ETA MODEL
# ============================================================

def load_eta_model():

    print("\n" + "=" * 60)
    print("LOADING ETA MODEL")
    print("=" * 60)

    model_path = (
        MODELS_DIR
        / "eta"
        / "eta_regressor.joblib"
    )

    print("\nLoading model from:")
    print(model_path)

    model = joblib.load(
        model_path
    )

    print("\nETA model loaded successfully.")

    return model


# ============================================================
# CREATE SAMPLE ORDER
# ============================================================

def create_sample_order():

    print("\n" + "=" * 60)
    print("CREATING SAMPLE ORDER")
    print("=" * 60)

    sample_order = {

        "quantity": 2,

        "origin_hub": "HUB-001",

        "origin_city": "Chennai",

        "destination_city": "Bengaluru",

        "destination_state": "Karnataka",

        "destination_lat": 12.9716,

        "destination_lon": 77.5946,

        "is_fragile": 0,

        "is_hazmat": 0,

        "cold_chain_required": 0,

        "delivery_priority": "standard",

        "payment_mode": "COD",

        "order_value_inr": 2500,

        "weather_condition_at_dest": "Clear",

        "dimension_length_cm": 30,

        "dimension_width_cm": 20,

        "dimension_height_cm": 15,

        "order_year": 2026,

        "order_month": 9,

        "order_day": 21,

        "order_day_of_week": 0,

        "order_week": 39,

        "is_weekend": 0,

        "package_volume_cm3": 9000,

        "package_weight_kg": 3.5,

        "delivery_distance_km": 350,

        "distance_category": "Medium",

        "priority_score": 2,

        "temp_celsius": 28.5,

        "humidity_%": 68,

        "Precipitation (mm)": 0.0,

        "wind_speed_kmph": 12.5,

        "Visibility_KM": 9.5,

        "condition": "Clear",

        "traffic_speed_kmph": 32.5,

        "traffic_level": 10.2,

        "high_precipitation": 0,

        "high_wind": 0,

        "low_visibility": 0,

        "transport_mode": "Truck"
    }

    order_df = pd.DataFrame(
        [sample_order]
    )

    print("\nSample order created.")

    print(
        f"Feature shape: "
        f"{order_df.shape}"
    )

    return order_df


# ============================================================
# VALIDATE INPUT FEATURES
# ============================================================

def validate_input_features(
    order_df,
    model
):

    print("\n" + "=" * 60)
    print("VALIDATING INPUT FEATURES")
    print("=" * 60)

    expected_features = (
        model
        .named_steps[
            "preprocessor"
        ]
        .feature_names_in_
    )

    missing_features = [
        feature
        for feature in expected_features
        if feature not in order_df.columns
    ]

    extra_features = [
        feature
        for feature in order_df.columns
        if feature not in expected_features
    ]

    if missing_features:

        print("\nMissing features:")

        for feature in missing_features:
            print(
                f"- {feature}"
            )

        raise ValueError(
            "Input data is missing "
            "required ETA model features."
        )

    if extra_features:

        print("\nExtra features detected:")

        for feature in extra_features:
            print(
                f"- {feature}"
            )

        order_df = order_df.drop(
            columns=extra_features
        )

    order_df = order_df[
        expected_features
    ]

    print(
        "\nInput validation successful."
    )

    print(
        f"Final input shape: "
        f"{order_df.shape}"
    )

    return order_df


# ============================================================
# PREDICT ETA
# ============================================================

def predict_eta(
    model,
    order_df
):

    print("\n" + "=" * 60)
    print("PREDICTING ETA")
    print("=" * 60)

    prediction = model.predict(
        order_df
    )

    predicted_hours = float(
        prediction[0]
    )

    # Prevent negative ETA
    predicted_hours = max(
        predicted_hours,
        0
    )

    print(
        f"\nPredicted ETA: "
        f"{predicted_hours:.2f} hours"
    )

    return predicted_hours


# ============================================================
# CONVERT HOURS TO HOURS AND MINUTES
# ============================================================

def convert_hours_to_time(
    hours
):

    total_minutes = round(
        hours * 60
    )

    hours_part = (
        total_minutes // 60
    )

    minutes_part = (
        total_minutes % 60
    )

    return (
        hours_part,
        minutes_part
    )


# ============================================================
# DISPLAY ETA SUMMARY
# ============================================================

def display_eta_summary(
    predicted_hours
):

    print("\n" + "=" * 60)
    print("ETA PREDICTION SUMMARY")
    print("=" * 60)

    (
        hours_part,
        minutes_part
    ) = convert_hours_to_time(
        predicted_hours
    )

    print(
        f"\nPredicted ETA      : "
        f"{predicted_hours:.2f} hours"
    )

    print(
        f"Approximate Time  : "
        f"{hours_part} hours "
        f"{minutes_part} minutes"
    )

    if predicted_hours <= 2:

        eta_category = (
            "Very Short Delivery"
        )

    elif predicted_hours <= 6:

        eta_category = (
            "Short Delivery"
        )

    elif predicted_hours <= 24:

        eta_category = (
            "Standard Delivery"
        )

    else:

        eta_category = (
            "Long Delivery"
        )

    print(
        f"ETA Category       : "
        f"{eta_category}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print(
        "SMARTLOGIX AI - ETA PREDICTION"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Load model
    # --------------------------------------------------------

    model = load_eta_model()

    # --------------------------------------------------------
    # Step 2: Create sample order
    # --------------------------------------------------------

    order_df = create_sample_order()

    # --------------------------------------------------------
    # Step 3: Validate input
    # --------------------------------------------------------

    order_df = validate_input_features(
        order_df,
        model
    )

    # --------------------------------------------------------
    # Step 4: Predict ETA
    # --------------------------------------------------------

    predicted_hours = predict_eta(
        model,
        order_df
    )

    # --------------------------------------------------------
    # Step 5: Display summary
    # --------------------------------------------------------

    display_eta_summary(
        predicted_hours
    )

    print("\n" + "=" * 60)
    print(
        "ETA PREDICTION "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()