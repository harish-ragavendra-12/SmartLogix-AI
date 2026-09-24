# ============================================================
# SMARTLOGIX AI - TRANSPORTATION MODE PREDICTION
# ============================================================

import joblib
import pandas as pd

from src.config.config import MODELS_DIR


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_transportation_model():
    print("\n" + "=" * 60)
    print("LOADING TRANSPORTATION MODEL")
    print("=" * 60)

    model_path = (
        MODELS_DIR
        / "transportation"
        / "transportation_classifier.joblib"
    )

    print(f"\nLoading model from:")
    print(model_path)

    model = joblib.load(model_path)

    print("\nTransportation model loaded successfully.")

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
        "package_weight_kg": 3.5,
        "package_volume_cm3": 9000,
        "delivery_distance_km": 350,
        "priority_score": 2,
        "temp_celsius": 28.5,
        "humidity_%": 68,
        "Precipitation (mm)": 0.0,
        "wind_speed_kmph": 12.5,
        "Visibility_KM": 9.5,
        "traffic_speed_kmph": 32.5,
        "traffic_level": 0.03,
        "weight_category": "Light",
        "distance_category": "Medium",
        "high_precipitation": 0,
        "high_wind": 0,
        "low_visibility": 0
    }

    order_df = pd.DataFrame([sample_order])

    print("\nSample order created.")
    print(f"Feature shape: {order_df.shape}")

    return order_df


# ============================================================
# VALIDATE INPUT FEATURES
# ============================================================

def validate_input_features(order_df, model):
    print("\n" + "=" * 60)
    print("VALIDATING INPUT FEATURES")
    print("=" * 60)

    expected_features = (
        model.named_steps["preprocessor"]
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
            print(f"- {feature}")

        raise ValueError(
            "Input data is missing required model features."
        )

    if extra_features:
        print("\nExtra features detected:")
        for feature in extra_features:
            print(f"- {feature}")

        order_df = order_df.drop(
            columns=extra_features
        )

    order_df = order_df[expected_features]

    print("\nInput validation successful.")
    print(f"Final input shape: {order_df.shape}")

    return order_df


# ============================================================
# PREDICT TRANSPORTATION MODE
# ============================================================

def predict_transport_mode(model, order_df):
    print("\n" + "=" * 60)
    print("PREDICTING TRANSPORTATION MODE")
    print("=" * 60)

    prediction = model.predict(order_df)

    predicted_mode = prediction[0]

    print(f"\nPredicted Transportation Mode:")
    print(f">>> {predicted_mode}")

    return predicted_mode


# ============================================================
# GET PREDICTION PROBABILITIES
# ============================================================

def get_prediction_probabilities(model, order_df):
    print("\n" + "=" * 60)
    print("PREDICTION PROBABILITIES")
    print("=" * 60)

    probabilities = model.predict_proba(order_df)[0]

    classes = model.named_steps["model"].classes_

    probability_df = pd.DataFrame(
        {
            "transport_mode": classes,
            "probability": probabilities
        }
    )

    probability_df = probability_df.sort_values(
        by="probability",
        ascending=False
    ).reset_index(drop=True)

    probability_df["probability_percent"] = (
        probability_df["probability"] * 100
    )

    print("\nPrediction probabilities:")

    for _, row in probability_df.iterrows():
        print(
            f"{row['transport_mode']:<12} "
            f"{row['probability_percent']:.2f}%"
        )

    return probability_df


# ============================================================
# DISPLAY FINAL PREDICTION
# ============================================================

def display_prediction_summary(
    predicted_mode,
    probability_df
):
    print("\n" + "=" * 60)
    print("TRANSPORTATION PREDICTION SUMMARY")
    print("=" * 60)

    top_probability = probability_df.iloc[0]["probability"]

    print(
        f"\nPredicted Mode : {predicted_mode}"
    )

    print(
        f"Confidence     : "
        f"{top_probability * 100:.2f}%"
    )

    print("\nRanked transportation modes:")

    for index, row in probability_df.iterrows():
        rank = index + 1

        print(
            f"{rank}. "
            f"{row['transport_mode']:<12} "
            f"{row['probability_percent']:.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("SMARTLOGIX AI - TRANSPORTATION MODE PREDICTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Load trained model
    # --------------------------------------------------------

    model = load_transportation_model()

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
    # Step 4: Predict transportation mode
    # --------------------------------------------------------

    predicted_mode = predict_transport_mode(
        model,
        order_df
    )

    # --------------------------------------------------------
    # Step 5: Get probabilities
    # --------------------------------------------------------

    probability_df = get_prediction_probabilities(
        model,
        order_df
    )

    # --------------------------------------------------------
    # Step 6: Display summary
    # --------------------------------------------------------

    display_prediction_summary(
        predicted_mode,
        probability_df
    )

    print("\n" + "=" * 60)
    print(
        "TRANSPORTATION PREDICTION "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()