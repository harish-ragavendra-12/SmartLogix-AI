# ============================================================
# SMARTLOGIX AI - ETA FEATURE ENGINEERING
# ============================================================

import pandas as pd
import numpy as np

from src.config.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR
)


# ============================================================
# LOAD ORDERS DATA
# ============================================================

def load_orders_data():
    print("\n" + "=" * 60)
    print("LOADING ORDERS DATA")
    print("=" * 60)

    file_path = PROCESSED_DATA_DIR / "orders_cleaned.csv"

    print(f"\nLoading orders from:")
    print(file_path)

    df = pd.read_csv(file_path)

    print(f"\nOrders shape: {df.shape}")

    return df


# ============================================================
# LOAD WEATHER DATA
# ============================================================

def load_weather_data():
    print("\n" + "=" * 60)
    print("LOADING WEATHER DATA")
    print("=" * 60)

    file_path = PROCESSED_DATA_DIR / "weather_data_clean.csv"

    print(f"\nLoading weather data from:")
    print(file_path)

    weather_df = pd.read_csv(file_path)

    print(f"\nWeather shape: {weather_df.shape}")

    return weather_df


# ============================================================
# LOAD TRAFFIC DATA
# ============================================================

def load_traffic_data():
    print("\n" + "=" * 60)
    print("LOADING TRAFFIC DATA")
    print("=" * 60)

    file_path = PROCESSED_DATA_DIR / "traffic_data_clean.csv"

    print(f"\nLoading traffic data from:")
    print(file_path)

    traffic_df = pd.read_csv(file_path)

    print(f"\nTraffic shape: {traffic_df.shape}")

    return traffic_df


# ============================================================
# PREPARE DATE FEATURES
# ============================================================

def create_date_features(df):
    print("\n" + "=" * 60)
    print("CREATING DATE FEATURES")
    print("=" * 60)

    df = df.copy()

    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce"
    )

    invalid_dates = df["order_date"].isna().sum()

    print(
        f"\nInvalid order dates: {invalid_dates}"
    )

    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_day"] = df["order_date"].dt.day
    df["order_day_of_week"] = df["order_date"].dt.dayofweek
    df["order_week"] = (
        df["order_date"]
        .dt.isocalendar()
        .week
        .astype("Int64")
    )
    df["is_weekend"] = (
        df["order_day_of_week"] >= 5
    ).astype(int)

    print("\nDate features created:")
    print("- order_year")
    print("- order_month")
    print("- order_day")
    print("- order_day_of_week")
    print("- order_week")
    print("- is_weekend")

    return df


# ============================================================
# CREATE PACKAGE FEATURES
# ============================================================

def create_package_features(df):
    print("\n" + "=" * 60)
    print("CREATING PACKAGE FEATURES")
    print("=" * 60)

    df = df.copy()

    df["package_volume_cm3"] = (
        df["dimension_length_cm"]
        * df["dimension_width_cm"]
        * df["dimension_height_cm"]
    )

    df["package_weight_kg"] = df["package_weight"]

    print("\nPackage features created:")
    print("- package_weight_kg")
    print("- package_volume_cm3")

    return df


# ============================================================
# CREATE DISTANCE FEATURES
# ============================================================

def create_distance_features(df):
    print("\n" + "=" * 60)
    print("CREATING DISTANCE FEATURES")
    print("=" * 60)

    df = df.copy()

    df["delivery_distance_km"] = df["distance_km"]

    df["distance_category"] = pd.cut(
        df["delivery_distance_km"],
        bins=[
            -np.inf,
            50,
            200,
            500,
            1000,
            np.inf
        ],
        labels=[
            "Very Short",
            "Short",
            "Medium",
            "Long",
            "Very Long"
        ]
    )

    print("\nDistance features created:")
    print("- delivery_distance_km")
    print("- distance_category")

    return df


# ============================================================
# CREATE PRIORITY FEATURES
# ============================================================

def create_priority_features(df):
    print("\n" + "=" * 60)
    print("CREATING PRIORITY FEATURES")
    print("=" * 60)

    df = df.copy()

    priority_mapping = {
        "economy": 1,
        "standard": 2,
        "express": 3,
        "same_day": 4,
        "low": 1,
        "medium": 2,
        "normal": 2,
        "high": 3,
        "urgent": 4,
        "critical": 5
    }

    df["priority_score"] = (
        df["delivery_priority"]
        .astype(str)
        .str.lower()
        .map(priority_mapping)
    )

    print("\nPriority score created.")

    return df


# ============================================================
# PREPARE WEATHER DATA
# ============================================================

def prepare_weather_data(weather_df):
    print("\n" + "=" * 60)
    print("PREPARING WEATHER DATA")
    print("=" * 60)

    weather_df = weather_df.copy()

    weather_df["Date"] = pd.to_datetime(
        weather_df["Date"],
        errors="coerce"
    )

    weather_df = weather_df.rename(
        columns={
            "Date": "weather_date",
            "City Name": "weather_city"
        }
    )

    required_columns = [
        "weather_date",
        "weather_city",
        "temp_celsius",
        "humidity_%",
        "Precipitation (mm)",
        "wind_speed_kmph",
        "Visibility_KM",
        "condition"
    ]

    existing_columns = [
        column
        for column in required_columns
        if column in weather_df.columns
    ]

    weather_df = weather_df[existing_columns]

    weather_df = weather_df.drop_duplicates(
        subset=["weather_date", "weather_city"]
    )

    print("\nWeather data prepared.")
    print(f"Weather columns used: {existing_columns}")

    return weather_df


# ============================================================
# MERGE WEATHER DATA
# ============================================================

def merge_weather_data(df, weather_df):
    print("\n" + "=" * 60)
    print("MERGING WEATHER DATA")
    print("=" * 60)

    df = df.copy()

    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce"
    )

    weather_df = prepare_weather_data(
        weather_df
    )

    df = df.merge(
        weather_df,
        left_on=[
            "order_date",
            "destination_city"
        ],
        right_on=[
            "weather_date",
            "weather_city"
        ],
        how="left"
    )

    df = df.drop(
        columns=[
            "weather_date",
            "weather_city"
        ],
        errors="ignore"
    )

    print("\nWeather merge completed.")

    return df


# ============================================================
# PREPARE TRAFFIC DATA
# ============================================================

def prepare_traffic_data(traffic_df):
    print("\n" + "=" * 60)
    print("PREPARING TRAFFIC DATA")
    print("=" * 60)

    traffic_df = traffic_df.copy()

    city_candidates = [
        "city",
        "City",
        "city_name",
        "City Name"
    ]

    speed_candidates = [
        "avg_speed_kmph",
        "average_speed_kmph",
        "speed_kmph",
        "avg_speed"
    ]

    congestion_candidates = [
        "congestion_index",
        "traffic_level",
        "congestion"
    ]

    city_column = next(
        (
            column
            for column in city_candidates
            if column in traffic_df.columns
        ),
        None
    )

    speed_column = next(
        (
            column
            for column in speed_candidates
            if column in traffic_df.columns
        ),
        None
    )

    congestion_column = next(
        (
            column
            for column in congestion_candidates
            if column in traffic_df.columns
        ),
        None
    )

    if city_column is None:
        raise ValueError(
            "Traffic city column could not be identified."
        )

    if speed_column is None:
        raise ValueError(
            "Traffic speed column could not be identified."
        )

    if congestion_column is None:
        raise ValueError(
            "Traffic congestion column could not be identified."
        )

    traffic_summary = (
        traffic_df[
            [
                city_column,
                speed_column,
                congestion_column
            ]
        ]
        .groupby(city_column, as_index=False)
        .agg(
            {
                speed_column: "mean",
                congestion_column: "mean"
            }
        )
    )

    traffic_summary = traffic_summary.rename(
        columns={
            city_column: "traffic_city",
            speed_column: "traffic_speed_kmph",
            congestion_column: "traffic_level"
        }
    )

    print("\nTraffic data prepared.")

    print("\nTraffic summary:")
    print(traffic_summary.head())

    return traffic_summary


# ============================================================
# MERGE TRAFFIC DATA
# ============================================================

def merge_traffic_data(df, traffic_df):
    print("\n" + "=" * 60)
    print("MERGING TRAFFIC DATA")
    print("=" * 60)

    df = df.copy()

    traffic_summary = prepare_traffic_data(
        traffic_df
    )

    df = df.merge(
        traffic_summary,
        left_on="destination_city",
        right_on="traffic_city",
        how="left"
    )

    df = df.drop(
        columns=["traffic_city"],
        errors="ignore"
    )

    print("\nTraffic merge completed.")

    return df


# ============================================================
# CREATE WEATHER IMPACT FEATURES
# ============================================================

def create_weather_impact_features(df):
    print("\n" + "=" * 60)
    print("CREATING WEATHER IMPACT FEATURES")
    print("=" * 60)

    df = df.copy()

    df["high_precipitation"] = (
        df["Precipitation (mm)"] > 10
    ).astype(int)

    df["high_wind"] = (
        df["wind_speed_kmph"] > 30
    ).astype(int)

    df["low_visibility"] = (
        df["Visibility_KM"] < 5
    ).astype(int)

    print("\nWeather impact features created:")
    print("- high_precipitation")
    print("- high_wind")
    print("- low_visibility")

    return df


# ============================================================
# CREATE TRAFFIC IMPACT FEATURES
# ============================================================

def create_traffic_impact_features(df):
    print("\n" + "=" * 60)
    print("CREATING TRAFFIC IMPACT FEATURES")
    print("=" * 60)

    df = df.copy()

    df["low_traffic_speed"] = (
        df["traffic_speed_kmph"] < 25
    ).astype(int)

    return df


# ============================================================
# CREATE TRANSPORTATION FEATURES
# ============================================================

def create_transportation_features(df):
    print("\n" + "=" * 60)
    print("CREATING TRANSPORTATION FEATURES")
    print("=" * 60)

    df = df.copy()

    if "transport_mode" in df.columns:
        df["transport_mode"] = (
            df["transport_mode"]
            .fillna("Unknown")
        )

    print("\nTransportation mode retained as a feature.")

    return df


# ============================================================
# IMPUTE MISSING VALUES
# ============================================================

def impute_missing_values(df):
    print("\n" + "=" * 60)
    print("IMPUTING MISSING VALUES")
    print("=" * 60)

    df = df.copy()

    numerical_columns = df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in numerical_columns:
        if df[column].isnull().any():
            df[column] = df[column].fillna(
                df[column].median()
            )

    for column in categorical_columns:
        if df[column].isnull().any():
            mode = df[column].mode()

            if not mode.empty:
                df[column] = df[column].fillna(
                    mode.iloc[0]
                )

    print("\nMissing values handled.")

    return df


# ============================================================
# FINAL MODEL FEATURE CLEANUP
# ============================================================

def final_model_feature_cleanup(df, target_column):
    print("\n" + "=" * 60)
    print("FINAL ETA MODEL FEATURE CLEANUP")
    print("=" * 60)

    df = df.copy()

    columns_to_drop = [
        # Identifiers
        "order_id",
        "customer_id",
        "product_id",

        # Original/redundant columns
        "order_date",
        "package_weight",
        "distance_km",
        "destination_pincode",

        # Post-delivery / leakage columns
        "promised_eta_hours",
        "delivery_cost_inr",
        "order_status",
        "assigned_vehicle_id"
    ]

    existing_columns = [
        column
        for column in columns_to_drop
        if column in df.columns
    ]

    df = df.drop(
        columns=existing_columns
    )

    # Remove rows where target is missing
    if target_column in df.columns:
        before_rows = len(df)

        df = df.dropna(
            subset=[target_column]
        )

        removed_rows = before_rows - len(df)

        print(
            f"\nRows removed due to missing target: "
            f"{removed_rows}"
        )

    # Remove constant columns
    constant_columns = [
        column
        for column in df.columns
        if column != target_column
        and df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        print("\nConstant columns removed:")
        for column in constant_columns:
            print(f"- {column}")

        df = df.drop(
            columns=constant_columns
        )

    print(f"\nFinal ETA feature shape: {df.shape}")

    print("\nFinal columns:")
    print(df.columns.tolist())

    return df


# ============================================================
# SAVE FEATURE DATASET
# ============================================================

def save_feature_dataset(df):
    print("\n" + "=" * 60)
    print("SAVING ETA FEATURE DATASET")
    print("=" * 60)

    output_path = (
        PROCESSED_DATA_DIR
        / "eta_features.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\nETA feature dataset saved to:")
    print(output_path)

    return output_path


# ============================================================
# FINAL DATASET INSPECTION
# ============================================================

def final_dataset_inspection(df, target_column):
    print("\n" + "=" * 60)
    print("FINAL ETA DATASET INSPECTION")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nMissing values:")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
    else:
        print(missing)

    print("\nTarget statistics:")

    print(
        df[target_column].describe()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("SMARTLOGIX AI - ETA FEATURE ENGINEERING")
    print("=" * 60)

    TARGET_COLUMN = "actual_delivery_hours"

    # --------------------------------------------------------
    # Step 1: Load datasets
    # --------------------------------------------------------

    orders_df = load_orders_data()

    weather_df = load_weather_data()

    traffic_df = load_traffic_data()

    # --------------------------------------------------------
    # Step 2: Create date features
    # --------------------------------------------------------

    orders_df = create_date_features(
        orders_df
    )

    # --------------------------------------------------------
    # Step 3: Create package features
    # --------------------------------------------------------

    orders_df = create_package_features(
        orders_df
    )

    # --------------------------------------------------------
    # Step 4: Create distance features
    # --------------------------------------------------------

    orders_df = create_distance_features(
        orders_df
    )

    # --------------------------------------------------------
    # Step 5: Create priority features
    # --------------------------------------------------------

    orders_df = create_priority_features(
        orders_df
    )

    # --------------------------------------------------------
    # Step 6: Merge weather
    # --------------------------------------------------------

    orders_df = merge_weather_data(
        orders_df,
        weather_df
    )

    # --------------------------------------------------------
    # Step 7: Merge traffic
    # --------------------------------------------------------

    orders_df = merge_traffic_data(
        orders_df,
        traffic_df
    )

    # --------------------------------------------------------
    # Step 8: Create weather impact features
    # --------------------------------------------------------

    orders_df = create_weather_impact_features(
        orders_df
    )

    # --------------------------------------------------------
    # Step 9: Create traffic impact features
    # --------------------------------------------------------

    orders_df = create_traffic_impact_features(
        orders_df
    )

    # --------------------------------------------------------
    # Step 10: Transportation feature
    # --------------------------------------------------------

    orders_df = create_transportation_features(
        orders_df
    )

    # --------------------------------------------------------
    # Step 11: Impute missing values
    # --------------------------------------------------------

    orders_df = impute_missing_values(
        orders_df
    )

    # --------------------------------------------------------
    # Step 12: Final cleanup
    # --------------------------------------------------------

    orders_df = final_model_feature_cleanup(
        orders_df,
        TARGET_COLUMN
    )

    # --------------------------------------------------------
    # Step 13: Final inspection
    # --------------------------------------------------------

    final_dataset_inspection(
        orders_df,
        TARGET_COLUMN
    )

    # --------------------------------------------------------
    # Step 14: Save
    # --------------------------------------------------------

    save_feature_dataset(
        orders_df
    )

    print("\n" + "=" * 60)
    print(
        "ETA FEATURE ENGINEERING "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()