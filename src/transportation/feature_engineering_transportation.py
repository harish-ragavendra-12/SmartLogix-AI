import numpy as np
import pandas as pd

from src.config.config import PROCESSED_DATA_DIR


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_dataset(file_name):

    file_path = PROCESSED_DATA_DIR / file_name

    print(f"\nLoading: {file_path}")

    df = pd.read_csv(file_path)

    print(
        f"Shape: {df.shape}"
    )

    return df


# ============================================================
# 2. FIND COLUMN HELPER
# ============================================================

def find_column(df, candidates):

    column_map = {
        column.lower().strip(): column
        for column in df.columns
    }

    # Exact match
    for candidate in candidates:

        candidate_lower = (
            candidate.lower().strip()
        )

        if candidate_lower in column_map:

            return column_map[
                candidate_lower
            ]

    # Partial match
    for candidate in candidates:

        candidate_lower = (
            candidate.lower().strip()
        )

        for column in df.columns:

            if candidate_lower in (
                column.lower().strip()
            ):

                return column

    return None


# ============================================================
# 3. BASIC DATA INSPECTION
# ============================================================

def inspect_orders(df):

    print("\n" + "=" * 60)
    print("TRANSPORTATION DATASET INSPECTION")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")

    for column in df.columns:

        print(
            f"- {column}"
        )

    print("\nData types:")
    print(df.dtypes)

    target_column = find_column(
        df,
        [
            "transport_mode",
            "transportation_mode",
            "delivery_mode",
            "mode"
        ]
    )

    if target_column:

        print(
            f"\nTarget column detected: "
            f"{target_column}"
        )

        print(
            "\nTarget distribution:"
        )

        print(
            df[target_column]
            .value_counts(dropna=False)
        )

    else:

        print(
            "\nWARNING: transport mode "
            "target column was not found."
        )


# ============================================================
# 4. CLEAN TRANSPORTATION TARGET
# ============================================================

def clean_transport_target(df):

    print("\n" + "=" * 60)
    print("CLEANING TRANSPORTATION TARGET")
    print("=" * 60)

    target_column = find_column(
        df,
        [
            "transport_mode",
            "transportation_mode",
            "delivery_mode",
            "mode"
        ]
    )

    if target_column is None:

        raise ValueError(
            "Transportation target column "
            "was not found."
        )

    # Remove leading/trailing spaces
    df[target_column] = (
        df[target_column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Standardize known variations
    transport_mapping = {

        "drone": "Drone",

        "bike": "Bike",
        "motorbike": "Bike",
        "motor bike": "Bike",
        "two wheeler": "Bike",

        "van": "Van",

        "truck": "Truck",

        "air cargo": "Air Cargo",
        "air_cargo": "Air Cargo",
        "aircargo": "Air Cargo",

        "ship": "Ship",
        "cargo ship": "Ship"
    }

    df[target_column] = (
        df[target_column]
        .map(transport_mapping)
        .fillna(
            df[target_column]
            .str.title()
        )
    )

    print(
        "\nStandardized target distribution:"
    )

    print(
        df[target_column]
        .value_counts(dropna=False)
    )

    return df, target_column


# ============================================================
# 5. DATE FEATURES
# ============================================================

def create_date_features(df):

    print("\n" + "=" * 60)
    print("CREATING DATE FEATURES")
    print("=" * 60)

    date_column = find_column(
        df,
        [
            "order_date",
            "delivery_date",
            "shipment_date",
            "created_at",
            "order_datetime",
            "timestamp",
            "date"
        ]
    )

    if date_column is None:

        print(
            "No suitable date column found."
        )

        return df

    print(
        f"Using date column: {date_column}"
    )

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df["order_year"] = (
        df[date_column].dt.year
    )

    df["order_month"] = (
        df[date_column].dt.month
    )

    df["order_day"] = (
        df[date_column].dt.day
    )

    df["order_day_of_week"] = (
        df[date_column].dt.dayofweek
    )

    df["order_week"] = (
        df[date_column].dt.isocalendar().week
        .astype("float")
    )

    df["is_weekend"] = (
        df["order_day_of_week"] >= 5
    ).astype(int)

    print(
        "Date features created:"
    )

    print(
        [
            "order_year",
            "order_month",
            "order_day",
            "order_day_of_week",
            "order_week",
            "is_weekend"
        ]
    )

    return df


# ============================================================
# 6. PACKAGE FEATURES
# ============================================================

def create_package_features(df):

    print("\n" + "=" * 60)
    print("CREATING PACKAGE FEATURES")
    print("=" * 60)

    weight_column = find_column(
        df,
        [
            "package_weight",
            "weight_kg",
            "package_weight_kg",
            "weight"
        ]
    )

    length_column = find_column(
        df,
        [
            "package_length",
            "length_cm",
            "length"
        ]
    )

    width_column = find_column(
        df,
        [
            "package_width",
            "width_cm",
            "width"
        ]
    )

    height_column = find_column(
        df,
        [
            "package_height",
            "height_cm",
            "height"
        ]
    )

    if weight_column:

        df[weight_column] = pd.to_numeric(
            df[weight_column],
            errors="coerce"
        )

        df["package_weight_kg"] = (
            df[weight_column]
        )

        print(
            f"Weight feature: "
            f"{weight_column}"
        )

    if (
        length_column
        and width_column
        and height_column
    ):

        df[length_column] = pd.to_numeric(
            df[length_column],
            errors="coerce"
        )

        df[width_column] = pd.to_numeric(
            df[width_column],
            errors="coerce"
        )

        df[height_column] = pd.to_numeric(
            df[height_column],
            errors="coerce"
        )

        df["package_volume_cm3"] = (
            df[length_column]
            * df[width_column]
            * df[height_column]
        )

        print(
            "Package volume feature created."
        )

    else:

        print(
            "Complete package dimensions "
            "were not found."
        )

    return df


# ============================================================
# 7. DISTANCE FEATURES
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    earth_radius_km = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    c = (
        2
        * np.arcsin(
            np.sqrt(a)
        )
    )

    return earth_radius_km * c


def create_distance_features(df):

    print("\n" + "=" * 60)
    print("CREATING DISTANCE FEATURES")
    print("=" * 60)

    existing_distance_column = find_column(
        df,
        [
            "distance_km",
            "delivery_distance_km",
            "distance",
            "route_distance_km"
        ]
    )

    if existing_distance_column:

        df[existing_distance_column] = (
            pd.to_numeric(
                df[existing_distance_column],
                errors="coerce"
            )
        )

        df["delivery_distance_km"] = (
            df[existing_distance_column]
        )

        print(
            f"Existing distance column used: "
            f"{existing_distance_column}"
        )

        return df

    # Try calculating distance
    # from origin/destination coordinates

    origin_lat = find_column(
        df,
        [
            "origin_lat",
            "source_lat",
            "pickup_lat"
        ]
    )

    origin_lon = find_column(
        df,
        [
            "origin_lon",
            "source_lon",
            "pickup_lon"
        ]
    )

    destination_lat = find_column(
        df,
        [
            "destination_lat",
            "dest_lat",
            "delivery_lat"
        ]
    )

    destination_lon = find_column(
        df,
        [
            "destination_lon",
            "dest_lon",
            "delivery_lon"
        ]
    )

    if (
        origin_lat
        and origin_lon
        and destination_lat
        and destination_lon
    ):

        for column in [
            origin_lat,
            origin_lon,
            destination_lat,
            destination_lon
        ]:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df["delivery_distance_km"] = (
            haversine_distance(
                df[origin_lat],
                df[origin_lon],
                df[destination_lat],
                df[destination_lon]
            )
        )

        print(
            "Distance calculated using "
            "Haversine formula."
        )

    else:

        print(
            "No distance or coordinate "
            "columns found."
        )

    return df


# ============================================================
# 8. PRIORITY FEATURES
# ============================================================

def create_priority_features(df):

    print("\n" + "=" * 60)
    print("CREATING PRIORITY FEATURES")
    print("=" * 60)

    priority_column = find_column(
        df,
        [
            "delivery_priority",
            "priority",
            "order_priority"
        ]
    )

    if priority_column is None:

        print(
            "No priority column found."
        )

        return df

    print(
        f"Priority column: "
        f"{priority_column}"
    )

    df[priority_column] = (
        df[priority_column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    priority_mapping = {
        "economy": 1,
        "standard": 2,
        "express": 3,
        "same_day": 4,

        # Keep these in case other records appear later
        "low": 1,
        "medium": 2,
        "normal": 2,
        "high": 3,
        "urgent": 4,
        "critical": 5
    }

    df["priority_score"] = (
        df[priority_column]
        .map(priority_mapping)
    )

    print(
        "\nPriority distribution:"
    )

    print(
        df[priority_column]
        .value_counts(dropna=False)
    )

    return df


# ============================================================
# 9. MERGE WEATHER FEATURES
# ============================================================

def merge_weather_features(
    orders_df,
    weather_df
):

    print("\n" + "=" * 60)
    print("MERGING WEATHER FEATURES")
    print("=" * 60)

    city_column_orders = find_column(
        orders_df,
        [
            "destination_city",
            "delivery_city",
            "city",
            "location_city"
        ]
    )

    city_column_weather = find_column(
        weather_df,
        [
            "City Name",
            "city_name",
            "city"
        ]
    )

    date_column_orders = find_column(
        orders_df,
        [
            "order_date",
            "delivery_date",
            "created_at",
            "date"
        ]
    )

    date_column_weather = find_column(
        weather_df,
        [
            "Date",
            "date"
        ]
    )

    if (
        city_column_orders is None
        or city_column_weather is None
    ):

        print(
            "Weather merge skipped: "
            "city columns not available."
        )

        return orders_df

    # Standardize city values

    orders_df["_merge_city"] = (
        orders_df[city_column_orders]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    weather_df["_merge_city"] = (
        weather_df[city_column_weather]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Date-based merge when both dates exist
    # --------------------------------------------------------

    if (
        date_column_orders
        and date_column_weather
    ):

        orders_df["_merge_date"] = pd.to_datetime(
            orders_df[date_column_orders],
            errors="coerce"
        ).dt.date

        weather_df["_merge_date"] = pd.to_datetime(
            weather_df[date_column_weather],
            errors="coerce"
        ).dt.date

        # Select useful weather features

        weather_features = [
            "temp_celsius",
            "humidity_%",
            "Precipitation (mm)",
            "wind_speed_kmph",
            "Visibility_KM"
        ]

        available_weather_features = [
            column
            for column in weather_features
            if column in weather_df.columns
        ]

        weather_subset = weather_df[
            [
                "_merge_city",
                "_merge_date"
            ]
            + available_weather_features
        ].copy()

        # If duplicate city/date records exist,
        # aggregate them.

        weather_subset = (
            weather_subset
            .groupby(
                [
                    "_merge_city",
                    "_merge_date"
                ],
                as_index=False
            )[available_weather_features]
            .mean()
        )

        orders_df = orders_df.merge(
            weather_subset,
            on=[
                "_merge_city",
                "_merge_date"
            ],
            how="left"
        )

        print(
            "Weather merged using city + date."
        )

    else:

        # City-level fallback
        weather_features = [
            "temp_celsius",
            "humidity_%",
            "Precipitation (mm)",
            "wind_speed_kmph",
            "Visibility_KM"
        ]

        available_weather_features = [
            column
            for column in weather_features
            if column in weather_df.columns
        ]

        weather_subset = (
            weather_df[
                ["_merge_city"]
                + available_weather_features
            ]
            .groupby(
                "_merge_city",
                as_index=False
            )
            .mean(
                numeric_only=True
            )
        )

        orders_df = orders_df.merge(
            weather_subset,
            on="_merge_city",
            how="left"
        )

        print(
            "Weather merged using city-level "
            "aggregates."
        )

    # Remove temporary columns

    for column in [
        "_merge_city",
        "_merge_date"
    ]:

        if column in orders_df.columns:

            orders_df.drop(
                columns=column,
                inplace=True
            )

    return orders_df


# ============================================================
# 10. MERGE TRAFFIC FEATURES
# ============================================================

def merge_traffic_features(
    orders_df,
    traffic_df
):

    print("\n" + "=" * 60)
    print("MERGING TRAFFIC FEATURES")
    print("=" * 60)

    # --------------------------------------------------------
    # Remove temporary columns if they already exist
    # --------------------------------------------------------

    orders_df = orders_df.drop(
        columns=["_traffic_city"],
        errors="ignore"
    ).copy()

    traffic_df = traffic_df.drop(
        columns=["_traffic_city"],
        errors="ignore"
    ).copy()

    # --------------------------------------------------------
    # Find city columns
    # --------------------------------------------------------

    order_city = find_column(
        orders_df,
        [
            "destination_city",
            "delivery_city",
            "city",
            "location_city"
        ]
    )

    traffic_city = find_column(
        traffic_df,
        [
            "city",
            "location_city",
            "City Name"
        ]
    )

    print(
        f"Order city column: {order_city}"
    )

    print(
        f"Traffic city column: {traffic_city}"
    )

    if (
        order_city is None
        or traffic_city is None
    ):

        print(
            "Traffic merge skipped: "
            "city columns not available."
        )

        return orders_df

    # --------------------------------------------------------
    # Create standardized city keys
    # --------------------------------------------------------

    orders_df["_traffic_city"] = (
        orders_df[order_city]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    traffic_df["_traffic_city"] = (
        traffic_df[traffic_city]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Find traffic speed column
    # --------------------------------------------------------

    speed_column = find_column(
        traffic_df,
        [
            "speed_kmph",
            "avg_speed_kmph",
            "average_speed_kmph",
            "average_speed",
            "vehicle_speed_kmph"
        ]
    )

    # --------------------------------------------------------
    # Find traffic level column
    # IMPORTANT:
    # Do NOT use generic "traffic" here because
    # it can match "_traffic_city".
    # --------------------------------------------------------

    traffic_level_column = find_column(
        traffic_df,
        [
            "traffic_level",
            "traffic_condition",
            "congestion_level",
            "congestion",
            "congestion_category"
        ]
    )

    # Safety check
    if speed_column == "_traffic_city":

        speed_column = None

    if traffic_level_column == "_traffic_city":

        traffic_level_column = None

    print(
        f"Traffic speed column: "
        f"{speed_column}"
    )

    print(
        f"Traffic level column: "
        f"{traffic_level_column}"
    )

    # --------------------------------------------------------
    # Make sure we have useful traffic information
    # --------------------------------------------------------

    traffic_columns = []

    if (
        speed_column
        and speed_column != "_traffic_city"
    ):

        traffic_columns.append(
            speed_column
        )

    if (
        traffic_level_column
        and traffic_level_column != "_traffic_city"
        and traffic_level_column
        not in traffic_columns
    ):

        traffic_columns.append(
            traffic_level_column
        )

    if not traffic_columns:

        print(
            "No useful traffic columns found."
        )

        orders_df.drop(
            columns="_traffic_city",
            inplace=True
        )

        return orders_df

    # --------------------------------------------------------
    # Create traffic subset
    # --------------------------------------------------------

    traffic_subset = traffic_df[
        ["_traffic_city"]
        + traffic_columns
    ].copy()

    # --------------------------------------------------------
    # Convert speed to numeric
    # --------------------------------------------------------

    if speed_column:

        traffic_subset[speed_column] = (
            pd.to_numeric(
                traffic_subset[speed_column],
                errors="coerce"
            )
        )

    # --------------------------------------------------------
    # Aggregate traffic by city
    # --------------------------------------------------------

    aggregation = {}

    if speed_column:

        aggregation[speed_column] = "mean"

    if traffic_level_column:

        aggregation[
            traffic_level_column
        ] = (
            lambda x:
            x.mode().iloc[0]
            if not x.mode().empty
            else "Unknown"
        )

    traffic_subset = (
        traffic_subset
        .groupby(
            "_traffic_city",
            as_index=False
        )
        .agg(aggregation)
    )

    print(
        "\nTraffic summary:"
    )

    print(
        traffic_subset.head(10)
    )

    # --------------------------------------------------------
    # Rename traffic columns
    # --------------------------------------------------------

    rename_mapping = {}

    if speed_column:

        rename_mapping[
            speed_column
        ] = "traffic_speed_kmph"

    if traffic_level_column:

        rename_mapping[
            traffic_level_column
        ] = "traffic_level"

    traffic_subset = (
        traffic_subset
        .rename(
            columns=rename_mapping
        )
    )

    # --------------------------------------------------------
    # Merge with orders
    # --------------------------------------------------------

    orders_df = orders_df.merge(
        traffic_subset,
        on="_traffic_city",
        how="left"
    )

    print(
        "\nTraffic features merged "
        "successfully using city-level "
        "aggregates."
    )

    # --------------------------------------------------------
    # Remove temporary column
    # --------------------------------------------------------

    orders_df.drop(
        columns="_traffic_city",
        inplace=True
    )

    return orders_df

# ============================================================
# 11. MERGE FLEET AVAILABILITY
# ============================================================

def merge_fleet_features(
    orders_df,
    fleet_df
):

    print("\n" + "=" * 60)
    print("MERGING FLEET FEATURES")
    print("=" * 60)

    fleet_mode_column = find_column(
        fleet_df,
        [
            "transport_mode",
            "vehicle_type",
            "vehicle_mode",
            "mode"
        ]
    )

    if fleet_mode_column is None:

        print(
            "Fleet merge skipped: "
            "vehicle/transport mode column "
            "not found."
        )

        return orders_df

    fleet_df["_fleet_mode"] = (
        fleet_df[fleet_mode_column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    fleet_mode_mapping = {
        "drone": "Drone",
        "bike": "Bike",
        "motorbike": "Bike",
        "van": "Van",
        "truck": "Truck",
        "air cargo": "Air Cargo",
        "air_cargo": "Air Cargo",
        "ship": "Ship"
    }

    fleet_df["_fleet_mode"] = (
        fleet_df["_fleet_mode"]
        .map(fleet_mode_mapping)
        .fillna(
            fleet_df["_fleet_mode"]
            .str.title()
        )
    )

    # Count available fleet units by mode

    availability_column = find_column(
        fleet_df,
        [
            "availability",
            "available",
            "status",
            "vehicle_status"
        ]
    )

    if availability_column:

        availability = (
            fleet_df[
                availability_column
            ]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        fleet_df["_is_available"] = (
            availability.isin(
                [
                    "available",
                    "active",
                    "ready",
                    "yes",
                    "true"
                ]
            )
            .astype(int)
        )

    else:

        fleet_df["_is_available"] = 1

    fleet_summary = (
        fleet_df
        .groupby("_fleet_mode")
        .agg(
            fleet_count=(
                "_fleet_mode",
                "size"
            ),
            available_fleet_count=(
                "_is_available",
                "sum"
            )
        )
        .reset_index()
    )

    print(
        "\nFleet summary:"
    )

    print(
        fleet_summary
    )

    # We don't merge fleet counts directly
    # into every order because transport_mode
    # is the target and doing so using the
    # actual target would cause leakage.
    #
    # Instead, create overall fleet features.

    total_fleet = (
        fleet_summary["fleet_count"]
        .sum()
    )

    total_available = (
        fleet_summary["available_fleet_count"]
        .sum()
    )

    orders_df["total_fleet_count"] = (
        total_fleet
    )

    orders_df["total_available_fleet"] = (
        total_available
    )

    print(
        "\nOverall fleet features added:"
    )

    print(
        f"Total fleet: {total_fleet}"
    )

    print(
        f"Available fleet: {total_available}"
    )

    return orders_df


# ============================================================
# 12. CREATE ADDITIONAL MODEL FEATURES
# ============================================================

def create_additional_features(df):

    print("\n" + "=" * 60)
    print("CREATING ADDITIONAL FEATURES")
    print("=" * 60)

    # --------------------------------------------------------
    # Weight category
    # --------------------------------------------------------

    weight_column = find_column(
        df,
        [
            "package_weight_kg",
            "package_weight",
            "weight_kg",
            "weight"
        ]
    )

    if weight_column:

        df[weight_column] = pd.to_numeric(
            df[weight_column],
            errors="coerce"
        )

        df["weight_category"] = pd.cut(
            df[weight_column],
            bins=[
                -np.inf,
                1,
                5,
                10,
                25,
                np.inf
            ],
            labels=[
                "Very Light",
                "Light",
                "Medium",
                "Heavy",
                "Very Heavy"
            ]
        )

        print(
            "Weight category created."
        )

    # --------------------------------------------------------
    # Distance category
    # --------------------------------------------------------

    if "delivery_distance_km" in df.columns:

        df["distance_category"] = pd.cut(
            df["delivery_distance_km"],
            bins=[
                -np.inf,
                5,
                20,
                50,
                100,
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

        print(
            "Distance category created."
        )

    # --------------------------------------------------------
    # Weather risk indicators
    # --------------------------------------------------------

    if "Precipitation (mm)" in df.columns:

        df["high_precipitation"] = (
            df["Precipitation (mm)"] >= 20
        ).astype(int)

        print(
            "High precipitation feature created."
        )

    if "wind_speed_kmph" in df.columns:

        df["high_wind"] = (
            df["wind_speed_kmph"] >= 35
        ).astype(int)

        print(
            "High wind feature created."
        )

    if "Visibility_KM" in df.columns:

        df["low_visibility"] = (
            df["Visibility_KM"] <= 3
        ).astype(int)

        print(
            "Low visibility feature created."
        )

    return df


# ============================================================
# 13. REMOVE DATA LEAKAGE
# ============================================================

def remove_target_leakage(
    df,
    target_column
):

    print("\n" + "=" * 60)
    print("REMOVING DATA LEAKAGE")
    print("=" * 60)

    leakage_columns = []

    for column in df.columns:

        column_lower = (
            column.lower()
        )

        # ----------------------------------------------------
        # Actual target
        # ----------------------------------------------------

        if column == target_column:

            continue

        # ----------------------------------------------------
        # Direct transportation-mode information
        # ----------------------------------------------------

        if (
            "transport_mode" in column_lower
            or "transportation_mode"
            in column_lower
            or "delivery_mode"
            in column_lower
        ):

            leakage_columns.append(
                column
            )

        # ----------------------------------------------------
        # Vehicle assignment
        # ----------------------------------------------------

        if column_lower in [
            "assigned_vehicle_id",
            "assigned_driver_id",
            "assigned_drone_id"
        ]:

            leakage_columns.append(
                column
            )

        # ----------------------------------------------------
        # Post-delivery information
        # ----------------------------------------------------

        if column_lower in [
            "actual_delivery_hours",
            "actual_delivery_time",
            "delivery_time",
            "delivery_status",
            "delivered_at",
            "actual_eta",
            "delivery_cost_inr",
            "order_status"
        ]:

            leakage_columns.append(
                column
            )

        # ----------------------------------------------------
        # Promised ETA can be downstream of mode selection
        # ----------------------------------------------------

        if column_lower in [
            "promised_eta_hours",
            "promised_eta",
            "estimated_delivery_hours",
            "estimated_eta"
        ]:

            leakage_columns.append(
                column
            )

    # Remove duplicates from list
    leakage_columns = list(
        dict.fromkeys(
            leakage_columns
        )
    )

    if leakage_columns:

        print(
            "\nRemoving potential "
            "target-leakage columns:"
        )

        for column in leakage_columns:

            print(
                f"- {column}"
            )

        df = df.drop(
            columns=leakage_columns,
            errors="ignore"
        )

    else:

        print(
            "No obvious leakage columns found."
        )

    return df


# ============================================================
# 14. HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(
    df,
    target_column
):

    print("\n" + "=" * 60)
    print("HANDLING MISSING VALUES")
    print("=" * 60)

    # Do not impute target
    # Missing target rows cannot be used
    # for supervised classification.

    before_rows = len(df)

    df = df.dropna(
        subset=[target_column]
    )

    removed_rows = (
        before_rows - len(df)
    )

    print(
        f"Rows removed because of "
        f"missing target: {removed_rows}"
    )

    # Numerical columns
    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numerical_columns:

        if column == target_column:

            continue

        if df[column].isnull().any():

            median_value = (
                df[column].median()
            )

            df[column] = (
                df[column]
                .fillna(median_value)
            )

            print(
                f"{column}: "
                f"filled with median "
                f"{median_value:.2f}"
            )

    # Categorical columns
    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    ).columns

    for column in categorical_columns:

        if column == target_column:

            continue

        if df[column].isnull().any():

            df[column] = (
                df[column]
                .astype("string")
                .fillna("Unknown")
            )

            print(
                f"{column}: "
                f"filled with 'Unknown'"
            )

    return df


# ============================================================
# 15. FINAL MODEL FEATURE CLEANUP
# ============================================================

def final_model_feature_cleanup(
    df,
    target_column
):

    print("\n" + "=" * 60)
    print("FINAL MODEL FEATURE CLEANUP")
    print("=" * 60)

    columns_to_drop = [
        # Identifier columns
        "order_id",
        "customer_id",
        "product_id",

        # Original date column
        "order_date",

        # Duplicate engineered features
        "package_weight",
        "distance_km",

        # Non-informative constant fleet features
        "total_fleet_count",
        "total_available_fleet",

        # Pincode represented as numeric
        "destination_pincode"
    ]

    existing_columns = [
        column
        for column in columns_to_drop
        if column in df.columns
    ]

    print(
        "\nColumns removed:"
    )

    for column in existing_columns:

        print(
            f"- {column}"
        )

    df = df.drop(
        columns=existing_columns,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Remove columns with only one unique value
    # --------------------------------------------------------

    constant_columns = [
        column
        for column in df.columns
        if column != target_column
        and df[column].nunique(
            dropna=False
        ) <= 1
    ]

    if constant_columns:

        print(
            "\nConstant columns removed:"
        )

        for column in constant_columns:

            print(
                f"- {column}"
            )

        df = df.drop(
            columns=constant_columns
        )

    # --------------------------------------------------------
    # Final missing-value check
    # --------------------------------------------------------

    print(
        "\nRemaining missing values:"
    )

    missing = (
        df.isnull()
        .sum()
    )

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:

        print(
            "No missing values."
        )

    else:

        print(
            missing
        )

    print(
        "\nFinal feature shape:"
    )

    print(
        df.shape
    )

    print(
        "\nFinal columns:"
    )

    print(
        df.columns.tolist()
    )

    return df


# ============================================================
# 16. SAVE FEATURE DATASET
# ============================================================

def save_feature_dataset(df):

    output_file = (
        PROCESSED_DATA_DIR
        / "transportation_features.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nFeature dataset saved to:"
    )

    print(
        output_file
    )

    return output_file


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 60
    )

    print(
        "SMARTLOGIX AI - "
        "TRANSPORTATION FEATURE ENGINEERING"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Load base dataset
    # --------------------------------------------------------

    orders_df = load_dataset(
        "orders_cleaned.csv"
    )

    inspect_orders(
        orders_df
    )

    # --------------------------------------------------------
    # Clean target
    # --------------------------------------------------------

    orders_df, target_column = (
        clean_transport_target(
            orders_df
        )
    )

    # --------------------------------------------------------
    # Create order-level features
    # --------------------------------------------------------

    orders_df = create_date_features(
        orders_df
    )

    orders_df = create_package_features(
        orders_df
    )

    orders_df = create_distance_features(
        orders_df
    )

    orders_df = create_priority_features(
        orders_df
    )

    # --------------------------------------------------------
    # Load external datasets
    # --------------------------------------------------------

    weather_df = load_dataset(
        "weather_data_clean.csv"
    )

    traffic_df = load_dataset(
        "traffic_data_clean.csv"
    )

    fleet_df = load_dataset(
        "fleet_vehicles_cleaned.csv"
    )

    # --------------------------------------------------------
    # Merge weather
    # --------------------------------------------------------

    orders_df = merge_weather_features(
        orders_df,
        weather_df
    )

    # --------------------------------------------------------
    # Merge traffic
    # --------------------------------------------------------

    orders_df = merge_traffic_features(
        orders_df,
        traffic_df
    )

    # --------------------------------------------------------
    # Merge fleet information
    # --------------------------------------------------------

    orders_df = merge_fleet_features(
        orders_df,
        fleet_df
    )

    # --------------------------------------------------------
    # Additional features
    # --------------------------------------------------------

    orders_df = create_additional_features(
        orders_df
    )

    # --------------------------------------------------------
    # Remove target leakage
    # --------------------------------------------------------

    orders_df = remove_target_leakage(
        orders_df,
        target_column
    )

    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    orders_df = handle_missing_values(
        orders_df,
        target_column
    )

    orders_df = final_model_feature_cleanup(
        orders_df,
        target_column
    )

    # --------------------------------------------------------
    # Final review
    # --------------------------------------------------------

    final_model_feature_cleanup(
        orders_df,
        target_column
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_feature_dataset(
        orders_df
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "TRANSPORTATION FEATURE ENGINEERING "
        "COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()