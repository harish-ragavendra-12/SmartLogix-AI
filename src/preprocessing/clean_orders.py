import re

import pandas as pd

from src.config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


# ==========================================================
# LOAD ORDERS
# ==========================================================

def load_orders():
    """
    Load the raw orders dataset.
    """

    file_path = RAW_DATA_DIR / "orders.csv"

    df = pd.read_csv(file_path)

    return df


# ==========================================================
# CLEAN PACKAGE WEIGHT
# ==========================================================

def clean_package_weight(value):
    """
    Convert package weight values to kilograms.
    """

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"", "-", "nan", "none", "unknown"}:
        return None

    try:
        numeric_value = float(
            re.sub(r"[^0-9.\-]", "", value)
        )
    except ValueError:
        return None

    if numeric_value <= 0:
        return None

    if "g" in value and "kg" not in value:
        numeric_value = numeric_value / 1000

    return numeric_value


# ==========================================================
# CLEAN DISTANCE
# ==========================================================

def clean_distance(value):
    """
    Convert distance values to kilometers.
    """

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"", "-", "nan", "none", "unknown"}:
        return None

    value = value.replace("km", "").strip()

    try:
        numeric_value = float(value)
    except ValueError:
        return None

    if numeric_value < 0:
        return None

    return numeric_value


# ==========================================================
# CLEAN CURRENCY
# ==========================================================

def clean_currency(value):
    """
    Convert currency-formatted values to numeric INR.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in {"", "-", "nan", "none"}:
        return None

    value = (
        value
        .replace("₹", "")
        .replace(",", "")
        .strip()
    )

    try:
        return float(value)
    except ValueError:
        return None


# ==========================================================
# CLEAN DIMENSIONS
# ==========================================================

def parse_dimensions(value):
    """
    Parse package dimensions into length, width and height
    in centimeters.
    """

    if pd.isna(value):
        return pd.Series(
            [None, None, None]
        )

    value = str(value).strip()

    if value in {"", "-", "nan", "none"}:
        return pd.Series(
            [None, None, None]
        )

    value = value.replace("*", "x")

    parts = re.split(
        r"\s*x\s*",
        value,
        flags=re.IGNORECASE
    )

    if len(parts) != 3:
        return pd.Series(
            [None, None, None]
        )

    try:
        dimensions = [
            float(part.strip())
            for part in parts
        ]
    except ValueError:
        return pd.Series(
            [None, None, None]
        )

    if any(value <= 0 for value in dimensions):
        return pd.Series(
            [None, None, None]
        )

    return pd.Series(dimensions)


# ==========================================================
# CLEAN ORDERS
# ==========================================================

def clean_orders(df):
    """
    Clean and standardize the orders dataset.
    """

    df = df.copy()

    # ------------------------------------------------------
    # Standardize column names
    # ------------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # ------------------------------------------------------
    # Remove duplicate orders
    # ------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset="order_id"
    )

    after_duplicates = len(df)

    print(
        f"\nDuplicate orders removed: "
        f"{before_duplicates - after_duplicates}"
    )

    # ------------------------------------------------------
    # Clean IDs
    # ------------------------------------------------------

    id_columns = [
        "order_id",
        "customer_id",
        "product_id",
        "assigned_vehicle_id"
    ]

    for column in id_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # ------------------------------------------------------
    # Convert quantity
    # ------------------------------------------------------

    df["quantity"] = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    )

    # ------------------------------------------------------
    # Convert order date
    # ------------------------------------------------------

    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce",
        format="mixed",
        dayfirst=True,
        utc=True
    )

    df["order_date"] = df["order_date"].dt.tz_localize(None)

    # ------------------------------------------------------
    # Convert coordinates
    # ------------------------------------------------------

    coordinate_columns = [
        "destination_lat",
        "destination_lon"
    ]

    for column in coordinate_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ------------------------------------------------------
    # Validate coordinates
    # ------------------------------------------------------

    df.loc[
        ~df["destination_lat"].between(-90, 90),
        "destination_lat"
    ] = pd.NA

    df.loc[
        ~df["destination_lon"].between(-180, 180),
        "destination_lon"
    ] = pd.NA

    # ------------------------------------------------------
    # Convert distance
    # ------------------------------------------------------

    df["distance_km"] = (
        df["distance_km"]
        .apply(clean_distance)
    )

    # ------------------------------------------------------
    # Convert package weight
    # ------------------------------------------------------

    df["package_weight"] = (
        df["package_weight"]
        .apply(clean_package_weight)
    )

    df["package_weight"] = pd.to_numeric(
        df["package_weight"],
        errors="coerce"
    )

    # Remove invalid negative package weights
    df.loc[
        df["package_weight"] <= 0,
        "package_weight"
    ] = pd.NA

    # ------------------------------------------------------
    # Parse package dimensions
    # ------------------------------------------------------

    dimensions = (
        df["package_dimensions_cm"]
        .apply(parse_dimensions)
    )

    dimensions.columns = [
        "dimension_length_cm",
        "dimension_width_cm",
        "dimension_height_cm"
    ]

    df = pd.concat(
        [
            df,
            dimensions
        ],
        axis=1
    )

    # ------------------------------------------------------
    # Remove original dimensions column
    # ------------------------------------------------------

    df = df.drop(
        columns=["package_dimensions_cm"]
    )

    # ------------------------------------------------------
    # Convert order value
    # ------------------------------------------------------

    df["order_value_inr"] = (
        df["order_value_inr"]
        .apply(clean_currency)
    )

    # ------------------------------------------------------
    # Convert delivery cost
    # ------------------------------------------------------

    df["delivery_cost_inr"] = (
        df["delivery_cost_inr"]
        .apply(clean_currency)
    )

    # Remove invalid negative delivery costs
    df.loc[
        df["delivery_cost_inr"] < 0,
        "delivery_cost_inr"
    ] = pd.NA

    # ------------------------------------------------------
    # Convert delivery hours
    # ------------------------------------------------------

    df["promised_eta_hours"] = pd.to_numeric(
        df["promised_eta_hours"],
        errors="coerce"
    )

    df["actual_delivery_hours"] = pd.to_numeric(
        df["actual_delivery_hours"],
        errors="coerce"
    )

    # ------------------------------------------------------
    # Standardize boolean fields
    # ------------------------------------------------------

    boolean_mapping = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "1": True,
        "0": False
    }

    boolean_columns = [
        "is_fragile",
        "is_hazmat",
        "cold_chain_required"
    ]

    for column in boolean_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .map(boolean_mapping)
            .astype("boolean")
        )

    # ------------------------------------------------------
    # Standardize transport mode
    # ------------------------------------------------------

    transport_mapping = {
        "truck": "truck",
        "bike": "bike",
        "van": "van",
        "air cargo": "air_cargo",
        "air_cargo": "air_cargo",
        "drone": "drone",
        "ship": "ship",
        "unknown": None,
        "-": None,
        "": None
    }

    df["transport_mode"] = (
        df["transport_mode"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(transport_mapping)
    )

    # ------------------------------------------------------
    # Standardize order status
    # ------------------------------------------------------

    status_mapping = {
        "delivered": "delivered",
        "in transit": "in_transit",
        "pending": "pending",
        "cancelled": "cancelled",
        "returned": "returned"
    }

    df["order_status"] = (
        df["order_status"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(status_mapping)
    )

    # ------------------------------------------------------
    # Standardize payment mode
    # ------------------------------------------------------

    payment_mapping = {
        "cod": "cod",
        "card": "card",
        "upi": "upi",
        "prepaid": "prepaid"
    }

    df["payment_mode"] = (
        df["payment_mode"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(payment_mapping)
    )

    # ------------------------------------------------------
    # Standardize delivery priority
    # ------------------------------------------------------

    priority_mapping = {
        "standard": "standard",
        "std": "standard",
        "express": "express",
        "exp": "express",
        "same-day": "same_day",
        "sameday": "same_day",
        "economy": "economy",
        "eco": "economy"
    }

    df["delivery_priority"] = (
        df["delivery_priority"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(priority_mapping)
    )

    # ------------------------------------------------------
    # Standardize weather condition
    # ------------------------------------------------------

    df["weather_condition_at_dest"] = (
        df["weather_condition_at_dest"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # ------------------------------------------------------
    # Clean destination pincode
    # ------------------------------------------------------

    df["destination_pincode"] = (
        df["destination_pincode"]
        .astype("string")
        .str.strip()
    )

    # ------------------------------------------------------
    # Check missing values
    # ------------------------------------------------------

    print("\nMissing values:")
    print(df.isnull().sum())

    # ------------------------------------------------------
    # Reset index
    # ------------------------------------------------------

    df = df.reset_index(drop=True)

    return df


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    orders_df = load_orders()

    orders_df = clean_orders(orders_df)

    output_path = PROCESSED_DATA_DIR / "orders_cleaned.csv"

    orders_df.to_csv(
        output_path,
        index=False
    )

    print("\nOrders dataset cleaned successfully.")
    print("Shape:", orders_df.shape)
    print("Cleaned dataset saved to:", output_path)

    print("\nData types:")
    print(orders_df.dtypes)

    print("\nFirst 5 records:")
    print(orders_df.head())