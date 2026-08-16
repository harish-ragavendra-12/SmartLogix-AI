import pandas as pd

from src.config.config import RAW_DATA_DIR


# ==========================================================
# LOAD FLEET VEHICLES
# ==========================================================

def load_fleet_vehicles():
    """
    Load the raw fleet vehicles dataset.
    """

    file_path = RAW_DATA_DIR / "fleet_vehicles.csv"

    df = pd.read_csv(file_path)

    return df


# ==========================================================
# CLEAN DATE
# ==========================================================

def clean_date(value):
    """
    Convert multiple date formats into a common datetime format.
    """

    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    if value in {"", "-", "nan", "none"}:
        return pd.NaT

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ]

    for date_format in formats:

        try:
            parsed_date = pd.to_datetime(
                value,
                format=date_format,
                errors="coerce"
            )

            if not pd.isna(parsed_date):
                return parsed_date

        except (ValueError, TypeError):
            continue

    return pd.NaT


# ==========================================================
# CLEAN FLEET VEHICLES
# ==========================================================

def clean_fleet_vehicles(df):
    """
    Clean and standardize the fleet vehicles dataset.
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
    # Remove exact duplicate records
    # ------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates()

    after_duplicates = len(df)

    print(
        f"\nDuplicate rows removed: "
        f"{before_duplicates - after_duplicates}"
    )

    # ------------------------------------------------------
    # Clean ID columns
    # ------------------------------------------------------

    id_columns = [
        "vehicle_id",
        "hub_code",
        "driver_id"
    ]

    for column in id_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # ------------------------------------------------------
    # Clean text columns
    # ------------------------------------------------------

    text_columns = [
        "vehicle_type",
        "model_name",
        "ownership"
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # ------------------------------------------------------
    # Standardize vehicle type
    # ------------------------------------------------------

    df["vehicle_type"] = (
        df["vehicle_type"]
        .str.strip()
        .str.title()
    )

    # ------------------------------------------------------
    # Standardize fleet status
    # ------------------------------------------------------

    df["fleet_status"] = (
        df["fleet_status"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
    )

    # Standardize known status values
    status_mapping = {
        "in service": "In Service",
        "under maintenance": "Maintenance",
        "maintenance": "Maintenance",
        "idle": "Idle",
        "retired": "Retired",
        "active": "Active"
    }

    df["fleet_status"] = (
        df["fleet_status"]
        .replace(status_mapping)
    )

    # ------------------------------------------------------
    # Standardize ownership
    # ------------------------------------------------------

    ownership_mapping = {
        "owned": "Owned",
        "leased": "Leased",
        "3pl partner": "3PL Partner"
    }

    df["ownership"] = (
        df["ownership"]
        .str.strip()
        .str.lower()
        .replace(ownership_mapping)
    )

    # ------------------------------------------------------
    # Convert numeric columns
    # ------------------------------------------------------

    numeric_columns = [
        "capacity_kg",
        "max_range_km",
        "avg_speed_kmph",
        "odometer_km",
        "battery_capacity_wh"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ------------------------------------------------------
    # Convert date columns
    # ------------------------------------------------------

    date_columns = [
        "purchase_date",
        "last_service_date",
        "insurance_expiry"
    ]

    for column in date_columns:

        df[column] = (
            df[column]
            .apply(clean_date)
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

    fleet_df = load_fleet_vehicles()

    fleet_df = clean_fleet_vehicles(
        fleet_df
    )

    print(
        "\nFleet vehicles dataset cleaned successfully."
    )

    print(
        "Shape:",
        fleet_df.shape
    )

    print("\nData types:")
    print(fleet_df.dtypes)

    print("\nFirst 5 records:")
    print(fleet_df.head())