import pandas as pd

from src.config.config import RAW_DATA_DIR


# ==========================================================
# LOAD DELIVERY LOGS
# ==========================================================

def load_delivery_logs():
    """
    Load the raw delivery logs dataset.
    """

    file_path = RAW_DATA_DIR / "delivery_logs.csv"

    df = pd.read_csv(file_path)

    return df


# ==========================================================
# CLEAN EVENT TIMESTAMP
# ==========================================================

def clean_event_timestamp(value):
    """
    Convert Unix timestamps and multiple string timestamp
    formats into a common datetime format.
    """

    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    if value in {"", "-", "nan", "none"}:
        return pd.NaT

    # ------------------------------------------------------
    # Unix timestamp
    # ------------------------------------------------------

    if value.isdigit():
        try:
            return pd.to_datetime(
                int(value),
                unit="s",
                errors="coerce"
            )
        except (ValueError, TypeError):
            return pd.NaT

    # ------------------------------------------------------
    # ISO / standard datetime formats
    # ------------------------------------------------------

    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ]

    for date_format in formats:

        try:
            parsed_date = pd.to_datetime(
                value,
                format=date_format,
                errors="coerce",
                utc=True
            )

            if not pd.isna(parsed_date):
                return parsed_date.tz_localize(None)

        except (ValueError, TypeError):
            continue

    return pd.NaT


# ==========================================================
# CLEAN DELIVERY LOGS
# ==========================================================

def clean_delivery_logs(df):
    """
    Clean and standardize the delivery logs dataset.
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
        "log_id",
        "order_id",
        "hub_code",
        "vehicle_id",
        "scanned_by_emp"
    ]

    for column in id_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # ------------------------------------------------------
    # Clean event sequence
    # ------------------------------------------------------

    df["event_seq"] = pd.to_numeric(
        df["event_seq"],
        errors="coerce"
    )

    # ------------------------------------------------------
    # Clean event timestamp
    # ------------------------------------------------------

    df["event_timestamp"] = (
        df["event_timestamp"]
        .apply(clean_event_timestamp)
    )

    # ------------------------------------------------------
    # Clean categorical fields
    # ------------------------------------------------------

    categorical_columns = [
        "event_type",
        "location_city",
        "exception_code"
    ]

    for column in categorical_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # ------------------------------------------------------
    # Clean remarks
    # ------------------------------------------------------

    df["remarks"] = (
        df["remarks"]
        .astype("string")
        .str.strip()
    )

    # Convert empty strings to missing values
    df["remarks"] = df["remarks"].replace(
        "",
        pd.NA
    )

    df["exception_code"] = df["exception_code"].replace(
        "",
        pd.NA
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

    delivery_logs_df = load_delivery_logs()

    delivery_logs_df = clean_delivery_logs(
        delivery_logs_df
    )

    print(
        "\nDelivery logs dataset cleaned successfully."
    )

    print(
        "Shape:",
        delivery_logs_df.shape
    )

    print("\nData types:")
    print(delivery_logs_df.dtypes)

    print("\nFirst 5 records:")
    print(delivery_logs_df.head())