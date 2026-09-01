import pandas as pd

from src.config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


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

    if value.lower() in {"", "-", "nan", "none"}:
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
    # Different date formats found in the dataset
    # ------------------------------------------------------

    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",

        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y",

        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",

        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",

        "%b %d %Y %I:%M %p",
        "%B %d %Y %I:%M %p",
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
# CLEAN EVENT TYPE
# ==========================================================

def clean_event_type(value):
    """
    Standardize event type values.

    Example:
        Order Placed -> ORDER_PLACED
        Picked Up    -> PICKED_UP
    """

    if pd.isna(value):
        return pd.NA

    value = str(value).strip().upper()

    # Replace spaces with underscores
    value = value.replace(" ", "_")

    return value


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
    # Clean event type
    # ------------------------------------------------------

    df["event_type"] = (
        df["event_type"]
        .apply(clean_event_type)
        .astype("string")
    )

    # ------------------------------------------------------
    # Clean categorical fields
    # ------------------------------------------------------

    categorical_columns = [
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

    # ------------------------------------------------------
    # Save cleaned dataset
    # ------------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PROCESSED_DATA_DIR /
        "delivery_logs_cleaned.csv"
    )

    delivery_logs_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nDelivery logs dataset cleaned successfully."
    )

    print(
        "Shape:",
        delivery_logs_df.shape
    )

    print(
        "Cleaned dataset saved to:",
        output_path
    )

    print("\nData types:")
    print(delivery_logs_df.dtypes)

    print("\nFirst 5 records:")
    print(delivery_logs_df.head())

    print("\nEvent types after cleaning:")
    print(
        delivery_logs_df["event_type"]
        .value_counts(dropna=False)
    )
