import pandas as pd

from src.config.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)


# ==========================================================
# LOAD CUSTOMERS
# ==========================================================

def load_customers():
    """
    Load the raw customers dataset.
    """

    file_path = RAW_DATA_DIR / "customers.csv"

    df = pd.read_csv(file_path)

    return df


# ==========================================================
# CLEAN SIGNUP DATE
# ==========================================================

def clean_signup_date(value):
    """
    Convert multiple signup date formats into a common
    datetime format.
    """

    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    if value.lower() in {"", "-", "nan", "none", "unknown"}:
        return pd.NaT

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",

        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",

        "%d-%m-%Y %H:%M",
        "%d-%m-%Y",
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
# CLEAN CUSTOMERS
# ==========================================================

def clean_customers(df):
    """
    Clean and standardize the customers dataset.
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
    # Remove exact duplicate customer records
    # ------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset="customer_id"
    )

    after_duplicates = len(df)

    print(
        f"\nDuplicate customer records removed: "
        f"{before_duplicates - after_duplicates}"
    )

    # ------------------------------------------------------
    # Clean customer ID
    # ------------------------------------------------------

    df["customer_id"] = (
        df["customer_id"]
        .astype("string")
        .str.strip()
    )

    # ------------------------------------------------------
    # Clean text columns
    # ------------------------------------------------------

    text_columns = [
        "customer_name",
        "city",
        "state"
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # ------------------------------------------------------
    # Clean email
    # ------------------------------------------------------

    df["email"] = (
        df["email"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # ------------------------------------------------------
    # Clean phone numbers
    # ------------------------------------------------------

    df["phone"] = (
        df["phone"]
        .astype("string")
        .str.strip()
    )

    # ------------------------------------------------------
    # Clean pincode
    # ------------------------------------------------------

    df["pincode"] = (
        df["pincode"]
        .astype("string")
        .str.strip()
    )

    # ------------------------------------------------------
    # Standardize signup date
    # ------------------------------------------------------

    df["signup_date"] = (
        df["signup_date"]
        .apply(clean_signup_date)
    )

    # ------------------------------------------------------
    # Standardize customer segment
    # ------------------------------------------------------

    df["customer_segment"] = (
        df["customer_segment"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # ------------------------------------------------------
    # Standardize prime membership
    # ------------------------------------------------------

    prime_mapping = {
        "yes": True,
        "y": True,
        "true": True,
        "1": True,

        "no": False,
        "n": False,
        "false": False,
        "0": False
    }

    df["is_prime_member"] = (
        df["is_prime_member"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(prime_mapping)
    )

    # ------------------------------------------------------
    # Convert lifetime orders to numeric
    # ------------------------------------------------------

    df["lifetime_orders"] = pd.to_numeric(
        df["lifetime_orders"],
        errors="coerce"
    )

    # ------------------------------------------------------
    # Convert average rating to numeric
    # ------------------------------------------------------

    df["avg_rating_given"] = pd.to_numeric(
        df["avg_rating_given"],
        errors="coerce"
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

    customers_df = load_customers()

    customers_df = clean_customers(
        customers_df
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
        "customers_cleaned.csv"
    )

    customers_df.to_csv(
        output_path,
        index=False
    )

    # ------------------------------------------------------
    # Validation output
    # ------------------------------------------------------

    print(
        "\nCustomers dataset cleaned successfully."
    )

    print(
        "Shape:",
        customers_df.shape
    )

    print(
        "Cleaned dataset saved to:",
        output_path
    )

    print("\nData types:")
    print(customers_df.dtypes)

    print("\nCustomer segments:")
    print(
        customers_df["customer_segment"]
        .value_counts(dropna=False)
    )

    print("\nPrime membership:")
    print(
        customers_df["is_prime_member"]
        .value_counts(dropna=False)
    )

    print("\nFirst 5 records:")
    print(customers_df.head())