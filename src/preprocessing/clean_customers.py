import pandas as pd

from src.config.config import RAW_DATA_DIR


def load_customers():
    """
    Load the raw customers dataset.
    """

    file_path = RAW_DATA_DIR / "customers.csv"

    df = pd.read_csv(file_path)

    return df


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
    # Remove duplicate customers
    # ------------------------------------------------------

    df = df.drop_duplicates(
        subset="customer_id"
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
        "state",
        "customer_segment"
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

    df["signup_date"] = pd.to_datetime(
        df["signup_date"],
        errors="coerce"
    )

    # ------------------------------------------------------
    # Standardize customer segment
    # ------------------------------------------------------

    df["customer_segment"] = (
        df["customer_segment"]
        .str.strip()
        .str.lower()
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
    # Check Missing Values
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

    customers_df = clean_customers(customers_df)

    print("Customers dataset cleaned successfully.")
    print("Shape:", customers_df.shape)
    print("\nData types:")
    print(customers_df.dtypes)
    print("\nFirst 5 records:")
    print(customers_df.head())
