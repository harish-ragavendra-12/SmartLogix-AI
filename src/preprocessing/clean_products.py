import re

import pandas as pd

from src.config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


# ==========================================================
# LOAD PRODUCT CATALOG
# ==========================================================

def load_products():
    """
    Load the raw product catalog JSON file.
    """

    file_path = RAW_DATA_DIR / "product_catalog.json"

    import json

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    products = data.get("products", [])

    return pd.DataFrame(products)


# ==========================================================
# FLATTEN PRODUCT DATA
# ==========================================================

def flatten_products(df):
    """
    Flatten nested product specifications and price fields.
    """

    flattened_products = []

    for product in df.to_dict("records"):

        specs = product.get("specs") or {}
        price = product.get("price") or {}
        dimensions = specs.get("dimensions_cm") or {}

        flattened_products.append(
            {
                "product_id": product.get("product_id"),
                "product_name": product.get("product_name"),
                "category": product.get("category"),
                "sub_category": product.get("sub_category"),

                "is_fragile": product.get("is_fragile"),
                "is_hazmat": product.get("is_hazmat"),
                "requires_cold_chain": product.get(
                    "requires_cold_chain"
                ),

                "stock_qty": product.get("stock_qty"),
                "avg_rating": product.get("avg_rating"),
                "tags": product.get("tags"),

                "launch_date": product.get("launch_date"),

                # Specs
                "weight_kg": specs.get("weight_kg"),
                "battery_included": specs.get(
                    "battery_included"
                ),

                # Dimensions
                "dimension_length_cm": dimensions.get(
                    "length"
                ),
                "dimension_width_cm": dimensions.get(
                    "width"
                ),
                "dimension_height_cm": dimensions.get(
                    "height"
                ),

                # Price
                "price_currency": price.get("currency"),
                "price_amount": price.get("amount"),
            }
        )

    return pd.DataFrame(flattened_products)


# ==========================================================
# CLEAN WEIGHT
# ==========================================================

def clean_weight(value):
    """
    Convert product weight to kilograms.
    """

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"", "-", "nan", "none"}:
        return None

    # Remove kg text and other non-numeric characters
    value = re.sub(r"[^0-9.\-]", "", value)

    try:
        numeric_value = float(value)
    except ValueError:
        return None

    # Weight cannot be negative or zero
    if numeric_value <= 0:
        return None

    return numeric_value


# ==========================================================
# CLEAN PRICE
# ==========================================================

def clean_price(value):
    """
    Convert price values to numeric INR.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in {"", "-", "nan", "none"}:
        return None

    value = (
        value
        .replace(",", "")
        .replace("₹", "")
        .strip()
    )

    try:
        numeric_value = float(value)
    except ValueError:
        return None

    if numeric_value <= 0:
        return None

    return numeric_value


# ==========================================================
# CLEAN PRODUCTS
# ==========================================================

def clean_products(df):
    """
    Clean and standardize the products dataset.
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
    # Remove duplicate products
    # ------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset="product_id"
    )

    after_duplicates = len(df)

    print(
        f"\nDuplicate products removed: "
        f"{before_duplicates - after_duplicates}"
    )

    # ------------------------------------------------------
    # Clean product ID
    # ------------------------------------------------------

    df["product_id"] = (
        df["product_id"]
        .astype("string")
        .str.strip()
    )

    # ------------------------------------------------------
    # Clean text columns
    # ------------------------------------------------------

    text_columns = [
        "product_name",
        "category",
        "sub_category"
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # ------------------------------------------------------
    # Standardize category
    # ------------------------------------------------------

    df["category"] = (
        df["category"]
        .str.lower()
        .str.title()
    )

    # ------------------------------------------------------
    # Clean currency
    # ------------------------------------------------------

    df["price_currency"] = (
        df["price_currency"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # ------------------------------------------------------
    # Clean price
    # ------------------------------------------------------

    df["price_amount"] = (
        df["price_amount"]
        .apply(clean_price)
    )

    df["price_amount"] = pd.to_numeric(
        df["price_amount"],
        errors="coerce"
    )

    # ------------------------------------------------------
    # Clean weight
    # ------------------------------------------------------

    df["weight_kg"] = (
        df["weight_kg"]
        .apply(clean_weight)
    )

    df["weight_kg"] = pd.to_numeric(
        df["weight_kg"],
        errors="coerce"
    )

    # ------------------------------------------------------
    # Clean dimensions
    # ------------------------------------------------------

    dimension_columns = [
        "dimension_length_cm",
        "dimension_width_cm",
        "dimension_height_cm"
    ]

    for column in dimension_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Dimensions cannot be zero or negative
        df.loc[
            df[column] <= 0,
            column
        ] = None

    # ------------------------------------------------------
    # Clean stock quantity
    # ------------------------------------------------------

    df["stock_qty"] = pd.to_numeric(
        df["stock_qty"],
        errors="coerce"
    )

    # Negative stock is invalid
    negative_stock = (
        df["stock_qty"] < 0
    ).sum()

    print(
        f"Invalid negative stock values: "
        f"{negative_stock}"
    )

    df.loc[
        df["stock_qty"] < 0,
        "stock_qty"
    ] = None

    # ------------------------------------------------------
    # Clean ratings
    # ------------------------------------------------------

    df["avg_rating"] = pd.to_numeric(
        df["avg_rating"],
        errors="coerce"
    )

    # Ratings must be between 0 and 5
    invalid_ratings = (
        (df["avg_rating"] < 0)
        | (df["avg_rating"] > 5)
    ).sum()

    print(
        f"Invalid rating values: "
        f"{invalid_ratings}"
    )

    df.loc[
        (df["avg_rating"] < 0)
        | (df["avg_rating"] > 5),
        "avg_rating"
    ] = None

    # ------------------------------------------------------
    # Standardize boolean columns
    # ------------------------------------------------------

    boolean_columns = [
        "is_fragile",
        "is_hazmat",
        "requires_cold_chain",
        "battery_included"
    ]

    for column in boolean_columns:

        df[column] = (
            df[column]
            .astype("boolean")
        )

    # ------------------------------------------------------
    # Standardize launch date
    # ------------------------------------------------------

    df["launch_date"] = pd.to_datetime(
        df["launch_date"],
        errors="coerce",
        dayfirst=True
    )

    # ------------------------------------------------------
    # Standardize tags
    # ------------------------------------------------------

    df["tags"] = df["tags"].apply(
        lambda x: x if isinstance(x, list) else []
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

    # Load raw catalog
    catalog_df = load_products()

    print(
        "Raw products loaded:",
        len(catalog_df)
    )

    # Flatten nested JSON
    products_df = flatten_products(
        catalog_df
    )

    # Clean products
    products_df = clean_products(
        products_df
    )

    print(
        "\nProducts dataset cleaned successfully."
    )

    print(
        "Shape:",
        products_df.shape
    )

    print("\nData types:")
    print(products_df.dtypes)

    print("\nCategories:")
    print(
        products_df["category"]
        .value_counts()
    )

    print("\nFirst 5 records:")
    print(
        products_df.head().to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # SAVE PROCESSED DATASET
    # ------------------------------------------------------

    output_path = (
        PROCESSED_DATA_DIR
        / "products_cleaned.csv"
    )

    products_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nCleaned products dataset saved to:"
    )

    print(output_path)