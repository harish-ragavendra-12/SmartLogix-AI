import pandas as pd

from src.config.config import RAW_DATA_DIR


# ==========================================================
# LOAD PRODUCT CATALOG
# ==========================================================

def load_products():
    """
    Load the raw product catalog JSON file.
    """

    file_path = RAW_DATA_DIR / "product_catalog.json"

    catalog_df = pd.read_json(file_path)

    return catalog_df


# ==========================================================
# FLATTEN PRODUCT DATA
# ==========================================================

def flatten_products(catalog_df):
    """
    Extract product records and flatten nested fields.
    """

    products = catalog_df["products"].tolist()

    flattened_products = []

    for product in products:

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
    # Clean price currency
    # ------------------------------------------------------

    df["price_currency"] = (
        df["price_currency"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # ------------------------------------------------------
    # Clean price amount
    # ------------------------------------------------------

    df["price_amount"] = (
        df["price_amount"]
        .astype("string")
        .str.replace(",", "", regex=False)
    )

    df["price_amount"] = pd.to_numeric(
        df["price_amount"],
        errors="coerce"
    )

    # ------------------------------------------------------
    # Convert weight
    # ------------------------------------------------------

    df["weight_kg"] = pd.to_numeric(
        df["weight_kg"],
        errors="coerce"
    )

    # ------------------------------------------------------
    # Convert dimensions
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

        df[column] = df[column].astype("boolean")

    # ------------------------------------------------------
    # Convert stock quantity
    # ------------------------------------------------------

    df["stock_qty"] = pd.to_numeric(
        df["stock_qty"],
        errors="coerce"
    )

    # ------------------------------------------------------
    # Convert average rating
    # ------------------------------------------------------

    df["avg_rating"] = pd.to_numeric(
        df["avg_rating"],
        errors="coerce"
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

    catalog_df = load_products()

    products_df = flatten_products(catalog_df)

    products_df = clean_products(products_df)

    print("\nProducts dataset cleaned successfully.")

    print("Shape:", products_df.shape)

    print("\nData types:")
    print(products_df.dtypes)

    print("\nFirst 5 records:")
    print(products_df.head())
