import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

input_path = project_root / "data" / "raw" / "customer_reviews.jsonl"

output_directory = project_root / "data" / "processed"
output_directory.mkdir(parents=True, exist_ok=True)

output_path = output_directory / "customer_reviews_clean.csv"


# --------------------------------------------------
# 2. Load the JSONL file
# --------------------------------------------------

df = pd.read_json(input_path, lines=True)

print("Original shape:", df.shape)


# --------------------------------------------------
# 3. Remove duplicate rows
# --------------------------------------------------

duplicate_count = df.duplicated().sum()

df = df.drop_duplicates()

print("Duplicate rows removed:", duplicate_count)


# --------------------------------------------------
# 4. Clean order_id
# --------------------------------------------------

missing_order_id = df["order_id"].isna().sum()

df["order_id"] = df["order_id"].fillna("UNKNOWN")

print("Missing order IDs filled:", missing_order_id)


# --------------------------------------------------
# 5. Clean rating
# --------------------------------------------------

# --------------------------------------------------
# 5. Clean rating
# --------------------------------------------------

df["rating"] = (
    df["rating"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Remove "/5"
df["rating"] = df["rating"].str.replace(
    "/5", "", regex=False
)

# Remove "stars"
df["rating"] = df["rating"].str.replace(
    "stars", "", regex=False
)

# Remove "star"
df["rating"] = df["rating"].str.replace(
    "star", "", regex=False
)

# Convert text ratings to numbers
df["rating"] = df["rating"].replace({
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
})

# Convert rating to numeric
df["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce"
)


# --------------------------------------------------
# 6. Clean review title
# --------------------------------------------------

df["review_title"] = df["review_title"].fillna(
    "No title"
)


# --------------------------------------------------
# 7. Clean review text
# --------------------------------------------------

df["review_text"] = df["review_text"].fillna(
    "No review text"
)


# --------------------------------------------------
# 8. Clean review date
# --------------------------------------------------

def parse_review_date(value):

    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    # Unix timestamp
    if value.isdigit() and len(value) == 10:
        return pd.to_datetime(
            int(value),
            unit="s",
            utc=True
        )

    # Other date formats
    return pd.to_datetime(
        value,
        errors="coerce",
        format="mixed",
        dayfirst=True,
        utc=True
    )


df["review_date"] = df["review_date"].apply(
    parse_review_date
)


# --------------------------------------------------
# 9. Clean verified_purchase
# --------------------------------------------------

df["verified_purchase"] = (
    df["verified_purchase"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["verified_purchase"] = df["verified_purchase"].replace({
    "1": "yes",
    "true": "yes",
    "yes": "yes",
    "0": "no",
    "false": "no",
    "no": "no"
})


# --------------------------------------------------
# 10. Clean helpful_votes
# --------------------------------------------------

df["helpful_votes"] = pd.to_numeric(
    df["helpful_votes"],
    errors="coerce"
)

df["helpful_votes"] = df["helpful_votes"].fillna(0)


# --------------------------------------------------
# 11. Clean delivery mode
# --------------------------------------------------

df["delivery_mode_experienced"] = (
    df["delivery_mode_experienced"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df["delivery_mode_experienced"] = (
    df["delivery_mode_experienced"]
    .fillna("unknown")
)


# --------------------------------------------------
# 12. Clean sentiment label
# --------------------------------------------------

df["sentiment_label"] = (
    df["sentiment_label"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df["sentiment_label"] = (
    df["sentiment_label"]
    .fillna("unknown")
)


# --------------------------------------------------
# 13. Validation
# --------------------------------------------------

print("\nCleaning completed successfully!")

print("\nFinal shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nRatings:")
print(
    df["rating"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nVerified purchase:")
print(
    df["verified_purchase"]
    .value_counts(dropna=False)
)

print("\nDelivery modes:")
print(
    df["delivery_mode_experienced"]
    .value_counts(dropna=False)
)

print("\nSentiment labels:")
print(
    df["sentiment_label"]
    .value_counts(dropna=False)
)


# --------------------------------------------------
# 14. Save cleaned data
# --------------------------------------------------

df.to_csv(output_path, index=False)

print("\nSaved file:")
print(output_path)