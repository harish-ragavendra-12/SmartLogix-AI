import pandas as pd

from src.data_ingestion.data_loader import load_csv


# --------------------------------------------------
# 1. Load raw maintenance data
# --------------------------------------------------

df = load_csv("maintenance_history.csv")

print("Original shape:", df.shape)


# --------------------------------------------------
# 2. Remove duplicate rows
# --------------------------------------------------

duplicates = df.duplicated().sum()

df = df.drop_duplicates()

print("Duplicate rows removed:", duplicates)


# --------------------------------------------------
# 3. Clean service_date
# --------------------------------------------------

df["service_date"] = pd.to_datetime(
    df["service_date"],
    errors="coerce",
    format="mixed",
    dayfirst=True
)


# --------------------------------------------------
# 4. Clean labour_hours
# --------------------------------------------------

df["labour_hours"] = pd.to_numeric(
    df["labour_hours"],
    errors="coerce"
)

labour_median = df["labour_hours"].median()

df["labour_hours"] = df["labour_hours"].fillna(labour_median)


# --------------------------------------------------
# 5. Clean cost_inr
# --------------------------------------------------

df["cost_inr"] = (
    df["cost_inr"]
    .astype(str)
    .str.replace("INR", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["cost_inr"] = pd.to_numeric(
    df["cost_inr"],
    errors="coerce"
)


# --------------------------------------------------
# 6. Clean downtime_hours
# --------------------------------------------------

df["downtime_hours"] = pd.to_numeric(
    df["downtime_hours"],
    errors="coerce"
)


# --------------------------------------------------
# 7. Clean parts_replaced
# --------------------------------------------------

df["parts_replaced"] = df["parts_replaced"].fillna("None")


# --------------------------------------------------
# 8. Clean failure_reported
# --------------------------------------------------

df["failure_reported"] = (
    df["failure_reported"]
    .astype("string")
    .str.lower()
    .str.strip()
)

df["failure_reported"] = df["failure_reported"].replace(
    {
        "y": "yes",
        "n": "no"
    }
)

df["failure_reported"] = df["failure_reported"].fillna("unknown")


# --------------------------------------------------
# 9. Save cleaned data
# --------------------------------------------------

output_path = r"D:\Data Science\SmartLogix_AI\data\processed\maintenance_history_clean.csv"

df.to_csv(output_path, index=False)


# --------------------------------------------------
# 10. Final validation
# --------------------------------------------------

print("\nCleaning completed successfully!")

print("Final shape:", df.shape)

print("\nFinal data types:")
print(df.dtypes)

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nFailure reported values:")
print(df["failure_reported"].value_counts())

print("\nSaved to:")
print(output_path)