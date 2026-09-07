import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

input_path = project_root / "data" / "raw" / "traffic_data.csv"
output_directory = project_root / "data" / "processed"

output_directory.mkdir(parents=True, exist_ok=True)

output_path = output_directory / "traffic_data_clean.csv"


# --------------------------------------------------
# 2. Load raw traffic data
# --------------------------------------------------

df = pd.read_csv(input_path)

print("Original shape:", df.shape)


# --------------------------------------------------
# 3. Remove duplicate rows
# --------------------------------------------------

duplicate_count = df.duplicated().sum()

df = df.drop_duplicates()

print("Duplicate rows removed:", duplicate_count)


# --------------------------------------------------
# 4. Clean record_date
# --------------------------------------------------

df["record_date"] = pd.to_datetime(
    df["record_date"],
    errors="coerce"
)


# --------------------------------------------------
# 5. Clean hour_of_day
# --------------------------------------------------

# Convert values like "7:00" to "7"
df["hour_of_day"] = (
    df["hour_of_day"]
    .astype(str)
    .str.split(":")
    .str[0]
)

df["hour_of_day"] = pd.to_numeric(
    df["hour_of_day"],
    errors="coerce"
)


# --------------------------------------------------
# 6. Clean avg_speed_kmph
# --------------------------------------------------

df["avg_speed_kmph"] = pd.to_numeric(
    df["avg_speed_kmph"],
    errors="coerce"
)

missing_speed = df["avg_speed_kmph"].isna().sum()

# Fill missing speed with median
speed_median = df["avg_speed_kmph"].median()

df["avg_speed_kmph"] = df["avg_speed_kmph"].fillna(
    speed_median
)

print("Missing speeds filled:", missing_speed)
print("Speed median used:", speed_median)


# --------------------------------------------------
# 7. Clean incident_reported
# --------------------------------------------------

df["incident_reported"] = (
    df["incident_reported"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["incident_reported"] = df["incident_reported"].replace({
    "1": "yes",
    "true": "yes",
    "yes": "yes",
    "0": "no",
    "false": "no",
    "no": "no"
})


# --------------------------------------------------
# 8. Clean road_closure
# --------------------------------------------------

df["road_closure"] = (
    df["road_closure"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df["road_closure"] = df["road_closure"].replace({
    "y": "yes",
    "n": "no"
})

df["road_closure"] = df["road_closure"].fillna("unknown")


# --------------------------------------------------
# 9. Convert congestion_index to numeric
# --------------------------------------------------

df["congestion_index"] = pd.to_numeric(
    df["congestion_index"],
    errors="coerce"
)


# --------------------------------------------------
# 10. Final validation
# --------------------------------------------------

print("\nCleaning completed successfully!")

print("\nFinal shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nIncident reported values:")
print(df["incident_reported"].value_counts())

print("\nRoad closure values:")
print(df["road_closure"].value_counts())


# --------------------------------------------------
# 11. Save cleaned data
# --------------------------------------------------

df.to_csv(output_path, index=False)

print("\nSaved file:")
print(output_path)