import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

input_path = BASE_DIR / "data" / "raw" / "drone_telemetry.csv"
output_path = BASE_DIR / "data" / "processed" / "drone_telemetry_cleaned.csv"

df = pd.read_csv(input_path)


# 1. Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
after = len(df)

print("Duplicate rows removed:", before - after)


# 2. Clean flight timestamp

# Replace invalid placeholder values with missing values
df["flight_timestamp"] = df["flight_timestamp"].replace(
    ["unknown", "-", ""], pd.NA
)

# Convert timestamp column to string
timestamp = df["flight_timestamp"].astype("string").str.strip()

# Create empty datetime column
clean_timestamp = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


# A. Unix timestamps (10-digit numbers)
unix_mask = timestamp.str.fullmatch(r"\d{10}", na=False)

clean_timestamp.loc[unix_mask] = pd.to_datetime(
    timestamp.loc[unix_mask].astype("int64"),
    unit="s",
    errors="coerce"
)


# B. Normal date/time formats
normal_mask = ~unix_mask & timestamp.notna()

clean_timestamp.loc[normal_mask] = pd.to_datetime(
    timestamp.loc[normal_mask],
    errors="coerce",
    dayfirst=True,
    format="mixed",
    utc=True
).dt.tz_localize(None)


# Replace original column
df["flight_timestamp"] = clean_timestamp


# 3. Fill missing flight duration with median
median_duration = df["flight_duration_min"].median()

df["flight_duration_min"] = df["flight_duration_min"].fillna(
    median_duration
)


# 4. Clean payload values containing "kg"

df["payload_kg"] = (
    df["payload_kg"]
    .astype(str)
    .str.replace("kg", "", regex=False)
)

df["payload_kg"] = pd.to_numeric(
    df["payload_kg"],
    errors="coerce"
)


# 5. Convert other numeric columns

numeric_columns = [
    "battery_end_pct",
    "motor_temp_c",
    "vibration_rms",
    "rotor_rpm_avg"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# 6. Standardize GPS signal quality

df["gps_signal_quality"] = (
    df["gps_signal_quality"]
    .str.strip()
    .str.title()
)


# 7. Handle missing error codes

df["error_codes"] = df["error_codes"].fillna("NONE")


# 8. Save cleaned dataset

df.to_csv(output_path, index=False)


# 9. Final information

print("\nCleaned dataset shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nData types after cleaning:")
print(df.dtypes)

print("\nCleaned file saved to:")
print(output_path)