import re
from pathlib import Path

import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

input_path = (
    BASE_DIR
    / "data"
    / "raw"
    / "drone_telemetry.csv"
)

output_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "drone_telemetry_cleaned.csv"
)


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(input_path)

print("Raw dataset shape:", df.shape)


# ==========================================================
# STANDARDIZE COLUMN NAMES
# ==========================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)


# ==========================================================
# REMOVE DUPLICATE ROWS
# ==========================================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print(
    "\nDuplicate rows removed:",
    before - after
)


# ==========================================================
# CLEAN FLIGHT TIMESTAMP
# ==========================================================

df["flight_timestamp"] = (
    df["flight_timestamp"]
    .astype("string")
    .str.strip()
)

# Invalid placeholders
df["flight_timestamp"] = (
    df["flight_timestamp"]
    .replace(
        {
            "": pd.NA,
            "-": pd.NA,
            "unknown": pd.NA,
            "Unknown": pd.NA,
            "UNKNOWN": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
        }
    )
)

timestamp = df["flight_timestamp"]

clean_timestamp = pd.Series(
    pd.NaT,
    index=df.index,
    dtype="datetime64[ns]"
)


# ----------------------------------------------------------
# Unix timestamps
# ----------------------------------------------------------

unix_mask = timestamp.str.fullmatch(
    r"\d{10}",
    na=False
)

clean_timestamp.loc[unix_mask] = (
    pd.to_datetime(
        timestamp.loc[unix_mask]
        .astype("int64"),
        unit="s",
        errors="coerce"
    )
)


# ----------------------------------------------------------
# Normal date/time values
# ----------------------------------------------------------

normal_mask = (
    ~unix_mask
    & timestamp.notna()
)

parsed_normal = pd.to_datetime(
    timestamp.loc[normal_mask],
    errors="coerce",
    format="mixed",
    dayfirst=True,
    utc=True
)

clean_timestamp.loc[normal_mask] = (
    parsed_normal
    .dt.tz_localize(None)
)


df["flight_timestamp"] = clean_timestamp


# ==========================================================
# CLEAN NUMERIC COLUMNS
# ==========================================================

numeric_columns = [
    "flight_duration_min",
    "payload_kg",
    "battery_end_pct",
    "motor_temp_c",
    "vibration_rms",
    "rotor_rpm_avg"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ==========================================================
# CLEAN PAYLOAD
# ==========================================================

# Handle values such as:
# 25 kg
# 25kg
# 25000 g
# 1,250 kg

def clean_payload(value):

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {
        "",
        "-",
        "unknown",
        "none",
        "nan"
    }:
        return None

    value = value.replace(",", "")

    try:

        if "kg" in value:

            numeric_value = float(
                re.sub(
                    r"[^0-9.\-]",
                    "",
                    value
                )
            )

            return (
                numeric_value
                if numeric_value > 0
                else None
            )

        if "g" in value:

            numeric_value = float(
                re.sub(
                    r"[^0-9.\-]",
                    "",
                    value
                )
            )

            return (
                numeric_value / 1000
                if numeric_value > 0
                else None
            )

        return float(value)

    except ValueError:
        return None


df["payload_kg"] = (
    df["payload_kg"]
    .apply(clean_payload)
)

df["payload_kg"] = pd.to_numeric(
    df["payload_kg"],
    errors="coerce"
)


# ==========================================================
# VALIDATE TELEMETRY VALUES
# ==========================================================

# Battery percentage must be 0–100

invalid_battery = (
    (df["battery_end_pct"] < 0)
    | (df["battery_end_pct"] > 100)
).sum()

print(
    "Invalid battery percentage values:",
    invalid_battery
)

df.loc[
    (df["battery_end_pct"] < 0)
    | (df["battery_end_pct"] > 100),
    "battery_end_pct"
] = None


# ----------------------------------------------------------
# Flight duration
# ----------------------------------------------------------

invalid_duration = (
    df["flight_duration_min"] <= 0
).sum()

print(
    "Invalid flight duration values:",
    invalid_duration
)

df.loc[
    df["flight_duration_min"] <= 0,
    "flight_duration_min"
] = None


# Fill missing duration with median

median_duration = (
    df["flight_duration_min"]
    .median()
)

df["flight_duration_min"] = (
    df["flight_duration_min"]
    .fillna(median_duration)
)


# ----------------------------------------------------------
# Payload
# ----------------------------------------------------------

invalid_payload = (
    df["payload_kg"] < 0
).sum()

print(
    "Invalid payload values:",
    invalid_payload
)

df.loc[
    df["payload_kg"] < 0,
    "payload_kg"
] = None


# ----------------------------------------------------------
# Motor temperature
# ----------------------------------------------------------

invalid_temperature = (
    df["motor_temp_c"] < 0
).sum()

print(
    "Invalid motor temperature values:",
    invalid_temperature
)

df.loc[
    df["motor_temp_c"] < 0,
    "motor_temp_c"
] = None


# ----------------------------------------------------------
# Vibration
# ----------------------------------------------------------

invalid_vibration = (
    df["vibration_rms"] < 0
).sum()

print(
    "Invalid vibration values:",
    invalid_vibration
)

df.loc[
    df["vibration_rms"] < 0,
    "vibration_rms"
] = None


# ----------------------------------------------------------
# Rotor RPM
# ----------------------------------------------------------

invalid_rpm = (
    df["rotor_rpm_avg"] < 0
).sum()

print(
    "Invalid rotor RPM values:",
    invalid_rpm
)

df.loc[
    df["rotor_rpm_avg"] < 0,
    "rotor_rpm_avg"
] = None


# ==========================================================
# STANDARDIZE GPS SIGNAL QUALITY
# ==========================================================

df["gps_signal_quality"] = (
    df["gps_signal_quality"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df["gps_signal_quality"] = (
    df["gps_signal_quality"]
    .replace(
        {
            "": pd.NA,
            "-": pd.NA,
            "unknown": pd.NA,
            "none": pd.NA,
            "nan": pd.NA
        }
    )
    .str.title()
)


# ==========================================================
# CLEAN ERROR CODES
# ==========================================================

df["error_codes"] = (
    df["error_codes"]
    .astype("string")
    .str.strip()
    .str.upper()
)

df["error_codes"] = (
    df["error_codes"]
    .replace(
        {
            "": pd.NA,
            "-": pd.NA,
            "UNKNOWN": pd.NA,
            "NONE": pd.NA,
            "NAN": pd.NA
        }
    )
    .fillna("NONE")
)


# ==========================================================
# RESET INDEX
# ==========================================================

df = df.reset_index(drop=True)


# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# SAVE CLEANED DATA
# ==========================================================

df.to_csv(
    output_path,
    index=False
)


# ==========================================================
# FINAL VALIDATION
# ==========================================================

print(
    "\nCleaned dataset shape:",
    df.shape
)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nData types after cleaning:")
print(df.dtypes)

print("\nCleaned dataset preview:")
print(
    df.head().to_string(
        index=False
    )
)

print("\nCleaned file saved to:")
print(output_path)