import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

input_path = project_root / "data" / "raw" / "weather_data.csv"
output_directory = project_root / "data" / "processed"

output_directory.mkdir(parents=True, exist_ok=True)

output_path = output_directory / "weather_data_clean.csv"


# --------------------------------------------------
# 2. Load raw weather data
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
# 4. Clean Date
# --------------------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
    format="mixed",
    dayfirst=True
)


# --------------------------------------------------
# 5. Convert temperature to numeric
# --------------------------------------------------

df["temp"] = pd.to_numeric(
    df["temp"],
    errors="coerce"
)


# --------------------------------------------------
# 6. Clean temperature units
# --------------------------------------------------

df["Temp Unit"] = (
    df["Temp Unit"]
    .astype("string")
    .str.strip()
    .str.upper()
)


# For missing temperature units:
# Values <= 45 are treated as Celsius.
# Values > 45 are treated as Fahrenheit.

missing_unit = df["Temp Unit"].isna()

df.loc[
    missing_unit & (df["temp"] <= 45),
    "Temp Unit"
] = "C"

df.loc[
    missing_unit & (df["temp"] > 45),
    "Temp Unit"
] = "F"


# --------------------------------------------------
# 7. Create standardized temperature in Celsius
# --------------------------------------------------

df["temp_celsius"] = df["temp"]

fahrenheit_mask = df["Temp Unit"] == "F"

df.loc[fahrenheit_mask, "temp_celsius"] = (
    (df.loc[fahrenheit_mask, "temp"] - 32) * 5 / 9
)


# --------------------------------------------------
# 8. Fill missing temperature values
# --------------------------------------------------

missing_temperature = df["temp_celsius"].isna()

temperature_median = df["temp_celsius"].median()

df.loc[missing_temperature, "temp_celsius"] = temperature_median

print("Missing temperatures filled:", missing_temperature.sum())
print("Temperature median used:", temperature_median)


# --------------------------------------------------
# 9. Clean humidity
# --------------------------------------------------

df["humidity_%"] = pd.to_numeric(
    df["humidity_%"],
    errors="coerce"
)

missing_humidity = df["humidity_%"].isna()

humidity_median = df["humidity_%"].median()

df["humidity_%"] = df["humidity_%"].fillna(
    humidity_median
)

print("Missing humidity values filled:", missing_humidity.sum())
print("Humidity median used:", humidity_median)


# --------------------------------------------------
# 10. Clean wind speed
# --------------------------------------------------

df["wind_speed_kmph"] = pd.to_numeric(
    df["wind_speed_kmph"],
    errors="coerce"
)

missing_wind = df["wind_speed_kmph"].isna()

wind_median = df["wind_speed_kmph"].median()

df["wind_speed_kmph"] = df["wind_speed_kmph"].fillna(
    wind_median
)

print("Missing wind speed values filled:", missing_wind.sum())
print("Wind speed median used:", wind_median)


# --------------------------------------------------
# 11. Clean weather condition
# --------------------------------------------------

df["condition"] = (
    df["condition"]
    .astype("string")
    .str.strip()
    .str.lower()
)


# --------------------------------------------------
# 12. Clean storm alert
# --------------------------------------------------

df["storm_alert"] = (
    df["storm_alert"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df["storm_alert"] = df["storm_alert"].replace({
    "y": "yes",
    "yes": "yes",
    "n": "no",
    "no": "no"
})

df["storm_alert"] = df["storm_alert"].fillna("unknown")


# --------------------------------------------------
# 13. Final validation
# --------------------------------------------------

print("\nCleaning completed successfully!")

print("\nFinal shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTemperature units:")
print(df["Temp Unit"].value_counts(dropna=False))

print("\nWeather conditions:")
print(df["condition"].value_counts())

print("\nStorm alert:")
print(df["storm_alert"].value_counts())


# --------------------------------------------------
# 14. Save cleaned data
# --------------------------------------------------

df.to_csv(output_path, index=False)

print("\nSaved file:")
print(output_path)