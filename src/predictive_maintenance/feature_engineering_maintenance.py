import pandas as pd
import numpy as np
from pathlib import Path

from src.config.config import PROCESSED_DATA_DIR


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE_CANDIDATES = [
    PROCESSED_DATA_DIR / "maintenance_history_clean.csv",
    PROCESSED_DATA_DIR / "maintenance_history_cleaned.csv"
]

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "maintenance_features.csv"
)

TARGET_COLUMN = "failure_reported"


# ============================================================
# FIND INPUT FILE
# ============================================================

def find_input_file():

    print("\n" + "=" * 70)
    print("SEARCHING FOR MAINTENANCE DATA")
    print("=" * 70)

    for file_path in INPUT_FILE_CANDIDATES:

        if file_path.exists():

            print(
                f"Input file found:\n"
                f"{file_path}"
            )

            return file_path

    raise FileNotFoundError(
        "Maintenance history cleaned file was not found."
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    input_file = find_input_file()

    df = pd.read_csv(input_file)

    print("\n" + "=" * 70)
    print("LOADING MAINTENANCE DATA")
    print("=" * 70)

    print(f"Shape : {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    return df


# ============================================================
# BASIC DATA CLEANING
# ============================================================

def clean_basic_data(df):

    print("\n" + "=" * 70)
    print("BASIC DATA CLEANING")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        column.strip()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    numerical_columns = [
        "labour_hours",
        "cost_inr",
        "downtime_hours"
    ]

    for column in numerical_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Convert service date
    # --------------------------------------------------------

    if "service_date" in df.columns:

        df["service_date"] = pd.to_datetime(
            df["service_date"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Clean categorical columns
    # --------------------------------------------------------

    categorical_columns = [
        "service_type",
        "parts_replaced",
        "failure_reported"
    ]

    for column in categorical_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .str.lower()
            )

    print(
        f"Rows after basic cleaning : "
        f"{len(df)}"
    )

    return df


# ============================================================
# NORMALIZE TARGET
# ============================================================

def normalize_target(df):

    print("\n" + "=" * 70)
    print("NORMALIZING FAILURE TARGET")
    print("=" * 70)

    df = df.copy()

    target_mapping = {

        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,

        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0

    }

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(target_mapping)
    )

    print("\nTarget distribution before removing unknowns:")

    print(
        df[TARGET_COLUMN]
        .value_counts(dropna=False)
    )

    # --------------------------------------------------------
    # Remove records where target cannot be determined
    # --------------------------------------------------------

    before_rows = len(df)

    df = df.dropna(
        subset=[TARGET_COLUMN]
    ).copy()

    removed_rows = (
        before_rows - len(df)
    )

    print(
        f"\nRows removed because target "
        f"was unknown/missing : {removed_rows}"
    )

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(int)
    )

    print("\nFinal target distribution:")

    print(
        df[TARGET_COLUMN]
        .value_counts()
    )

    return df


# ============================================================
# CREATE DATE FEATURES
# ============================================================

def create_date_features(df):

    print("\n" + "=" * 70)
    print("CREATING DATE FEATURES")
    print("=" * 70)

    df = df.copy()

    if "service_date" not in df.columns:

        print(
            "service_date column not found."
        )

        return df

    df["service_year"] = (
        df["service_date"].dt.year
    )

    df["service_month"] = (
        df["service_date"].dt.month
    )

    df["service_day"] = (
        df["service_date"].dt.day
    )

    df["service_day_of_week"] = (
        df["service_date"].dt.dayofweek
    )

    df["service_week"] = (
        df["service_date"]
        .dt.isocalendar()
        .week
        .astype("Int64")
    )

    df["is_weekend"] = (
        df["service_day_of_week"] >= 5
    ).astype(int)

    print(
        "Created date-based maintenance features."
    )

    return df


# ============================================================
# CREATE VEHICLE MAINTENANCE FEATURES
# ============================================================

def create_vehicle_maintenance_features(df):

    print("\n" + "=" * 70)
    print("CREATING VEHICLE MAINTENANCE FEATURES")
    print("=" * 70)

    df = df.copy()

    if "vehicle_id" not in df.columns:

        print(
            "vehicle_id column not found."
        )

        return df

    # --------------------------------------------------------
    # Historical maintenance frequency
    # --------------------------------------------------------

    df["vehicle_maintenance_count"] = (
        df.groupby("vehicle_id")["vehicle_id"]
        .transform("count")
    )

    # --------------------------------------------------------
    # Historical average maintenance cost
    # --------------------------------------------------------

    if "cost_inr" in df.columns:

        df["vehicle_avg_maintenance_cost"] = (
            df.groupby("vehicle_id")["cost_inr"]
            .transform("mean")
        )

    # --------------------------------------------------------
    # Historical average downtime
    # --------------------------------------------------------

    if "downtime_hours" in df.columns:

        df["vehicle_avg_downtime_hours"] = (
            df.groupby("vehicle_id")[
                "downtime_hours"
            ]
            .transform("mean")
        )

    # --------------------------------------------------------
    # Historical average labour hours
    # --------------------------------------------------------

    if "labour_hours" in df.columns:

        df["vehicle_avg_labour_hours"] = (
            df.groupby("vehicle_id")[
                "labour_hours"
            ]
            .transform("mean")
        )

    print(
        "Created vehicle-level maintenance features."
    )

    return df


# ============================================================
# CREATE MAINTENANCE COST FEATURES
# ============================================================

def create_cost_features(df):

    print("\n" + "=" * 70)
    print("CREATING COST FEATURES")
    print("=" * 70)

    df = df.copy()

    if "cost_inr" in df.columns:

        df["cost_per_labour_hour"] = (
            df["cost_inr"]
            /
            df["labour_hours"].replace(
                0,
                np.nan
            )
        )

    if (
        "cost_inr" in df.columns
        and "downtime_hours" in df.columns
    ):

        df["cost_per_downtime_hour"] = (
            df["cost_inr"]
            /
            df["downtime_hours"].replace(
                0,
                np.nan
            )
        )

    print(
        "Created maintenance cost features."
    )

    return df


# ============================================================
# CREATE LABOUR / DOWNTIME FEATURES
# ============================================================

def create_operational_features(df):

    print("\n" + "=" * 70)
    print("CREATING OPERATIONAL FEATURES")
    print("=" * 70)

    df = df.copy()

    if "labour_hours" in df.columns:

        df["high_labour_hours"] = (
            df["labour_hours"]
            >
            df["labour_hours"].median()
        ).astype(int)

    if "downtime_hours" in df.columns:

        df["high_downtime"] = (
            df["downtime_hours"]
            >
            df["downtime_hours"].median()
        ).astype(int)

    if "cost_inr" in df.columns:

        df["high_maintenance_cost"] = (
            df["cost_inr"]
            >
            df["cost_inr"].median()
        ).astype(int)

    print(
        "Created operational risk indicators."
    )

    return df


# ============================================================
# CREATE SERVICE TYPE FEATURES
# ============================================================

def create_service_features(df):

    print("\n" + "=" * 70)
    print("CREATING SERVICE FEATURES")
    print("=" * 70)

    df = df.copy()

    if "service_type" in df.columns:

        service_frequency = (
            df["service_type"]
            .value_counts()
        )

        df["service_type_frequency"] = (
            df["service_type"]
            .map(service_frequency)
        )

    if "parts_replaced" in df.columns:

        df["parts_replaced_count"] = (
            df["parts_replaced"]
            .fillna("")
            .astype(str)
            .apply(
                lambda value:
                0
                if value.strip() == ""
                else len(
                    [
                        part
                        for part in value.split(",")
                        if part.strip()
                    ]
                )
            )
        )

    print(
        "Created service-related features."
    )

    return df


# ============================================================
# IMPUTE NUMERICAL FEATURES
# ============================================================

def impute_numerical_features(df):

    print("\n" + "=" * 70)
    print("IMPUTING NUMERICAL FEATURES")
    print("=" * 70)

    df = df.copy()

    numerical_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    numerical_columns = [
        column
        for column in numerical_columns
        if column != TARGET_COLUMN
    ]

    for column in numerical_columns:

        median_value = df[column].median()

        df[column] = (
            df[column]
            .fillna(median_value)
        )

    print(
        f"Numerical columns processed : "
        f"{len(numerical_columns)}"
    )

    return df


# ============================================================
# IMPUTE CATEGORICAL FEATURES
# ============================================================

def impute_categorical_features(df):

    print("\n" + "=" * 70)
    print("IMPUTING CATEGORICAL FEATURES")
    print("=" * 70)

    df = df.copy()

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    for column in categorical_columns:

        mode_values = df[column].mode()

        if len(mode_values) > 0:

            fill_value = mode_values.iloc[0]

        else:

            fill_value = "unknown"

        df[column] = (
            df[column]
            .fillna(fill_value)
        )

    print(
        f"Categorical columns processed : "
        f"{len(categorical_columns)}"
    )

    return df


# ============================================================
# REMOVE IDENTIFIERS
# ============================================================

def remove_identifier_columns(df):

    print("\n" + "=" * 70)
    print("REMOVING IDENTIFIER COLUMNS")
    print("=" * 70)

    df = df.copy()

    columns_to_drop = [
        "work_order_id",
        "vehicle_id",
        "technician_id",
        "service_date"
    ]

    existing_columns = [
        column
        for column in columns_to_drop
        if column in df.columns
    ]

    df = df.drop(
        columns=existing_columns
    )

    print(
        "Removed columns:"
    )

    print(existing_columns)

    return df


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

def remove_constant_features(df):

    print("\n" + "=" * 70)
    print("REMOVING CONSTANT FEATURES")
    print("=" * 70)

    df = df.copy()

    constant_columns = [
        column
        for column in df.columns
        if column != TARGET_COLUMN
        and df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:

        df = df.drop(
            columns=constant_columns
        )

    print(
        "Constant columns removed:"
    )

    print(constant_columns)

    return df


# ============================================================
# FINAL FEATURE CLEANUP
# ============================================================

def final_feature_cleanup(df):

    print("\n" + "=" * 70)
    print("FINAL FEATURE CLEANUP")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Re-impute numerical values created by divisions
    # --------------------------------------------------------

    numerical_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    for column in numerical_columns:

        if column == TARGET_COLUMN:
            continue

        median_value = df[column].median()

        df[column] = (
            df[column]
            .fillna(median_value)
        )

    # --------------------------------------------------------
    # Re-impute categorical values
    # --------------------------------------------------------

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    for column in categorical_columns:

        mode_values = df[column].mode()

        if len(mode_values) > 0:

            df[column] = (
                df[column]
                .fillna(mode_values.iloc[0])
            )

        else:

            df[column] = (
                df[column]
                .fillna("unknown")
            )

    return df


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(df):

    print("\n" + "=" * 70)
    print("SAVING MAINTENANCE FEATURES")
    print("=" * 70)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Feature dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(df):

    print("\n" + "=" * 70)
    print("FINAL MAINTENANCE FEATURE SUMMARY")
    print("=" * 70)

    print(
        f"Final shape : {df.shape}"
    )

    print("\nFinal columns:")

    for index, column in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{index:2d}. {column}"
        )

    print("\nTarget distribution:")

    print(
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nMissing values:")

    missing_values = (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        missing_values[
            missing_values > 0
        ]
    )

    print("\nData types:")

    print(
        df.dtypes
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("SMARTLOGIX AI - PREDICTIVE MAINTENANCE")
    print("FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Basic cleaning
    # --------------------------------------------------------

    df = clean_basic_data(
        df
    )

    # --------------------------------------------------------
    # 3. Normalize target
    # --------------------------------------------------------

    df = normalize_target(
        df
    )

    # --------------------------------------------------------
    # 4. Create date features
    # --------------------------------------------------------

    df = create_date_features(
        df
    )

    # --------------------------------------------------------
    # 5. Vehicle maintenance features
    # --------------------------------------------------------

    df = create_vehicle_maintenance_features(
        df
    )

    # --------------------------------------------------------
    # 6. Cost features
    # --------------------------------------------------------

    df = create_cost_features(
        df
    )

    # --------------------------------------------------------
    # 7. Operational features
    # --------------------------------------------------------

    df = create_operational_features(
        df
    )

    # --------------------------------------------------------
    # 8. Service features
    # --------------------------------------------------------

    df = create_service_features(
        df
    )

    # --------------------------------------------------------
    # 9. Numerical imputation
    # --------------------------------------------------------

    df = impute_numerical_features(
        df
    )

    # --------------------------------------------------------
    # 10. Categorical imputation
    # --------------------------------------------------------

    df = impute_categorical_features(
        df
    )

    # --------------------------------------------------------
    # 11. Remove identifiers
    # --------------------------------------------------------

    df = remove_identifier_columns(
        df
    )

    # --------------------------------------------------------
    # 12. Remove constant columns
    # --------------------------------------------------------

    df = remove_constant_features(
        df
    )

    # --------------------------------------------------------
    # 13. Final cleanup
    # --------------------------------------------------------

    df = final_feature_cleanup(
        df
    )

    # --------------------------------------------------------
    # 14. Save
    # --------------------------------------------------------

    save_features(
        df
    )

    # --------------------------------------------------------
    # 15. Summary
    # --------------------------------------------------------

    print_final_summary(
        df
    )

    print("\n" + "=" * 70)
    print("PREDICTIVE MAINTENANCE FEATURE ENGINEERING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()