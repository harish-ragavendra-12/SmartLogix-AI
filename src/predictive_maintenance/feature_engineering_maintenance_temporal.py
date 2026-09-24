import pandas as pd
import numpy as np

from src.config.config import (
    PROCESSED_DATA_DIR
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "maintenance_history_clean.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "maintenance_temporal_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING MAINTENANCE HISTORY")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Input shape : {df.shape}"
    )

    print("\nColumns:")

    print(
        df.columns.tolist()
    )

    return df


# ============================================================
# CLEAN TARGET
# ============================================================

def clean_target(df):

    print("\n" + "=" * 70)
    print("CLEANING FAILURE TARGET")
    print("=" * 70)

    df["failure_reported"] = (
        df["failure_reported"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    target_mapping = {

        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0
    }

    df["failure_reported"] = (
        df["failure_reported"]
        .map(target_mapping)
    )

    print(
        "Target distribution before "
        "removing unknown values:"
    )

    print(
        df["failure_reported"]
        .value_counts(
            dropna=False
        )
    )

    before_count = len(df)

    df = df.dropna(
        subset=["failure_reported"]
    ).copy()

    df["failure_reported"] = (
        df["failure_reported"]
        .astype(int)
    )

    removed_count = (
        before_count - len(df)
    )

    print(
        f"\nRows removed due to "
        f"unknown/missing target : "
        f"{removed_count}"
    )

    print(
        f"Rows remaining : {len(df)}"
    )

    print("\nFinal target distribution:")

    print(
        df["failure_reported"]
        .value_counts()
        .sort_index()
    )

    return df


# ============================================================
# PREPARE DATE
# ============================================================

def prepare_date(df):

    print("\n" + "=" * 70)
    print("PREPARING SERVICE DATE")
    print("=" * 70)

    df["service_date"] = pd.to_datetime(
        df["service_date"],
        errors="coerce"
    )

    invalid_dates = (
        df["service_date"]
        .isna()
        .sum()
    )

    print(
        f"Invalid service dates : "
        f"{invalid_dates}"
    )

    df = df.dropna(
        subset=["service_date"]
    ).copy()

    df = df.sort_values(
        by=[
            "vehicle_id",
            "service_date"
        ]
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# CREATE DATE FEATURES
# ============================================================

def create_date_features(df):

    print("\n" + "=" * 70)
    print("CREATING DATE FEATURES")
    print("=" * 70)

    df["service_year"] = (
        df["service_date"].dt.year
    )

    df["service_month"] = (
        df["service_date"].dt.month
    )

    df["service_day_of_week"] = (
        df["service_date"].dt.dayofweek
    )

    df["service_week"] = (
        df["service_date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["is_weekend"] = (
        df["service_day_of_week"]
        >= 5
    ).astype(int)

    return df


# ============================================================
# CREATE PREVIOUS MAINTENANCE FEATURES
# ============================================================

def create_temporal_features(df):

    print("\n" + "=" * 70)
    print("CREATING TEMPORAL MAINTENANCE FEATURES")
    print("=" * 70)

    grouped = df.groupby(
        "vehicle_id",
        group_keys=False
    )

    # --------------------------------------------------------
    # Previous maintenance count
    # --------------------------------------------------------

    df["previous_maintenance_count"] = (
        grouped.cumcount()
    )

    # --------------------------------------------------------
    # Previous failure count
    # --------------------------------------------------------

    df["previous_failure_count"] = (
        grouped["failure_reported"]
        .transform(
            lambda x:
            x.shift(1)
            .fillna(0)
            .cumsum()
        )
    )

    # --------------------------------------------------------
    # Previous failure rate
    # --------------------------------------------------------

    df["previous_failure_rate"] = np.where(

        df["previous_maintenance_count"] > 0,

        (
            df["previous_failure_count"]
            /
            df["previous_maintenance_count"]
        ),

        0
    )

    # --------------------------------------------------------
    # Previous maintenance cost
    # --------------------------------------------------------

    df["previous_maintenance_cost"] = (
        grouped["cost_inr"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Previous average cost
    # --------------------------------------------------------

    df["previous_average_cost"] = (
        grouped["cost_inr"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )

    # --------------------------------------------------------
    # Previous maximum cost
    # --------------------------------------------------------

    df["previous_max_cost"] = (
        grouped["cost_inr"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .max()
        )
    )

    # --------------------------------------------------------
    # Previous downtime
    # --------------------------------------------------------

    df["previous_downtime_hours"] = (
        grouped["downtime_hours"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Previous average downtime
    # --------------------------------------------------------

    df["previous_average_downtime"] = (
        grouped["downtime_hours"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )

    # --------------------------------------------------------
    # Previous maximum downtime
    # --------------------------------------------------------

    df["previous_max_downtime"] = (
        grouped["downtime_hours"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .max()
        )
    )

    # --------------------------------------------------------
    # Previous labour hours
    # --------------------------------------------------------

    df["previous_labour_hours"] = (
        grouped["labour_hours"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Previous average labour hours
    # --------------------------------------------------------

    df["previous_average_labour_hours"] = (
        grouped["labour_hours"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )

    # --------------------------------------------------------
    # Previous maximum labour hours
    # --------------------------------------------------------

    df["previous_max_labour_hours"] = (
        grouped["labour_hours"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .max()
        )
    )

    # --------------------------------------------------------
    # Days since previous maintenance
    # --------------------------------------------------------

    previous_date = (
        grouped["service_date"]
        .shift(1)
    )

    df["days_since_previous_maintenance"] = (
        (
            df["service_date"]
            - previous_date
        )
        .dt.total_seconds()
        / (60 * 60 * 24)
    )

    # --------------------------------------------------------
    # Frequent maintenance
    # --------------------------------------------------------

    df["frequent_maintenance"] = (
        (
            df["days_since_previous_maintenance"]
            <= 30
        )
        .fillna(False)
        .astype(int)
    )

    # --------------------------------------------------------
    # Previous failure
    # --------------------------------------------------------

    df["previous_failure"] = (
        grouped["failure_reported"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Cost compared with historical average
    # --------------------------------------------------------

    df["cost_vs_previous_average"] = (
        df["previous_maintenance_cost"]
        /
        df["previous_average_cost"]
    )

    # --------------------------------------------------------
    # Downtime compared with historical average
    # --------------------------------------------------------

    df["downtime_vs_previous_average"] = (
        df["previous_downtime_hours"]
        /
        df["previous_average_downtime"]
    )

    # --------------------------------------------------------
    # Labour compared with historical average
    # --------------------------------------------------------

    df["labour_vs_previous_average"] = (
        df["previous_labour_hours"]
        /
        df["previous_average_labour_hours"]
    )

    print(
        "Temporal features created successfully."
    )

    return df


# ============================================================
# REMOVE RECORDS WITHOUT HISTORY
# ============================================================

def remove_records_without_history(df):

    print("\n" + "=" * 70)
    print("REMOVING RECORDS WITHOUT PREVIOUS HISTORY")
    print("=" * 70)

    before_count = len(df)

    df = df[
        df["previous_maintenance_count"] > 0
    ].copy()

    removed_count = (
        before_count - len(df)
    )

    print(
        f"Rows removed : {removed_count}"
    )

    print(
        f"Rows remaining : {len(df)}"
    )

    return df


# ============================================================
# HANDLE NUMERICAL MISSING VALUES
# ============================================================

def handle_missing_values(df):

    print("\n" + "=" * 70)
    print("HANDLING NUMERICAL MISSING VALUES")
    print("=" * 70)

    numerical_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    for column in numerical_columns:

        if df[column].isna().any():

            median_value = (
                df[column]
                .median()
            )

            df[column] = (
                df[column]
                .fillna(median_value)
            )

    print(
        "Numerical missing values "
        "handled."
    )

    return df


# ============================================================
# CREATE FINAL DATASET
# ============================================================

def create_final_dataset(df):

    print("\n" + "=" * 70)
    print("CREATING FINAL TEMPORAL DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # These columns are retained for downstream analysis.
    # They MUST NOT be used as ML features.
    # --------------------------------------------------------

    identifier_columns = [
        "vehicle_id",
        "service_date"
    ]

    # --------------------------------------------------------
    # Current-event information must not be used as features.
    # --------------------------------------------------------

    leakage_columns = [
        "work_order_id",
        "technician_id",
        "labour_hours",
        "cost_inr",
        "downtime_hours",
        "service_type",
        "parts_replaced"
    ]

    columns_to_remove = [
        column
        for column in leakage_columns
        if column in df.columns
    ]

    df = df.drop(
        columns=columns_to_remove
    )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    preferred_columns = [

        "vehicle_id",
        "service_date",

        "failure_reported",

        "service_year",
        "service_month",
        "service_day_of_week",
        "service_week",
        "is_weekend",

        "previous_maintenance_count",
        "previous_failure_count",
        "previous_failure_rate",

        "previous_maintenance_cost",
        "previous_average_cost",
        "previous_max_cost",

        "previous_downtime_hours",
        "previous_average_downtime",
        "previous_max_downtime",

        "previous_labour_hours",
        "previous_average_labour_hours",
        "previous_max_labour_hours",

        "days_since_previous_maintenance",

        "frequent_maintenance",

        "previous_failure",

        "cost_vs_previous_average",
        "downtime_vs_previous_average",
        "labour_vs_previous_average"
    ]

    final_columns = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    df = df[
        final_columns
    ].copy()

    print(
        f"Final dataset shape : {df.shape}"
    )

    print("\nFinal columns:")

    print(
        df.columns.tolist()
    )

    return df


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(df):

    print("\n" + "=" * 70)
    print("VALIDATING TEMPORAL DATASET")
    print("=" * 70)

    print(
        f"Shape : {df.shape}"
    )

    print(
        f"Duplicate rows : "
        f"{df.duplicated().sum()}"
    )

    print("\nMissing values:")

    missing_values = (
        df.isna()
        .sum()
    )

    missing_values = (
        missing_values[
            missing_values > 0
        ]
    )

    if missing_values.empty:

        print(
            "No missing values."
        )

    else:

        print(
            missing_values
        )

    print("\nTarget distribution:")

    print(
        df["failure_reported"]
        .value_counts()
        .sort_index()
    )

    print("\nUnique vehicles:")

    print(
        df["vehicle_id"]
        .nunique()
    )

    print("\nService date range:")

    print(
        f"Minimum : "
        f"{df['service_date'].min()}"
    )

    print(
        f"Maximum : "
        f"{df['service_date'].max()}"
    )


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df):

    print("\n" + "=" * 70)
    print("SAVING TEMPORAL FEATURES")
    print("=" * 70)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Saved to:\n"
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("SMARTLOGIX AI")
    print("TEMPORAL MAINTENANCE FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Clean target
    # --------------------------------------------------------

    df = clean_target(
        df
    )

    # --------------------------------------------------------
    # 3. Prepare dates
    # --------------------------------------------------------

    df = prepare_date(
        df
    )

    # --------------------------------------------------------
    # 4. Create date features
    # --------------------------------------------------------

    df = create_date_features(
        df
    )

    # --------------------------------------------------------
    # 5. Create historical features
    # --------------------------------------------------------

    df = create_temporal_features(
        df
    )

    # --------------------------------------------------------
    # 6. Remove records without history
    # --------------------------------------------------------

    df = remove_records_without_history(
        df
    )

    # --------------------------------------------------------
    # 7. Handle numerical missing values
    # --------------------------------------------------------

    df = handle_missing_values(
        df
    )

    # --------------------------------------------------------
    # 8. Create final dataset
    # --------------------------------------------------------

    df = create_final_dataset(
        df
    )

    # --------------------------------------------------------
    # 9. Validate
    # --------------------------------------------------------

    validate_dataset(
        df
    )

    # --------------------------------------------------------
    # 10. Save
    # --------------------------------------------------------

    save_dataset(
        df
    )

    print("\n" + "=" * 70)
    print(
        "TEMPORAL MAINTENANCE FEATURE "
        "ENGINEERING COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()