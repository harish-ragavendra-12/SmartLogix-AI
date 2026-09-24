import pandas as pd

from src.config.config import PROCESSED_DATA_DIR


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "maintenance_history_clean.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "operational_maintenance_profile.csv"
)


# ============================================================
# LOAD MAINTENANCE HISTORY
# ============================================================

def load_maintenance_history():

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

    print("\nUnique vehicles:")

    print(
        df["vehicle_id"]
        .nunique()
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print("\n" + "=" * 70)
    print("PREPARING MAINTENANCE DATA")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Prepare service date
    # --------------------------------------------------------

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
        f"Invalid service dates : {invalid_dates}"
    )

    # --------------------------------------------------------
    # Normalize failure status
    # --------------------------------------------------------

    df["failure_reported"] = (
        df["failure_reported"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print("\nFailure status distribution:")

    print(
        df["failure_reported"]
        .value_counts(dropna=False)
    )

    return df


# ============================================================
# CREATE VEHICLE PROFILE
# ============================================================

def create_vehicle_profile(df):

    print("\n" + "=" * 70)
    print("CREATING OPERATIONAL VEHICLE MAINTENANCE PROFILE")
    print("=" * 70)

    # --------------------------------------------------------
    # Sort records chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            "vehicle_id",
            "service_date"
        ]
    ).copy()

    # --------------------------------------------------------
    # Aggregate vehicle-level metrics
    # --------------------------------------------------------

    profile_df = (
        df.groupby("vehicle_id")
        .agg(
            maintenance_count=(
                "work_order_id",
                "count"
            ),

            first_service_date=(
                "service_date",
                "min"
            ),

            last_service_date=(
                "service_date",
                "max"
            ),

            total_maintenance_cost=(
                "cost_inr",
                "sum"
            ),

            average_maintenance_cost=(
                "cost_inr",
                "mean"
            ),

            total_downtime_hours=(
                "downtime_hours",
                "sum"
            ),

            average_downtime_hours=(
                "downtime_hours",
                "mean"
            ),

            total_labour_hours=(
                "labour_hours",
                "sum"
            ),

            average_labour_hours=(
                "labour_hours",
                "mean"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Failure statistics
    # --------------------------------------------------------

    df["failure_flag"] = (
        df["failure_reported"]
        .map({
            "yes": 1,
            "no": 0
        })
    )

    failure_stats = (
        df.groupby("vehicle_id")
        .agg(
            failure_count=(
                "failure_flag",
                "sum"
            ),

            known_failure_records=(
                "failure_flag",
                "count"
            ),

            unknown_failure_count=(
                "failure_reported",
                lambda x: (x == "unknown").sum()
            )
        )
        .reset_index()
    )

    profile_df = profile_df.merge(
        failure_stats,
        on="vehicle_id",
        how="left"
    )

    # --------------------------------------------------------
    # Failure rate based only on known failure records
    # --------------------------------------------------------

    profile_df["failure_rate"] = (
        profile_df["failure_count"]
        /
        profile_df["known_failure_records"]
    )

    profile_df["failure_rate"] = (
        profile_df["failure_rate"]
        .fillna(0)
    )

    profile_df["failure_count"] = (
        profile_df["failure_count"]
        .fillna(0)
    )

    profile_df["unknown_failure_count"] = (
        profile_df["unknown_failure_count"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Failure rate
    # --------------------------------------------------------

    profile_df["failure_rate"] = (
        profile_df["failure_count"]
        /
        profile_df["maintenance_count"]
    )

    # --------------------------------------------------------
    # Last maintenance record
    # --------------------------------------------------------

    last_records = (
        df.sort_values(
            by=[
                "vehicle_id",
                "service_date"
            ]
        )
        .groupby(
            "vehicle_id",
            as_index=False
        )
        .tail(1)
    )

    last_records = (
        last_records[
            [
                "vehicle_id",
                "service_type",
                "failure_reported"
            ]
        ]
        .rename(
            columns={
                "service_type":
                    "last_service_type",

                "failure_reported":
                    "last_failure_reported"
            }
        )
    )

    profile_df = profile_df.merge(
        last_records,
        on="vehicle_id",
        how="left"
    )

    # --------------------------------------------------------
    # Maintenance history classification
    # --------------------------------------------------------

    profile_df["history_level"] = (
        profile_df["maintenance_count"]
        .apply(
            lambda count:
            "LIMITED"
            if count == 1
            else
            "MODERATE"
            if count <= 3
            else
            "STRONG"
        )
    )

    # --------------------------------------------------------
    # Evidence level
    # --------------------------------------------------------

    profile_df["evidence_level"] = (
        profile_df.apply(
            classify_evidence_level,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    preferred_columns = [

        "vehicle_id",

        "maintenance_count",

        "history_level",
        "evidence_level",

        "first_service_date",
        "last_service_date",

        "total_maintenance_cost",
        "average_maintenance_cost",

        "total_downtime_hours",
        "average_downtime_hours",

        "total_labour_hours",
        "average_labour_hours",

        "failure_count",
        "known_failure_records",
        "unknown_failure_count",
        "failure_rate",

        "last_service_type",
        "last_failure_reported"
    ]

    profile_df = profile_df[
        [
            column
            for column in preferred_columns
            if column in profile_df.columns
        ]
    ]

    # --------------------------------------------------------
    # Sort by vehicle
    # --------------------------------------------------------

    profile_df = profile_df.sort_values(
        by="vehicle_id"
    ).reset_index(
        drop=True
    )

    print(
        f"\nOperational profile shape : "
        f"{profile_df.shape}"
    )

    print(
        f"Unique vehicles : "
        f"{profile_df['vehicle_id'].nunique()}"
    )

    return profile_df


# ============================================================
# CLASSIFY EVIDENCE LEVEL
# ============================================================

def classify_evidence_level(row):

    maintenance_count = (
        row["maintenance_count"]
    )

    unknown_failure_count = (
        row["unknown_failure_count"]
    )

    # --------------------------------------------------------
    # Unknown failure information exists
    # --------------------------------------------------------

    if unknown_failure_count > 0:

        return "PARTIAL"

    # --------------------------------------------------------
    # Only one maintenance record
    # --------------------------------------------------------

    if maintenance_count == 1:

        return "LIMITED"

    # --------------------------------------------------------
    # Multiple records with known failure status
    # --------------------------------------------------------

    return "STRONG"


# ============================================================
# VALIDATE PROFILE
# ============================================================

def validate_profile(profile_df):

    print("\n" + "=" * 70)
    print("VALIDATING OPERATIONAL MAINTENANCE PROFILE")
    print("=" * 70)

    print(
        f"\nShape : {profile_df.shape}"
    )

    print(
        f"Unique vehicles : "
        f"{profile_df['vehicle_id'].nunique()}"
    )

    print(
        f"Duplicate vehicles : "
        f"{profile_df['vehicle_id'].duplicated().sum()}"
    )

    print("\nHistory level distribution:")

    print(
        profile_df["history_level"]
        .value_counts()
    )

    print("\nEvidence level distribution:")

    print(
        profile_df["evidence_level"]
        .value_counts()
    )

    print("\nMissing values:")

    missing_values = (
        profile_df
        .isna()
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

    # --------------------------------------------------------
    # Verify VEH-0650
    # --------------------------------------------------------

    vehicle_id = "VEH-0650"

    vehicle = profile_df[
        profile_df["vehicle_id"] == vehicle_id
    ]

    print(
        f"\nChecking vehicle: {vehicle_id}"
    )

    if vehicle.empty:

        print(
            f"{vehicle_id} NOT FOUND"
        )

    else:

        print(
            vehicle.to_string(
                index=False
            )
        )


# ============================================================
# SAVE PROFILE
# ============================================================

def save_profile(profile_df):

    print("\n" + "=" * 70)
    print("SAVING OPERATIONAL MAINTENANCE PROFILE")
    print("=" * 70)

    profile_df.to_csv(
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
    print("OPERATIONAL MAINTENANCE PROFILE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_maintenance_history()

    # --------------------------------------------------------
    # 2. Prepare
    # --------------------------------------------------------

    df = prepare_data(
        df
    )

    # --------------------------------------------------------
    # 3. Create vehicle profile
    # --------------------------------------------------------

    profile_df = create_vehicle_profile(
        df
    )

    # --------------------------------------------------------
    # 4. Validate
    # --------------------------------------------------------

    validate_profile(
        profile_df
    )

    # --------------------------------------------------------
    # 5. Save
    # --------------------------------------------------------

    save_profile(
        profile_df
    )

    print("\n" + "=" * 70)
    print(
        "OPERATIONAL MAINTENANCE PROFILE "
        "COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()