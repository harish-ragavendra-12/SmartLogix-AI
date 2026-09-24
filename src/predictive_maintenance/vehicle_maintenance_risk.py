import pandas as pd
import numpy as np

from src.config.config import PROCESSED_DATA_DIR


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "maintenance_temporal_features.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "vehicle_maintenance_risk.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING TEMPORAL MAINTENANCE DATA")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["service_date"]
    )

    print(
        f"Dataset shape : {df.shape}"
    )

    print(
        f"Unique vehicles : "
        f"{df['vehicle_id'].nunique()}"
    )

    return df


# ============================================================
# AGGREGATE VEHICLE METRICS
# ============================================================

def aggregate_vehicle_metrics(df):

    print("\n" + "=" * 70)
    print("AGGREGATING MAINTENANCE DATA BY VEHICLE")
    print("=" * 70)

    vehicle_df = (
        df.groupby("vehicle_id")
        .agg(
            maintenance_count=(
                "failure_reported",
                "count"
            ),

            failure_count=(
                "failure_reported",
                "sum"
            ),

            average_maintenance_cost=(
                "previous_average_cost",
                "mean"
            ),

            maximum_maintenance_cost=(
                "previous_max_cost",
                "max"
            ),

            average_downtime_hours=(
                "previous_average_downtime",
                "mean"
            ),

            maximum_downtime_hours=(
                "previous_max_downtime",
                "max"
            ),

            average_labour_hours=(
                "previous_average_labour_hours",
                "mean"
            ),

            maximum_labour_hours=(
                "previous_max_labour_hours",
                "max"
            ),

            average_failure_rate=(
                "previous_failure_rate",
                "mean"
            ),

            frequent_maintenance_count=(
                "frequent_maintenance",
                "sum"
            ),

            previous_failure_count=(
                "previous_failure_count",
                "max"
            ),

            latest_maintenance_date=(
                "service_date",
                "max"
            ),

            days_since_previous_maintenance=(
                "days_since_previous_maintenance",
                "last"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Vehicle-level failure rate
    # --------------------------------------------------------

    vehicle_df["failure_rate"] = (
        vehicle_df["failure_count"]
        /
        vehicle_df["maintenance_count"]
    )

    # --------------------------------------------------------
    # Current/latest maintenance recency
    # --------------------------------------------------------

    latest_date = (
        df["service_date"].max()
    )

    vehicle_df["days_since_last_maintenance"] = (
        latest_date
        -
        vehicle_df["latest_maintenance_date"]
    ).dt.days

    print(
        f"Vehicle-level dataset shape : "
        f"{vehicle_df.shape}"
    )

    return vehicle_df


# ============================================================
# NORMALIZE SCORE
# ============================================================

def normalize_score(
    series,
    minimum,
    maximum
):

    if maximum == minimum:

        return pd.Series(
            0.0,
            index=series.index
        )

    score = (
        (series - minimum)
        /
        (maximum - minimum)
    )

    score = (
        score
        .clip(0, 1)
        * 100
    )

    return score


# ============================================================
# CALCULATE VEHICLE RISK SCORE
# ============================================================

def calculate_vehicle_risk_score(
    vehicle_df
):

    print("\n" + "=" * 70)
    print("CALCULATING VEHICLE RISK SCORE")
    print("=" * 70)

    # --------------------------------------------------------
    # Component 1: Failure rate
    # Weight = 30%
    # --------------------------------------------------------

    failure_score = (
        vehicle_df["failure_rate"]
        .clip(0, 1)
        * 100
    )

    # --------------------------------------------------------
    # Component 2: Previous failure history
    # Weight = 20%
    # --------------------------------------------------------

    previous_failure_score = (
        vehicle_df["previous_failure_count"]
        > 0
    ).astype(int) * 100

    # --------------------------------------------------------
    # Component 3: Average downtime
    # Weight = 15%
    # --------------------------------------------------------

    downtime_score = normalize_score(
        vehicle_df[
            "average_downtime_hours"
        ],
        vehicle_df[
            "average_downtime_hours"
        ].min(),
        vehicle_df[
            "average_downtime_hours"
        ].max()
    )

    # --------------------------------------------------------
    # Component 4: Maintenance cost
    # Weight = 15%
    # --------------------------------------------------------

    cost_score = normalize_score(
        vehicle_df[
            "average_maintenance_cost"
        ],
        vehicle_df[
            "average_maintenance_cost"
        ].min(),
        vehicle_df[
            "average_maintenance_cost"
        ].max()
    )

    # --------------------------------------------------------
    # Component 5: Frequent maintenance
    # Weight = 10%
    # --------------------------------------------------------

    frequency_score = normalize_score(
        vehicle_df[
            "frequent_maintenance_count"
        ],
        vehicle_df[
            "frequent_maintenance_count"
        ].min(),
        vehicle_df[
            "frequent_maintenance_count"
        ].max()
    )

    # --------------------------------------------------------
    # Component 6: Maximum downtime
    # Weight = 10%
    # --------------------------------------------------------

    max_downtime_score = normalize_score(
        vehicle_df[
            "maximum_downtime_hours"
        ],
        vehicle_df[
            "maximum_downtime_hours"
        ].min(),
        vehicle_df[
            "maximum_downtime_hours"
        ].max()
    )

    # --------------------------------------------------------
    # Weighted risk score
    # --------------------------------------------------------

    vehicle_df["risk_score"] = (
        failure_score * 0.30
        +
        previous_failure_score * 0.20
        +
        downtime_score * 0.15
        +
        cost_score * 0.15
        +
        frequency_score * 0.10
        +
        max_downtime_score * 0.10
    )

    vehicle_df["risk_score"] = (
        vehicle_df["risk_score"]
        .clip(0, 100)
        .round(2)
    )

    return vehicle_df


# ============================================================
# ASSIGN RISK LEVEL
# ============================================================

def assign_risk_level(
    risk_score
):

    if risk_score >= 70:

        return "HIGH"

    elif risk_score >= 40:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# ADD RISK LEVEL
# ============================================================

def add_risk_levels(
    vehicle_df
):

    print("\n" + "=" * 70)
    print("ASSIGNING VEHICLE RISK LEVELS")
    print("=" * 70)

    vehicle_df["risk_level"] = (
        vehicle_df["risk_score"]
        .apply(
            assign_risk_level
        )
    )

    return vehicle_df


# ============================================================
# GENERATE RISK REASONS
# ============================================================

def generate_risk_reasons(
    row
):

    reasons = []

    # --------------------------------------------------------
    # Failure rate
    # --------------------------------------------------------

    if row["failure_rate"] >= 0.50:

        reasons.append(
            "High historical failure rate"
        )

    elif row["failure_rate"] >= 0.30:

        reasons.append(
            "Moderate historical failure rate"
        )

    # --------------------------------------------------------
    # Previous failures
    # --------------------------------------------------------

    if row["previous_failure_count"] > 0:

        reasons.append(
            "Previous maintenance failures recorded"
        )

    # --------------------------------------------------------
    # Downtime
    # --------------------------------------------------------

    if row["average_downtime_hours"] >= 40:

        reasons.append(
            "High average maintenance downtime"
        )

    elif row["average_downtime_hours"] >= 25:

        reasons.append(
            "Elevated average maintenance downtime"
        )

    # --------------------------------------------------------
    # Maintenance cost
    # --------------------------------------------------------

    if row["average_maintenance_cost"] >= 30000:

        reasons.append(
            "High average maintenance cost"
        )

    elif row["average_maintenance_cost"] >= 25000:

        reasons.append(
            "Elevated average maintenance cost"
        )

    # --------------------------------------------------------
    # Frequent maintenance
    # --------------------------------------------------------

    if row["frequent_maintenance_count"] >= 2:

        reasons.append(
            "Repeated maintenance within short intervals"
        )

    # --------------------------------------------------------
    # Maximum downtime
    # --------------------------------------------------------

    if row["maximum_downtime_hours"] >= 70:

        reasons.append(
            "Very high historical downtime event"
        )

    # --------------------------------------------------------
    # Maintenance recency
    # --------------------------------------------------------

    if row["days_since_last_maintenance"] <= 30:

        reasons.append(
            "Maintenance activity occurred recently"
        )

    elif row["days_since_last_maintenance"] > 180:

        reasons.append(
            "Long interval since last maintenance"
        )

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    if not reasons:

        reasons.append(
            "No major historical maintenance risk indicators"
        )

    return "; ".join(reasons)


# ============================================================
# GENERATE RECOMMENDATION
# ============================================================

def generate_recommendation(
    row
):

    risk_level = row["risk_level"]

    if risk_level == "HIGH":

        return (
            "Schedule preventive maintenance inspection "
            "before assigning the vehicle to critical deliveries."
        )

    elif risk_level == "MEDIUM":

        return (
            "Monitor the vehicle closely and schedule "
            "maintenance during the next available service window."
        )

    else:

        return (
            "Continue regular monitoring and follow "
            "the standard preventive maintenance schedule."
        )


# ============================================================
# ADD REASONS AND RECOMMENDATIONS
# ============================================================

def add_risk_explanations(
    vehicle_df
):

    print("\n" + "=" * 70)
    print("GENERATING RISK REASONS AND RECOMMENDATIONS")
    print("=" * 70)

    vehicle_df["risk_reasons"] = (
        vehicle_df.apply(
            generate_risk_reasons,
            axis=1
        )
    )

    vehicle_df["recommendation"] = (
        vehicle_df.apply(
            generate_recommendation,
            axis=1
        )
    )

    return vehicle_df


# ============================================================
# ROUND NUMERICAL COLUMNS
# ============================================================

def format_output(
    vehicle_df
):

    numerical_columns = [
        "failure_rate",
        "average_maintenance_cost",
        "maximum_maintenance_cost",
        "average_downtime_hours",
        "maximum_downtime_hours",
        "average_labour_hours",
        "maximum_labour_hours",
        "average_failure_rate",
        "risk_score"
    ]

    for column in numerical_columns:

        if column in vehicle_df.columns:

            vehicle_df[column] = (
                vehicle_df[column]
                .round(2)
            )

    return vehicle_df


# ============================================================
# DISPLAY RISK DISTRIBUTION
# ============================================================

def display_risk_distribution(
    vehicle_df
):

    print("\n" + "=" * 70)
    print("VEHICLE RISK DISTRIBUTION")
    print("=" * 70)

    distribution = (
        vehicle_df["risk_level"]
        .value_counts()
    )

    print(
        distribution.to_string()
    )

    print("\nRisk percentages:")

    percentages = (
        vehicle_df["risk_level"]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    print(
        percentages.to_string()
    )


# ============================================================
# DISPLAY TOP HIGH-RISK VEHICLES
# ============================================================

def display_high_risk_vehicles(
    vehicle_df,
    number_of_records=15
):

    print("\n" + "=" * 70)
    print("TOP HIGH-RISK VEHICLES")
    print("=" * 70)

    high_risk = (
        vehicle_df[
            vehicle_df["risk_level"]
            == "HIGH"
        ]
        .sort_values(
            by="risk_score",
            ascending=False
        )
        .head(number_of_records)
    )

    if high_risk.empty:

        print(
            "No HIGH-risk vehicles found."
        )

        return

    columns = [

        "vehicle_id",
        "risk_score",
        "risk_level",
        "maintenance_count",
        "failure_count",
        "failure_rate",
        "average_maintenance_cost",
        "average_downtime_hours",
        "days_since_last_maintenance"
    ]

    print(
        high_risk[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# DISPLAY RISK SUMMARY
# ============================================================

def display_risk_summary(
    vehicle_df
):

    print("\n" + "=" * 70)
    print("VEHICLE MAINTENANCE RISK SUMMARY")
    print("=" * 70)

    columns = [

        "vehicle_id",
        "risk_score",
        "risk_level",
        "failure_rate",
        "average_maintenance_cost",
        "average_downtime_hours",
        "risk_reasons",
        "recommendation"
    ]

    summary = (
        vehicle_df
        .sort_values(
            by="risk_score",
            ascending=False
        )
        .head(10)
    )

    for _, row in summary.iterrows():

        print(
            f"\nVehicle: "
            f"{row['vehicle_id']}"
        )

        print(
            f"Risk Score: "
            f"{row['risk_score']:.2f}/100"
        )

        print(
            f"Risk Level: "
            f"{row['risk_level']}"
        )

        print(
            f"Failure Rate: "
            f"{row['failure_rate']:.2%}"
        )

        print(
            f"Average Maintenance Cost: "
            f"₹{row['average_maintenance_cost']:,.2f}"
        )

        print(
            f"Average Downtime: "
            f"{row['average_downtime_hours']:.2f} hours"
        )

        print(
            f"Reasons: "
            f"{row['risk_reasons']}"
        )

        print(
            f"Recommendation: "
            f"{row['recommendation']}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    vehicle_df
):

    print("\n" + "=" * 70)
    print("SAVING VEHICLE MAINTENANCE RISK")
    print("=" * 70)

    vehicle_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Output file:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"Records saved : "
        f"{len(vehicle_df)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("SMARTLOGIX AI")
    print("VEHICLE MAINTENANCE RISK ENGINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Aggregate by vehicle
    # --------------------------------------------------------

    vehicle_df = aggregate_vehicle_metrics(
        df
    )

    # --------------------------------------------------------
    # 3. Calculate normalized risk score
    # --------------------------------------------------------

    vehicle_df = calculate_vehicle_risk_score(
        vehicle_df
    )

    # --------------------------------------------------------
    # 4. Assign risk levels
    # --------------------------------------------------------

    vehicle_df = add_risk_levels(
        vehicle_df
    )

    # --------------------------------------------------------
    # 5. Generate explanations
    # --------------------------------------------------------

    vehicle_df = add_risk_explanations(
        vehicle_df
    )

    # --------------------------------------------------------
    # 6. Format output
    # --------------------------------------------------------

    vehicle_df = format_output(
        vehicle_df
    )

    # --------------------------------------------------------
    # 7. Display distribution
    # --------------------------------------------------------

    display_risk_distribution(
        vehicle_df
    )

    # --------------------------------------------------------
    # 8. Display high-risk vehicles
    # --------------------------------------------------------

    display_high_risk_vehicles(
        vehicle_df
    )

    # --------------------------------------------------------
    # 9. Display detailed summary
    # --------------------------------------------------------

    display_risk_summary(
        vehicle_df
    )

    # --------------------------------------------------------
    # 10. Save
    # --------------------------------------------------------

    save_results(
        vehicle_df
    )

    print("\n" + "=" * 70)
    print(
        "VEHICLE MAINTENANCE RISK ENGINE "
        "COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()