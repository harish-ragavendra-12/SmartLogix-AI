import pandas as pd

from src.config.config import PROCESSED_DATA_DIR


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "maintenance_temporal_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_maintenance_data():

    print("\n" + "=" * 70)
    print("LOADING MAINTENANCE TEMPORAL DATA")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Dataset shape : {df.shape}"
    )

    return df


# ============================================================
# CALCULATE RISK SCORE
# ============================================================

def calculate_risk_score(row):

    score = 0

    # --------------------------------------------------------
    # Previous failure
    # --------------------------------------------------------

    if row["previous_failure"] == 1:
        score += 30

    # --------------------------------------------------------
    # Previous failure rate
    # --------------------------------------------------------

    if row["previous_failure_rate"] >= 0.50:
        score += 20

    elif row["previous_failure_rate"] >= 0.30:
        score += 10

    # --------------------------------------------------------
    # Maintenance frequency
    # --------------------------------------------------------

    if row["frequent_maintenance"] == 1:
        score += 15

    # --------------------------------------------------------
    # Days since previous maintenance
    # --------------------------------------------------------

    if row["days_since_previous_maintenance"] <= 30:
        score += 10

    elif row["days_since_previous_maintenance"] <= 60:
        score += 5

    # --------------------------------------------------------
    # Previous downtime
    # --------------------------------------------------------

    if row["previous_average_downtime"] >= 40:
        score += 15

    elif row["previous_average_downtime"] >= 25:
        score += 8

    # --------------------------------------------------------
    # Previous maintenance cost
    # --------------------------------------------------------

    if row["previous_average_cost"] >= 30000:
        score += 10

    elif row["previous_average_cost"] >= 25000:
        score += 5

    # --------------------------------------------------------
    # Maximum historical downtime
    # --------------------------------------------------------

    if row["previous_max_downtime"] >= 70:
        score += 10

    # --------------------------------------------------------
    # Maximum historical maintenance cost
    # --------------------------------------------------------

    if row["previous_max_cost"] >= 40000:
        score += 10

    return score


# ============================================================
# ASSIGN RISK LEVEL
# ============================================================

def assign_risk_level(score):

    if score >= 70:
        return "HIGH"

    elif score >= 40:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# GENERATE RISK REASONS
# ============================================================

def generate_risk_reasons(row):

    reasons = []

    # --------------------------------------------------------
    # Failure history
    # --------------------------------------------------------

    if row["previous_failure"] == 1:

        reasons.append(
            "Previous maintenance failure recorded"
        )

    # --------------------------------------------------------
    # Failure rate
    # --------------------------------------------------------

    if row["previous_failure_rate"] >= 0.50:

        reasons.append(
            "High historical failure rate"
        )

    elif row["previous_failure_rate"] >= 0.30:

        reasons.append(
            "Moderate historical failure rate"
        )

    # --------------------------------------------------------
    # Maintenance frequency
    # --------------------------------------------------------

    if row["frequent_maintenance"] == 1:

        reasons.append(
            "Frequent maintenance pattern detected"
        )

    # --------------------------------------------------------
    # Maintenance recency
    # --------------------------------------------------------

    if row["days_since_previous_maintenance"] <= 30:

        reasons.append(
            "Recent maintenance activity detected"
        )

    elif row["days_since_previous_maintenance"] > 90:

        reasons.append(
            "Long interval since previous maintenance"
        )

    # --------------------------------------------------------
    # Downtime
    # --------------------------------------------------------

    if row["previous_average_downtime"] >= 40:

        reasons.append(
            "High historical average downtime"
        )

    elif row["previous_average_downtime"] >= 25:

        reasons.append(
            "Elevated historical downtime"
        )

    # --------------------------------------------------------
    # Maintenance cost
    # --------------------------------------------------------

    if row["previous_average_cost"] >= 30000:

        reasons.append(
            "High historical maintenance cost"
        )

    elif row["previous_average_cost"] >= 25000:

        reasons.append(
            "Elevated historical maintenance cost"
        )

    # --------------------------------------------------------
    # Maximum downtime
    # --------------------------------------------------------

    if row["previous_max_downtime"] >= 70:

        reasons.append(
            "Very high historical downtime observed"
        )

    # --------------------------------------------------------
    # Maximum maintenance cost
    # --------------------------------------------------------

    if row["previous_max_cost"] >= 40000:

        reasons.append(
            "High historical maintenance cost event observed"
        )

    # --------------------------------------------------------
    # Default reason
    # --------------------------------------------------------

    if not reasons:

        reasons.append(
            "No major historical maintenance risk indicators detected"
        )

    return reasons


# ============================================================
# GENERATE RECOMMENDATION
# ============================================================

def generate_recommendation(
    risk_level,
    reasons
):

    if risk_level == "HIGH":

        recommendation = (
            "Schedule preventive maintenance inspection "
            "before the vehicle is assigned to a critical delivery."
        )

    elif risk_level == "MEDIUM":

        recommendation = (
            "Monitor the vehicle closely and schedule "
            "maintenance inspection during the next service window."
        )

    else:

        recommendation = (
            "Continue normal monitoring and follow the "
            "regular preventive maintenance schedule."
        )

    return recommendation


# ============================================================
# ANALYZE SINGLE RECORD
# ============================================================

def analyze_vehicle_record(row):

    score = calculate_risk_score(
        row
    )

    risk_level = assign_risk_level(
        score
    )

    reasons = generate_risk_reasons(
        row
    )

    recommendation = generate_recommendation(
        risk_level,
        reasons
    )

    return {

        "risk_score": score,

        "risk_level": risk_level,

        "risk_reasons": reasons,

        "recommendation": recommendation

    }


# ============================================================
# ADD RISK ANALYSIS TO DATASET
# ============================================================

def generate_risk_analysis(df):

    print("\n" + "=" * 70)
    print("GENERATING MAINTENANCE RISK ANALYSIS")
    print("=" * 70)

    analysis_results = []

    for _, row in df.iterrows():

        result = analyze_vehicle_record(
            row
        )

        analysis_results.append(
            result
        )

    risk_df = pd.DataFrame(
        analysis_results
    )

    result_df = pd.concat(
        [
            df.reset_index(drop=True),
            risk_df
        ],
        axis=1
    )

    return result_df


# ============================================================
# DISPLAY RISK DISTRIBUTION
# ============================================================

def display_risk_distribution(
    result_df
):

    print("\n" + "=" * 70)
    print("MAINTENANCE RISK DISTRIBUTION")
    print("=" * 70)

    distribution = (
        result_df["risk_level"]
        .value_counts()
    )

    print(
        distribution.to_string()
    )

    print("\nRisk percentages:")

    percentages = (
        result_df["risk_level"]
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
# DISPLAY SAMPLE RISK RECORDS
# ============================================================

def display_sample_risks(
    result_df,
    number_of_records=10
):

    print("\n" + "=" * 70)
    print("SAMPLE MAINTENANCE RISK ANALYSIS")
    print("=" * 70)

    columns = [
        "risk_score",
        "risk_level",
        "previous_failure",
        "previous_failure_rate",
        "previous_maintenance_count",
        "previous_average_cost",
        "previous_average_downtime",
        "days_since_previous_maintenance"
    ]

    sample = (
        result_df[
            columns
        ]
        .head(number_of_records)
    )

    print(
        sample.to_string(
            index=False
        )
    )


# ============================================================
# DISPLAY HIGH-RISK RECORDS
# ============================================================

def display_high_risk_records(
    result_df,
    number_of_records=10
):

    print("\n" + "=" * 70)
    print("HIGH-RISK MAINTENANCE RECORDS")
    print("=" * 70)

    high_risk = (
        result_df[
            result_df["risk_level"]
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
            "No HIGH-risk records found."
        )

        return

    columns = [
        "risk_score",
        "risk_level",
        "previous_failure",
        "previous_failure_rate",
        "frequent_maintenance",
        "previous_average_cost",
        "previous_average_downtime",
        "days_since_previous_maintenance"
    ]

    print(
        high_risk[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# SAVE RISK RESULTS
# ============================================================

def save_risk_results(
    result_df
):

    output_file = (
        PROCESSED_DATA_DIR
        / "maintenance_risk_analysis.csv"
    )

    result_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("RISK ANALYSIS SAVED")
    print("=" * 70)

    print(
        f"Output file:\n"
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("SMARTLOGIX AI")
    print("MAINTENANCE RISK ENGINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    df = load_maintenance_data()

    # --------------------------------------------------------
    # 2. Generate risk analysis
    # --------------------------------------------------------

    result_df = generate_risk_analysis(
        df
    )

    # --------------------------------------------------------
    # 3. Display risk distribution
    # --------------------------------------------------------

    display_risk_distribution(
        result_df
    )

    # --------------------------------------------------------
    # 4. Display sample records
    # --------------------------------------------------------

    display_sample_risks(
        result_df
    )

    # --------------------------------------------------------
    # 5. Display high-risk records
    # --------------------------------------------------------

    display_high_risk_records(
        result_df
    )

    # --------------------------------------------------------
    # 6. Save results
    # --------------------------------------------------------

    save_risk_results(
        result_df
    )

    print("\n" + "=" * 70)
    print("MAINTENANCE RISK ENGINE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()