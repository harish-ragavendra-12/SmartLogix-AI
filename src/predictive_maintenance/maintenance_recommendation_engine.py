import pandas as pd

from src.config.config import PROCESSED_DATA_DIR


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "vehicle_maintenance_risk.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "maintenance_recommendations.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_vehicle_risk_data():

    print("\n" + "=" * 70)
    print("LOADING VEHICLE MAINTENANCE RISK DATA")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE
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
# ASSIGN PRIORITY
# ============================================================

def assign_priority(row):

    risk_level = row["risk_level"]
    risk_score = row["risk_score"]

    # --------------------------------------------------------
    # HIGH risk
    # --------------------------------------------------------

    if risk_level == "HIGH":

        if risk_score >= 85:

            return "CRITICAL"

        return "HIGH"

    # --------------------------------------------------------
    # MEDIUM risk
    # --------------------------------------------------------

    elif risk_level == "MEDIUM":

        if risk_score >= 55:

            return "MEDIUM"

        return "LOW"

    # --------------------------------------------------------
    # LOW risk
    # --------------------------------------------------------

    return "LOW"


# ============================================================
# GENERATE PRIMARY ACTION
# ============================================================

def generate_primary_action(row):

    priority = row["priority"]

    # --------------------------------------------------------
    # Critical
    # --------------------------------------------------------

    if priority == "CRITICAL":

        return (
            "Inspect vehicle before assigning it "
            "to critical deliveries."
        )

    # --------------------------------------------------------
    # High
    # --------------------------------------------------------

    if priority == "HIGH":

        return (
            "Schedule preventive maintenance inspection "
            "before the next major delivery assignment."
        )

    # --------------------------------------------------------
    # Medium
    # --------------------------------------------------------

    if priority == "MEDIUM":

        return (
            "Monitor vehicle condition and schedule "
            "maintenance during the next service window."
        )

    # --------------------------------------------------------
    # Low
    # --------------------------------------------------------

    return (
        "Continue routine monitoring and follow "
        "the standard preventive maintenance schedule."
    )


# ============================================================
# GENERATE FAILURE-RELATED ACTION
# ============================================================

def generate_failure_action(row):

    failure_rate = row["failure_rate"]
    failure_count = row["failure_count"]

    if failure_rate >= 0.75:

        return (
            f"Review failure history closely; "
            f"{failure_count} recorded failure(s) "
            f"across {row['maintenance_count']} "
            f"maintenance record(s)."
        )

    if failure_rate >= 0.50:

        return (
            "Review previous failure events and "
            "inspect failure-prone components."
        )

    if failure_rate >= 0.30:

        return (
            "Monitor failure history during the "
            "next maintenance inspection."
        )

    return (
        "No strong historical failure pattern detected."
    )


# ============================================================
# GENERATE DOWNTIME ACTION
# ============================================================

def generate_downtime_action(row):

    downtime = row[
        "average_downtime_hours"
    ]

    if downtime >= 60:

        return (
            "Investigate causes of prolonged downtime "
            "and prioritize downtime-reduction measures."
        )

    if downtime >= 40:

        return (
            "Review recent maintenance events to identify "
            "sources of elevated downtime."
        )

    if downtime >= 25:

        return (
            "Monitor maintenance downtime for recurring issues."
        )

    return (
        "Historical downtime is within the lower-risk range."
    )


# ============================================================
# GENERATE COST ACTION
# ============================================================

def generate_cost_action(row):

    cost = row[
        "average_maintenance_cost"
    ]

    if cost >= 40000:

        return (
            "Review high maintenance-cost events and "
            "evaluate recurring component replacement costs."
        )

    if cost >= 30000:

        return (
            "Monitor maintenance expenditure and "
            "investigate recurring high-cost repairs."
        )

    if cost >= 25000:

        return (
            "Track maintenance cost trends during "
            "future service events."
        )

    return (
        "No major historical maintenance-cost concern detected."
    )


# ============================================================
# GENERATE RECENCY ACTION
# ============================================================

def generate_recency_action(row):

    days = row[
        "days_since_last_maintenance"
    ]

    if days > 365:

        return (
            "Maintenance has not been recorded for over a year; "
            "verify whether inspection is overdue."
        )

    if days > 180:

        return (
            "Long interval since the last recorded maintenance; "
            "verify current vehicle condition."
        )

    if days > 90:

        return (
            "Review the vehicle's maintenance schedule."
        )

    if days <= 30:

        return (
            "Recent maintenance activity is recorded; "
            "monitor subsequent vehicle performance."
        )

    return (
        "Maintenance recency does not currently require "
        "additional action."
    )


# ============================================================
# GENERATE OVERALL RECOMMENDATION
# ============================================================

def generate_overall_recommendation(row):

    priority = row["priority"]

    if priority == "CRITICAL":

        return (
            "Immediate preventive inspection recommended "
            "before further critical delivery assignments."
        )

    if priority == "HIGH":

        return (
            "Prioritize preventive maintenance and review "
            "failure, downtime, and cost history."
        )

    if priority == "MEDIUM":

        return (
            "Continue monitoring and address identified "
            "maintenance indicators during the next service window."
        )

    return (
        "Continue routine preventive maintenance and monitoring."
    )


# ============================================================
# GENERATE ALL RECOMMENDATIONS
# ============================================================

def generate_recommendations(df):

    print("\n" + "=" * 70)
    print("GENERATING MAINTENANCE RECOMMENDATIONS")
    print("=" * 70)

    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

    df["priority"] = (
        df.apply(
            assign_priority,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    df["primary_action"] = (
        df.apply(
            generate_primary_action,
            axis=1
        )
    )

    df["failure_action"] = (
        df.apply(
            generate_failure_action,
            axis=1
        )
    )

    df["downtime_action"] = (
        df.apply(
            generate_downtime_action,
            axis=1
        )
    )

    df["cost_action"] = (
        df.apply(
            generate_cost_action,
            axis=1
        )
    )

    df["recency_action"] = (
        df.apply(
            generate_recency_action,
            axis=1
        )
    )

    df["overall_recommendation"] = (
        df.apply(
            generate_overall_recommendation,
            axis=1
        )
    )

    return df


# ============================================================
# ASSIGN MAINTENANCE CATEGORY
# ============================================================

def assign_maintenance_category(row):

    if row["priority"] in [
        "CRITICAL",
        "HIGH"
    ]:

        if (
            row["failure_rate"] >= 0.50
            and
            row["average_downtime_hours"] >= 40
        ):

            return "Failure & Downtime Investigation"

        if row["failure_rate"] >= 0.50:

            return "Failure History Review"

        if row["average_downtime_hours"] >= 40:

            return "Downtime Investigation"

        if row["average_maintenance_cost"] >= 30000:

            return "Maintenance Cost Investigation"

        return "Preventive Maintenance"

    if row["priority"] == "MEDIUM":

        return "Scheduled Monitoring"

    return "Routine Monitoring"


# ============================================================
# ADD MAINTENANCE CATEGORY
# ============================================================

def add_maintenance_category(df):

    print("\n" + "=" * 70)
    print("ASSIGNING MAINTENANCE ACTION CATEGORY")
    print("=" * 70)

    df["maintenance_category"] = (
        df.apply(
            assign_maintenance_category,
            axis=1
        )
    )

    return df


# ============================================================
# DISPLAY PRIORITY DISTRIBUTION
# ============================================================

def display_priority_distribution(df):

    print("\n" + "=" * 70)
    print("MAINTENANCE PRIORITY DISTRIBUTION")
    print("=" * 70)

    distribution = (
        df["priority"]
        .value_counts()
    )

    print(
        distribution.to_string()
    )

    print("\nPriority percentages:")

    percentages = (
        df["priority"]
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
# DISPLAY ACTION CATEGORY DISTRIBUTION
# ============================================================

def display_category_distribution(df):

    print("\n" + "=" * 70)
    print("MAINTENANCE ACTION CATEGORY DISTRIBUTION")
    print("=" * 70)

    distribution = (
        df["maintenance_category"]
        .value_counts()
    )

    print(
        distribution.to_string()
    )


# ============================================================
# DISPLAY CRITICAL VEHICLES
# ============================================================

def display_critical_vehicles(
    df,
    number_of_records=10
):

    print("\n" + "=" * 70)
    print("CRITICAL MAINTENANCE VEHICLES")
    print("=" * 70)

    critical = (
        df[
            df["priority"]
            == "CRITICAL"
        ]
        .sort_values(
            by="risk_score",
            ascending=False
        )
        .head(number_of_records)
    )

    if critical.empty:

        print(
            "No CRITICAL vehicles found."
        )

        return

    columns = [
        "vehicle_id",
        "risk_score",
        "risk_level",
        "priority",
        "maintenance_count",
        "failure_count",
        "failure_rate",
        "average_maintenance_cost",
        "average_downtime_hours",
        "maintenance_category"
    ]

    print(
        critical[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# DISPLAY DETAILED RECOMMENDATIONS
# ============================================================

def display_detailed_recommendations(
    df,
    number_of_records=10
):

    print("\n" + "=" * 70)
    print("TOP VEHICLE MAINTENANCE RECOMMENDATIONS")
    print("=" * 70)

    priority_order = {
        "CRITICAL": 1,
        "HIGH": 2,
        "MEDIUM": 3,
        "LOW": 4
    }

    display_df = (
        df.assign(
            priority_order=
            df["priority"]
            .map(priority_order)
        )
        .sort_values(
            by=[
                "priority_order",
                "risk_score"
            ],
            ascending=[
                True,
                False
            ]
        )
        .head(number_of_records)
    )

    for _, row in display_df.iterrows():

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
            f"Priority: "
            f"{row['priority']}"
        )

        print(
            f"Category: "
            f"{row['maintenance_category']}"
        )

        print(
            f"Failure Rate: "
            f"{row['failure_rate']:.2%}"
        )

        print(
            f"Average Downtime: "
            f"{row['average_downtime_hours']:.2f} hours"
        )

        print(
            f"Average Maintenance Cost: "
            f"₹{row['average_maintenance_cost']:,.2f}"
        )

        print(
            f"Primary Action: "
            f"{row['primary_action']}"
        )

        print(
            f"Failure Action: "
            f"{row['failure_action']}"
        )

        print(
            f"Downtime Action: "
            f"{row['downtime_action']}"
        )

        print(
            f"Cost Action: "
            f"{row['cost_action']}"
        )

        print(
            f"Recency Action: "
            f"{row['recency_action']}"
        )

        print(
            f"Overall Recommendation: "
            f"{row['overall_recommendation']}"
        )


# ============================================================
# SAVE RECOMMENDATIONS
# ============================================================

def save_recommendations(df):

    print("\n" + "=" * 70)
    print("SAVING MAINTENANCE RECOMMENDATIONS")
    print("=" * 70)

    # --------------------------------------------------------
    # Remove temporary sorting column if present
    # --------------------------------------------------------

    if "priority_order" in df.columns:

        df = df.drop(
            columns=["priority_order"]
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Output file:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"Vehicles saved : "
        f"{len(df)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("SMARTLOGIX AI")
    print("MAINTENANCE RECOMMENDATION ENGINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load vehicle risk data
    # --------------------------------------------------------

    df = load_vehicle_risk_data()

    # --------------------------------------------------------
    # 2. Generate recommendations
    # --------------------------------------------------------

    df = generate_recommendations(
        df
    )

    # --------------------------------------------------------
    # 3. Assign action category
    # --------------------------------------------------------

    df = add_maintenance_category(
        df
    )

    # --------------------------------------------------------
    # 4. Display priority distribution
    # --------------------------------------------------------

    display_priority_distribution(
        df
    )

    # --------------------------------------------------------
    # 5. Display category distribution
    # --------------------------------------------------------

    display_category_distribution(
        df
    )

    # --------------------------------------------------------
    # 6. Display critical vehicles
    # --------------------------------------------------------

    display_critical_vehicles(
        df
    )

    # --------------------------------------------------------
    # 7. Display detailed recommendations
    # --------------------------------------------------------

    display_detailed_recommendations(
        df
    )

    # --------------------------------------------------------
    # 8. Save
    # --------------------------------------------------------

    save_recommendations(
        df
    )

    print("\n" + "=" * 70)
    print(
        "MAINTENANCE RECOMMENDATION ENGINE "
        "COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()