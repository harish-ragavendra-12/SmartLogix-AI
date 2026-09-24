
import matplotlib.pyplot as plt
import pandas as pd

from src.config.config import PROCESSED_DATA_DIR, FIGURES_DIR


def load_cleaned_data(file_name):

    df = pd.read_csv(PROCESSED_DATA_DIR / file_name)

    df["service_date"] = pd.to_datetime(df["service_date"])

    return df

def basic_inspection(df):

    print("Shape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nStatistical summary:")
    print(df.describe(include="all"))

def data_quality_checks(df):

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nUnique values:")
    print(df.nunique())

    # Example categorical column
    print("\nFailure status:")
    print(df["failure_reported"].value_counts(dropna=False))

def univariate_analysis(df):

    # Numerical variable
    print("\nMaintenance cost statistics:")
    print(df["cost_inr"].describe())

    # Categorical variable
    print("\nFailure distribution:")
    print(df["failure_reported"].value_counts(dropna=False))

def bivariate_analysis(df):

    # failure status vs maintenance cost
    print("\nAverage maintenance cost by failure status:")
    print(df.groupby("failure_reported")["cost_inr"].mean())

    # failure status vs downtime
    print("\nAverage downtime by failure status:")
    print(df.groupby("failure_reported")["downtime_hours"].mean())

    # service type vs maintenance cost
    print("\nAverage maintenance cost by service type:")
    print(df.groupby("service_type")["cost_inr"].mean())

    # service type vs downtime
    print("\nAverage downtime by service type:")
    print(df.groupby("service_type")["downtime_hours"].mean())

    # labour hours vs maintenance cost
    print("\nCorrelation between labour hours and maintenance cost:")
    print(df["labour_hours"].corr(df["cost_inr"]))

def multivariate_analysis(df):

    print("\nAverage maintenance cost by service type and failure status:")
    print(df.groupby(["service_type", "failure_reported"])["cost_inr"].mean())

    print("\nAverage downtime by service type and failure status:")
    print(df.groupby(["service_type", "failure_reported"])["downtime_hours"].mean())

    print("\nAverage labour hours and maintenance cost by service type:")
    print(df.groupby("service_type")[["labour_hours", "cost_inr"]].mean())

    print("\nAverage labour hours by service type and failure status:")
    print(df.groupby(["service_type", "failure_reported"])["labour_hours"].mean())

    print("\nMaintenance records and average cost by vehicle:")
    print(
        df.groupby("vehicle_id")
        .agg(
            maintenance_count=("work_order_id", "count"),
            average_cost=("cost_inr", "mean")
        )
        .sort_values("maintenance_count", ascending=False)
        .head(10)
    )


def visualizations(df):

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Visualization 1
    print("\nGenerating failure status distribution...")

    failure_counts = df["failure_reported"].value_counts()

    plt.figure(figsize=(8, 5))

    failure_counts.plot(kind="bar")

    plt.title("Failure Status Distribution")

    plt.xlabel("Failure Status")
    plt.ylabel("Number of Maintenance Records")

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "maintenance_failure_status.png", dpi=300)

    plt.show()

    plt.close()


    # Visualization 2
    print("\nGenerating maintenance cost distribution...")

    plt.figure(figsize=(8, 5))

    plt.hist(df["cost_inr"], bins=20)

    plt.title("Maintenance Cost Distribution")

    plt.xlabel("Maintenance Cost (INR)")
    plt.ylabel("Number of Maintenance Records")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "maintenance_cost_distribution.png",
        dpi=300
    )

    plt.show()

    plt.close()


    # Visualization 3
    print("\nGenerating maintenance cost boxplot...")

    plt.figure(figsize=(8, 5))

    plt.boxplot(df["cost_inr"])

    plt.title("Maintenance Cost Boxplot")

    plt.ylabel("Maintenance Cost (INR)")

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "maintenance_cost_boxplot.png", dpi=300)

    plt.show()

    plt.close()


    # Visualization 4
    print("\nGenerating average maintenance cost by service type...")

    service_cost = df.groupby("service_type")["cost_inr"].mean().sort_values()

    plt.figure(figsize=(10, 6))

    service_cost.plot(kind="barh")

    plt.title("Average Maintenance Cost by Service Type")

    plt.xlabel("Average Maintenance Cost (INR)")
    plt.ylabel("Service Type")

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "maintenance_cost_by_service_type.png", dpi=300)

    plt.show()

    plt.close()


    # Visualization 5
    print("\nGenerating average downtime by service type...")

    service_downtime = (df.groupby("service_type")["downtime_hours"].mean().sort_values())

    plt.figure(figsize=(10, 6))

    service_downtime.plot(kind="barh")

    plt.title("Average Downtime by Service Type")

    plt.xlabel("Average Downtime (Hours)")
    plt.ylabel("Service Type")

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "downtime_by_service_type.png", dpi=300)

    plt.show()

    plt.close()


    # Visualization 6
    print("\nGenerating labour hours vs maintenance cost...")

    plt.figure(figsize=(8, 5))

    plt.scatter(df["labour_hours"], df["cost_inr"])

    plt.title("Labour Hours vs Maintenance Cost")

    plt.xlabel("Labour Hours")
    plt.ylabel("Maintenance Cost (INR)")

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "labour_hours_vs_cost.png", dpi=300)

    plt.show()

    plt.close()


    # Visualization 7
    print("\nGenerating failure status by service type...")

    failure_service = pd.crosstab(
        df["service_type"],
        df["failure_reported"]
    )

    failure_service.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6)
    )

    plt.title("Failure Status by Service Type")

    plt.xlabel("Service Type")
    plt.ylabel("Number of Maintenance Records")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "failure_status_by_service_type.png", dpi=300)

    plt.show()

    plt.close()


    # Visualization 8
    print("\nGenerating monthly maintenance activity...")

    monthly_maintenance = (
        df.groupby(df["service_date"].dt.to_period("M"))
        .size()
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        monthly_maintenance.index.astype(str),
        monthly_maintenance.values
    )

    plt.title("Monthly Maintenance Activity")

    plt.xlabel("Month")
    plt.ylabel("Number of Maintenance Records")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "monthly_maintenance_activity.png", dpi=300)

    plt.show()

    plt.close()

def business_insights(df):

    print("\n========== BUSINESS INSIGHTS ==========")

    # 1. Overall maintenance cost
    average_cost = df["cost_inr"].mean()
    median_cost = df["cost_inr"].median()

    print("\n1. Overall Maintenance Cost:")
    print(f"Average maintenance cost: ₹{average_cost:,.2f}")
    print(f"Median maintenance cost: ₹{median_cost:,.2f}")

    # 2. Highest-cost service
    service_cost = (
        df.groupby("service_type")["cost_inr"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n2. Service Type with Highest Average Cost:")
    print(f"{service_cost.index[0]}: ₹{service_cost.iloc[0]:,.2f}")

    # 3. Highest-downtime service
    service_downtime = (
        df.groupby("service_type")["downtime_hours"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n3. Service Type with Highest Average Downtime:")
    print(
        f"{service_downtime.index[0]}: "
        f"{service_downtime.iloc[0]:.2f} hours"
    )

    # 4. Failure distribution
    failure_distribution = (
        df["failure_reported"]
        .value_counts()
    )

    print("\n4. Failure Status Distribution:")

    for status, count in failure_distribution.items():
        percentage = count / len(df) * 100

        print(
            f"{status}: {count} records "
            f"({percentage:.2f}%)"
        )

    # 5. Vehicle maintenance frequency and average cost
    vehicle_summary = (
        df.groupby("vehicle_id")
        .agg(
            maintenance_count=("work_order_id", "count"),
            average_cost=("cost_inr", "mean")
        )
    )

    filtered_vehicle_summary = (
        vehicle_summary[
            vehicle_summary["maintenance_count"] >= 3
            ]
        .sort_values("average_cost", ascending=False)
    )

    print("\n5. Highest Average-Cost Vehicles (minimum 3 records):")
    print(filtered_vehicle_summary.head(10))

    # 6. Labour hours and maintenance cost relationship
    labour_cost_correlation = (
        df["labour_hours"].corr(df["cost_inr"])
    )

    print("\n6. Labour Hours vs Maintenance Cost:")
    print(
        f"Correlation: {labour_cost_correlation:.4f}"
    )

    # 7. Highest-downtime vehicles
    vehicle_downtime = (
        df.groupby("vehicle_id")
        .agg(
            maintenance_count=("work_order_id", "count"),
            average_downtime=("downtime_hours", "mean")
        )
    )

    filtered_vehicle_downtime = (
        vehicle_downtime[
            vehicle_downtime["maintenance_count"] >= 3
            ]
        .sort_values("average_downtime", ascending=False)
    )

    print("\n7. Highest Average-Downtime Vehicles (minimum 3 records):")
    print(filtered_vehicle_downtime.head(10))

    # 8. Monthly maintenance activity

    monthly_maintenance = (
        df.groupby(df["service_date"].dt.to_period("M"))
        .size()
    )

    highest_month = monthly_maintenance.idxmax()
    highest_count = monthly_maintenance.max()

    lowest_month = monthly_maintenance.idxmin()
    lowest_count = monthly_maintenance.min()

    print("\n8. Monthly Maintenance Activity:")
    print(
        f"Highest maintenance activity: "
        f"{highest_month} ({highest_count} records)"
    )
    print(
        f"Lowest maintenance activity: "
        f"{lowest_month} ({lowest_count} records)"
    )

def main():

    df = load_cleaned_data("maintenance_history_clean.csv")

    basic_inspection(df)

    data_quality_checks(df)

    univariate_analysis(df)

    bivariate_analysis(df)

    multivariate_analysis(df)

    visualizations(df)

    business_insights(df)

if __name__ == "__main__":
    main()