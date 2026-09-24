import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from src.config.config import (
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    RANDOM_STATE
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "maintenance_features.csv"
)

TARGET_COLUMN = "failure_reported"

FAILURE_LABELS = {
    0: "No Failure",
    1: "Failure Reported"
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING MAINTENANCE DATA")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Dataset shape : {df.shape}"
    )

    return df


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def analyze_target_distribution(df):

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    percentages = (
        df[TARGET_COLUMN]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    for target_value in counts.index:

        label = FAILURE_LABELS.get(
            target_value,
            str(target_value)
        )

        print(
            f"{label:20s} : "
            f"{counts[target_value]:4d} "
            f"({percentages[target_value]:.2f}%)"
        )

    return counts


# ============================================================
# FAILURE RATE BY SERVICE TYPE
# ============================================================

def analyze_service_type(df):

    print("\n" + "=" * 70)
    print("FAILURE RATE BY SERVICE TYPE")
    print("=" * 70)

    result = (
        df.groupby("service_type")
        .agg(
            maintenance_count=(
                TARGET_COLUMN,
                "count"
            ),
            failures=(
                TARGET_COLUMN,
                "sum"
            ),
            failure_rate=(
                TARGET_COLUMN,
                "mean"
            )
        )
        .reset_index()
    )

    result["failure_rate_percent"] = (
        result["failure_rate"] * 100
    )

    result = result.sort_values(
        "failure_rate",
        ascending=False
    )

    print(
        result[
            [
                "service_type",
                "maintenance_count",
                "failures",
                "failure_rate_percent"
            ]
        ].to_string(
            index=False
        )
    )

    return result


# ============================================================
# FAILURE RATE BY NUMERICAL FEATURE BINS
# ============================================================

def analyze_numeric_feature(
    df,
    column,
    number_of_bins=5
):

    print("\n" + "=" * 70)
    print(
        f"FAILURE RATE BY {column.upper()}"
    )
    print("=" * 70)

    try:

        df_temp = df[
            [
                column,
                TARGET_COLUMN
            ]
        ].copy()

        df_temp["bin"] = pd.qcut(
            df_temp[column],
            q=number_of_bins,
            duplicates="drop"
        )

        result = (
            df_temp.groupby(
                "bin",
                observed=True
            )
            .agg(
                record_count=(
                    TARGET_COLUMN,
                    "count"
                ),
                failures=(
                    TARGET_COLUMN,
                    "sum"
                ),
                failure_rate=(
                    TARGET_COLUMN,
                    "mean"
                )
            )
            .reset_index()
        )

        result["failure_rate_percent"] = (
            result["failure_rate"] * 100
        )

        print(
            result[
                [
                    "bin",
                    "record_count",
                    "failures",
                    "failure_rate_percent"
                ]
                .to_string(index=False)
            ]
        )

        return result

    except Exception as error:

        print(
            f"Unable to analyze {column}: "
            f"{error}"
        )

        return pd.DataFrame()


# ============================================================
# FAILURE RATE BY PARTS REPLACED
# ============================================================

def analyze_parts_replaced(df):

    print("\n" + "=" * 70)
    print("FAILURE RATE BY PARTS REPLACED")
    print("=" * 70)

    result = (
        df.groupby("parts_replaced")
        .agg(
            maintenance_count=(
                TARGET_COLUMN,
                "count"
            ),
            failures=(
                TARGET_COLUMN,
                "sum"
            ),
            failure_rate=(
                TARGET_COLUMN,
                "mean"
            )
        )
        .reset_index()
    )

    result["failure_rate_percent"] = (
        result["failure_rate"] * 100
    )

    result = result.sort_values(
        "maintenance_count",
        ascending=False
    )

    print(
        result[
            [
                "parts_replaced",
                "maintenance_count",
                "failures",
                "failure_rate_percent"
            ]
        ].to_string(
            index=False
        )
    )

    return result


# ============================================================
# NUMERICAL CORRELATION
# ============================================================

def analyze_correlations(df):

    print("\n" + "=" * 70)
    print("NUMERICAL FEATURE CORRELATION")
    print("=" * 70)

    numerical_df = (
        df.select_dtypes(
            include=["number"]
        )
        .copy()
    )

    correlation = (
        numerical_df.corr(
            method="pearson"
        )[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(
            key=lambda series:
            series.abs(),
            ascending=False
        )
    )

    print(
        "Correlation with failure_reported:"
    )

    print(
        correlation.to_string()
    )

    return correlation


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

def analyze_feature_importance(df):

    print("\n" + "=" * 70)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)

    model_df = df.copy()

    # --------------------------------------------------------
    # One-hot encode categorical variables
    # --------------------------------------------------------

    model_df = pd.get_dummies(
        model_df,
        columns=[
            "service_type",
            "parts_replaced"
        ],
        drop_first=False
    )

    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    X = model_df.drop(
        columns=[TARGET_COLUMN]
    )

    y = model_df[TARGET_COLUMN]

    # --------------------------------------------------------
    # Replace non-finite values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    # Any remaining missing values
    X = X.fillna(0)

    # --------------------------------------------------------
    # Train diagnostic Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X,
        y
    )

    importance_df = pd.DataFrame({

        "feature": X.columns,

        "importance":
            model.feature_importances_

    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        importance_df.head(20)
        .to_string(index=False)
    )

    return importance_df


# ============================================================
# CREATE FAILURE RATE VISUALIZATION
# ============================================================

def create_service_failure_plot(
    service_result
):

    print("\n" + "=" * 70)
    print("CREATING FAILURE RATE VISUALIZATION")
    print("=" * 70)

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        service_result["service_type"],
        service_result["failure_rate_percent"]
    )

    plt.xlabel(
        "Service Type"
    )

    plt.ylabel(
        "Failure Rate (%)"
    )

    plt.title(
        "Failure Rate by Maintenance Service Type"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "maintenance_failure_rate_by_service.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Visualization saved to:\n"
        f"{output_file}"
    )


# ============================================================
# CREATE NUMERICAL FAILURE RATE PLOTS
# ============================================================

def create_numeric_failure_plot(
    df,
    column,
    number_of_bins=5
):

    try:

        df_temp = df[
            [
                column,
                TARGET_COLUMN
            ]
        ].copy()

        df_temp["bin"] = pd.qcut(
            df_temp[column],
            q=number_of_bins,
            duplicates="drop"
        )

        result = (
            df_temp.groupby(
                "bin",
                observed=True
            )[TARGET_COLUMN]
            .mean()
            * 100
        )

        plt.figure(
            figsize=(9, 6)
        )

        plt.bar(
            range(len(result)),
            result.values
        )

        plt.xlabel(
            f"{column} Range"
        )

        plt.ylabel(
            "Failure Rate (%)"
        )

        plt.title(
            f"Failure Rate by {column}"
        )

        plt.xticks(
            range(len(result)),
            [
                str(value)
                for value in result.index
            ],
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        output_file = (
            FIGURES_DIR
            / f"maintenance_failure_rate_{column}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_file}"
        )

    except Exception as error:

        print(
            f"Could not create plot for "
            f"{column}: {error}"
        )


# ============================================================
# PRINT KEY FINDINGS
# ============================================================

def print_key_findings(
    service_result,
    correlation,
    importance_df
):

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # --------------------------------------------------------
    # Service type
    # --------------------------------------------------------

    if not service_result.empty:

        highest_service = (
            service_result.iloc[0]
        )

        lowest_service = (
            service_result
            .sort_values(
                "failure_rate"
            )
            .iloc[0]
        )

        print(
            "\nService type:"
        )

        print(
            f"Highest observed failure rate : "
            f"{highest_service['service_type']} "
            f"({highest_service['failure_rate_percent']:.2f}%)"
        )

        print(
            f"Lowest observed failure rate  : "
            f"{lowest_service['service_type']} "
            f"({lowest_service['failure_rate_percent']:.2f}%)"
        )

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    print(
        "\nStrongest numerical correlations "
        "with failure_reported:"
    )

    print(
        correlation.head(5).to_string()
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print(
        "\nTop Random Forest diagnostic features:"
    )

    print(
        importance_df.head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Overall interpretation
    # --------------------------------------------------------

    print(
        "\nInterpretation:"
    )

    print(
        "Use these relationships to determine whether "
        "the current maintenance dataset contains "
        "predictive patterns for failure_reported."
    )

    print(
        "A feature with high correlation or importance "
        "is not automatically causal or leakage-free."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("SMARTLOGIX AI - MAINTENANCE TARGET ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Target distribution
    # --------------------------------------------------------

    analyze_target_distribution(
        df
    )

    # --------------------------------------------------------
    # 3. Service type analysis
    # --------------------------------------------------------

    service_result = analyze_service_type(
        df
    )

    # --------------------------------------------------------
    # 4. Numerical feature analysis
    # --------------------------------------------------------

    numerical_features = [
        "labour_hours",
        "cost_inr",
        "downtime_hours",
        "vehicle_maintenance_count",
        "vehicle_avg_maintenance_cost",
        "vehicle_avg_downtime_hours",
        "vehicle_avg_labour_hours",
        "cost_per_labour_hour",
        "cost_per_downtime_hour",
        "service_type_frequency",
        "parts_replaced_count"
    ]

    numeric_results = {}

    for column in numerical_features:

        if column in df.columns:

            numeric_results[column] = (
                analyze_numeric_feature(
                    df,
                    column
                )
            )

    # --------------------------------------------------------
    # 5. Parts analysis
    # --------------------------------------------------------

    parts_result = analyze_parts_replaced(
        df
    )

    # --------------------------------------------------------
    # 6. Correlation analysis
    # --------------------------------------------------------

    correlation = analyze_correlations(
        df
    )

    # --------------------------------------------------------
    # 7. Feature importance
    # --------------------------------------------------------

    importance_df = analyze_feature_importance(
        df
    )

    # --------------------------------------------------------
    # 8. Visualizations
    # --------------------------------------------------------

    create_service_failure_plot(
        service_result
    )

    plot_features = [
        "labour_hours",
        "cost_inr",
        "downtime_hours"
    ]

    for column in plot_features:

        if column in df.columns:

            create_numeric_failure_plot(
                df,
                column
            )

    # --------------------------------------------------------
    # 9. Key findings
    # --------------------------------------------------------

    print_key_findings(
        service_result,
        correlation,
        importance_df
    )

    print("\n" + "=" * 70)
    print("MAINTENANCE TARGET ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()