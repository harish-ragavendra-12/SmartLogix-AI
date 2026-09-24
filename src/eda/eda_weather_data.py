import matplotlib.pyplot as plt
import pandas as pd

from src.config.config import PROCESSED_DATA_DIR, FIGURES_DIR


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

def load_cleaned_data(file_name):
    file_path = PROCESSED_DATA_DIR / file_name

    print(f"\nLoading dataset from: {file_path}")

    df = pd.read_csv(file_path)

    # Convert actual date/timestamp columns
    date_columns = [
        column
        for column in df.columns
        if "date" in column.lower()
        or "timestamp" in column.lower()
        or "datetime" in column.lower()
    ]

    for column in date_columns:
        try:
            df[column] = pd.to_datetime(df[column], errors="coerce")
        except Exception:
            pass

    return df


# ============================================================
# 2. BASIC INSPECTION
# ============================================================

def basic_inspection(df):

    print("\n" + "=" * 60)
    print("BASIC INSPECTION")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nNumerical columns:")
    print(
        df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
    )

    print("\nCategorical columns:")
    print(
        df.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
    )


# ============================================================
# 3. DATA QUALITY CHECKS
# ============================================================

def data_quality_checks(df):

    print("\n" + "=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)

    print("\nMissing values:")
    missing_values = df.isnull().sum()

    missing_percentage = (
        df.isnull().mean() * 100
    ).round(2)

    missing_summary = pd.DataFrame({
        "Missing Count": missing_values,
        "Missing Percentage": missing_percentage
    })

    print(
        missing_summary[
            missing_summary["Missing Count"] > 0
        ]
    )

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nUnique values:")
    for column in df.columns:
        print(
            f"{column}: "
            f"{df[column].nunique(dropna=True)}"
        )


# ============================================================
# 4. UNIVARIATE ANALYSIS
# ============================================================

def univariate_analysis(df):

    print("\n" + "=" * 60)
    print("UNIVARIATE ANALYSIS")
    print("=" * 60)

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    # Numerical analysis
    print("\n--- Numerical Variables ---")

    for column in numerical_columns:

        print(f"\n{column}")

        print(
            df[column].describe()
        )

    # Categorical analysis
    print("\n--- Categorical Variables ---")

    for column in categorical_columns:

        print(f"\n{column}")

        print(
            df[column]
            .value_counts(dropna=False)
            .head(15)
        )


# ============================================================
# 5. BIVARIATE ANALYSIS
# ============================================================

def bivariate_analysis(df):

    print("\n" + "=" * 60)
    print("BIVARIATE ANALYSIS")
    print("=" * 60)

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    # --------------------------------------------------------
    # Numerical vs Numerical
    # --------------------------------------------------------

    if len(numerical_columns) >= 2:

        print("\n--- Numerical vs Numerical Correlation ---")

        correlation_matrix = df[
            numerical_columns
        ].corr()

        print(
            correlation_matrix.round(3)
        )

    # --------------------------------------------------------
    # Remove high-cardinality identifiers
    # --------------------------------------------------------

    identifier_columns = [
        column
        for column in categorical_columns
        if column.lower().endswith("_id")
        or column.lower() in [
            "id",
            "weather_id"
        ]
    ]

    analysis_categorical_columns = [
        column
        for column in categorical_columns
        if column not in identifier_columns
    ]

    # --------------------------------------------------------
    # Categorical vs Numerical
    # --------------------------------------------------------

    if (
        len(analysis_categorical_columns) > 0
        and len(numerical_columns) > 0
    ):

        print(
            "\n--- Categorical vs Numerical ---"
        )

        for categorical_column in analysis_categorical_columns:

            # Limit to reasonable cardinality
            if (
                df[categorical_column]
                .nunique(dropna=True) <= 20
            ):

                for numerical_column in numerical_columns:

                    print(
                        f"\nAverage "
                        f"{numerical_column} "
                        f"by "
                        f"{categorical_column}:"
                    )

                    result = (
                        df.groupby(
                            categorical_column,
                            observed=True
                        )[numerical_column]
                        .mean()
                        .sort_values(
                            ascending=False
                        )
                    )

                    print(
                        result.round(2)
                    )


# ============================================================
# 6. MULTIVARIATE ANALYSIS
# ============================================================

def multivariate_analysis(df):

    print("\n" + "=" * 60)
    print("MULTIVARIATE ANALYSIS")
    print("=" * 60)

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    if len(numerical_columns) >= 2:

        print("\nCorrelation Matrix:")

        correlation_matrix = (
            df[numerical_columns]
            .corr()
            .round(3)
        )

        print(correlation_matrix)

        # Find strongest correlations
        correlation_pairs = []

        for i in range(
            len(correlation_matrix.columns)
        ):

            for j in range(i + 1,
                           len(correlation_matrix.columns)):

                column_1 = (
                    correlation_matrix.columns[i]
                )

                column_2 = (
                    correlation_matrix.columns[j]
                )

                correlation_value = (
                    correlation_matrix.iloc[i, j]
                )

                correlation_pairs.append(
                    (
                        column_1,
                        column_2,
                        correlation_value
                    )
                )

        correlation_pairs.sort(
            key=lambda x: abs(x[2]),
            reverse=True
        )

        print(
            "\nStrongest numerical relationships:"
        )

        for pair in correlation_pairs[:10]:

            print(
                f"{pair[0]} vs {pair[1]}: "
                f"{pair[2]:.3f}"
            )


# ============================================================
# 7. VISUALIZATIONS
# ============================================================

def visualizations(df):

    print("\n" + "=" * 60)
    print("VISUALIZATIONS")
    print("=" * 60)

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    # --------------------------------------------------------
    # Visualization 1: Numerical Distributions
    # --------------------------------------------------------

    if len(numerical_columns) > 0:

        column = numerical_columns[0]

        plt.figure(figsize=(10, 6))

        plt.hist(
            df[column].dropna(),
            bins=30
        )

        plt.title(
            f"Distribution of {column}"
        )

        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.savefig(
            FIGURES_DIR /
            "weather_numeric_distribution.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Visualization 1 saved."
        )

    # --------------------------------------------------------
    # Visualization 2: Numerical Boxplot
    # --------------------------------------------------------

    if len(numerical_columns) > 0:

        column = numerical_columns[0]

        plt.figure(figsize=(10, 6))

        plt.boxplot(
            df[column].dropna()
        )

        plt.title(
            f"Boxplot of {column}"
        )

        plt.ylabel(column)

        plt.savefig(
            FIGURES_DIR /
            "weather_numeric_boxplot.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Visualization 2 saved."
        )

    # --------------------------------------------------------
    # Visualization 3: Categorical Distribution
    # --------------------------------------------------------

    identifier_columns = [
        column
        for column in categorical_columns
        if column.lower().endswith("_id")
        or column.lower() in [
            "id",
            "weather_id"
        ]
    ]

    analysis_categorical_columns = [
        column
        for column in categorical_columns
        if column not in identifier_columns
    ]

    if len(analysis_categorical_columns) > 0:

        column = (
            analysis_categorical_columns[0]
        )

        value_counts = (
            df[column]
            .value_counts()
            .head(15)
        )

        plt.figure(figsize=(10, 6))

        value_counts.plot(
            kind="bar"
        )

        plt.title(
            f"Distribution of {column}"
        )

        plt.xlabel(column)
        plt.ylabel("Count")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.subplots_adjust(
            left=0.10,
            right=0.95,
            top=0.90,
            bottom=0.30
        )

        plt.savefig(
            FIGURES_DIR /
            "weather_categorical_distribution.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Visualization 3 saved."
        )

    # --------------------------------------------------------
    # Visualization 4: Numerical Relationship
    # --------------------------------------------------------

    if len(numerical_columns) >= 2:

        x_column = numerical_columns[0]
        y_column = numerical_columns[1]

        plot_df = df[
            [x_column, y_column]
        ].dropna()

        plt.figure(figsize=(10, 6))

        plt.scatter(
            plot_df[x_column],
            plot_df[y_column],
            alpha=0.5
        )

        plt.title(
            f"{x_column} vs {y_column}"
        )

        plt.xlabel(x_column)
        plt.ylabel(y_column)

        plt.savefig(
            FIGURES_DIR /
            "weather_numeric_relationship.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Visualization 4 saved."
        )

    # --------------------------------------------------------
    # Visualization 5: Categorical vs Numerical
    # --------------------------------------------------------

    if (
        len(analysis_categorical_columns) > 0
        and len(numerical_columns) > 0
    ):

        categorical_column = (
            analysis_categorical_columns[0]
        )

        numerical_column = (
            numerical_columns[0]
        )

        if (
            df[categorical_column]
            .nunique(dropna=True) <= 15
        ):

            grouped_data = (
                df.groupby(
                    categorical_column,
                    observed=True
                )[numerical_column]
                .mean()
                .sort_values(
                    ascending=False
                )
            )

            plt.figure(
                figsize=(10, 6)
            )

            grouped_data.plot(
                kind="bar"
            )

            plt.title(
                f"Average {numerical_column} "
                f"by {categorical_column}"
            )

            plt.xlabel(
                categorical_column
            )

            plt.ylabel(
                f"Average {numerical_column}"
            )

            plt.xticks(
                rotation=45,
                ha="right"
            )

            plt.subplots_adjust(
                left=0.10,
                right=0.95,
                top=0.90,
                bottom=0.30
            )

            plt.savefig(
                FIGURES_DIR /
                "weather_categorical_relationship.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print(
                "Visualization 5 saved."
            )


# ============================================================
# 8. BUSINESS INSIGHTS
# ============================================================

def business_insights(df):

    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS")
    print("=" * 60)

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    # --------------------------------------------------------
    # Insight 1: Numerical Summary
    # --------------------------------------------------------

    if len(numerical_columns) > 0:

        print(
            "\n1. Numerical Variable Summary:"
        )

        for column in numerical_columns:

            print(
                f"{column}: "
                f"Mean = "
                f"{df[column].mean():.2f}, "
                f"Median = "
                f"{df[column].median():.2f}, "
                f"Min = "
                f"{df[column].min():.2f}, "
                f"Max = "
                f"{df[column].max():.2f}"
            )

    # --------------------------------------------------------
    # Insight 2: Highest Average Numerical Value
    # --------------------------------------------------------

    if len(numerical_columns) > 0:

        print(
            "\n2. Highest Average Numerical Variable:"
        )

        averages = (
            df[numerical_columns]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        print(
            f"{averages.index[0]}: "
            f"{averages.iloc[0]:.2f}"
        )

    # --------------------------------------------------------
    # Insight 3: Lowest Average Numerical Value
    # --------------------------------------------------------

    if len(numerical_columns) > 0:

        print(
            "\n3. Lowest Average Numerical Variable:"
        )

        averages = (
            df[numerical_columns]
            .mean()
            .sort_values()
        )

        print(
            f"{averages.index[0]}: "
            f"{averages.iloc[0]:.2f}"
        )

    # --------------------------------------------------------
    # Insight 4: Strongest Correlation
    # --------------------------------------------------------

    if len(numerical_columns) >= 2:

        correlation_matrix = (
            df[numerical_columns]
            .corr()
        )

        correlation_pairs = []

        for i in range(
            len(correlation_matrix.columns)
        ):

            for j in range(
                i + 1,
                len(correlation_matrix.columns)
            ):

                column_1 = (
                    correlation_matrix.columns[i]
                )

                column_2 = (
                    correlation_matrix.columns[j]
                )

                value = (
                    correlation_matrix.iloc[i, j]
                )

                correlation_pairs.append(
                    (
                        column_1,
                        column_2,
                        value
                    )
                )

        if correlation_pairs:

            strongest_pair = max(
                correlation_pairs,
                key=lambda x: abs(x[2])
            )

            print(
                "\n4. Strongest Numerical "
                "Relationship:"
            )

            print(
                f"{strongest_pair[0]} vs "
                f"{strongest_pair[1]}: "
                f"{strongest_pair[2]:.3f}"
            )

    # --------------------------------------------------------
    # Insight 5: Most Common Category
    # --------------------------------------------------------

    identifier_columns = [
        column
        for column in categorical_columns
        if column.lower().endswith("_id")
        or column.lower() in [
            "id",
            "weather_id"
        ]
    ]

    analysis_categorical_columns = [
        column
        for column in categorical_columns
        if column not in identifier_columns
    ]

    if len(analysis_categorical_columns) > 0:

        print(
            "\n5. Most Common Categories:"
        )

        for column in analysis_categorical_columns:

            if (
                df[column]
                .nunique(dropna=True) <= 20
            ):

                most_common = (
                    df[column]
                    .value_counts()
                    .idxmax()
                )

                count = (
                    df[column]
                    .value_counts()
                    .max()
                )

                print(
                    f"{column}: "
                    f"{most_common} "
                    f"({count} records)"
                )

    # --------------------------------------------------------
    # Insight 6: Missing Data
    # --------------------------------------------------------

    missing = (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    print(
        "\n6. Missing Data:"
    )

    if len(missing) > 0:

        for column, count in missing.items():

            percentage = (
                count / len(df) * 100
            )

            print(
                f"{column}: "
                f"{count} records "
                f"({percentage:.2f}%)"
            )

    else:

        print(
            "No missing values found."
        )

    # --------------------------------------------------------
    # Insight 7: Dataset Size
    # --------------------------------------------------------

    print(
        "\n7. Dataset Overview:"
    )

    print(
        f"Total records: {len(df):,}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )

    print(
        f"Duplicate rows: "
        f"{df.duplicated().sum():,}"
    )

    # --------------------------------------------------------
    # Insight 8: Categorical Diversity
    # --------------------------------------------------------

    if len(analysis_categorical_columns) > 0:

        print(
            "\n8. Categorical Diversity:"
        )

        for column in analysis_categorical_columns:

            unique_count = (
                df[column]
                .nunique(dropna=True)
            )

            print(
                f"{column}: "
                f"{unique_count} unique values"
            )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 60
    )

    print(
        "SMARTLOGIX AI - WEATHER DATA EDA"
    )

    print(
        "=" * 60
    )

    # Load dataset
    df = load_cleaned_data(
        "weather_data_clean.csv"
    )

    # Basic inspection
    basic_inspection(df)

    # Data quality
    data_quality_checks(df)

    # Univariate analysis
    univariate_analysis(df)

    # Bivariate analysis
    bivariate_analysis(df)

    # Multivariate analysis
    multivariate_analysis(df)

    # Visualizations
    visualizations(df)

    # Business insights
    business_insights(df)

    print(
        "\n" + "=" * 60
    )

    print(
        "WEATHER DATA EDA COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()