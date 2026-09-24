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

    # Convert timestamp/date columns
    date_columns = [
        column
        for column in df.columns
        if "date" in column.lower()
        or "timestamp" in column.lower()
        or "datetime" in column.lower()
    ]

    for column in date_columns:

        try:
            df[column] = pd.to_datetime(df[column])

        except (ValueError, TypeError):
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

    print("\nNumerical summary:")
    print(df.describe())


# ============================================================
# 3. DATA QUALITY CHECKS
# ============================================================

def data_quality_checks(df):

    print("\n" + "=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nMissing value percentage:")

    print(
        (df.isnull().sum() / len(df) * 100)
        .round(2)
    )

    print("\nDuplicate rows:")

    print(
        df.duplicated().sum()
    )

    print("\nUnique values:")

    print(
        df.nunique()
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

    print("\nNumerical Columns:")

    for column in numerical_columns:

        print(f"\n--- {column} ---")

        print(
            df[column].describe()
        )

    print("\nCategorical Columns:")

    for column in categorical_columns:

        if (
            column.lower().endswith("_id")
            or column.lower() in [
                "id",
                "date",
                "timestamp",
                "datetime"
            ]
        ):
            continue

        print(f"\n--- {column} ---")

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
    # Remove identifier columns
    # --------------------------------------------------------

    identifier_columns = [
        column
        for column in categorical_columns
        if (
            column.lower().endswith("_id")
            or column.lower() in [
                "id",
                "date",
                "timestamp",
                "datetime"
            ]
        )
    ]

    analysis_categorical_columns = [
        column
        for column in categorical_columns
        if column not in identifier_columns
    ]

    # --------------------------------------------------------
    # Numerical vs Numerical
    # --------------------------------------------------------

    if len(numerical_columns) >= 2:

        print("\nNumerical Correlation:")

        correlation = df[
            numerical_columns
        ].corr()

        print(
            correlation.round(3)
        )

    # --------------------------------------------------------
    # Categorical vs Numerical
    # --------------------------------------------------------

    if (
        len(analysis_categorical_columns) >= 1
        and len(numerical_columns) >= 1
    ):

        category = (
            analysis_categorical_columns[0]
        )

        numerical = numerical_columns[0]

        print(
            f"\nAverage {numerical} "
            f"by {category}:"
        )

        grouped_summary = (
            df.groupby(
                category,
                dropna=False
            )[numerical]
            .mean()
            .sort_values(
                ascending=False
            )
            .head(15)
        )

        print(
            grouped_summary
        )

    # --------------------------------------------------------
    # Categorical vs Categorical
    # --------------------------------------------------------

    if len(analysis_categorical_columns) >= 2:

        first_category = (
            analysis_categorical_columns[0]
        )

        second_category = (
            analysis_categorical_columns[1]
        )

        print(
            f"\nCategorical relationship:"
            f"\n{first_category} vs "
            f"{second_category}"
        )

        cross_tab = pd.crosstab(
            df[first_category],
            df[second_category]
        )

        print(
            cross_tab.head(15)
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

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    # Remove identifier columns
    identifier_columns = [
        column
        for column in categorical_columns
        if (
            column.lower().endswith("_id")
            or column.lower() in [
                "id",
                "date",
                "timestamp",
                "datetime"
            ]
        )
    ]

    analysis_categorical_columns = [
        column
        for column in categorical_columns
        if column not in identifier_columns
    ]

    # --------------------------------------------------------
    # Correlation Matrix
    # --------------------------------------------------------

    if len(numerical_columns) >= 2:

        print(
            "\nNumerical correlation matrix:"
        )

        correlation = df[
            numerical_columns
        ].corr()

        print(
            correlation.round(3)
        )

    # --------------------------------------------------------
    # Grouped Numerical Analysis
    # --------------------------------------------------------

    if (
        len(analysis_categorical_columns) >= 1
        and len(numerical_columns) >= 2
    ):

        category = (
            analysis_categorical_columns[0]
        )

        print(
            f"\nAverage numerical metrics "
            f"by {category}:"
        )

        grouped_summary = (
            df.groupby(
                category,
                dropna=False
            )[numerical_columns]
            .mean()
            .sort_values(
                by=numerical_columns[0],
                ascending=False
            )
            .head(15)
        )

        print(
            grouped_summary
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

    # Remove identifier columns
    identifier_columns = [
        column
        for column in categorical_columns
        if (
            column.lower().endswith("_id")
            or column.lower() in [
                "id",
                "date",
                "timestamp",
                "datetime"
            ]
        )
    ]

    categorical_columns = [
        column
        for column in categorical_columns
        if column not in identifier_columns
    ]

    # --------------------------------------------------------
    # Visualization 1
    # Numerical Distribution
    # --------------------------------------------------------

    if len(numerical_columns) >= 1:

        column = numerical_columns[0]

        print(
            f"\nGenerating distribution "
            f"for {column}..."
        )

        plt.figure(figsize=(10, 6))

        plt.hist(
            df[column].dropna(),
            bins=30
        )

        plt.title(
            f"{column} Distribution"
        )

        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR /
            "traffic_data_numeric_distribution.png",
            dpi=300
        )

        plt.show()
        plt.close()

    # --------------------------------------------------------
    # Visualization 2
    # Numerical Boxplot
    # --------------------------------------------------------

    if len(numerical_columns) >= 1:

        column = numerical_columns[0]

        print(
            f"\nGenerating boxplot "
            f"for {column}..."
        )

        plt.figure(figsize=(10, 6))

        plt.boxplot(
            df[column].dropna()
        )

        plt.title(
            f"{column} Boxplot"
        )

        plt.ylabel(column)

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR /
            "traffic_data_numeric_boxplot.png",
            dpi=300
        )

        plt.show()
        plt.close()

    # --------------------------------------------------------
    # Visualization 3
    # Categorical Distribution
    # --------------------------------------------------------

    if len(categorical_columns) >= 1:

        column = categorical_columns[0]

        print(
            f"\nGenerating distribution "
            f"for {column}..."
        )

        value_counts = (
            df[column]
            .value_counts()
            .head(10)
        )

        plt.figure(figsize=(10, 6))

        value_counts.plot(
            kind="bar"
        )

        plt.title(
            f"Top Categories of {column}"
        )

        plt.xlabel(column)
        plt.ylabel("Number of Records")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.subplots_adjust(
            bottom=0.30
        )

        plt.savefig(
            FIGURES_DIR /
            "traffic_data_categorical_distribution.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

    # --------------------------------------------------------
    # Visualization 4
    # Numerical Relationship
    # --------------------------------------------------------

    if len(numerical_columns) >= 2:

        x_column = numerical_columns[0]
        y_column = numerical_columns[1]

        print(
            f"\nGenerating relationship "
            f"between {x_column} "
            f"and {y_column}..."
        )

        plt.figure(figsize=(10, 6))

        plt.scatter(
            df[x_column],
            df[y_column],
            alpha=0.5
        )

        plt.title(
            f"{x_column} vs {y_column}"
        )

        plt.xlabel(x_column)
        plt.ylabel(y_column)

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR /
            "traffic_data_numeric_relationship.png",
            dpi=300
        )

        plt.show()
        plt.close()

    # --------------------------------------------------------
    # Visualization 5
    # Categorical Relationship
    # --------------------------------------------------------

    if len(categorical_columns) >= 2:

        first_category = (
            categorical_columns[0]
        )

        second_category = (
            categorical_columns[1]
        )

        cross_tab = pd.crosstab(
            df[first_category],
            df[second_category]
        )

        print(
            "\nGenerating categorical relationship..."
        )

        cross_tab.head(10).plot(
            kind="bar",
            stacked=True,
            figsize=(14, 7)
        )

        plt.title(
            f"{first_category} "
            f"by {second_category}"
        )

        plt.xlabel(
            first_category
        )

        plt.ylabel(
            "Number of Records"
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
            "traffic_data_categorical_relationship.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()


# ============================================================
# 8. BUSINESS INSIGHTS
# ============================================================

def business_insights(df):

    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS")
    print("=" * 60)

    # --------------------------------------------------------
    # Dataset Size
    # --------------------------------------------------------

    print(
        f"\n1. Total Traffic Records: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Missing Values
    # --------------------------------------------------------

    missing_values = (
        df.isnull()
        .sum()
        .sum()
    )

    print(
        f"\n2. Total Missing Values: "
        f"{missing_values:,}"
    )

    # --------------------------------------------------------
    # Duplicate Records
    # --------------------------------------------------------

    duplicate_count = (
        df.duplicated().sum()
    )

    print(
        f"\n3. Duplicate Records: "
        f"{duplicate_count:,}"
    )

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    # --------------------------------------------------------
    # Numerical Summary
    # --------------------------------------------------------

    if len(numerical_columns) > 0:

        print(
            "\n4. Numerical Column Summary:"
        )

        for column in numerical_columns:

            print(
                f"\n{column}:"
            )

            print(
                f"   Mean   : "
                f"{df[column].mean():.2f}"
            )

            print(
                f"   Median : "
                f"{df[column].median():.2f}"
            )

            print(
                f"   Minimum: "
                f"{df[column].min():.2f}"
            )

            print(
                f"   Maximum: "
                f"{df[column].max():.2f}"
            )

    # --------------------------------------------------------
    # Categorical Summary
    # --------------------------------------------------------

    if len(categorical_columns) > 0:

        print(
            "\n5. Most Common Categories:"
        )

        for column in categorical_columns:

            if (
                column.lower().endswith("_id")
                or column.lower() in [
                    "id",
                    "date",
                    "timestamp",
                    "datetime"
                ]
            ):
                continue

            top_value = (
                df[column]
                .value_counts(
                    dropna=False
                )
                .head(1)
            )

            if len(top_value) > 0:

                print(
                    f"\n{column}:"
                )

                print(
                    top_value
                )

    # --------------------------------------------------------
    # Date Range
    # --------------------------------------------------------

    date_columns = [
        column
        for column in df.columns
        if "date" in column.lower()
        or "timestamp" in column.lower()
        or "datetime" in column.lower()
    ]

    if len(date_columns) > 0:

        print(
            "\n6. Date Range:"
        )

        for column in date_columns:

            print(
                f"\n{column}:"
            )

            print(
                f"   Start: "
                f"{df[column].min()}"
            )

            print(
                f"   End  : "
                f"{df[column].max()}"
            )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    df = load_cleaned_data(
        "traffic_data_clean.csv"
    )

    basic_inspection(df)

    data_quality_checks(df)

    univariate_analysis(df)

    bivariate_analysis(df)

    multivariate_analysis(df)

    visualizations(df)

    business_insights(df)


if __name__ == "__main__":
    main()