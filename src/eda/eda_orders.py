import matplotlib.pyplot as plt
import pandas as pd

from src.config.config import PROCESSED_DATA_DIR, FIGURES_DIR


def load_cleaned_data(file_name):

    df = pd.read_csv(PROCESSED_DATA_DIR / file_name)

    # Convert date/time columns when available
    date_columns = [
        column
        for column in df.columns
        if "date" in column.lower()
        or "time" in column.lower()
    ]

    for column in date_columns:

        try:
            df[column] = pd.to_datetime(df[column])
        except (ValueError, TypeError):
            pass

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

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    print("\nCategorical columns:")

    for column in categorical_columns:

        print(f"\n{column}:")

        print(
            df[column]
            .value_counts(dropna=False)
            .head(10)
        )


def univariate_analysis(df):

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    print("\nNumerical variable analysis:")

    for column in numerical_columns:

        print(f"\n{column} statistics:")

        print(
            df[column].describe()
        )

    print("\nCategorical variable analysis:")

    for column in categorical_columns:

        print(f"\n{column} distribution:")

        print(
            df[column]
            .value_counts(dropna=False)
            .head(10)
        )


def bivariate_analysis(df):

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    if len(numerical_columns) >= 2:

        print("\nCorrelation between numerical variables:")

        print(
            df[numerical_columns].corr()
        )

    if (
        len(categorical_columns) >= 1
        and len(numerical_columns) >= 1
    ):

        print(
            "\nNumerical variables by "
            "categorical variables:"
        )

        for categorical_column in categorical_columns[:5]:

            for numerical_column in numerical_columns[:5]:

                print(
                    f"\nAverage {numerical_column} "
                    f"by {categorical_column}:"
                )

                print(
                    df.groupby(categorical_column)[
                        numerical_column
                    ]
                    .mean()
                    .sort_values(
                        ascending=False
                    )
                    .head(10)
                )


def multivariate_analysis(df):

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    if len(numerical_columns) >= 2:

        print(
            "\nMultivariate numerical correlation:"
        )

        print(
            df[numerical_columns].corr()
        )

    if (
        len(categorical_columns) >= 2
        and len(numerical_columns) >= 1
    ):

        print(
            "\nCategorical combination "
            "with numerical analysis:"
        )

        print(
            df.groupby(
                [
                    categorical_columns[0],
                    categorical_columns[1]
                ]
            )[numerical_columns[0]]
            .mean()
            .sort_values(
                ascending=False
            )
            .head(20)
        )

    if (
        len(categorical_columns) >= 1
        and len(numerical_columns) >= 2
    ):

        print(
            "\nAverage numerical metrics "
            "by category:"
        )

        print(
            df.groupby(
                categorical_columns[0]
            )[numerical_columns[:3]]
            .mean()
            .sort_values(
                numerical_columns[0],
                ascending=False
            )
            .head(10)
        )


def visualizations(df):

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

    # Visualization 1
    if len(numerical_columns) >= 1:

        column = numerical_columns[0]

        print(
            f"\nGenerating {column} distribution..."
        )

        plt.figure(figsize=(8, 5))

        plt.hist(
            df[column].dropna(),
            bins=20
        )

        plt.title(
            f"{column} Distribution"
        )

        plt.xlabel(column)
        plt.ylabel("Number of Orders")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR /
            f"orders_{column}_distribution.png",
            dpi=300
        )

        plt.show()

        plt.close()

    # Visualization 2
    if len(numerical_columns) >= 1:

        column = numerical_columns[0]

        print(
            f"\nGenerating {column} boxplot..."
        )

        plt.figure(figsize=(8, 5))

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
            f"orders_{column}_boxplot.png",
            dpi=300
        )

        plt.show()

        plt.close()

    # Visualization 3
    if len(categorical_columns) >= 1:

        column = categorical_columns[0]

        print(
            f"\nGenerating {column} distribution..."
        )

        counts = (
            df[column]
            .value_counts()
            .head(10)
        )

        plt.figure(figsize=(10, 6))

        counts.plot(
            kind="bar"
        )

        plt.title(
            f"Top {column} Distribution"
        )

        plt.xlabel(column)
        plt.ylabel("Number of Orders")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR /
            f"orders_{column}_distribution.png",
            dpi=300
        )

        plt.show()

        plt.close()

    # Visualization 4
    if len(numerical_columns) >= 2:

        print(
            "\nGenerating numerical relationship..."
        )

        plt.figure(figsize=(8, 5))

        plt.scatter(
            df[numerical_columns[0]],
            df[numerical_columns[1]]
        )

        plt.title(
            f"{numerical_columns[0]} vs "
            f"{numerical_columns[1]}"
        )

        plt.xlabel(
            numerical_columns[0]
        )

        plt.ylabel(
            numerical_columns[1]
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR /
            "orders_numerical_relationship.png",
            dpi=300
        )

        plt.show()

        plt.close()

    # Visualization 5
    if len(categorical_columns) >= 2:

        cross_tab = pd.crosstab(
            df[categorical_columns[0]],
            df[categorical_columns[1]]
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
            f"{categorical_columns[0]} "
            f"by {categorical_columns[1]}"
        )

        plt.xlabel(
            categorical_columns[0]
        )

        plt.ylabel(
            "Number of Orders"
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
            "orders_categorical_relationship.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        plt.close()


def business_insights(df):

    print(
        "\n========== BUSINESS INSIGHTS =========="
    )

    # 1. Order volume
    print("\n1. Order Dataset Size:")

    print(
        f"Total order records: {len(df)}"
    )

    # 2. Missing values
    missing_values = df.isnull().sum()

    missing_columns = (
        missing_values[
            missing_values > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    print(
        "\n2. Columns with Missing Values:"
    )

    if len(missing_columns) > 0:

        print(
            missing_columns
        )

    else:

        print(
            "No missing values found."
        )

    # 3. Duplicate records
    duplicate_count = df.duplicated().sum()

    print(
        "\n3. Duplicate Records:"
    )

    print(
        f"Duplicate records: "
        f"{duplicate_count}"
    )

    # 4. Numerical summary
    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    print(
        "\n4. Numerical Variable Summary:"
    )

    for column in numerical_columns:

        print(
            f"{column}: "
            f"average = "
            f"{df[column].mean():.2f}, "
            f"minimum = "
            f"{df[column].min():.2f}, "
            f"maximum = "
            f"{df[column].max():.2f}"
        )

    # 5. Categorical summary
    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    print(
        "\n5. Categorical Variable Summary:"
    )

    for column in categorical_columns[:5]:

        print(
            f"{column}: "
            f"{df[column].nunique()} "
            f"unique values"
        )

    # 6. Most common categories
    if len(categorical_columns) > 0:

        print(
            "\n6. Most Common Categories:"
        )

        for column in categorical_columns[:5]:

            counts = (
                df[column]
                .value_counts()
            )

            if len(counts) > 0:

                print(
                    f"{column}: "
                    f"{counts.index[0]} "
                    f"({counts.iloc[0]} records)"
                )

    # 7. Date range
    date_columns = [
        column
        for column in df.columns
        if "date" in column.lower()
        or "time" in column.lower()
    ]

    if len(date_columns) > 0:

        print(
            "\n7. Date Range:"
        )

        for column in date_columns:

            if pd.api.types.is_datetime64_any_dtype(
                df[column]
            ):

                print(
                    f"{column}: "
                    f"{df[column].min()} "
                    f"to "
                    f"{df[column].max()}"
                )


def main():

    df = load_cleaned_data(
        "orders_cleaned.csv"
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