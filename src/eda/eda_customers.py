import matplotlib.pyplot as plt
import pandas as pd

from src.config.config import PROCESSED_DATA_DIR, FIGURES_DIR


def load_cleaned_data(file_name):

    df = pd.read_csv(PROCESSED_DATA_DIR / file_name)

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

    print("\nCategorical columns:")
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False).head(10))


def univariate_analysis(df):

    print("\nNumerical columns:")

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numerical_columns:
        print(f"\n{column} statistics:")
        print(df[column].describe())

    print("\nCategorical columns:")

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        print(f"\n{column} distribution:")
        print(df[column].value_counts(dropna=False).head(10))


def bivariate_analysis(df):

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    if len(numerical_columns) >= 2:

        print("\nCorrelation between numerical variables:")

        print(
            df[numerical_columns]
            .corr()
        )

    if len(categorical_columns) >= 1 and len(numerical_columns) >= 1:

        print("\nAverage numerical values by categorical variables:")

        for categorical_column in categorical_columns[:3]:

            for numerical_column in numerical_columns[:3]:

                print(
                    f"\nAverage {numerical_column} by "
                    f"{categorical_column}:"
                )

                print(
                    df.groupby(categorical_column)[numerical_column]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                )


def multivariate_analysis(df):

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    print("\nMultivariate numerical correlation:")

    if len(numerical_columns) >= 2:

        print(
            df[numerical_columns]
            .corr()
        )

    if len(categorical_columns) >= 2 and len(numerical_columns) >= 1:

        print("\nCategorical combinations with numerical values:")

        print(
            df.groupby(
                [
                    categorical_columns[0],
                    categorical_columns[1]
                ]
            )[numerical_columns[0]]
            .mean()
            .head(20)
        )


def visualizations(df):

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    # Visualization 1
    if len(numerical_columns) >= 1:

        column = numerical_columns[0]

        print(f"\nGenerating {column} distribution...")

        plt.figure(figsize=(8, 5))

        plt.hist(df[column].dropna(), bins=20)

        plt.title(f"{column} Distribution")

        plt.xlabel(column)
        plt.ylabel("Number of Records")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"customers_{column}_distribution.png",
            dpi=300
        )

        plt.show()

        plt.close()

    # Visualization 2
    if len(numerical_columns) >= 1:

        column = numerical_columns[0]

        print(f"\nGenerating {column} boxplot...")

        plt.figure(figsize=(8, 5))

        plt.boxplot(df[column].dropna())

        plt.title(f"{column} Boxplot")

        plt.ylabel(column)

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"customers_{column}_boxplot.png",
            dpi=300
        )

        plt.show()

        plt.close()

    # Visualization 3
    if len(categorical_columns) >= 1:

        column = categorical_columns[0]

        print(f"\nGenerating {column} distribution...")

        counts = df[column].value_counts().head(10)

        plt.figure(figsize=(10, 6))

        counts.plot(kind="bar")

        plt.title(f"{column} Distribution")

        plt.xlabel(column)
        plt.ylabel("Number of Customers")

        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"customers_{column}_distribution.png",
            dpi=300
        )

        plt.show()

        plt.close()

    # Visualization 4
    if len(numerical_columns) >= 2:

        print("\nGenerating numerical correlation...")

        plt.figure(figsize=(8, 5))

        plt.scatter(
            df[numerical_columns[0]],
            df[numerical_columns[1]]
        )

        plt.title(
            f"{numerical_columns[0]} vs "
            f"{numerical_columns[1]}"
        )

        plt.xlabel(numerical_columns[0])
        plt.ylabel(numerical_columns[1])

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / "customers_numerical_relationship.png",
            dpi=300
        )

        plt.show()

        plt.close()


def business_insights(df):

    print("\n========== BUSINESS INSIGHTS ==========")

    print("\n1. Customer Dataset Size:")
    print(f"Total customer records: {len(df)}")

    missing_values = df.isnull().sum()

    print("\n2. Columns with Missing Values:")

    missing_columns = missing_values[
        missing_values > 0
    ].sort_values(ascending=False)

    if len(missing_columns) > 0:
        print(missing_columns)
    else:
        print("No missing values found.")

    duplicate_count = df.duplicated().sum()

    print("\n3. Duplicate Records:")
    print(f"Duplicate records: {duplicate_count}")

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    if len(numerical_columns) > 0:

        print("\n4. Numerical Variable Summary:")

        for column in numerical_columns:
            print(
                f"{column}: "
                f"average = {df[column].mean():.2f}"
            )

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    if len(categorical_columns) > 0:

        print("\n5. Categorical Variable Summary:")

        for column in categorical_columns[:5]:

            unique_count = df[column].nunique()

            print(
                f"{column}: "
                f"{unique_count} unique values"
            )


def main():

    df = load_cleaned_data("customers_cleaned.csv")

    basic_inspection(df)

    data_quality_checks(df)

    univariate_analysis(df)

    bivariate_analysis(df)

    multivariate_analysis(df)

    visualizations(df)

    business_insights(df)


if __name__ == "__main__":
    main()