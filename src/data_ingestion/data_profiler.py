from pathlib import Path
import json

import pandas as pd

from src.config.config import RAW_DATA_DIR


# ==========================================================
# DATASET LOADING
# ==========================================================

def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV, JSON, or JSONL dataset into a DataFrame.
    """

    file_extension = file_path.suffix.lower()

    # ------------------------------------------------------
    # CSV
    # ------------------------------------------------------

    if file_extension == ".csv":
        return pd.read_csv(file_path)

    # ------------------------------------------------------
    # JSONL
    # ------------------------------------------------------

    if file_extension == ".jsonl":
        return pd.read_json(file_path, lines=True)

    # ------------------------------------------------------
    # JSON
    # ------------------------------------------------------

    if file_extension == ".json":

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Case 1:
        # JSON contains a list of records
        #
        # [
        #     {...},
        #     {...}
        # ]

        if isinstance(data, list):
            return pd.json_normalize(data)

        # Case 2:
        # JSON contains a dictionary
        #
        # {
        #     "products": [
        #         {...},
        #         {...}
        #     ]
        # }

        if isinstance(data, dict):

            # Look for a nested list containing records
            for key, value in data.items():

                if isinstance(value, list):

                    if all(isinstance(item, dict) for item in value):
                        return pd.json_normalize(value)

            # Case 3:
            # JSON contains a single dictionary
            return pd.json_normalize(data)

    raise ValueError(
        f"Unsupported file format: {file_extension}"
    )


# ==========================================================
# JSON STRUCTURE INSPECTION
# ==========================================================

def inspect_json_structure(file_path: Path) -> None:
    """
    Display the high-level structure of a JSON file.

    This is useful for nested JSON datasets such as
    product catalogs and GPS route information.
    """

    if file_path.suffix.lower() != ".json":
        return

    try:

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        print("\nJSON Structure:")

        if isinstance(data, list):

            print("  Root type: list")
            print(f"  Number of records: {len(data):,}")

            if data and isinstance(data[0], dict):
                print("  Record fields:")
                for key in data[0].keys():
                    print(f"    - {key}")

        elif isinstance(data, dict):

            print("  Root type: dictionary")

            print("  Top-level fields:")

            for key, value in data.items():

                if isinstance(value, list):

                    print(
                        f"    - {key}: "
                        f"list ({len(value):,} items)"
                    )

                    if value and isinstance(value[0], dict):
                        print("      Record fields:")

                        for nested_key in value[0].keys():
                            print(
                                f"        - {nested_key}"
                            )

                elif isinstance(value, dict):

                    print(
                        f"    - {key}: dictionary"
                    )

                else:

                    print(
                        f"    - {key}: "
                        f"{type(value).__name__}"
                    )

        else:

            print(
                f"  Root type: "
                f"{type(data).__name__}"
            )

    except Exception as error:

        print(
            f"  Could not inspect JSON structure: "
            f"{error}"
        )


# ==========================================================
# DUPLICATE CHECK
# ==========================================================

def count_duplicates(df: pd.DataFrame) -> int:
    """
    Safely count duplicate rows.

    Some JSON datasets may contain lists or dictionaries
    inside cells. Pandas cannot directly hash those values,
    so we convert values to strings for duplicate detection.
    """

    try:
        return int(df.duplicated().sum())

    except TypeError:

        safe_df = df.map(
            lambda value: repr(value)
            if isinstance(value, (list, dict, set))
            else value
        )

        return int(safe_df.duplicated().sum())


# ==========================================================
# DATASET PROFILING
# ==========================================================

def profile_dataset(file_path: Path) -> None:
    """
    Display basic information about a dataset.
    """

    print("\n" + "=" * 80)
    print(f"DATASET: {file_path.name}")
    print("=" * 80)

    try:

        # --------------------------------------------------
        # JSON STRUCTURE
        # --------------------------------------------------

        inspect_json_structure(file_path)

        # --------------------------------------------------
        # LOAD DATASET
        # --------------------------------------------------

        df = load_dataset(file_path)

        # --------------------------------------------------
        # BASIC INFORMATION
        # --------------------------------------------------

        print("\nDataset Information:")

        print(f"  Rows       : {df.shape[0]:,}")
        print(f"  Columns    : {df.shape[1]}")

        # --------------------------------------------------
        # COLUMN NAMES
        # --------------------------------------------------

        print("\nColumn Names:")

        for column in df.columns:
            print(f"  - {column}")

        # --------------------------------------------------
        # DATA TYPES
        # --------------------------------------------------

        print("\nData Types:")

        for column, dtype in df.dtypes.items():
            print(f"  {column}: {dtype}")

        # --------------------------------------------------
        # MISSING VALUES
        # --------------------------------------------------

        print("\nMissing Values:")

        missing_values = df.isnull().sum()

        missing_values = missing_values[
            missing_values > 0
        ]

        if missing_values.empty:

            print("  No missing values.")

        else:

            for column, count in missing_values.items():

                percentage = (
                    count / len(df) * 100
                    if len(df) > 0
                    else 0
                )

                print(
                    f"  {column}: "
                    f"{count:,} "
                    f"({percentage:.2f}%)"
                )

        # --------------------------------------------------
        # DUPLICATES
        # --------------------------------------------------

        duplicate_count = count_duplicates(df)

        print(
            f"\nDuplicate Rows: "
            f"{duplicate_count:,}"
        )

        # --------------------------------------------------
        # SAMPLE RECORDS
        # --------------------------------------------------

        print("\nSample Records:")

        if df.empty:

            print("  Dataset is empty.")

        else:

            print(
                df.head(3).to_string(
                    index=False
                )
            )

    except Exception as error:

        print(
            f"\nERROR: Could not profile "
            f"{file_path.name}"
        )

        print(f"Reason: {error}")


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """
    Profile all supported datasets in the raw data directory.
    """

    supported_extensions = {
        ".csv",
        ".json",
        ".jsonl"
    }

    if not RAW_DATA_DIR.exists():

        print(
            f"Raw data directory does not exist:\n"
            f"{RAW_DATA_DIR}"
        )

        return

    dataset_files = sorted(
        file_path
        for file_path in RAW_DATA_DIR.iterdir()
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in supported_extensions
        )
    )

    if not dataset_files:

        print(
            "No supported datasets found "
            "in the raw data directory."
        )

        return

    print("=" * 80)
    print("SMARTLOGIX AI - DATASET PROFILER")
    print("=" * 80)

    print(
        f"\nRaw Data Directory:"
        f"\n{RAW_DATA_DIR}"
    )

    print(
        f"\nFound {len(dataset_files)} dataset(s)."
    )

    for file_path in dataset_files:

        profile_dataset(file_path)

    print("\n" + "=" * 80)
    print("DATASET PROFILING COMPLETED")
    print("=" * 80)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
