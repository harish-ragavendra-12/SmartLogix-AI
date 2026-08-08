import pandas as pd

from src.config.config import RAW_DATA_DIR


def load_csv(file_path):
    """
    Load a CSV file from the raw data directory.

    Parameters:
        file_path (str): Name of the CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """

    file_path = RAW_DATA_DIR / file_path

    df = pd.read_csv(file_path)

    return df


def load_json(file_path):
    """
    Load a JSON file from the raw data directory.

    Parameters:
        file_path (str): Name of the JSON file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """

    file_path = RAW_DATA_DIR / file_path

    df = pd.read_json(file_path)

    return df


def load_jsonl(file_path):
    """
    Load a JSON Lines file from the raw data directory.

    Parameters:
        file_path (str): Name of the JSONL file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """

    file_path = RAW_DATA_DIR / file_path

    df = pd.read_json(file_path, lines=True)

    return df
