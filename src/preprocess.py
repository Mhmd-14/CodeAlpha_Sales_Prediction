import os
import pandas as pd


def preprocess_data(input_path: str, output_dir: str) -> str:
    """
    Loads a CSV file, drops the first column, and saves the cleaned data.

    :param input_path: Path to the input CSV file.
    :param output_dir: Directory to save the processed CSV file.
    :return: Path to the saved processed CSV file.
    """
    df = pd.read_csv(input_path)
    df = df.iloc[:, 1:]  # Drop first column (e.g., unnamed index)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'processed.csv')
    df.to_csv(output_path, index=False)

    print(f"[INFO] Preprocessed data saved to: {output_path}")
    return output_path