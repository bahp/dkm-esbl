import sys
import yaml
import pandas as pd
from pathlib import Path

# Setup path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.metrics import ClinicalEvaluator

def get_latest_data_dir():
    """Finds the most recently created folder in data/synthetic/"""
    synthetic_dir = project_root / 'data' / 'synthetic'
    if not synthetic_dir.exists():
        raise FileNotFoundError("No synthetic data found. Run 01_generate_data.py first.")

    dirs = sorted([d for d in synthetic_dir.iterdir() if d.is_dir()])
    if not dirs:
        raise FileNotFoundError("No synthetic data directories found.")

    return dirs[-1]


def get_latest_processed_file():
    """Finds the most recently processed dataset (any .csv file in the latest folder)."""
    processed_dir = project_root / 'data' / 'processed'
    if not processed_dir.exists():
        raise FileNotFoundError("No processed data found. Run 02_build_features.py first.")

    dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir()])
    if not dirs:
        raise FileNotFoundError("No processed data directories found.")

    latest_dir = dirs[-1]
    csv_files = list(latest_dir.glob('*.csv'))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the directory: {latest_dir}")

    # Grab the first CSV file found
    latest_file = csv_files[0]

    # Log the file we are about to read
    print(f"📄 Loading processed dataset: '{latest_file.name}' from run '{latest_dir.name}'")

    return latest_file


import logging

def validate_required_columns(df,
                              required_cols,
                              score_name="Clinical Score",
                              strict=False):
    """
    Validates the presence of required clinical variables in a DataFrame.

    This utility prevents the 'silent zero' problem, where a missing column
    leads to an incorrectly low clinical score (under-scoring). It logs
    missing columns to the console to ensure data integrity during
    feature engineering.

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataset being evaluated.
    required_cols : list of str
        The column names mandatory for the specific scoring system.
    score_name : str, default="Clinical Score"
        The name of the score (e.g., 'INCREMENT-ESBL') for log identification.
    strict : bool, default=False
        If True, raises a ValueError when columns are missing.
        If False, prints a warning and allows the pipeline to continue.

    Returns
    -------
    bool
        True if all required columns are present; False if any are missing.

    Raises
    ------
    ValueError
        If 'strict' is True and required columns are missing from the input DataFrame.
    """
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        error_msg = f"[{score_name}] Missing critical variables: {missing_cols}"

        if strict:
            raise ValueError(f"❌ {error_msg}. Pipeline halted to prevent data corruption.")

        print(f"⚠️ Warning: {error_msg}. Results will be underestimated for these records.")
        return False

    return True


def check_col_bool(df, col_name):
    """Returns a boolean Series checking if a binary column is 1/True."""
    if col_name not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col_name].fillna(0).astype(int) == 1


def check_col_contains(df, col_name, keywords):
    """Returns a boolean Series checking if text in a column contains certain keywords."""
    if col_name not in df.columns:
        return pd.Series(False, index=df.index)

    # Create a regex pattern: 'keyword1|keyword2|keyword3'
    pattern = '|'.join(keywords)
    return df[col_name].astype(str).str.lower().str.contains(pattern, na=False)


def check_col_threshold(df, col_name, threshold, operator='>'):
    """Returns a boolean Series checking if a numerical column meets a threshold."""
    if col_name not in df.columns:
        return pd.Series(False, index=df.index)

    # Temporarily fill NaNs with a safe value that won't trigger the threshold
    temp_col = df[col_name].fillna(-9999 if operator == '>' else 9999)

    if operator == '>': return temp_col > threshold
    if operator == '>=': return temp_col >= threshold
    if operator == '<': return temp_col < threshold
    if operator == '<=': return temp_col <= threshold
    return pd.Series(False, index=df.index)


def check_col_icd10(df, col_name, target_codes):
    """Returns a boolean Series checking if any ICD codes match the target list."""
    if col_name not in df.columns:
        return pd.Series(False, index=df.index)

    def match_codes(patient_codes):
        if pd.isna(patient_codes): return False
        patient_codes = [c.strip() for c in str(patient_codes).split(',')]
        return any(any(str(pc).startswith(str(tc)) for tc in target_codes) for pc in patient_codes)

    return df[col_name].apply(match_codes)