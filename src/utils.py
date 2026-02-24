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
    """Finds the most recently processed dataset."""
    processed_dir = project_root / 'data' / 'processed'
    if not processed_dir.exists():
        raise FileNotFoundError("No processed data found. Run 02_build_features.py first.")

    dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir()])
    if not dirs:
        raise FileNotFoundError("No processed data directories found.")

    latest_file = dirs[-1] / 'features_engineered.csv'
    if not latest_file.exists():
        raise FileNotFoundError(f"features_engineered.csv not found in {dirs[-1]}")

    return latest_file


import logging

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