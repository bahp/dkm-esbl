# src/phenotypes.py
import pandas as pd
import numpy as np


def map_medical_codes(df, code_col, target_codes):
    """
    Generic function to identify if any code in a patient's record
    matches a list of target diagnostic codes (ICD-10, ICD-9, etc.).

    Parameters:
    -----------
    df : pd.DataFrame
    code_col : str
        The column containing patient codes (can be a list or a string).
    target_codes : list
        The list of codes that define the clinical category (e.g., ['I50', 'I11.0']).
    """

    def check_match(patient_codes):
        if pd.isna(patient_codes):
            return 0
        # Convert to list if it's a string (e.g., "I10, I50")
        if isinstance(patient_codes, str):
            patient_codes = [c.strip() for c in patient_codes.split(',')]

        # Check if there is any intersection between patient codes and targets
        # This handles partial matches too (e.g., 'E11' matches 'E11.9')
        match = any(any(str(pc).startswith(str(tc)) for tc in target_codes)
                    for pc in patient_codes)
        return 1 if match else 0

    return df[code_col].apply(check_match)

def derive_diabetes_status(df, insulin_col='med_insulin_given', glucose_col='Glucose_24h_max'):
    """
    Derives a diabetes diagnosis (1=Yes, 0=No) based on clinical proxies
    rather than a direct medical history column.

    Rules for positive phenotype:
    1. Patient was administered insulin.
    2. OR, Patient had a severely elevated rolling glucose reading (> 200).
    """
    # Start by assuming the patient does NOT have diabetes (fill with 0)
    status = pd.Series(0, index=df.index)

    # Rule A: Check for the medication proxy
    if insulin_col in df.columns:
        # If insulin is 1, set status to 1. Otherwise, keep the current status.
        status = np.where(df[insulin_col] == 1, 1, status)

    # Rule B: Check for the lab value proxy
    if glucose_col in df.columns:
        # If glucose is > 200, set status to 1. Otherwise, keep current status.
        status = np.where(df[glucose_col] > 200.0, 1, status)

    return status


def derive_liver_disease(df, ast_col='AST', alt_col='ALT', cirrhosis_med_col='med_lactulose'):
    """
    Example 2: Deriving Liver Disease based on severely elevated liver enzymes
    or the administration of Lactulose (a common medication for hepatic encephalopathy).
    """
    status = pd.Series(0, index=df.index)

    # Rule A: Medication proxy
    if cirrhosis_med_col in df.columns:
        status = np.where(df[cirrhosis_med_col] == 1, 1, status)

    # Rule B: Lab proxy (Enzymes > 3x the upper limit of normal)
    if ast_col in df.columns and alt_col in df.columns:
        status = np.where((df[ast_col] > 120) | (df[alt_col] > 150), 1, status)

    return status