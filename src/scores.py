# src/scores.py
import pandas as pd
import numpy as np

from src.utils import validate_required_columns


def audit_log(condition, points, description, verbose=False, logger=None):
    """
    Observer helper: Evaluates a condition to log a message.
    Does NOT calculate, modify, or return any scores.
    """
    if verbose and logger:
        # Check if the condition is True for the audited patient
        is_met = condition.iloc[0] if hasattr(condition, 'iloc') else bool(np.array(condition)[0])
        if is_met:
            logger.info(f"    [+] {description:<30} +{points}")

def calculate_mews(df, rr_col='RR', hr_col='HR', sbp_col='SBP', temp_col='Temp'):
    """
    Computes a simplified Modified Early Warning Score (MEWS).
    Expects a pandas DataFrame and the column names to use.
    """
    # Initialize a score series of zeros matching the dataframe index
    score = pd.Series(0, index=df.index)

    # 1. Respiratory Rate (RR)
    score += np.where(df[rr_col] <= 8, 2, 0)
    score += np.where((df[rr_col] >= 9) & (df[rr_col] <= 14), 0, 0)
    score += np.where((df[rr_col] >= 15) & (df[rr_col] <= 20), 1, 0)
    score += np.where((df[rr_col] >= 21) & (df[rr_col] <= 29), 2, 0)
    score += np.where(df[rr_col] >= 30, 3, 0)

    # 2. Heart Rate (HR)
    score += np.where(df[hr_col] <= 40, 2, 0)
    score += np.where((df[hr_col] >= 41) & (df[hr_col] <= 50), 1, 0)
    score += np.where((df[hr_col] >= 51) & (df[hr_col] <= 100), 0, 0)
    score += np.where((df[hr_col] >= 101) & (df[hr_col] <= 110), 1, 0)
    score += np.where((df[hr_col] >= 111) & (df[hr_col] <= 129), 2, 0)
    score += np.where(df[hr_col] >= 130, 3, 0)

    # (You would add SBP and Temp logic here following the same pattern)

    return score


import pandas as pd
import numpy as np


def calculate_charlson(df, comorbidities):
    """
    Computes the Charlson Comorbidity Index (CCI).
    Expects a dictionary mapping dataframe columns (binary 1/0) to their CCI weights.
    """
    score = pd.Series(0, index=df.index)

    for col, weight in comorbidities.items():
        if col in df.columns:
            score += np.where(df[col] == 1, weight, 0)

    # Optional Age-adjusted CCI: +1 point for every decade over 40
    if 'age' in df.columns:
        age_points = np.where(df['age'] > 40, (df['age'] - 30) // 10, 0)
        score += np.clip(age_points, 0, None)  # Ensure no negative points

    return score


def calculate_charlson_quan(df, age_col='age'):
    """
    Computes the Charlson Comorbidity Index using Quan et al. weights
    and hierarchy rules.
    """
    score = pd.Series(0, index=df.index)

    # 1. Age Component
    if age_col in df.columns:
        score += np.where((df[age_col] >= 50) & (df[age_col] <= 59), 1, 0)
        score += np.where((df[age_col] >= 60) & (df[age_col] <= 69), 2, 0)
        score += np.where((df[age_col] >= 70) & (df[age_col] <= 79), 3, 0)
        score += np.where(df[age_col] >= 80, 4, 0)

    # 2. Apply Points and Hierarchies
    # Simple 1-point categories
    for col in ['mi', 'chf', 'pvd', 'stroke', 'dementia', 'pulmonary', 'rheum', 'pud']:
        if col in df.columns:
            score += df[col].fillna(0).astype(int)

    # Hierarchical Categories (Higher weight trumps lower)
    if 'liver_mod_sev' in df.columns:
        score += np.where(df['liver_mod_sev'] == 1, 3, df.get('liver_mild', 0))

    if 'diabetes_comp' in df.columns:
        score += np.where(df['diabetes_comp'] == 1, 2, df.get('diabetes_uncomp', 0))

    if 'cancer_met' in df.columns:
        score += np.where(df['cancer_met'] == 1, 6, (df.get('cancer_solid', 0) * 2))

    if 'aids' in df.columns:
        score += np.where(df['aids'] == 1, 6, df.get('hiv', 0))

    if 'renal_mod_sev' in df.columns:
        score += np.where(df['renal_mod_sev'] == 1, 2, 0)  # Quan weight for renal

    return score


def calculate_pitt_score(df, temp_col='Temp', hypotens_col='hypotension',
                         vent_col='mech_vent', arrest_col='cardiac_arrest',
                         mental_col='mental_status_score'):
    """
    Computes the Pitt Bacteremia Score.
    Mental status score expects: 0=Alert, 1=Disoriented, 2=Stuporous, 4=Comatose.
    """
    score = pd.Series(0, index=df.index)

    # 1. Temperature (Celsius)
    if temp_col in df.columns:
        score += np.where(df[temp_col] <= 35.0, 2, 0)
        score += np.where((df[temp_col] > 35.0) & (df[temp_col] <= 36.0), 1, 0)
        score += np.where((df[temp_col] >= 39.0) & (df[temp_col] <= 39.9), 1, 0)
        score += np.where(df[temp_col] >= 40.0, 2, 0)

    # 2. Hypotension (Drop in BP or vasopressor requirement)
    if hypotens_col in df.columns:
        score += np.where(df[hypotens_col] == 1, 2, 0)

    # 3. Mechanical Ventilation
    if vent_col in df.columns:
        score += np.where(df[vent_col] == 1, 2, 0)

    # 4. Cardiac Arrest
    if arrest_col in df.columns:
        score += np.where(df[arrest_col] == 1, 4, 0)

    # 5. Mental Status
    if mental_col in df.columns:
        score += df[mental_col].fillna(0)

    return score


def calculate_sirs(df, temp_col='Temp', hr_col='HR', rr_col='RR', wbc_col='WBC'):
    """
    Computes the number of SIRS criteria met (0 to 4).
    """
    sirs_count = pd.Series(0, index=df.index)

    if temp_col in df.columns:
        sirs_count += np.where((df[temp_col] > 38.0) | (df[temp_col] < 36.0), 1, 0)

    if hr_col in df.columns:
        sirs_count += np.where(df[hr_col] > 90, 1, 0)

    if rr_col in df.columns:
        sirs_count += np.where(df[rr_col] > 20, 1, 0)

    if wbc_col in df.columns:
        sirs_count += np.where((df[wbc_col] > 12.0) | (df[wbc_col] < 4.0), 1, 0)

    return sirs_count


def calculate_increment_esbl_v2(df, age_col='age',
                             charlson_col='charlson_score',
                             pitt_col='pitt_score',
                             sirs_col='sirs_count',
                             bsi_source_col='bsi_source',
                             microorganism_col='microorganism',
                             inapprop_abx_col='inappropriate_abx'):
    """
    Computes the INCREMENT-ESBL predictive score for mortality.

    Reference: Palacios-Baena et al., J Antimicrob Chemother, 2017.

    This score is designed to predict 30-day mortality in patients with bloodstream
    infections (BSI) due to extended-spectrum beta-lactamase (ESBL)-producing
    Enterobacteriaceae. The function calculates a total score based on demographic,
    clinical severity, microbiological, and treatment variables.

    Clinical Criteria & Point Allocation:
    | Clinical Variable                      | Condition Evaluated               | Points |
    |----------------------------------------|-----------------------------------|--------|
    | Demographics                           | Age > 50 years                    |   +3   |
    | Chronic Conditions / Comorbidities     | Severe (e.g., Charlson > 3)       |   +4   |
    | Acute Underlying Severity              | High Pitt bacteremia score        |   +3   |
    | Source of Bloodstream Infection (BSI)  | Origin is NOT urinary             |   +3   |
    | SIRS Severity                          | Severe SIRS / Shock present       |   +4   |
    | Microorganism                          | Non-E. coli (e.g., Klebsiella)    |   +2   |
    | Antibiotic Therapy                     | Inappropriate empirical/targeted  |   +2   |

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataframe containing the required clinical columns.
    age_col : str, default='age'
        Column name for patient age (numeric).
    severe_comorb_col : str, default='severe_comorbidities'
        Column name for severe comorbidities flag (binary 1/0).
    high_pitt_col : str, default='high_pitt_score'
        Column name for high Pitt bacteremia score flag (binary 1/0).
    bsi_source_col : str, default='bsi_source'
        Column name for infection source (string).
    severe_sirs_col : str, default='severe_sirs_or_shock'
        Column name for severe SIRS/shock flag (binary 1/0).
    microorganism_col : str, default='microorganism'
        Column name for isolated organism (string).
    inapprop_abx_col : str, default='inappropriate_abx'
        Column name for inappropriate antibiotic therapy flag (binary 1/0).

    Returns
    -------
    pd.Series
        A pandas Series containing the computed INCREMENT-ESBL score (integers)
        for each patient, matching the input DataFrame index.
    """


    score = pd.Series(0, index=df.index)

    # Demographics
    if age_col in df.columns:
        score += np.where(df[age_col] > 50, 3, 0)

    # Severe Comorbidities (Defined by paper as Charlson > 3)
    if charlson_col in df.columns:
        score += np.where(df[charlson_col] > 3, 4, 0)

    # Acute Severity (Defined by paper as Pitt score >= 6)
    if pitt_col in df.columns:
        score += np.where(df[pitt_col] >= 6, 3, 0)

    # Severe SIRS / Shock (Defined as >= 2 SIRS criteria + hypotension/shock)
    if sirs_col in df.columns:
        score += np.where(df[sirs_col] >= 2, 4, 0)  # Simplified for example

    # BSI Source
    if bsi_source_col in df.columns:
        is_urinary = df[bsi_source_col].astype(str) \
            .str.lower().str.contains('urinary|uti', na=False)
        score += np.where(~is_urinary, 3, 0)

    # Microorganism
    if microorganism_col in df.columns:
        is_kleb = df[microorganism_col].astype(str) \
            .str.lower().str.contains('kleb|other', na=False)
        score += np.where(is_kleb, 2, 0)

    # Inappropriate Abx
    if inapprop_abx_col in df.columns: score += np.where(df[inapprop_abx_col] == 1, 2, 0)

    return score


def calculate_increment_esbl(df, age_col='age',
                             charlson_col='charlson_score',
                             pitt_col='pitt_score',
                             sirs_col='sirs_count',
                             bsi_not_urinary_col='bsi_not_urinary',
                             is_non_ecoli_col='is_non_ecoli',
                             inapprop_abx_col='inappropriate_abx',
                             verbose=False,
                             logger=None):
    """
    Computes the INCREMENT-ESBL predictive score for mortality.
    Reference: Palacios-Baena et al., J Antimicrob Chemother, 2017.

    This score is designed to predict 30-day mortality in patients with bloodstream
    infections (BSI) due to extended-spectrum beta-lactamase (ESBL)-producing
    Enterobacteriaceae. The function calculates a total score based on demographic,
    clinical severity, microbiological, and treatment variables.

    Clinical Criteria & Point Allocation:
    | Clinical Variable                      | Condition Evaluated               | Points |
    |----------------------------------------|-----------------------------------|--------|
    | Demographics                           | Age > 50 years                    |   +3   |
    | Chronic Conditions / Comorbidities     | Severe (e.g., Charlson > 3)       |   +4   |
    | Acute Underlying Severity              | High Pitt bacteremia score (>= 6) |   +3   |
    | Source of Bloodstream Infection (BSI)  | Origin is NOT urinary             |   +3   |
    | SIRS Severity                          | Severe SIRS / Shock (SIRS >= 2)   |   +4   |
    | Microorganism                          | Non-E. coli (e.g., Klebsiella)    |   +2   |
    | Antibiotic Therapy                     | Inappropriate empirical/targeted  |   +2   |

    Parameters
    ----------
    df : pd.DataFrame
        The patient dataframe containing the required clinical columns.
    age_col : str, default='age'
        Column name for patient age (numeric).
    charlson_col : str, default='charlson_score'
        Column name for Charlson Comorbidity Index (numeric).
    pitt_col : str, default='pitt_score'
        Column name for Pitt bacteremia score (numeric).
    sirs_col : str, default='sirs_count'
        Column name for SIRS criteria count (numeric).
    bsi_source_col : str, default='bsi_source'
        Column name for infection source (string).
    microorganism_col : str, default='microorganism'
        Column name for isolated organism (string).
    inapprop_abx_col : str, default='inappropriate_abx'
        Column name for inappropriate antibiotic therapy flag (binary 1/0).

    Returns
    -------
    pd.Series
        A pandas Series containing the computed INCREMENT-ESBL score (integers)
        for each patient, matching the input DataFrame index.
    """
    # 1. Validation: Check for missing columns before calculating
    validate_required_columns(df,
        required_cols=[age_col, charlson_col, pitt_col, sirs_col,
            bsi_not_urinary_col, is_non_ecoli_col, inapprop_abx_col],
        score_name="Palacios-Baena 2019"
    )

    # For audit purposes
    if verbose and logger:
        logger.info("  [Palacios-Baena 2019 Score Breakdown]")

    score = pd.Series(0, index=df.index)

    # 2. Calculation logic with safety checks
    if age_col in df.columns:
        score += np.where(df[age_col] > 50, 3, 0)
        audit_log(df[age_col]>50, 1, "Age >= 50", verbose, logger)

    if charlson_col in df.columns:
        score += np.where(df[charlson_col] > 3, 4, 0)
        audit_log(df[charlson_col] > 3, 4, "Charlson > 3", verbose, logger)

    if pitt_col in df.columns:
        score += np.where(df[pitt_col] >= 6, 3, 0)
        audit_log(df[pitt_col] >= 6, 3, "Pitt Score >= 6", verbose, logger)

    if sirs_col in df.columns:
        score += np.where(df[sirs_col] >= 2, 4, 0)
        audit_log(df[sirs_col] >= 2, 4, "SIRS >= 2", verbose, logger)

    if bsi_not_urinary_col in df.columns:
        score += np.where(df[bsi_not_urinary_col] == 1, 3, 0)
        audit_log(df[bsi_not_urinary_col] == 1, 3, "Source NOT Urinary", verbose, logger)

    if is_non_ecoli_col in df.columns:
        score += np.where(df[is_non_ecoli_col] == 1, 2, 0)
        audit_log(df[is_non_ecoli_col] == 1, 2, "Non-E. coli (e.g., Klebsiella)", verbose, logger)

    if inapprop_abx_col in df.columns:
        score += np.where(df[inapprop_abx_col] == 1, 2, 0)
        audit_log(df[inapprop_abx_col] == 1, 2, "Inappropriate Abx", verbose, logger)

    if verbose and logger:
        logger.info(f"    {'-' * 38}")
        logger.info(f"    [=] TOTAL COMPUTED:            {score.iloc[0]}\n")

    return score

def calculate_holmgren_score(df,
                             hosp_abroad_col='hosp_abroad_12m',
                             prev_culture_col='prev_3gcr_culture',
                             prev_swab_col='prev_3gcr_rectal_swab',
                             verbose=False,
                             logger=None):
    """
    Computes the Holmgren score (2020) for 3GCR Enterobacterales bacteraemia.
    Reference: Holmgren et al., "An easy-to-use scoring system...", 2020.

    Clinical Criteria & Point Allocation:
    | Clinical Variable                    | Condition Evaluated                    | Points |
    |--------------------------------------|----------------------------------------|--------|
    | Hospital care abroad                 | Hospitalized abroad in last 12 months  |   +1   |
    | Previous 3GCR Culture                | Previous 3GCR in blood or urine        |   +1   |
    | Previous 3GCR Rectal Swab            | Previous 3GCR in rectal swab           |   +1   |

    Interpretation: Score >= 1 is considered "High Risk" in low-resistance settings.
    """
    validate_required_columns(df,
        required_cols=[hosp_abroad_col, prev_culture_col, prev_swab_col],
        score_name="Holmgren 2020"
    )

    score = pd.Series(0, index=df.index)
    if verbose and logger: logger.info("  [Holmgren 2020 Score Breakdown]")

    # 1. Previous hospital care abroad (12 months)
    if hosp_abroad_col in df.columns:
        cond = df[hosp_abroad_col] == 1
        score += np.where(cond, 1, 0)
        audit_log(cond, 1, "Hospital care abroad (12m)", verbose, logger)

    # 2. 3GCR in a previous blood or urine culture
    if prev_culture_col in df.columns:
        cond = df[prev_culture_col] == 1
        score += np.where(cond, 1, 0)
        audit_log(cond, 1, "Previous 3GCR Culture", verbose, logger)

    # 3. 3GCR in a previous rectal swab culture
    if prev_swab_col in df.columns:
        cond = df[prev_swab_col] == 1
        score += np.where(cond, 1, 0)
        audit_log(cond, 1, "Previous 3GCR Rectal Swab", verbose, logger)

    if verbose and logger:
        logger.info(f"    {'-' * 38}\n    [=] TOTAL COMPUTED:            {score.iloc[0]}\n")

    return score

# src/scores.py

def calculate_gavaghan_score(df,
                             age_col='age',
                             prior_esbl_col='prior_esbl_365d',
                             nursing_home_col='nursing_home_resident',
                             urinary_catheter_col='urinary_catheter_present',
                             prior_abx_col='prior_f_c_abx_90d',
                             verbose=False,
                             logger=None):
    """
    Computes the Gavaghan et al. (2025) ESBL Risk Score.
    Reference: Gavaghan et al., Antimicrob Steward Healthc Epidemiol, 2025.

    The Gavaghan et al. (2025) score is a contemporary tool developed specifically for
    risk assessment of ESBL-producing Enterobacterales bacteremia in a tertiary setting.

    Clinical Variable & Point Allocation:
    | Clinical Variable                    | Condition Evaluated                       | Points |
    |--------------------------------------|-------------------------------------------|--------|
    | Prior ESBL                           | Any ESBL organism within 365 days         |   +4   |
    | Age                                  | Age >= 65 years                           |   +1   |
    | Nursing Home Resident                | Lives in a long-term care facility        |   +2   |
    | Urinary Catheter                     | Indwelling catheter at presentation      |   +1   |
    | Prior Antibiotics                    | Fluoroquinolone or Cephalosporin (90d)    |   +2   |
    """
    validate_required_columns(df,
        required_cols=[age_col, prior_esbl_col, nursing_home_col,
            urinary_catheter_col, prior_abx_col],
        score_name="Holmgren 2020"
    )

    if verbose and logger:
        logger.info("  [Gavaghan 2025 Score Breakdown]")

    score = pd.Series(0, index=df.index)

    # 1. Prior ESBL (365 days)
    if prior_esbl_col in df.columns:
        cond = df[prior_esbl_col] == 1
        score += np.where(cond, 4, 0)
        audit_log(cond, 4, "Prior ESBL (365 days)", verbose, logger)

    # 2. Age >= 65
    if age_col in df.columns:
        cond = df[age_col] >= 65
        score += np.where(cond, 1, 0)
        audit_log(cond, 1, "Age >= 65", verbose, logger)

    # 3. Nursing Home Residency
    if nursing_home_col in df.columns:
        cond = df[nursing_home_col] == 1
        score += np.where(cond, 2, 0)
        audit_log(cond, 2, "Nursing Home Resident", verbose, logger)

    # 4. Urinary Catheter
    if urinary_catheter_col in df.columns:
        cond = df[urinary_catheter_col] == 1
        score += np.where(cond, 1, 0)
        audit_log(cond, 1, "Urinary Catheter Present", verbose, logger)

    # 5. Prior Antibiotics (FQs or 3G-Cephalosporins within 90 days)
    if prior_abx_col in df.columns:
        cond = df[prior_abx_col] == 1
        score += np.where(cond, 2, 0)
        audit_log(cond, 2, "Prior Antibiotics (90d)", verbose, logger)

    if (verbose and
        logger): logger.info(f"    {'-' * 38}\n    [=] TOTAL COMPUTED:            {score.iloc[0]}\n")

    return score


# src/scores.py

def calculate_jones_score(df,
                          prior_esbl_col='prior_esbl_180d',
                          prior_abx_col='prior_abx_30d',
                          chronic_dialysis_col='chronic_dialysis',
                          transfer_hosp_col='transfer_from_hosp',
                          verbose=False,
                          logger=None):
    """
    Computes the Jones et al. (2025) ESBL Risk Score for Non-Urinary Isolates.
    Reference: Jones et al., Pharmacotherapy, 2025.

    The Jones et al. (2025) score is a specialized tool designed specifically for
    non-urinary isolates. This is a crucial distinction in clinical practice, as
    risk factors for ESBL in bloodstream or respiratory infections often differ
    from those in simple UTIs.

    Clinical Variable & Point Allocation:
    | Clinical Variable           | Condition Evaluated                       | Points |
    |-----------------------------|-------------------------------------------|--------|
    | Prior ESBL                  | Positive ESBL culture within 180 days     |   +5   |
    | Prior Antibiotics           | Any antibiotic use within 30 days         |   +2   |
    | Chronic Dialysis            | Patient on hemodialysis or peritoneal     |   +2   |
    | Transfer from Hospital      | Admission via transfer from another hosp  |   +1   |
    """
    validate_required_columns(df,
        required_cols=[prior_esbl_col, prior_abx_col,
            chronic_dialysis_col, transfer_hosp_col],
        score_name="Jones 2025"
    )

    score = pd.Series(0, index=df.index)

    if verbose and logger:
        logger.info("  [Jones 2025 Score Breakdown]")

    # 1. Prior ESBL (180 days) - Heaviest weight
    if prior_esbl_col in df.columns:
        cond = df[prior_esbl_col] == 1
        score += np.where(cond, 5, 0)
        audit_log(cond, 5, "Prior ESBL (180 days)", verbose, logger)

    # 2. Prior Antibiotics (30 days)
    if prior_abx_col in df.columns:
        cond = df[prior_abx_col] == 1
        score += np.where(cond, 2, 0)
        audit_log(cond, 2, "Prior Antibiotics (30 days)", verbose, logger)

    # 3. Chronic Dialysis
    if chronic_dialysis_col in df.columns:
        cond = df[chronic_dialysis_col] == 1
        score += np.where(cond, 2, 0)
        audit_log(cond, 2, "Chronic Dialysis", verbose, logger)

    # 4. Transfer from Hospital
    if transfer_hosp_col in df.columns:
        cond = df[transfer_hosp_col] == 1
        score += np.where(cond, 1, 0)
        audit_log(cond, 1, "Transfer from Hospital", verbose, logger)

    if verbose and logger:
        logger.info(f"    {'-' * 38}\n    [=] TOTAL COMPUTED:            {score.iloc[0]}\n")

    return score

# src/scores.py

def calculate_tumbarello_score(df,
                               prior_esbl_col='prior_esbl_history',
                               hosp_90d_col='hosp_last_90d',
                               abx_90d_col='abx_last_90d',
                               urinary_catheter_col='urinary_catheter_present',
                               verbose=False,
                               logger=None):
    """
    Computes the Tumbarello/Utrecht-Stockholm ESBL Risk Score.
    Reference: Int J Antimicrob Agents, 2019 (Utrecht/Stockholm Cohort).

    This specific model is designed for community-onset sepsis, making it a vital
    baseline for patients arriving at the Emergency Department before hospital-acquired
    factors come into play.

    Clinical Variable & Point Allocation:
    | Clinical Variable           | Condition Evaluated                       | Points |
    |-----------------------------|-------------------------------------------|--------|
    | Prior ESBL                  | Known colonization/infection (any time)   |   +4   |
    | Recent Hospitalization      | Hospitalized within last 90 days          |   +2   |
    | Recent Antibiotics          | Beta-lactams/Quinolones within 90 days    |   +2   |
    | Urinary Catheter            | Permanent or recent urinary catheter      |   +1   |

    Interpretation: High risk is typically defined as a score >= 3 or 4.
    """
    validate_required_columns(df,
        required_cols=[prior_esbl_col, hosp_90d_col,
            abx_90d_col, urinary_catheter_col],
        score_name="Jones 2025"
    )

    score = pd.Series(0, index=df.index)

    if verbose and logger:
        logger.info("  [Tumbarello Score Breakdown]")

    # 1. Prior ESBL History
    if prior_esbl_col in df.columns:
        cond = df[prior_esbl_col] == 1
        score += np.where(cond, 4, 0)
        audit_log(cond, 4, "Prior ESBL History", verbose, logger)

    # 2. Recent Hospitalization (90 days)
    if hosp_90d_col in df.columns:
        cond = df[hosp_90d_col] == 1
        score += np.where(cond, 2, 0)
        audit_log(cond, 2, "Recent Hospitalization (90d)", verbose, logger)

    # 3. Recent Antibiotics (90 days)
    if abx_90d_col in df.columns:
        cond = df[abx_90d_col] == 1
        score += np.where(cond, 2, 0)
        audit_log(cond, 2, "Recent Antibiotics (90d)", verbose, logger)

    # 4. Urinary Catheter
    if urinary_catheter_col in df.columns:
        cond = df[urinary_catheter_col] == 1
        score += np.where(cond, 1, 0)
        audit_log(cond, 1, "Urinary Catheter", verbose, logger)

    if verbose and logger:
        logger.info(f"    {'-' * 38}\n    [=] TOTAL COMPUTED:            {score.iloc[0]}\n")
    return score


# src/scores.py

def calculate_kim_score(df,
                        prior_esbl_col='prior_esbl_history',
                        hosp_1y_col='hosp_last_1y',
                        nursing_home_col='nursing_home_resident',
                        urinary_catheter_col='urinary_catheter_present',
                        prior_abx_90d_col='prior_abx_90d',
                        verbose=False,
                        logger=None):
    """
    Computes the Kim et al. (2019) ESBL Risk Score.
    Reference: Kim et al., J Korean Med Sci, 2019.

    It focuses on identifying risk factors specifically for community-onset BSIs caused
    by ESBL-producing E. coli and Klebsiella species. This model is particularly useful
    for differentiating resistant from susceptible strains right at the point of admission.

    Clinical Variable & Point Allocation:
    | Clinical Variable           | Condition Evaluated                       | Points |
    |-----------------------------|-------------------------------------------|--------|
    | Prior ESBL                  | Prior ESBL colonization or infection      |   +5   |
    | Recent Hospitalization      | Hospitalization within the last 1 year    |   +2   |
    | Nursing Home Resident       | Resident in a long-term care facility     |   +2   |
    | Urinary Catheter            | Use of indwelling urinary catheter        |   +1   |
    | Prior Antibiotics           | Use of antibiotics within 90 days         |   +1   |
    """
    validate_required_columns(df,
        required_cols=[prior_esbl_col, hosp_1y_col,
             nursing_home_col, urinary_catheter_col, prior_abx_90d_col],
        score_name="Jones 2025"
    )

    score = pd.Series(0, index=df.index)

    if verbose and logger:
        logger.info("  [Kim 2019 Score Breakdown]")

    # 1. Prior ESBL History
    if prior_esbl_col in df.columns:
        cond = df[prior_esbl_col] == 1
        score += np.where(cond, 5, 0)
        audit_log(cond, 5, "Prior ESBL History", verbose, logger)

    # 2. Recent Hospitalization (1 year)
    if hosp_1y_col in df.columns:
        cond = df[hosp_1y_col] == 1
        score += np.where(cond, 2, 0)
        audit_log(cond, 2, "Recent Hospitalization (1y)", verbose, logger)

    # 3. Nursing Home Resident
    if nursing_home_col in df.columns:
        cond = df[nursing_home_col] == 1
        score += np.where(cond, 2, 0)
        audit_log(cond, 2, "Nursing Home Resident", verbose, logger)

    # 4. Urinary Catheter
    if urinary_catheter_col in df.columns:
        cond = df[urinary_catheter_col] == 1
        score += np.where(cond, 1, 0)
        audit_log(cond, 1, "Urinary Catheter", verbose, logger)

    # 5. Prior Antibiotic Use (90 days)
    if prior_abx_90d_col in df.columns:
        cond = df[prior_abx_90d_col] == 1
        score += np.where(cond, 1, 0)
        audit_log(cond, 1, "Prior Antibiotic Use (90d)", verbose, logger)

    if verbose and logger:
        logger.info(f"    {'-' * 38}\n    [=] TOTAL COMPUTED:            {score.iloc[0]}\n")
    return score

# src/scores.py

def calculate_consensus_2023_meta(df,
                                  prior_esbl_col='prior_esbl_history',
                                  prior_abx_col='prior_abx_90d',
                                  hosp_col='recent_hospitalization',
                                  invasive_proc_col='recent_procedure',
                                  verbose=False,
                                  logger=None):
    """
    Weighted Consensus Score based on the Timbrook & Fowler (2023) Meta-Analysis.
    Weights are derived from the most common aORs reported in the review.
    """
    score = pd.Series(0, index=df.index)

    if verbose and logger:
        logger.info("  [Consensus 2023 Meta Score Breakdown]")

    # 1. Prior ESBL (Heaviest weighted across all 10 studies)
    if prior_esbl_col in df.columns:
        cond = df[prior_esbl_col] == 1
        score += np.where(cond, 4, 0)
        audit_log(cond, 4, "Prior ESBL", verbose, logger)

    # 2. Recent Hospitalization
    if hosp_col in df.columns:
        cond = df[hosp_col] == 1
        score += np.where(cond, 2, 0)
        audit_log(cond, 2, "Recent Hospitalization", verbose, logger)

    # 3. Antibiotic Exposure (30-90 days)
    if prior_abx_col in df.columns:
        cond = df[prior_abx_col] == 1
        score += np.where(cond, 2, 0)
        audit_log(cond, 2, "Antibiotic Exposure", verbose, logger)

    # 4. Recent Invasive Procedure (GI/GU)
    if invasive_proc_col in df.columns:
        cond = df[invasive_proc_col] == 1
        score += np.where(cond, 1, 0)
        audit_log(cond, 1, "Recent Invasive Procedure", verbose, logger)

    if verbose and logger:
        logger.info(f"    {'-' * 38}\n    [=] TOTAL COMPUTED:            {score.iloc[0]}\n")
    return score