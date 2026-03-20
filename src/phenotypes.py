"""
Author:
Date:

This module acts as the "translation layer" between raw, messy Electronic
Health Record (EHR) data and clean, standardized clinical variables.

NAMING CONVENTIONS
==================
To keep the pipeline predictable and easy to read, all functions in this file
follow strict naming prefixes based on the exact type of data they return:

1. `has_` (Returns Integer 1 or 0)
   Used for past medical history, chronic conditions, or comorbidities.
   Example: `has_diabetes()`, `has_congestive_heart_failure()`

2. `is_` (Returns Integer 1 or 0)
   Used for acute, current patient states occurring during this specific encounter.
   Example: `is_mechanically_ventilated()`, `is_urinary_source()`

3. `derive_` (Returns Continuous, Categorical, or Multi-level Data)
   Used when the output is not a simple boolean flag. This includes extracting
   numbers, categorizing statuses, or performing date math.
   Example: `derive_age_at_admission()`, `derive_fever_status()`git
"""

# src/phenotypes.py
import pandas as pd
import numpy as np

from src.utils import (check_col_bool,
                       check_col_contains,
                       check_col_threshold,
                       check_col_icd10)


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

    Data Mapping for iCARE:
    -----------------------
    - Episodes: Table `icare_episodes_diagnosis_anon` 
      Cols: `diagnosis_code_icd`, `diagnosis_code_snomed`, `diagnosis_desc_icd`, 
            `diagnosis_desc_snomed`, `diagnosis_date`.
    - Problems: Table `icare_problem_anon`
      Cols: `problem_code` (SNOMED), `problem_desc`, `problem_dt_tm`.

    Required Logic:
    Must search BOTH the diagnosis episodes and the problem list for matches.
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

# -----------------------------------------------------------------------------------
#                 CHARLSON COMORBIDITY INDEX (CCI) EXTRACTORS
# -----------------------------------------------------------------------------------
# These methods extract the 17 chronic conditions required to compute the Charlson
# Comorbidity Index, a validated method of categorizing comorbidities of patients
# based on the International Classification of Diseases (ICD) diagnosis codes.
#
# Reference:
# Charlson ME, et al. "A new method of classifying prognostic comorbidity in
# longitudinal studies: development and validation." J Chronic Dis. 1987;40(5):373-83.
# Mapping Logic:
# - SNOMED codes are found in `icare_problem_anon.problem_code`.
# - ICD-10/SNOMED codes are found in `icare_episodes_diagnosis_anon`.
# - A list of all the snomed codes for all the comorbidities from publication (https://clinicalcodes.rss.mhs.man.ac.uk/medcodes/article/196/) can be based on 'res195-comorbidity-cci-gold.csv' but need to check with Damien on these.

def has_myocardial_infarction(df, **kwargs):
    """""
    Identifies history of MI.
    Codes: Use 'MI' category from CCI Gold CSV (e.g., 323..00, G30..00).
    Search: `icare_problem_anon` and `icare_episodes_diagnosis_anon`.
    pass
    """

def has_congestive_heart_failure(df, **kwargs):
    """
    @Example:

    Determines if a patient has a history of Congestive Heart Failure (CHF).

    Clinical Logic:
    A patient is flagged with CHF if ANY of the following criteria are met:
    1. SNOMED Codes: Take from es195-comorbidity-cci-gold.csv (Hypertensive heart disease with heart failure).
    2. Explicit Flag: The `hx_chf` boolean column is exactly 1.
    3. Medication Proxy: The patient is actively prescribed specific heart failure
       medications like 'entresto', 'milrinone', or 'dobutamine' in the `home_meds` column.

    Required Columns in df:
    - `diagnosis_codes` (String/List of codes)
    - `hx_chf` (Int/Float 1.0 or 0.0)
    - `home_meds` (String)

    Returns:
    pd.Series of integers (1 for has CHF, 0 for no CHF).
    """
    pass

def has_peripheral_vascular_disease(df, **kwargs):
    """"""
    pass

def has_cerebrovascular_disease(df, **kwargs):
    """"""
    pass

def has_dementia(df, **kwargs):
    """"""
    pass

def has_chronic_pulmonary_disease(df, **kwargs):
    """"""
    pass

def has_connective_tissue_disease(df, **kwargs):
    """"""
    pass

def has_peptic_ulcer_disease(df, **kwargs):
    """"""
    pass

def has_mild_liver_disease(df, **kwargs):
    """

    Clinical Logic:
    1. Explicit History: (Fill this out)
    2. ICD-10 Codes: (Fill this out - e.g., K70, K74)
    3. Medications: (Fill this out - e.g., lactulose)
    4. Lab Values: (Fill this out - e.g., AST > 3x normal, Bilirubin > 2.0)
    """

    # c1_history = check_col_bool(...)
    # c2_icd10 = check_col_icd10(...)
    # c3_meds = check_col_contains(...)
    # c4_labs = check_col_threshold(...)

    # combined_signal = c1_history | c2_icd10 | c3_meds | c4_labs
    # return combined_signal.astype(int)
    pass


def has_diabetes(df, config=None):
    """
    @Example:

    Determines if a patient has diabetes using a multi-modal data approach.

    Clinical Logic defined by the student:
    We consider a patient to have diabetes if ANY of the following are true:
    1. Explicit History: The `diabetes_history` boolean column is exactly 1.
    2. ICD-10 Codes: The `diagnosis_codes` column contains E10, E11, or E14.
    3. Medications: The `medications_given` column contains 'insulin' or 'metformin'.
    4. Lab Values: The rolling maximum glucose (`glucose_max_24h`) is > 200 mg/dL.

    Returns:
    pd.Series of integers (1 for has_diabetes, 0 for no diabetes).
    """
    # 1. Extract signals using helpers
    c1_history = check_col_bool(df, 'diabetes_history')
    c2_icd10 = check_col_icd10(df, 'diagnosis_codes', target_codes=['E10', 'E11', 'E14'])
    c3_meds = check_col_contains(df, 'medications_given', keywords=['insulin', 'metformin'])
    c4_labs = check_col_threshold(df, 'glucose_max_24h', threshold=200, operator='>')

    # 2. Combine signals (Logical OR: if any signal is True, the condition is met)
    combined_signal = c1_history | c2_icd10 | c3_meds | c4_labs

    # 3. Return as 1s and 0s
    return combined_signal.astype(int)

def has_diabetes_without_complications(df, **kwargs):
    """"""
    pass

def has_diabetes_with_complications(df, **kwargs):
    """"""
    pass

def has_hemiplegia_or_paraplegia(df, **kwargs):
    """"""
    pass

def has_moderate_to_severe_renal_disease(df, **kwargs):
    """"""
    pass

def has_malignancy(df, **kwargs):
    """"""
    pass

def has_moderate_to_severe_liver_disease(df, **kwargs):
    """"""
    pass

def has_metastatic_solid_tumor(df, **kwargs):
    """"""
    pass

def has_aids(df, **kwargs):
    """"""
    pass


# ----------------------------------------------------------------------------------------
#                          PITT BACTEREMIA SCORE EXTRACTORS
# ----------------------------------------------------------------------------------------
# These methods derive the 5 acute severity variables required to compute the Pitt
# Bacteremia Score, a validated measure of acute illness severity in patients with
# bloodstream infections (BSIs) assessed either at the day of positive culture or
# admission.
#
# Reference:
# Korvick JA, et al. "Prospective observational study of Klebsiella bacteremia..."
# Clin Infect Dis. 1992;15(5):795-801.
# (Also widely validated for ESBL by Paterson et al., 2004).
# Score Range: 0 to 14 points from https://onlinelibrary.wiley.com/doi/10.1155/2024/6996399?af=R#bib-0017

def derive_mental_status_score(df, **kwargs):
    """
    Clinical Logic:
    - Alertness: 0 pts
    - Disorientation: 1 pt
    - Stupor: 2 pts
    - Coma: 4 pts

    iCARE Mapping:
    - Table: `icare_vital_signs_anon`
    - Check Columns: `news_conscious_level_score`, `new_score_consciousness`, or 
      `glasgow_coma_score`.
    """
    pass

def derive_fever_status(df, **kwargs):
    """
    Clinical Logic:
    - 36.1°C – 38.9°C: 0 pts
    - 35.1°C – 36.0°C or 39.0°C – 39.9°C: 1 pt
    - <= 35°C or >= 40°C: 2 pts

    iCARE Mapping:
    - Table: `icare_vital_signs_anon`
    - Filter: `observation_code` IN (10933766, 486347689)
    - Value: `observation_result_clean`
    """

def derive_hypotension_status(df, **kwargs):
    """
    Clinical Logic:
    - Presence of hypotension: +2 pts
    - Systolic BP < 90 mmHg or requires vasopressors from icare_antitbiotic_prescribing whiuch contains vast range of drugs.
    iCARE Mapping:
    - Table: `icare_vital_signs_anon`
    - Filter: `observation_code` IN (13389125)
    - Value: `observation_result_clean`

    """
    pass

def is_mechanically_ventilated(df, **kwargs):
    """
    Clinical Logic:
    - Receiving mechanical ventilation: +2 pts
    - Search respiratory support flowsheets.
    """
    pass

def has_recent_cardiac_arrest(df, **kwargs):
    """
    Clinical Logic:
    - Cardiac arrest within window: +4 pts
    - Check resuscitation event codes.
    """
    pass


# ----------------------------------------------------------------------------------------
#                               SIRS CRITERIA EXTRACTORS                               ---
# ----------------------------------------------------------------------------------------
# These methods evaluate the 4 physiological parameters required to determine if a
# patient meets the criteria for Systemic Inflammatory Response Syndrome (SIRS),
# indicating a severe, systemic immune response to infection.

# Reference:
# Bone RC, et al. "Definitions for sepsis and organ failure and guidelines for the
# use of innovative therapies in sepsis." Chest. 1992;101(6):1644-55.
def derive_tachycardia(df, **kwargs):
    """Checks HR > 90"""
    """
    Clinical Logic: HR > 90
    iCARE Mapping:
    - Table: `icare_vital_signs_anon`
    - Filter: `observation_code` == 13472364
    - Value: `observation_result_clean`
    """
    pass

def derive_tachypnea(df, **kwargs):
    """Checks RR > 20 or PaCO2 < 32"""
    """
    Clinical Logic: RR > 20
    iCARE Mapping:
    - Table: `icare_vital_signs_anon`
    - Filter: `observation_code` == 9096705
    - Value: `observation_result_clean`
    """
    pass

def derive_abnormal_temp(df, **kwargs):
    """Checks Temp > 38°C or < 36°C"""
    """
    Clinical Logic: Temp > 38°C or < 36°C
    iCARE Mapping:
    - Table: `icare_vital_signs_anon`
    - Filter: `observation_code` IN (10933766, 486347689)
    - Value: `observation_result_clean`
    """
    pass

def derive_abnormal_wbc(df, **kwargs):
    """Checks WBC > 12k, < 4k, or > 10% bands"""
    """
    Clinical Logic: WBC > 12k or < 4k
    iCARE Mapping:
    - Table: `icare_pathology_blood_anon`
    - Filter: `test_code` contains 'wbc'
    - Value: `result_cleaned`
    """
    pass

# ----------------------------------------------------------------------------------------
#                        INCREMENT-ESBL DIRECT CLINICAL EXTRACTORS
# ----------------------------------------------------------------------------------------
# These methods extract the specific standalone clinical variables (source of infection,
# microorganism type, and antibiotic appropriateness) required to compute the final
# INCREMENT-ESBL risk score for 30-day mortality.

# Reference:
# Palacios-Baena Z, et al. "Development and validation of the INCREMENT-ESBL predictive
# score for mortality in patients with bloodstream infections..." J Antimicrob Chemother.
# 2017;72(3):906-913.
def derive_age_at_admission(df, **kwargs):
    """
    @Example:

    Calculates the patient's age in years at the time of hospital admission.

    Clinical Logic:
    1. Direct Extraction: If an `age` column already exists from the demographics table,
       extract it directly.
    2. Date Math: If `age` is missing, calculate it by subtracting the `date_of_birth`
       from the `admission_date` and extracting the years.

    Required Columns in df:
    - `age` (Int/Float, optional)
    - `date_of_birth` (Datetime, optional)
    - `admission_date` (Datetime, optional)

    Returns:
    pd.Series of floats/integers representing patient age in years.
    """
    """
    Requirement: INCREMENT score adds +2 points if age > 50.
    iCARE Mapping:
    - Table: `icare_episodes_anon`
    - Column: `age_at_admission` (Direct extraction, no date math needed).
    """
    pass

def determine_bsi_source(df, **kwargs):
    """
    @Example:

    Determines if the source of the Bloodstream Infection (BSI) is NON-urinary.
    In the INCREMENT-ESBL score, non-urinary sources (respiratory, abdominal, etc.)
    are associated with higher mortality and receive +3 points.

    Clinical Logic:
    A patient is flagged as having a NON-urinary source (returns 1) UNLESS we can
    prove the source was urinary. We assume urinary if:
    1. Explicit Flag: The `bsi_source` column explicitly equals 'Urinary'.
    2. Concurrent Cultures: A urine culture (`urine_culture_result`) drawn within 48h
       grew the same organism.
    3. ICD-10 Proxy: The patient was billed for a UTI (e.g., 'N39.0') on admission.

    If none of the urinary criteria are met, we default to NON-urinary (1).

    Returns:
    pd.Series of integers (1 for NON-urinary source, 0 for Urinary source).
    """
    """
    Requirement: Non-urinary sources receive +3 points.
    iCARE Mapping:
    - Table: `icare_episodes_diagnosis_anon`
    - Column: `diagnosis_code_snomed`
    - Logic: use the diagnosis icd-10 codes 
    - J15.8, J15.9, J18.0, J18.1, J18.9, J85.1, N39.0, N10, N13.6, N30.0, N30.9 (to cover sputum and urine, for example).
    """
    pass

def identify_microorganism_type(df, **kwargs):
    """
    @Example:

    Identifies if the ESBL-producing organism is a non-E. coli species
    (e.g., Klebsiella, Enterobacter, Serratia, Proteus).

    Clinical Logic:
    E. coli bacteremia generally has a better prognosis in this specific cohort.
    We flag the patient as 'Non-E. coli' if:
    1. Blood Culture Result: The `organism_bug`column
       contains keywords like 'klebsiella', 'enterobacter', 'serratia', or 'other'.
    2. Exclusion: Ensure we do NOT flag it if the string strictly says
       'escherichia coli' or 'e. coli'.

    Required Columns in df:
    - `organism_bug` or `sensitivity (String from microbiology LIS system)

    Returns:
    pd.Series of integers (1 for Non-E. coli, 0 for E. coli).
    """
    pass

def evaluate_antibiotic_appropriateness(df, **kwargs):
    """
    @Example:

    Determines if the empirical antibiotic therapy administered was INAPPROPRIATE.

    Clinical Logic:
    Empirical therapy is the drug given *before* the final lab results come back.
    Therapy is considered INAPPROPRIATE (returns 1) if:
    1. Resistance: The `empiric_abx_given` (medication given in the first 24h)
       matches a drug listed in the `abx_resistant_to` column (from the lab report).
    2. No Coverage: The patient received no active anti-ESBL antibiotics
       (like carbapenems) within the first 24 hours of blood culture collection.
    3. Explicit Flag: If an `inappropriate_abx_flag` already exists from a
       clinical pharmacist's manual review, use it.

    Required Columns in df:
    - `empiric_abx_given` (String/List of medications)
    - `abx_resistant_to` (String/List from susceptibility report)
    - `inappropriate_abx_flag` (Int 1 or 0, optional)

    .. note: Notice how evaluate_antibiotic_appropriateness requires us to look at Pharmacy
             data (empiric_abx_given) and Laboratory data (abx_resistant_to) at the same time.
             This is why we write the logic out first! If we don't have the Pharmacy data in
             our dataset, we immediately know we cannot compute this score, and we need to
             ask the data engineers for a new table.

    Returns:
    pd.Series of integers (1 for Inappropriate therapy, 0 for Appropriate therapy).
    """
    """
    Requirement: Inappropriate therapy receives +2 points.
    iCARE_mapping:
    - Merge `icare_microbiology_anon` with `icare_pharmacy_prescribing_anon`.
    - Compare `sample_collected_dt` (`icare_pathology_blood_anon`) with 
      `order_dt_tm` (`icare_pharmacy_prescribing_anon`).
    - Inappropriate if administration > 1 day from blood cultures or 
      > 4 days without targeted anti-ESBL therapy (e.g. Carbapenems).
    """

    pass
