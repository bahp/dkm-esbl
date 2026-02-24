# tests/test_scores.py
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure src is in the path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.scores import (
    calculate_mews,
    calculate_sirs,
    calculate_charlson,
    calculate_charlson_quan,
    calculate_pitt_score,
    calculate_increment_esbl,
    calculate_holmgren_score,
    calculate_gavaghan_score,
    calculate_jones_score,
    calculate_tumbarello_score,
    calculate_kim_score
)


# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.scores import calculate_charlson_quan, calculate_pitt_score

def get_test_data():
    """Helper to load the CSV test cases."""
    csv_path = Path(__file__).parent / 'cases.csv'
    return pd.read_csv(csv_path)


def test_charlson_quan_hierarchy():
    """Verify that Moderate Liver Disease trumps Mild Liver Disease (Scenario C)."""
    # 65-year-old (+2) + Mild (1) + Moderate (3) = 5
    data = pd.DataFrame({
        'age': [65],
        'liver_mild': [1],
        'liver_mod_sev': [1]
    })
    result = calculate_charlson_quan(data)
    assert result[0] == 5, f"Hierarchy failed: Expected 5, got {result[0]}"

def test_charlson_quan_metastatic():
    """Verify that Metastatic Cancer trumps non-metastatic (Scenario B)."""
    # 85-year-old (+4) + Metastatic (6) + CHF (1) = 11
    data = pd.DataFrame({
        'age': [85],
        'cancer_met': [1],
        'cancer_solid': [1],
        'chf': [1]
    })
    result = calculate_charlson_quan(data)
    assert result[0] == 11, f"Expected 11, got {result[0]}"

def test_pitt_score_extreme():
    """Test extreme Pitt score values."""
    data = pd.DataFrame({
        'Temp': [34.0],          # +2
        'hypotension': [1],      # +2
        'mech_vent': [1],        # +2
        'cardiac_arrest': [1],   # +4
        'mental_status_score': [4] # +4 (Comatose)
    })
    result = calculate_pitt_score(data)
    assert result[0] == 14


def test_charlson_logic():
    """Validates Charlson Comorbidity Index weights."""
    # Patient: Diabetes (1), CKD (2), Age 75 (+3 points for age > 40: (75-30)//10 = 4)
    # Total expected: 1 + 2 + 4 = 7
    data = pd.DataFrame({
        'diabetes': [1],
        'CKD': [1],
        'age': [75]
    })
    weights = {'diabetes': 1, 'CKD': 2}

    result = calculate_charlson(data, weights)
    assert result[0] == 7, f"Charlson Expected 7, got {result[0]}"
    print("✅ Charlson Index test passed!")


def test_pitt_logic():
    """Validates Pitt Bacteremia Score logic."""
    # Patient: Temp 34 (+2), Hypotension (+2), Vent (+2), Arrest (+4), Mental Status 1 (+1)
    # Total expected: 11
    data = pd.DataFrame({
        'Temp': [34.0],
        'hypotension': [1],
        'mech_vent': [1],
        'cardiac_arrest': [1],
        'mental_status_score': [1]
    })

    result = calculate_pitt_score(data, temp_col='Temp', hypotens_col='hypotension')
    assert result[0] == 11, f"Pitt Expected 11, got {result[0]}"
    print("✅ Pitt Score test passed!")


def test_increment_esbl_logic_v2():
    """Validates INCREMENT-ESBL specific point allocation."""
    # Patient:
    # Age 60 (>50: +3)
    # Charlson 5 (>3: +4)
    # Pitt 7 (>=6: +3)
    # SIRS 3 (>=2: +4)
    # Source: Respiratory (Not Urinary: +3)
    # Microorganism: Klebsiella (+2)
    # Inappropriate Abx: Yes (+2)
    # Total Expected: 3 + 4 + 3 + 4 + 3 + 2 + 2 = 21
    data = pd.DataFrame({
        'age': [60],
        'charlson_score': [5],
        'pitt_score': [7],
        'sirs_count': [3],
        'bsi_source': ['Respiratory'],
        'microorganism': ['Klebsiella spp.'],
        'inappropriate_abx': [1]
    })

    result = calculate_increment_esbl(data)
    assert result[0] == 21, f"INCREMENT-ESBL Expected 21, got {result[0]}"

    # Negative Test Case: Young, healthy patient with urinary BSI
    # All 0
    data_low = pd.DataFrame({
        'age': [25],
        'charlson_score': [0],
        'pitt_score': [0],
        'sirs_count': [0],
        'bsi_source': ['Urinary'],
        'microorganism': ['E. coli'],
        'inappropriate_abx': [0]
    })
    result_low = calculate_increment_esbl(data_low)
    assert result_low[0] == 0, f"INCREMENT-ESBL Expected 0, got {result_low[0]}"

    print("✅ INCREMENT-ESBL test passed!")


def test_increment_esbl_logic():
    """Validates INCREMENT-ESBL based on 2017 Palacios-Baena weights."""
    data = pd.DataFrame({
        'age': [60],  # +3 points
        'microorganism': ['Klebsiella pneumoniae'],  # +2 points
        'bsi_source': ['Respiratory'],  # +3 points
        'pitt_score': [7],  # +3 points (threshold is >3)
        'sirs_count': [3],  # +4 points (proxy for severe sepsis)
        'charlson_score': [5],  # +4 points (proxy for fatal disease)
        'inappropriate_abx': [1]  # +2 points
    })

    # Calculation: 3 + 2 + 3 + 3 + 4 + 4 + 2 = 21
    result = calculate_increment_esbl(data)
    assert result[0] == 21, f"INCREMENT-ESBL Expected 21, got {result[0]}"
    print("✅ INCREMENT-ESBL test passed!")

def test_holmgren_logic():
    """Validates Holmgren 2020 score for low-resistance settings."""
    # Patient with only hospital care abroad should get 1 point
    data = pd.DataFrame({
        'hosp_abroad_12m': [1, 0],
        'prev_3gcr_culture': [0, 0],
        'prev_3gcr_rectal_swab': [0, 1]
    })
    results = calculate_holmgren_score(data)
    assert results[0] == 1
    assert results[1] == 1
    print("✅ Holmgren Score test passed!")


def test_gavaghan_logic():
    """Verify Gavaghan 2025 point allocation."""
    # Patient: Age 70 (+1), Prior ESBL (+4), Nursing Home (+2) = 7
    data = pd.DataFrame({
        'age': [70],
        'prior_esbl_365d': [1],
        'nursing_home_resident': [1],
        'urinary_catheter_present': [0],
        'prior_f_c_abx_90d': [0]
    })
    result = calculate_gavaghan_score(data)
    assert result[0] == 7
    print("✅ Gavaghan Score test passed!")


# tests/test_scores.py

def test_jones_logic():
    """Verify Jones 2025 point allocation for non-urinary isolates."""
    # Patient: Prior ESBL (+5) + Transfer (+1) = 6
    data = pd.DataFrame({
        'prior_esbl_180d': [1],
        'prior_abx_30d': [0],
        'chronic_dialysis': [0],
        'transfer_from_hosp': [1]
    })
    result = calculate_jones_score(data)
    assert result[0] == 6, f"Expected 6, got {result[0]}"
    print("✅ Jones Score test passed!")

def test_kim_logic():
    """Verify Kim et al. 2019 point allocation."""
    # Patient: Prior ESBL (+5) + Nursing Home (+2) = 7
    data = pd.DataFrame({
        'prior_esbl_history': [1],
        'hosp_last_1y': [0],
        'nursing_home_resident': [1],
        'urinary_catheter_present': [0],
        'prior_abx_90d': [0]
    })
    result = calculate_kim_score(data)
    assert result[0] == 7
    print("✅ Kim Score test passed!")





def run_all_tests():
    print("--- 🧪 Starting Scoring System Validation ---")
    try:
        test_charlson_logic()
        test_pitt_logic()
        test_increment_esbl_logic()
        # Including your existing ones
        data_mews = pd.DataFrame({'RR': [30], 'HR': [130], 'SBP': [70], 'Temp': [34.0]})
        assert calculate_mews(data_mews)[0] == 10
        print("✅ MEWS test passed!")
        print("\n🎉 ALL SCORING TESTS PASSED!")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")


# tests/test_scores.py
# tests/test_scores.py

@pytest.mark.parametrize("index, row", get_test_data().iterrows())
def test_all_scores_from_csv(index, row):
    # CRITICAL FIX: Convert row to DataFrame and reset index to 0
    patient_df = pd.DataFrame([row]).reset_index(drop=True)

    # Now result[0] will always work because the index is reset to 0
    results = {
        'INCREMENT-ESBL': (calculate_increment_esbl(patient_df).iloc[0], row['exp_increment']),
        'Holmgren': (calculate_holmgren_score(patient_df).iloc[0], row['exp_holmgren']),
        'Gavaghan': (calculate_gavaghan_score(patient_df).iloc[0], row['exp_gavaghan']),
        'Jones': (calculate_jones_score(patient_df).iloc[0], row['exp_jones']),
        'Tumbarello': (calculate_tumbarello_score(patient_df).iloc[0], row['exp_tumbarello']),
        'Kim': (calculate_kim_score(patient_df).iloc[0], row['exp_kim'])
    }

    for score_name, (calculated, expected) in results.items():
        assert calculated == expected, \
            f"Row {index} ({row['description']}): {score_name} failed. Expected {expected}, got {calculated}"


if __name__ == "__main__":
    run_all_tests()