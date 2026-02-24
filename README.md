# 🏥 Clinical Scoring Evaluation Pipeline

Welcome to the Clinical Scoring Evaluation Pipeline! This project 
is a highly modular, configuration-driven architecture designed to 
generate synthetic patient data, engineer complex clinical features 
(like rolling averages and clinical phenotypes), and evaluate 
predictive scoring systems (like MEWS, SOFA, and INCREMENT-ESBL).

---

## 🚀 The 3-Step Workflow

This pipeline is broken into three distinct scripts. This means if 
you want to tweak a clinical score, you don't have to wait for the 
data generation step to run all over again!

Run these in order from your terminal at the root of the project:

---

### 1️⃣ Generate Synthetic Data

```bash
python scripts/01_generate_data.py
```

**What it does:**  
Reads `config/data_config.yaml`, generates a synthetic patient 
cohort with demographics, comorbidities, and time-series vitals/labs, 
and applies realistic missingness.

**Outputs to:**  
`data/synthetic/<timestamp>/`

---

### 2️⃣ Build Features & Compute Scores

```bash
python scripts/02_build_features.py
```

**What it does:**  
Grabs the latest synthetic data, handles missing value imputation, 
calculates time-windowed features (e.g., 24-hour max heart rate), 
and computes clinical scores based on `config/feature_config.yaml`.

**Outputs to:**  
`data/processed/<timestamp>/features_engineered.csv`

---

### 3️⃣ Evaluate Model Performance

```bash
python scripts/03_evaluate_scores.py
```

**What it does:**  
Evaluates how well your computed scores predict the target label 
(e.g., `sepsis_case`). It calculates AUROC, AUPRC, Sensitivity, 
and Specificity at the mathematically optimal threshold.

**Outputs to:**  
- `outputs/metrics/` (CSVs)  
- `outputs/plots/` (ROC/PR curves)

---

# ⚙️ How to Configure the Pipeline

You rarely need to touch the Python code! Almost everything is 
controlled via two simple YAML files in the `config/` directory.

---

## A. Configuring Data Generation (`config/data_config.yaml`)

Want to add a new lab test or demographic feature? Just add it to 
this file. The pipeline will automatically generate it, bound it 
within realistic human ranges, and apply missing data masks.

### Example: Adding a new lab test (e.g., Creatinine)

```yaml
ts_config:
  Creatinine:
    type: 'float'
    unit: 'mg/dL'
    prob: 0.20          # 80% chance this value is missing (only drawn 20% of the time)
    range: [0.3, 15.0]  # Impossible to generate values outside this range
```

### Example: Adding a new demographic or binary condition

### 🔧 Tuning the Population Prevalence
You can now control the "sickness" or "resistance" level of your synthetic population by editing the `conditions` in `config/data_config.yaml`.

* **To simulate a high-resistance setting:** Increase the probability of `prev_3gcr_culture`.
* **To simulate an elderly cohort:** Increase the probability of `CKD` and `malignancy`.

```bash
conditions:
  hosp_abroad_12m: 0.15  # Increased from 0.05 for a high-travel cohort
```

```yaml
conditions:
  - mech_vent         # Adds a binary 1/0 column for mechanical ventilation
  - cardiac_arrest    # Adds a binary 1/0 column for cardiac arrest
```

---

## B. Configuring Features & Scores (`config/feature_config.yaml`)

This file tells the pipeline exactly how to handle the data 
generated in Step 1.

---

### 1️⃣ Base Features (Imputation & Rolling Windows)

Define how to handle missing data and what statistical windows to compute.

```yaml
base_features:
  HR:
    impute: 'ffill'           # Forward-fill missing heart rates
    rolling:
      windows: ['24h', '7D']  # Calculate 24-hour and 7-day stats
      aggs: ['mean', 'max']   # Specifically, the mean and max
```

---

### 2️⃣ Computed Features (Simple Math)

You can write simple math directly as strings. The pipeline will 
compile and calculate it instantly.

```yaml
computed_features:
  Shock_Index: "HR / SBP"
  derived_diabetes: "(med_insulin_given == 1) or (Glucose_24h_max > 200)"
```

---

### 3️⃣ Custom Scores (Complex Clinical Logic)

For heavy clinical scoring (like MEWS or INCREMENT-ESBL), write a 
Python function in `src/scores.py`, then map the columns to it here.

```yaml
custom_scores:
  INCREMENT_ESBL_Score:
    module: 'scores'                     # Looks in src/scores.py
    function: 'calculate_increment_esbl' # Calls this specific function
    kwargs:                              # Passes these dataset columns as arguments
      age_col: 'age'
      charlson_col: 'Charlson_Score'
      pitt_col: 'Pitt_Score'
      inapprop_abx_col: 'inappropriate_abx'
```

---

# 🛠️ Adding a Brand New Clinical Score (A Guide for Juniors)

Let’s say your attending physician asks you to test the qSOFA score  
(Respiratory Rate ≥ 22, Systolic BP ≤ 100, Altered Mental Status).

---

## Step 1: Write the logic in `src/scores.py`

```python
# src/scores.py
import pandas as pd
import numpy as np

def calculate_qsofa(df, rr_col='RR', sbp_col='SBP', ams_col='altered_mental_status'):
    score = pd.Series(0, index=df.index)
    
    if rr_col in df.columns:
        score += np.where(df[rr_col] >= 22, 1, 0)
    if sbp_col in df.columns:
        score += np.where(df[sbp_col] <= 100, 1, 0)
    if ams_col in df.columns:
        score += np.where(df[ams_col] == 1, 1, 0)
        
    return score
```

---

## Step 2: Tell the YAML to use it in `config/feature_config.yaml`

```yaml
custom_scores:
  qSOFA_Score:
    module: 'scores'
    function: 'calculate_qsofa'
    kwargs:
      rr_col: 'RR'
      sbp_col: 'SBP'
      ams_col: 'GCS_less_than_15'  # Assuming you engineered a GCS flag!
```

---

## Step 3: Run Steps 02 and 03

```bash
python scripts/02_build_features.py
python scripts/03_evaluate_scores.py
```

You will instantly see your new `qSOFA_Score` evaluated against all 
the others in your `outputs/` folder!

---

## 🧪 Testing and Validation

We use `pytest` to ensure clinical scores (Charlson, Pitt, INCREMENT-ESBL) match the literature exactly and remain stable as the codebase evolves.

---

## ▶️ Running Tests

Run all tests from the project root directory:

```bash
py -m pytest tests                                                          # All
py -m pytest tests/test_scores.py -k "test_all_scores_from_csv"             # That function
py -m pytest tests/test_scores.py -k "test_all_scores_from_csv and index2"  # That patient
```



`pytest` will automatically:

- Discover all `test_*.py` files inside the `tests/` folder  
- Execute each test function  
- Provide a clean summary of passed/failed tests  
- Show detailed tracebacks if failures occur  

---

## ✅ Why We Test

### Hierarchy Verification
Ensures weighted conditions correctly override each other.
Example:
- Moderate/Severe Liver Disease (3 pts) correctly overrides Mild Liver Disease (1 pt).

### Edge Case Handling
Validates extreme physiological and clinical inputs:
- Cardiac arrest
- Mechanical ventilation
- Comatose Glasgow Coma Scale (GCS)
- Severe hypotension

### Scenario Matching
Confirms outputs match published validation scenarios from:
- Charlson Comorbidity Index (Quan ICD-10 update)
- Pitt Bacteremia Score
- INCREMENT-ESBL Score

### Regression Protection
Prevents silent scoring changes when:
- Updating ICD-10 mappings
- Refactoring score logic
- Adjusting condition hierarchies

---

## 📍 How and Where to Run

### 1️⃣ Where
Always run tests from the **project root directory**:

```
project-root/
├── src/
├── scripts/
├── tests/
└── README.md
```

### 2️⃣ How

In your terminal:

```bash
pytest
```

---

## 🧩 Adding ICD-10 Mapping Tests

To ensure ICD-10 codes correctly map to Charlson conditions, create:

```
tests/test_icd10_mapping.py
```

Example test:

```python
import pytest
from src.charlson import calculate_charlson_score

def test_mild_vs_moderate_liver_hierarchy():
    """
    Moderate liver disease should override mild liver disease.
    """
    icd_codes = ["K73.9", "K72.90"]  # Example mild + moderate
    score = calculate_charlson_score(icd_codes)
    assert score == 3  # Only moderate counted

def test_metastatic_cancer_weight():
    icd_codes = ["C78.7"]  # Secondary malignant neoplasm
    score = calculate_charlson_score(icd_codes)
    assert score == 6
```

---

## 📊 Test Coverage (Optional but Recommended)

Install coverage support:

```bash
pip install pytest-cov
```

Run tests with coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

Generate HTML coverage report:

```bash
pytest --cov=src --cov-report=html
```

Then open:

```
htmlcov/index.html
```

---

## 🚀 Continuous Integration (CI) with GitHub Actions

Create:

```
.github/workflows/tests.yml
```

Example CI configuration:

```yaml
name: Run Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest --cov=src --cov-report=term
```

This ensures:

- All pull requests are automatically tested  
- Score calculations cannot silently change  
- Coverage is continuously monitored  

---

## 🛡️ Best Practices

- Every scoring function must have:
  - At least one normal case test
  - One edge case test
  - One hierarchy/override test
- Any change to ICD-10 mappings requires:
  - Updating mapping tests
  - Running full regression suite
- CI must pass before merging to `main`

---

## 🎯 Summary

Our testing framework ensures:

- Literature-faithful score calculations  
- Correct ICD-10 condition mapping  
- Proper hierarchy enforcement  
- Safe refactoring  
- Reproducible validation  

Reliable clinical scoring requires deterministic, validated logic — and automated testing guarantees it.