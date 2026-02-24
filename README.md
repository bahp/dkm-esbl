# 🏥 Clinical Scoring Evaluation Pipeline

A modular, configuration-driven pipeline to generate synthetic patient 
data, engineer clinical features, and audit predictive scoring systems 
(e.g., INCREMENT-ESBL, Gavaghan, Jones, Holmgren).

---
## 📁 Key Directories

```
config/        → YAML configuration files controlling:
                   • Data generation probabilities
                   • Feature engineering rules
                 (No Python changes required)

src/           → Core logic modules
   ├── scores.py      → Pure mathematical clinical scoring functions
   ├── features.py    → Pipeline transformation logic
   └── generators.py  → Synthetic data creation utilities

scripts/       → Executable pipeline orchestration scripts
                   • Step 1
                   • Step 2
                   • Step 3

tests/         → Validation and regression testing
   ├── cases.csv      → "Ground Truth" patient profiles
   └── test_scores.py → Unit tests validating scoring accuracy
```

## 🚀 Quick Start: The pipeline

Run all commands from the root of the project directory using the -m (module) flag.

---

### 1️⃣ Generate Synthetic Data

Generates a synthetic patient cohort with demographics, comorbidities, 
and time-series vitals/labs based on your YAML configurations.

```bash
python -m scripts.01_generate_data
```

  - **Reads:** `config/data_config.yaml` and `config/icd_config.yaml` 
  - **Outputs:** `data/synthetic/<timestamp>/`

---

### 2️⃣ Build Features & Compute Scores

Merges raw data, handles missing value imputation, calculates time-windowed 
features, and runs the clinical scoring algorithms across the generated cohort.

```bash
python -m scripts.02_build_features
```

  - **Reads:** `config/feature_config.yaml`
  - **Outputs:** `data/processed/<timestamp>/features_engineered.csv`

---

### 3️⃣ Validate Scores (Audit Report)

Runs specific edge-case patients through the scoring algorithms to generate 
a point-by-point clinical audit trail, ensuring mathematical accuracy.

```bash
python -m scripts.05_validate_scores
```

  - **Reads:** `tests/cases.csv`
  - **Outputs:**  `reports/score_validation.log` (Check this file to see exactly how points were awarded for each patient).


### 3️⃣ Evaluate Model Performance (PENDING)

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

py -m scripts.05_validate_scores.py  
```

You will instantly see your new `qSOFA_Score` evaluated against all 
the others in your `outputs/` folder!

---

## Testing and Validation

We use `pytest` to ensure clinical scores (Charlson, Pitt, INCREMENT-ESBL) 
match the literature exactly and remain stable as the codebase evolves.

---

## Running Tests

Run all tests from the project root directory:

```bash
py -m pytest tests
```

`pytest` will automatically:

- Discover all `test_*.py` files inside the `tests/` folder  
- Execute each test function  
- Provide a clean summary of passed/failed tests  
- Show detailed tracebacks if failures occur  

### 🎯 Running Specific Tests (Advanced)

If you are debugging a specific score or a single patient case, 
you don't need to run the entire test suite. You can use the `-k` (keyword)
flag to isolate exactly what you want to test:

```bash
# Run ONLY the CSV-based scoring tests (all patients)
python -m pytest tests/test_scores.py -k "test_all_scores_from_csv"             

# Run ONLY the CSV test for a SPECIFIC patient (e.g., Row index 2)
python -m pytest tests/test_scores.py -k "test_all_scores_from_csv and index2"
```



