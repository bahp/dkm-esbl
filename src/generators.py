import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime


# ----------------------------------------------------------------------------------
# Helper methods
# ----------------------------------------------------------------------------------
def generate_clinical_data(config,
                           n_patients,
                           days=6,
                           freq='8h',
                           output_format='long',
                           icd_map={},
                           id_col_name='patient_id'):
    """Generates all data dynamically based on the provided schema configuration."""
    # Create unique ids
    unique_ids = np.arange(101, 101 + n_patients)

    # Generate demographics data using the schema
    df_static = generate_demographics(
        unique_ids=unique_ids,
        demo_schema=config.get('demographics_schema', {}),
        conditions=config.get('conditions', []),
        icd_map=icd_map,
        id_col_name=id_col_name
    )

    # Generate timeseries data
    df_ts = generate_ts(
        unique_ids=unique_ids,
        df_static=df_static,
        ts_config=config.get('ts_config', {}),
        days=days,
        freq=freq,
        output_format=output_format
    )

    return df_static, df_ts


def generate_demographics(unique_ids, demo_schema, conditions,
                          icd_map, id_col_name='patient_id'):
    """Generates static demographic data strictly based on the schema config."""
    n = len(unique_ids)
    df = pd.DataFrame({id_col_name: unique_ids})

    # 1. Generate based on schema definition
    for col, props in demo_schema.items():
        if props['type'] == 'int':
            df[col] = np.random.randint(props['range'][0], props['range'][1] + 1, size=n)
        elif props['type'] == 'float':
            # Rounding to 1 decimal place for realistic weights/measurements
            df[col] = np.round(np.random.uniform(props['range'][0], props['range'][1], size=n), 1)
        elif props['type'] == 'enumerated':
            df[col] = np.random.choice(props['values'], size=n)

    # 2. Generate Binary Conditions (Comorbidities)
    if isinstance(conditions, dict):
        for cond, prob in conditions.items():
            df[cond] = np.random.choice([0, 1], size=n, p=[1 - prob, prob])

    # 3. Logic-Driven ICD-10 Code Generation
    # We create a string of codes for each patient based on their assigned conditions
    def map_to_icd(row, condition_name):
        if row[condition_name] == 1 and condition_name in icd_map:
            return np.random.choice(icd_map[condition_name])
        return "N/A"

    # Apply mapping only for conditions that exist in the ICD map
    for condition in icd_map.keys():
        if condition in df.columns:
            df[f'{condition}_icd'] = df.apply(lambda r: map_to_icd(r, condition), axis=1)

    # 3. Risk score & label (You can externalize these weights to config later if desired)
    risk_score = (
            (df.get('age', 50) * 0.5) +
            (df.get('diabetes', 0) * 20) +
            (df.get('CKD', 0) * 15)
    )
    df['risk'] = risk_score
    df['sepsis_case'] = (risk_score > np.percentile(risk_score, 70)).astype(int)

    return df


def generate_ts(unique_ids, df_static, ts_config,
                days=6, freq='8h', output_format='long',
                id_col_name='patient_id'):
    """Generates clinical data with LEARNABLE patterns, bounded by schema ranges."""
    # 1. Setup Time Grid
    periods = days * (24 // int(freq[:-1]))
    timestamps = pd.date_range('2024-01-01', periods=periods, freq=freq)
    idx = pd.MultiIndex.from_product(
        [unique_ids, timestamps], names=[id_col_name, 'date']
    )
    df = pd.DataFrame(index=idx).reset_index()

    # 2. Determine Sepsis Onset
    septic_ids = df_static[df_static['sepsis_case'] == 1][id_col_name].values
    onset_map = {pid: np.random.randint(5, periods - 5) for pid in septic_ids}

    df['step'] = df.groupby(id_col_name).cumcount()
    df['onset_step'] = df[id_col_name].map(onset_map)
    df['is_sick'] = (df.step > df.onset_step).astype(int)

    # 3. Generate Learnable Signals
    t = df['step'].values
    steps_per_day = 24 // int(freq[:-1])

    for k, props in ts_config.items():
        # A. BASELINE: deterministic hash-like
        patient_base = df[id_col_name].map(lambda x: (x % 50) + 50)

        # B. PATTERN: Sine Wave (Circadian Rhythm)
        amp = 10 if k in ['hr', 'temp'] else 2
        seasonality = amp * np.sin(2 * np.pi * t / steps_per_day)

        # C. RANDOM WALK (Physiology is smooth)
        noise = np.random.normal(0, 1, size=len(df))
        smooth_noise = np.convolve(noise, np.ones(3) / 3, mode='same') * 5

        # D. SEPSIS DRIFT
        drift_intensity = 2.0 if k in ['hr', 'temp', 'lactate'] else 0.5
        drift = np.where(df['is_sick'], (df['step'] - df['onset_step']) * drift_intensity, 0)

        # Combine
        df[k] = patient_base + seasonality + smooth_noise + drift

        # E. SCHEMA BOUNDARIES (Replaces the hardcoded clip)
        if 'range' in props:
            df[k] = df[k].clip(lower=props['range'][0], upper=props['range'][1])

        # Optional: round floats to a sensible precision
        if props.get('type') == 'float':
            df[k] = np.round(df[k], 2)

    if output_format == 'tidy':
        return df.set_index([id_col_name, 'step'])

    # Melt logic to long format
    df = df.melt(id_vars=[id_col_name, 'date'], var_name='test', value_name='result')
    unit_map = {k: v.get('unit', '') for k, v in ts_config.items()}
    df['unit'] = df['test'].map(unit_map)

    return df.sort_values([id_col_name, 'date']).reset_index(drop=True)


def apply_missingness(df, ts_config, default_rate=0.5):
    """Randomly removes rows to simulate missing data based on config probabilities."""
    df_out = df.copy()
    target_cols = [c for c in df.columns if c in ts_config]

    # Extract prob from config, fallback to default_rate if missing
    probs = np.array([ts_config[c].get('prob', default_rate) for c in target_cols])

    random_matrix = np.random.rand(len(df), len(target_cols))
    mask_matrix = random_matrix > probs
    df_out[target_cols] = df_out[target_cols].mask(mask_matrix)
    return df_out


if __name__ == '__main__':
    # --------------------------
    # 1. Load Configuration
    # --------------------------
    # Assuming you run this from inside the src/ folder
    config_path = Path('../config/data_config.yaml')

    # Fallback/mock config if file doesn't exist yet for testing
    if not config_path.exists():
        print(f"Warning: {config_path} not found. Please ensure your YAML file is created.")
        exit(1)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    params = config.get('generation_params', {})
    n_patients = params.get('n_patients', 100)
    days = params.get('days', 10)
    freq = params.get('freq', '4h')
    output_format = params.get('output_format', 'tidy')

    # --------------------------
    # 2. Generate data
    # --------------------------
    print(f"Generating data for {n_patients} patients...")
    df_static, df_ts = generate_clinical_data(
        config=config,
        n_patients=n_patients,
        days=days,
        freq=freq,
        output_format=output_format
    )

    print('\n--- Sample: Static Demographics ---')
    print(df_static.head())
    print('\n--- Sample: Timeseries Data ---')
    print(df_ts.head())

    # --------------------------
    # 3. Add missingness
    # --------------------------
    print("\nApplying missingness masks...")
    df_ms = apply_missingness(df_ts, ts_config=config.get('ts_config', {}))

    # --------------------------
    # 4. Save data in structured FS
    # --------------------------
    date_str = datetime.now().strftime('%Y-%m-%d')
    output_dir = Path(f'../data/synthetic/{date_str}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving files to <{output_dir}>...")
    df_static.to_csv(output_dir / 'df_static.csv', index=False)
    df_ts.to_csv(output_dir / 'df_ts.csv', index=True)
    df_ms.to_csv(output_dir / 'df_ts_missing.csv', index=True)
    print("Done!")