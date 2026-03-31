# scripts/01_generate_data.py

import sys
import yaml
from pathlib import Path
from datetime import datetime

# Setup path so Python can find 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.generators import generate_clinical_data, apply_missingness


def main():

    # ---------------------------------------------------------
    # 1. Setup Paths and Load Configs
    # ---------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent
    config_dir = project_root / 'config'
    output_dir = project_root / 'data' / 'synthetic'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the three required config files
    with open(config_dir / 'data_config.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    with open(config_dir / 'icd_config.yaml', 'r') as f:
        icd_config = yaml.safe_load(f)


    print("==================================================")
    print("🧬 [Step 1] Synthetic Data Generation")
    print("==================================================\n")

    # 1. Load Config
    #config_path = project_root / 'config' / 'data_config.yaml'
    #with open(config_path, 'r') as file:
    #    data_config = yaml.safe_load(file)

    # 2. Setup Output Directory
    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    data_dir = project_root / 'data' / 'synthetic' / date_str
    data_dir.mkdir(parents=True, exist_ok=True)

    # 3. Generate Data
    params = data_config.get('generation_params', {})
    print(f"Generating cohort of {params.get('n_patients', 100)} patients...")

    df_static, df_ts = generate_clinical_data(
        config=data_config,
        n_patients=params.get('n_patients', 100),
        days=params.get('days', 10),
        freq=params.get('freq', '4h'),
        output_format=params.get('output_format', 'tidy'),
        icd_map=icd_config.get('icd_codes', {}),
        id_col_name=data_config.get('primary_key', 'patient_id')
    )

    # 3b. Generate Custom Relational Tables
    print("Generating custom relational tables...")
    custom_tables = {}
    unique_ids = df_static[data_config.get('primary_key', 'patient_id')].values

    if 'tables' in data_config:
        from src.generators import generate_custom_table  # ensure this is imported

        for table_name, table_config in data_config['tables'].items():
            print(f" -> Building {table_name}...")
            df_custom = generate_custom_table(unique_ids, table_config)
            custom_tables[table_name] = df_custom

            # Save it alongside the others
            df_custom.to_csv(data_dir / f'{table_name.lower()}.csv', index=False)

    # 4. Apply Missingness
    print("Applying missing data masks...")
    df_ts_ms = apply_missingness(df_ts, ts_config=data_config.get('ts_config', {}))

    # 5. Save Outputs
    df_static.to_csv(data_dir / 'df_static.csv', index=False)
    df_ts_ms.to_csv(data_dir / 'df_ts_missing.csv', index=True)

    print(f"\n✅ Success! Raw data saved to: {data_dir.relative_to(project_root)}")


if __name__ == "__main__":
    main()