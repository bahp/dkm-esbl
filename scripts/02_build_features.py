# scripts/02_build_features.py

import sys
import yaml
import pandas as pd
from pathlib import Path

# Setup path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.features import FeaturePipeline
from src.utils import get_latest_data_dir



def main():
    print("==================================================")
    print("⚙️  [Step 2] Feature Engineering & Scoring")
    print("==================================================\n")

    # 1. Locate Data
    latest_data_dir = get_latest_data_dir()
    run_id = latest_data_dir.name
    print(f"Loading raw data from run: {run_id}...")

    df_static = pd.read_csv(latest_data_dir / 'df_static.csv')
    df_ts_ms = pd.read_csv(latest_data_dir / 'df_ts_missing.csv', parse_dates=['date'])

    # 2. Setup Output Directory
    processed_dir = project_root / 'data' / 'processed' / run_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 3. Run Pipeline
    print("Executing feature pipeline (Imputation, Rolling Windows, Phenotypes, Scores)...")
    config_path = project_root / 'config' / 'feature_config.yaml'
    pipeline = FeaturePipeline(config_path=config_path)

    final_features = pipeline.process(df_static, df_ts_ms)

    # 4. Save Outputs
    output_file = processed_dir / 'features_engineered.csv'
    final_features.to_csv(output_file, index=False)

    print(f"\n✅ Success! Engineered features saved to: {output_file.relative_to(project_root)}")


if __name__ == "__main__":
    main()