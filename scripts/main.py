# scripts/main.py

import sys
import yaml
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------
# Path Setup
# ---------------------------------------------------------
# Ensure Python can find the 'src' module when running from the 'scripts' folder
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.generators import generate_clinical_data, apply_missingness
from src.features import FeaturePipeline
from src.metrics import ClinicalEvaluator


def load_config(config_name):
    """Utility to load YAML configuration files."""
    config_path = project_root / 'config' / config_name
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    print("==================================================")
    print("🏥 Clinical Scoring Pipeline Execution Started")
    print("==================================================\n")

    # ---------------------------------------------------------
    # 1. Configuration & Directories
    # ---------------------------------------------------------
    print("[1/4] Loading configurations...")
    data_config = load_config('data_config.yaml')
    feature_config = load_config('feature_config.yaml')

    date_str = datetime.now().strftime('%Y-%m-%d_%H%M')
    data_dir = project_root / 'data' / 'synthetic' / date_str
    processed_dir = project_root / 'data' / 'processed' / date_str

    data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Data Generation
    # ---------------------------------------------------------
    print("\n[2/4] Generating synthetic patient cohort...")
    params = data_config.get('generation_params', {})

    df_static, df_ts = generate_clinical_data(
        config=data_config,
        n_patients=params.get('n_patients', 100),
        days=params.get('days', 10),
        freq=params.get('freq', '4h'),
        output_format=params.get('output_format', 'tidy')
    )

    df_ts_ms = apply_missingness(df_ts, ts_config=data_config.get('ts_config', {}))

    # Save raw outputs
    df_static.to_csv(data_dir / 'df_static.csv', index=False)
    df_ts_ms.to_csv(data_dir / 'df_ts_missing.csv', index=True)
    print(f"      ✓ Raw data saved to: {data_dir.relative_to(project_root)}")

    # ---------------------------------------------------------
    # 3. Feature Engineering & Scoring
    # ---------------------------------------------------------
    print("\n[3/4] Running Feature Engineering Pipeline...")
    # Initialize the pipeline class we built
    pipeline = FeaturePipeline(config_path=project_root / 'config' / 'feature_config.yaml')

    # Process the data
    final_features = pipeline.process(df_static, df_ts_ms)

    # Save processed outputs
    processed_path = processed_dir / 'features_engineered.csv'
    final_features.to_csv(processed_path, index=False)
    print(f"      ✓ Processed features saved to: {processed_path.relative_to(project_root)}")

    # ---------------------------------------------------------
    # 4. Evaluation & Metrics
    # ---------------------------------------------------------
    print("\n[4/4] Evaluating Clinical Scores...")
    # Initialize the evaluator, routing outputs to the outputs/ directory
    evaluator = ClinicalEvaluator(output_dir=project_root / 'outputs')

    # Define what we are trying to predict and what scores we want to test
    target_label = 'sepsis_case'

    # We pull the list of scores dynamically from what we defined in feature_config.yaml
    computed_scores = list(feature_config.get('computed_features', {}).keys())
    custom_scores = list(feature_config.get('custom_scores', {}).keys())
    scores_to_test = computed_scores + custom_scores

    # Ensure the scores actually exist in the dataframe before evaluating
    valid_scores = [s for s in scores_to_test if s in final_features.columns]

    if not valid_scores:
        print("      ⚠ No valid scores found to evaluate. Check your feature_config.yaml.")
    else:
        # Run evaluations
        results_df = evaluator.evaluate_scores(final_features, target_label, valid_scores)
        evaluator.plot_roc_curves(final_features, target_label, valid_scores)
        evaluator.plot_pr_curves(final_features, target_label, valid_scores)

        print("\n--- Summary of Model Performance ---")
        print(results_df[['Score_Name', 'AUROC', 'AUPRC', 'Optimal_Cutoff']].to_string(index=False))
        print(f"\n      ✓ Full metrics and plots saved to: {evaluator.output_dir.relative_to(project_root)}")

    print("\n==================================================")
    print("✅ Pipeline Execution Complete!")
    print("==================================================")


if __name__ == "__main__":
    main()