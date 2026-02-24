# scripts/03_evaluate_scores.py

import sys
import yaml
import pandas as pd
from pathlib import Path

# Setup path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.metrics import ClinicalEvaluator
from src.utils import get_latest_processed_file


def main():
    print("==================================================")
    print("📊 [Step 3] Model Evaluation & Metrics")
    print("==================================================\n")

    # 1. Load Data & Config
    processed_file = get_latest_processed_file()
    print(f"Loading engineered features from: {processed_file.parent.name}...")
    final_features = pd.read_csv(processed_file)

    feature_config_path = project_root / 'config' / 'feature_config.yaml'
    with open(feature_config_path, 'r') as file:
        feature_config = yaml.safe_load(file)

    # 2. Identify Target and Scores
    target_label = 'sepsis_case'  # The true label we are trying to predict

    # Dynamically grab the names of the scores we created in the YAML
    computed_scores = list(feature_config.get('computed_features', {}).keys())
    custom_scores = list(feature_config.get('custom_scores', {}).keys())
    scores_to_test = computed_scores + custom_scores

    valid_scores = [s for s in scores_to_test if s in final_features.columns]

    if not valid_scores:
        print("⚠ No valid scores found in the dataset to evaluate.")
        return

    # 3. Evaluate
    print(f"\nEvaluating performance for: {', '.join(valid_scores)}")
    evaluator = ClinicalEvaluator(output_dir=project_root / 'outputs')

    results_df = evaluator.evaluate_scores(final_features, target_label, valid_scores)
    evaluator.plot_roc_curves(final_features, target_label, valid_scores)
    evaluator.plot_pr_curves(final_features, target_label, valid_scores)

    # 4. Display Results
    print("\n--- Summary of Predictive Performance ---")
    print(results_df[['Score_Name', 'AUROC', 'AUPRC', 'Optimal_Cutoff']].to_string(index=False))

    print(f"\n✅ Success! Full metrics and plots saved to: {evaluator.output_dir.relative_to(project_root)}")


if __name__ == "__main__":
    main()