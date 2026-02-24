# scripts/main.py

# ... (previous imports and setup) ...
import yaml


def main():
    # ... (Step 1-3: Configuration, Generation, and Feature Building) ...

    # ---------------------------------------------------------
    # 4. Evaluation & Metrics
    # ---------------------------------------------------------
    print("\n[4/4] Evaluating Clinical Scores...")
    evaluator = ClinicalEvaluator(output_dir=project_root / 'outputs')
    target_label = 'sepsis_case'

    # Identify which scores were computed
    computed_scores = list(feature_config.get('computed_features', {}).keys())
    custom_scores = list(feature_config.get('custom_scores', {}).keys())
    scores_to_test = [s for s in (computed_scores + custom_scores) if s in final_features.columns]

    if not scores_to_test:
        print("      ⚠ No valid scores found to evaluate.")
        return

    # A. Standard Evaluation (Optimal Thresholds via Youden's J)
    # This uses your existing evaluate_scores function
    results_df = evaluator.evaluate_scores(final_features, target_label, scores_to_test)
    evaluator.plot_roc_curves(final_features, target_label, scores_to_test)
    evaluator.plot_pr_curves(final_features, target_label, scores_to_test)

    # B. Stewardship Safety Evaluation (Recommended Thresholds)
    # Load the thresholds we defined in the new config file
    threshold_path = project_root / 'config' / 'threshold_config.yaml'
    if threshold_path.exists():
        with open(threshold_path, 'r') as file:
            threshold_config = yaml.safe_load(file)

        recommended_thresholds = threshold_config.get('thresholds', {})
        stewardship_results = []

        print("      Validating against literature-recommended thresholds...")
        for score_name, cutoff in recommended_thresholds.items():
            if score_name in final_features.columns:
                # This uses your existing evaluate_at_recommended_threshold function
                metrics = evaluator.evaluate_at_recommended_threshold(
                    final_features, target_label, score_name, cutoff
                )
                stewardship_results.append(metrics)

        if stewardship_results:
            steward_df = pd.DataFrame(stewardship_results)
            steward_df.to_csv(evaluator.metrics_dir / 'stewardship_validation.csv', index=False)
            print("\n--- Stewardship Safety (NPV Focus) ---")
            print(steward_df[['Score_Name', 'Recommended_Cutoff', 'Sensitivity', 'NPV']].to_string(index=False))

    print(f"\n✅ Pipeline Complete! Results saved to: {evaluator.output_dir.relative_to(project_root)}")


if __name__ == "__main__":
    main()