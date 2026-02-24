# src/metrics.py

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ClinicalEvaluator:
    def __init__(self, output_dir='../outputs'):
        """Initializes the evaluator and sets up the output directories."""
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / 'metrics'
        self.plots_dir = self.output_dir / 'plots'

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Set visualization style
        sns.set_theme(style="whitegrid")

    def evaluate_at_recommended_threshold(self, df, target_col, score_name, recommended_cutoff):
        """
        Calculates diagnostic metrics at a pre-defined clinical threshold.
        Useful for validating 'Rule-In' or 'Rule-Out' safety.
        """
        # Ensure no NaNs in the evaluation
        valid_data = df[[target_col, score_name]].dropna()
        y_true = valid_data[target_col]
        y_score = valid_data[score_name]

        # Apply the recommended cutoff
        y_pred_binary = (y_score >= recommended_cutoff).astype(int)

        # Generate Confusion Matrix
        #
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

        # Calculate Clinical Metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Rule-In)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value (Rule-Out)

        # Likelihood Ratios
        lr_plus = sensitivity / (1 - specificity) if (1 - specificity) > 0 else np.nan
        lr_minus = (1 - sensitivity) / specificity if specificity > 0 else np.nan

        return {
            'Score_Name': score_name,
            'Recommended_Cutoff': recommended_cutoff,
            'Sensitivity': round(sensitivity, 3),
            'Specificity': round(specificity, 3),
            'PPV': round(ppv, 3),
            'NPV': round(npv, 3),
            'LR+': round(lr_plus, 2),
            'LR-': round(lr_minus, 2)
        }

    def evaluate_scores(self, df, true_label_col, score_cols):
        """
        Evaluates multiple clinical scores against a true binary label.
        Returns a DataFrame comparing their performance.
        """
        results = []

        for score in score_cols:
            # Drop rows where the score or label is missing to ensure fair comparison
            valid_data = df[[true_label_col, score]].dropna()
            if valid_data.empty:
                print(f"Skipping {score}: No valid data.")
                continue

            y_true = valid_data[true_label_col]
            y_pred_score = valid_data[score]

            # 1. Area Under the Curves
            auroc = roc_auc_score(y_true, y_pred_score)
            auprc = average_precision_score(y_true, y_pred_score)
            #brier = brier_score_loss(y_true, y_pred_score)  # Lower is better (calibration)

            # 2. Find the optimal threshold using Youden's J statistic
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_score)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]

            # 3. Calculate discrete metrics at the optimal threshold
            y_pred_binary = (y_pred_score >= optimal_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            results.append({
                'Score_Name': score,
                'AUROC': round(auroc, 3),
                'AUPRC': round(auprc, 3),
                #'Brier_Score': round(brier, 3),
                'Optimal_Cutoff': round(optimal_threshold, 2),
                'Sensitivity': round(sensitivity, 3),
                'Specificity': round(specificity, 3),
                'PPV': round(ppv, 3),
                'NPV': round(npv, 3)
            })

        results_df = pd.DataFrame(results).sort_values(by='AUROC', ascending=False)

        # Save results to CSV
        results_df.to_csv(self.metrics_dir / 'score_comparison.csv', index=False)
        return results_df

    def plot_roc_curves(self, df, true_label_col, score_cols, filename='roc_curves.png'):
        """Generates and saves a combined ROC curve for all evaluated scores."""
        plt.figure(figsize=(8, 6))

        for score in score_cols:
            valid_data = df[[true_label_col, score]].dropna()
            if valid_data.empty: continue

            fpr, tpr, _ = roc_curve(valid_data[true_label_col], valid_data[score])
            auc = roc_auc_score(valid_data[true_label_col], valid_data[score])

            plt.plot(fpr, tpr, lw=2, label=f'{score} (AUC = {auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(self.plots_dir / filename, dpi=300)
        plt.close()

    def plot_pr_curves(self, df, true_label_col, score_cols, filename='pr_curves.png'):
        """Generates and saves a combined Precision-Recall curve."""
        plt.figure(figsize=(8, 6))

        for score in score_cols:
            valid_data = df[[true_label_col, score]].dropna()
            if valid_data.empty: continue

            precision, recall, _ = precision_recall_curve(valid_data[true_label_col], valid_data[score])
            auprc = average_precision_score(valid_data[true_label_col], valid_data[score])

            plt.plot(recall, precision, lw=2, label=f'{score} (AUPRC = {auprc:.2f})')

        # Baseline is the prevalence of the positive class
        baseline = df[true_label_col].mean()
        plt.axhline(y=baseline, color='navy', lw=2, linestyle='--', label=f'Baseline ({baseline:.2f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(self.plots_dir / filename, dpi=300)
        plt.close()


if __name__ == '__main__':
    # ---------------------------------------------------------
    # Example Usage / Testing
    # ---------------------------------------------------------

    # 1. Create dummy data to simulate the output of features.py
    np.random.seed(42)
    n = 1000
    dummy_df = pd.DataFrame({
        'sepsis_case': np.random.choice([0, 1], size=n, p=[0.85, 0.15]),  # 15% prevalence
        'MEWS_score': np.random.normal(loc=2, scale=1.5, size=n),
        'qSOFA_score': np.random.normal(loc=1, scale=1.0, size=n),
        'Shock_Index': np.random.normal(loc=0.7, scale=0.3, size=n)
    })

    # Artificially make scores slightly predictive for the test
    dummy_df.loc[dummy_df['sepsis_case'] == 1, 'MEWS_score'] += 2.5
    dummy_df.loc[dummy_df['sepsis_case'] == 1, 'qSOFA_score'] += 1.0
    dummy_df.loc[dummy_df['sepsis_case'] == 1, 'Shock_Index'] += 0.4

    # 2. Run the evaluator
    print("Evaluating clinical scores...")
    evaluator = ClinicalEvaluator()

    scores_to_test = ['MEWS_score', 'qSOFA_score', 'Shock_Index']
    target_label = 'sepsis_case'

    # Compute tabular metrics
    results_table = evaluator.evaluate_scores(dummy_df, target_label, scores_to_test)
    print("\n--- Evaluation Results ---")
    print(results_table.to_string(index=False))

    # Generate plots
    print(f"\nGenerating plots in {evaluator.plots_dir}...")
    evaluator.plot_roc_curves(dummy_df, target_label, scores_to_test)
    evaluator.plot_pr_curves(dummy_df, target_label, scores_to_test)
    print("Done!")