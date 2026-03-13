# src/features.py
import yaml
import importlib
import pandas as pd
import numpy as np
from pathlib import Path


class FeaturePipeline:
    def __init__(self, config_path='../config/feature_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def process(self, df_static, df_ts):
        """Main orchestration method."""
        df = df_ts.sort_values(['patient_id', 'date']) \
            .set_index(['patient_id', 'date'])
        print("Processing base features...")
        df = self._process_base_features(df)
        df = df.reset_index()
        df = pd.merge(df, df_static, on='patient_id', how='left')
        print("Computing pandas eval expressions...")
        df = self._compute_expressions(df)
        print("Executing custom Python scores...")
        df = self._compute_custom_scores(df)
        df = df.reset_index()
        return df

    def _process_base_features(self, df):
        """Handles missing indicators, imputation, deltas, and rolling statistics."""
        new_columns = []

        for col, rules in self.config.get('base_features', {}).items():
            if col not in df.columns:
                continue

            # A. Missing Indicators
            if rules.get('missing_indicator'):
                new_columns.append(df[col].isna().astype(int).rename(f"{col}_is_missing"))

            # B. Imputation (grouped by patient to prevent cross-contamination)
            grp = df.groupby(level='patient_id')[col]

            if rules.get('impute') == 'ffill':
                df[col] = grp.ffill()
            elif rules.get('impute') == 'constant':
                df[col] = df[col].fillna(rules.get('fill_value', 0))

            # Fallback for leading NaNs
            df[col] = df[col].fillna(df[col].median())

            # C. Deltas
            if rules.get('delta'):
                new_columns.append(grp.diff().fillna(0).rename(f"{col}_delta"))

            # D. Rolling Time-Windows
            if 'rolling' in rules:
                # For time-aware rolling, date must be the only index
                temp_df = df[[col]].reset_index(level='patient_id')

                for window in rules['rolling'].get('windows', []):
                    rolled = temp_df.groupby('patient_id')[col].rolling(window).agg(rules['rolling']['aggs'])
                    rolled.columns = [f"{col}_{window}_{agg}" for agg in rolled.columns]
                    new_columns.append(rolled)

        # Concatenate all newly generated features at once (Highly optimized memory usage)
        if new_columns:
            df = pd.concat([df] + new_columns, axis=1)

        return df

    def _compute_expressions(self, df):
        """Dynamically evaluates string expressions to create new features/scores.

        . note: The numexpr is fast, but does not handle True + True. So we can
                tell Pandas to use the standard Python engine so it can safely
                add boolean flags (True + True = 2) without numexpr complaining!
                df[feature_name] = df.eval(expr, engine='python')
        """
        expressions = self.config.get('computed_features', {})

        for feature_name, expr in expressions.items():
            try:
                df[feature_name] = df.eval(expr)
            except Exception as e:
                print(f"Warning: Failed to compute '{feature_name}'. Error: {e}")

        return df

    def _compute_custom_scores(self, df):
        """Dynamically imports and executes external Python scoring functions."""
        import importlib  # Ensure this is at the top of your file or method

        custom_scores = self.config.get('custom_scores', {})

        for score_name, meta in custom_scores.items():
            module_name = meta.get('module', 'scores')
            func_name = meta.get('function')
            kwargs = meta.get('kwargs', {})

            try:
                # Dynamically load the module and function
                module = importlib.import_module(module_name)
                custom_func = getattr(module, func_name)

                # Execute the function, passing the dataframe and the kwargs from YAML
                df[score_name] = custom_func(df, **kwargs)

            except ModuleNotFoundError:
                print(f"❌ Error computing '{score_name}': Cannot find module '{module_name}'.")
                # Add the custom hint for the most common mistake
                if not module_name.startswith('src.'):
                    print(
                        f"   💡 Hint: If your file is in the 'src' folder, update your YAML to use `module: 'src.{module_name}'`")

            except AttributeError:
                print(
                    f"❌ Error computing '{score_name}': Function '{func_name}' does not exist inside '{module_name}'.")

            except Exception as e:
                print(f"⚠️ Warning: Failed to compute custom score '{score_name}'. Error: {e}")

        return df

if __name__ == '__main__':

    # 1. Setup paths
    latest_dir = sorted([d for d in Path('../data/synthetic').iterdir() if d.is_dir()])[-1]

    # 2. Load data
    df_static = pd.read_csv(latest_dir / 'df_static.csv')
    df_ts_ms = pd.read_csv(latest_dir / 'df_ts_missing.csv', parse_dates=['date'])

    # 3. Run Pipeline
    pipeline = FeaturePipeline()
    final_features = pipeline.process(df_static, df_ts_ms)

    # 4. Show results
    print("\n--- Pipeline Complete! ---")
    print(f"Final Shape: {final_features.shape}")

    preview_cols = ['patient_id', 'date', 'HR', 'SBP', 'Shock_Index', 'qSOFA_score']
    available = [c for c in preview_cols if c in final_features.columns]
    print("\nSample of Computed Scores & Ratios:")
    print(final_features[available].dropna().head())