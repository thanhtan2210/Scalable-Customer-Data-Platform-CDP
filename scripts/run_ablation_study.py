import os
import io
import json
import tempfile
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Add root folder to sys.path so we can import from backend
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.profiler.orchestrator import run_profiling
from backend.app.core.profiler.column_profile import ColumnProfile, DataRole
from backend.app.core.profiler.target_analysis import CompositeTargetConfig, SynthesisStrategy
from backend.app.core.profiler.target_synthesizer import _pca_synthesis
from backend.app.core.training.automl import run_automl
from backend.app.core.training.continual_trainer import ReplayBuffer

def main():
    # Set local MLflow tracking URI to avoid polluting remote registries
    mlflow.set_tracking_uri("file:./mlruns_test")
    
    print("="*60)
    print("STARTING ABLATION STUDY FOR ACADEMIC EVALUATION")
    print("="*60)

    # 1. Load cleaned_telco dataset
    csv_path = "data/raw/cleaned_telco.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns.")

    # Target column is "Churn Value", composite auxiliary churn columns are Churn Value and Churn Score
    target_col = "Churn Value"
    
    # Pre-synthesize CPI score on full dataset for consistency
    variance, cpi_series = _pca_synthesis(df, ["Churn Value", "Churn Score"], target_col)
    df["cpi_score"] = cpi_series
    print(f"Pre-synthesized cpi_score via PCA. Variance explained: {variance:.4f}")

    # 2. Split dataset into A (Old Task) and B (New Task)
    # Shuffle full dataset first to ensure class balance in both splits A and B
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_A = df.iloc[:3500].copy()
    df_B = df.iloc[3500:].copy()
    
    print(f"Split data: Task A = {len(df_A)} rows, Task B = {len(df_B)} rows.")

    # 3. Profile Task A to get feature configurations
    profiles_A, suggested_target_A = run_profiling(df_A)
    
    # Overwrite/Force the primary target to be "Churn Value" in the profiles
    for p in profiles_A:
        if p.name == target_col:
            p.inferred_role = DataRole.TARGET
            p.confidence_score = 1.0
        elif p.name == "cpi_score":
            p.inferred_role = DataRole.NUMERIC

    # Force composite configuration to use Churn Value & Churn Score
    composite_config_A = CompositeTargetConfig(
        strategy=SynthesisStrategy.PCA,
        source_columns=["Churn Value", "Churn Score"],
        cpi_variance_explained=variance,
        weights=None,
        requires_confirmation=False
    )
    print(f"Forced composite target configuration: {composite_config_A}")

    # Prepare features list
    feature_cols = [p.name for p in profiles_A if p.name != target_col and p.inferred_role not in ["ID", "IGNORE", "TARGET"]]
    if "cpi_score" not in feature_cols:
        feature_cols.append("cpi_score")
            
    # Split Task A/B into Train/Test subsets for evaluation
    df_A_train, df_A_test = train_test_split(df_A, test_size=0.2, stratify=df_A[target_col], random_state=42)
    df_B_train, df_B_test = train_test_split(df_B, test_size=0.2, stratify=df_B[target_col], random_state=42)

    # Clean previous local replay buffers to avoid interference
    dataset_id = "ablation_study_dataset"
    replay_path = f"ml_artifacts/{dataset_id}/replay_buffer.parquet"
    if os.path.exists(f"data/{replay_path}"):
        try:
            os.remove(f"data/{replay_path}")
        except Exception:
            pass

    # 4. Train Model A (Initial task model on A)
    print("\n--- Training Initial Model on Task A ---")
    model_uri_A, _ = run_automl(
        df=df_A_train,
        confirmed_profiles=profiles_A,
        target_col=target_col,
        dataset_id=dataset_id,
        composite_config=composite_config_A
    )
    print(f"Initial Model A logged at: {model_uri_A}")

    # Measure AUC of Model A on Test set A
    pipeline_A = mlflow.sklearn.load_model(model_uri_A)
    y_A_test_pred = pipeline_A.predict_proba(df_A_test[feature_cols])[:, 1]
    auc_A_before = float(roc_auc_score(df_A_test[target_col], y_A_test_pred))
    print(f"AUC of Model A on Test A (Initial): {auc_A_before:.4f}")

    # Setup the Replay Buffer with initial Task A train data to represent history
    buffer = ReplayBuffer()
    buffer.update(dataset_id, df_A_train, target_col)
    print("Replay buffer populated with Task A training data.")

    # 5. Run Ablation study for 4 configurations on Task B
    scenarios = {
        "Baseline": {"lambda": 0.0, "ratio": 0.0, "use_replay": False},
        "Replay only": {"lambda": 0.0, "ratio": 0.2, "use_replay": True},
        "EWC only": {"lambda": 100.0, "ratio": 0.0, "use_replay": False},
        "Full (EWC+Replay)": {"lambda": 100.0, "ratio": 0.2, "use_replay": True}
    }

    results = []

    for name, config in scenarios.items():
        print(f"\n--- Running Scenario: {name} ---")
        
        # Configure EWC & Replay params using env vars
        os.environ["EWC_LAMBDA"] = str(config["lambda"])
        os.environ["REPLAY_BUFFER_RATIO"] = str(config["ratio"])
        
        # Adjust replay buffer state before training to mimic setup
        if not config["use_replay"]:
            original_load = buffer.load_from_r2
            buffer.load_from_r2 = lambda *args, **kwargs: None
            from backend.app.core.training.continual_trainer import ReplayBuffer as RBModule
            original_rb_load = RBModule.load_from_r2
            RBModule.load_from_r2 = lambda *args, **kwargs: None
        else:
            from backend.app.core.training.continual_trainer import ReplayBuffer as RBModule
            RBModule.load_from_r2 = original_load

        try:
            # Train model continuously on Task B using prior Model A checkpoint
            model_uri_B, _ = run_automl(
                df=df_B_train,
                confirmed_profiles=profiles_A,
                target_col=target_col,
                dataset_id=dataset_id,
                composite_config=composite_config_A,
                prior_model_uri=model_uri_A
            )
            
            # Load fine-tuned pipeline
            pipeline_B = mlflow.sklearn.load_model(model_uri_B)
            
            # Evaluate on Task A test set
            y_A_after_pred = pipeline_B.predict_proba(df_A_test[feature_cols])[:, 1]
            auc_A_after = float(roc_auc_score(df_A_test[target_col], y_A_after_pred))
            
            # Evaluate on Task B test set
            y_B_pred = pipeline_B.predict_proba(df_B_test[feature_cols])[:, 1]
            auc_B = float(roc_auc_score(df_B_test[target_col], y_B_pred))
            
            # Calculate forgetting rate
            forgetting_rate = (auc_A_before - auc_A_after) / auc_A_before
            
            results.append({
                "Scenario": name,
                "EWC": "Yes" if config["lambda"] > 0 else "No",
                "Replay": "Yes" if config["use_replay"] else "No",
                "AUC_A_before": auc_A_before,
                "AUC_A_after": auc_A_after,
                "AUC_B": auc_B,
                "Forgetting_Rate": forgetting_rate
            })
            
            print(f"Results for {name}:")
            print(f"  AUC A before: {auc_A_before:.4f}")
            print(f"  AUC A after : {auc_A_after:.4f}")
            print(f"  AUC B       : {auc_B:.4f}")
            print(f"  Forgetting  : {forgetting_rate*100:.2f}%")
            
        except Exception as e:
            print(f"Error running {name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if not config["use_replay"]:
                from backend.app.core.training.continual_trainer import ReplayBuffer as RBModule
                RBModule.load_from_r2 = original_rb_load

    # 6. Format and display results as Markdown Table for Academic Evaluation
    print("\n" + "="*80)
    print("EVALUATION SECTION: ABLATION STUDY COMPARISON TABLE")
    print("="*80)
    print("\n| Scenario | Use EWC | Use Replay | $AUC_{A\\_before}$ | $AUC_{A\\_after}$ | $AUC_{B}$ | Forgetting Rate (%) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        f_rate_str = f"{r['Forgetting_Rate']*100:.2f}%"
        if r["Scenario"] == "Full (EWC+Replay)":
            print(f"| **{r['Scenario']}** | {r['EWC']} | {r['Replay']} | {r['AUC_A_before']:.4f} | **{r['AUC_A_after']:.4f}** | {r['AUC_B']:.4f} | **{f_rate_str}** |")
        else:
            print(f"| {r['Scenario']} | {r['EWC']} | {r['Replay']} | {r['AUC_A_before']:.4f} | {r['AUC_A_after']:.4f} | {r['AUC_B']:.4f} | {f_rate_str} |")
            
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
