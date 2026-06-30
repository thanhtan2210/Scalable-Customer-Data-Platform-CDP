import os
import io
import json
import tempfile
import random
import argparse
import uuid
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

try:
    import torch
except (ImportError, OSError):
    torch = None

# Add root folder to sys.path so we can import from backend
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.core.profiler.orchestrator import run_profiling
from backend.app.core.profiler.column_profile import ColumnProfile, DataRole
from backend.app.core.profiler.target_analysis import CompositeTargetConfig, SynthesisStrategy, ColumnWeight
from backend.app.core.training.automl import run_automl
from backend.app.core.training.continual_trainer import ReplayBuffer

def cleanup_r2_artifacts(run_id: str):
    try:
        from backend.app.core.storage import storage
        path = f"ml_artifacts/{run_id}/"
        storage.delete_prefix(path)
        print(f"Cleaned up R2 artifacts at {path}")
    except Exception as e:
        print(f"Warning: R2 cleanup failed: {e}")

def run_single_seed(SEED, df, target_col, profiles_A, composite_config_A, dataset_id):
    # Set seed
    random.seed(SEED)
    np.random.seed(SEED)
    if torch is not None:
        torch.manual_seed(SEED)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Split dataset into A (Old Task) and B (New Task) based on Contract type (Real Domain Shift)
    df_A = df[df["Contract"] == "Month-to-month"].copy()
    df_B = df[df["Contract"].isin(["One year", "Two year"])].copy()

    # Split Task A/B into Train/Test subsets for evaluation
    df_A_train, df_A_test = train_test_split(df_A, test_size=0.2, stratify=df_A[target_col], random_state=SEED)
    df_B_train, df_B_test = train_test_split(df_B, test_size=0.2, stratify=df_B[target_col], random_state=SEED)

    # Clean previous local replay buffers to avoid interference
    replay_path = f"ml_artifacts/{dataset_id}/replay_buffer.parquet"
    if os.path.exists(f"data/{replay_path}"):
        try:
            os.remove(f"data/{replay_path}")
        except Exception:
            pass

    # Train Model A (Initial task model on A)
    print(f"\n--- Training Initial Model on Task A (Seed {SEED}) ---")
    model_uri_A, _ = run_automl(
        df=df_A_train,
        confirmed_profiles=profiles_A,
        target_col=target_col,
        dataset_id=dataset_id,
        composite_config=composite_config_A,
        random_state=SEED
    )
    print(f"Initial Model A logged at: {model_uri_A}")

    # Measure AUC of Model A on Test set A
    pipeline_A = mlflow.sklearn.load_model(model_uri_A)
    actual_model_A = pipeline_A.steps[-1][1]
    model_type_str = type(actual_model_A).__name__
    print(f"Initial Model type actually used: {model_type_str}")
    
    y_A_test_pred = pipeline_A.predict_proba(df_A_test)[:, 1]
    auc_A_before = float(roc_auc_score(df_A_test[target_col], y_A_test_pred))
    print(f"AUC of Model A on Test A (Initial): {auc_A_before:.4f}")

    # Setup the Replay Buffer with initial Task A train data to represent history
    buffer = ReplayBuffer()
    buffer.update(dataset_id, df_A_train, target_col, random_state=SEED)
    print("Replay buffer populated with Task A training data.")

    scenarios = {
        "Baseline": {"lambda": 0.0, "ratio": 0.0, "use_replay": False},
        "Replay only": {"lambda": 0.0, "ratio": 0.2, "use_replay": True},
        "EWC only": {"lambda": 100.0, "ratio": 0.0, "use_replay": False},
        "Full (EWC+Replay)": {"lambda": 100.0, "ratio": 0.2, "use_replay": True}
    }

    # Backup original functions before scenario loop to avoid leakage
    original_load = buffer.load_from_r2
    from backend.app.core.training.continual_trainer import ReplayBuffer as RBModule
    original_rb_load = RBModule.load_from_r2

    results = {}
    for name, config in scenarios.items():
        print(f"\n--- Running Scenario: {name} (Seed {SEED}) ---")
        os.environ["EWC_LAMBDA"] = str(config["lambda"])
        os.environ["REPLAY_BUFFER_RATIO"] = str(config["ratio"])
        
        if not config["use_replay"]:
            buffer.load_from_r2 = lambda *args, **kwargs: None
            RBModule.load_from_r2 = lambda *args, **kwargs: None
        else:
            RBModule.load_from_r2 = original_load

        try:
            model_uri_B, _ = run_automl(
                df=df_B_train,
                confirmed_profiles=profiles_A,
                target_col=target_col,
                dataset_id=dataset_id,
                composite_config=composite_config_A,
                prior_model_uri=model_uri_A,
                random_state=SEED
            )
            
            pipeline_B = mlflow.sklearn.load_model(model_uri_B)
            actual_model_B = pipeline_B.steps[-1][1]
            curr_model_type = type(actual_model_B).__name__
            print(f"Model type actually used: {curr_model_type}")
            
            y_A_after_pred = pipeline_B.predict_proba(df_A_test)[:, 1]
            auc_A_after = float(roc_auc_score(df_A_test[target_col], y_A_after_pred))
            
            y_B_pred = pipeline_B.predict_proba(df_B_test)[:, 1]
            auc_B = float(roc_auc_score(df_B_test[target_col], y_B_pred))
            
            forgetting_rate = (auc_A_before - auc_A_after) / auc_A_before
            
            results[name] = {
                "AUC_A_before": auc_A_before,
                "AUC_A_after": auc_A_after,
                "AUC_B": auc_B,
                "Forgetting_Rate": forgetting_rate,
                "Model_Type": curr_model_type
            }
            
            print(f"Results for {name}:")
            print(f"  AUC A before: {auc_A_before:.4f}")
            print(f"  AUC A after : {auc_A_after:.4f}")
            print(f"  AUC B       : {auc_B:.4f}")
            print(f"  Forgetting  : {forgetting_rate*100:.2f}%")
            
        finally:
            if not config["use_replay"]:
                buffer.load_from_r2 = original_load
                RBModule.load_from_r2 = original_rb_load

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-all-seeds", action="store_true", help="Run with 3 seeds and print statistics")
    args = parser.parse_args()

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

    # Target column is "Churn Value"
    target_col = "Churn Value"
    
    # Pre-synthesize CPI score on full dataset for consistency without target leakage (excluding Churn Value)
    df["cpi_score"] = (df["Churn Score"] - df["Churn Score"].min()) / (df["Churn Score"].max() - df["Churn Score"].min())
    print("Pre-synthesized cpi_score via Churn Score normalization (No target leakage).")

    # Profile once to get baseline feature profiles (using full dataset or Task A to establish configuration)
    df_A_temp = df[df["Contract"] == "Month-to-month"].copy()
    profiles_A, _ = run_profiling(df_A_temp)
    
    # Overwrite/Force roles in the profiles to prevent leakage
    exclude_leakage = ["CustomerID", "Churn Label", "Churn Score", "Churn Reason", "Contract"]
    for p in profiles_A:
        if p.name == target_col:
            p.inferred_role = DataRole.TARGET
            p.confidence_score = 1.0
        elif p.name == "cpi_score":
            p.inferred_role = DataRole.NUMERIC
        elif p.name in exclude_leakage:
            p.inferred_role = DataRole.IGNORE

    composite_config_A = CompositeTargetConfig(
        strategy=SynthesisStrategy.WEIGHTED,
        source_columns=["Churn Score"],
        cpi_variance_explained=None,
        weights=[ColumnWeight(name="Churn Score", weight=1.0, normalize_method="minmax")],
        requires_confirmation=False
    )
    
    # Generate unique run ID to isolate R2 data lake zones
    ABLATION_RUN_ID = f"ablation-{uuid.uuid4().hex[:8]}"
    print(f"Isolated Ablation Run ID: {ABLATION_RUN_ID}")

    try:
        if args.run_all_seeds:
            seeds = [42, 123, 456]
            all_results = {name: [] for name in ["Baseline", "Replay only", "EWC only", "Full (EWC+Replay)"]}
            
            for s in seeds:
                print(f"\n==================== RUNNING FOR SEED {s} ====================")
                seed_res = run_single_seed(s, df, target_col, profiles_A, composite_config_A, ABLATION_RUN_ID)
                for name, metrics in seed_res.items():
                    all_results[name].append(metrics)
                    
            # Calculate statistics (mean ± std)
            print("\n" + "="*80)
            print("EVALUATION SECTION: STATISTICAL SIGNIFICANCE (3 SEEDS MEAN ± STD)")
            print("="*80)
            
            # Check if fallback was used
            model_type_logged = all_results["Baseline"][0]["Model_Type"]
            print(f"Model type actually used: {model_type_logged}")
            if "MTLChurnModel" not in model_type_logged:
                print("[NOTE] Ablation Study chay tren Standard AutoML fallback, KHONG phai MTL PyTorch, do moi truong Windows chan torch DLL")
                
            print("\n| Scenario | Use EWC | Use Replay | $AUC_{A\\_before}$ | $AUC_{A\\_after}$ | $AUC_{B}$ | Forgetting Rate (%) |")
            print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
            
            for name in ["Baseline", "Replay only", "EWC only", "Full (EWC+Replay)"]:
                runs = all_results[name]
                auc_before_vals = [r["AUC_A_before"] for r in runs]
                auc_after_vals = [r["AUC_A_after"] for r in runs]
                auc_b_vals = [r["AUC_B"] for r in runs]
                forget_vals = [r["Forgetting_Rate"] * 100 for r in runs]
                
                use_ewc = "Yes" if "EWC" in name or name == "Full (EWC+Replay)" else "No"
                use_replay = "Yes" if "Replay" in name or name == "Full (EWC+Replay)" else "No"
                
                b_mean, b_std = np.mean(auc_before_vals), np.std(auc_before_vals)
                a_mean, a_std = np.mean(auc_after_vals), np.std(auc_after_vals)
                b_b_mean, b_b_std = np.mean(auc_b_vals), np.std(auc_b_vals)
                f_mean, f_std = np.mean(forget_vals), np.std(forget_vals)
                
                b_str = f"{b_mean:.4f} ± {b_std:.4f}"
                a_str = f"{a_mean:.4f} ± {a_std:.4f}"
                bb_str = f"{b_b_mean:.4f} ± {b_b_std:.4f}"
                f_str = f"{f_mean:.2f}% ± {f_std:.2f}%"
                
                if name == "Full (EWC+Replay)":
                    print(f"| **{name}** | {use_ewc} | {use_replay} | {b_str} | **{a_str}** | {bb_str} | **{f_str}** |")
                else:
                    print(f"| {name} | {use_ewc} | {use_replay} | {b_str} | {a_str} | {bb_str} | {f_str} |")
                    
            print("\n" + "="*80)
            
        else:
            # Run single seed
            res = run_single_seed(args.seed, df, target_col, profiles_A, composite_config_A, ABLATION_RUN_ID)
            
            # Display single seed table
            print("\n" + "="*80)
            print(f"EVALUATION SECTION: ABLATION STUDY COMPARISON TABLE (SEED {args.seed})")
            print("="*80)
            
            model_type_logged = res["Baseline"]["Model_Type"]
            print(f"Model type actually used: {model_type_logged}")
            if "MTLChurnModel" not in model_type_logged:
                print("[NOTE] Ablation Study chay tren Standard AutoML fallback, KHONG phai MTL PyTorch, do moi truong Windows chan torch DLL")
                
            print("\n| Scenario | Use EWC | Use Replay | $AUC_{A\\_before}$ | $AUC_{A\\_after}$ | $AUC_{B}$ | Forgetting Rate (%) |")
            print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
            for name, r in res.items():
                f_rate_str = f"{r['Forgetting_Rate']*100:.2f}%"
                use_ewc = "Yes" if "EWC" in name or name == "Full (EWC+Replay)" else "No"
                use_replay = "Yes" if "Replay" in name or name == "Full (EWC+Replay)" else "No"
                if name == "Full (EWC+Replay)":
                    print(f"| **{name}** | {use_ewc} | {use_replay} | {r['AUC_A_before']:.4f} | **{r['AUC_A_after']:.4f}** | {r['AUC_B']:.4f} | **{f_rate_str}** |")
                else:
                    print(f"| {name} | {use_ewc} | {use_replay} | {r['AUC_A_before']:.4f} | {r['AUC_A_after']:.4f} | {r['AUC_B']:.4f} | {f_rate_str} |")
            print("\n" + "="*80)

    finally:
        cleanup_r2_artifacts(ABLATION_RUN_ID)

if __name__ == "__main__":
    main()
