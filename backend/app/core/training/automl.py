import os
import json
import hashlib
import tempfile
import optuna
import mlflow
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from typing import List, Tuple
from mlflow.models.signature import infer_signature

from ..profiler.column_profile import ColumnProfile
from ..pipeline.builder import build_pipeline
from ..pipeline.schema_gen import generate_schema
from .model_router import route_models
from ..config import MODEL_NAME

def _hash_dataframe(df: pd.DataFrame) -> str:
    """Creates an MD5 hash of the dataframe for tracking."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def run_automl(df: pd.DataFrame, confirmed_profiles: List[ColumnProfile], target_col: str, dataset_id: str) -> Tuple[str, str]:
    """
    Executes the full AutoML flow:
    1. Validation
    2. Build Pipeline
    3. Generate Schema
    4. Route Models
    5. Optimize
    6. Log & Register Model
    """
    if not confirmed_profiles:
        raise ValueError("Empty profiles list provided.")
    if not target_col or target_col not in df.columns:
        raise ValueError("Invalid target column.")
    if df[target_col].nunique() <= 1:
        raise ValueError("Target column has only one unique class.")
        
    # Generate schema
    schema_path = generate_schema(confirmed_profiles, dataset_id, target_col)
    
    # Build preprocessor pipeline
    from ..exceptions import PipelineBuilderError
    try:
        base_pipeline = build_pipeline(confirmed_profiles, target_col)
    except PipelineBuilderError as e:
        raise e
    except Exception as e:
        raise ValueError(f"Pipeline build failed: {e}")
        
    # Get feature columns
    feature_cols = [p.name for p in confirmed_profiles if p.name != target_col and p.inferred_role not in ["ID", "IGNORE", "TARGET"]]
    
    # Route models
    routed_models = route_models(df, confirmed_profiles)
    if not routed_models:
        raise ValueError("No models routed for the given dataset.")
        
    X_full = df[feature_cols]
    y_full = df[target_col]
    
    # Split for threshold calibration to avoid leakage
    X, X_val, y, y_val = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)
    
    n_trials = int(os.getenv("OPTUNA_N_TRIALS", "15"))
    timeout = int(os.getenv("OPTUNA_TIMEOUT_SECONDS", "600"))
    dataset_hash = _hash_dataframe(df)
    
    best_overall_score = -1.0
    best_overall_pipeline = None
    best_model_info = None
    best_trial_params = None
    
    # Setup CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_info in routed_models:
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, (param_type, *args) in model_info["search_space"].items():
                if param_type == "int":
                    params[param_name] = trial.suggest_int(param_name, args[0], args[1])
                elif param_type == "loguniform":
                    params[param_name] = trial.suggest_float(param_name, args[0], args[1], log=True)
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, args[0])
                    
            # Instantiate model
            model = model_info["class"](**model_info["kwargs"], **params)
            
            # Clone base pipeline and set model
            trial_pipeline = clone(base_pipeline)
            trial_pipeline.steps[-1] = ('model', model)
            
            # Cross validate using explicit CV object
            scores = cross_val_score(trial_pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1, error_score='raise')
            auc = scores.mean()
            
            # MLflow logging per trial
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("roc_auc", auc)
                mlflow.set_tag("model_class", model_info["name"])
                
            return auc
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        if study.best_value > best_overall_score:
            best_overall_score = study.best_value
            best_model_info = model_info
            best_trial_params = study.best_params
            
    # Final fit with best params on training set
    final_model = best_model_info["class"](**best_model_info["kwargs"], **best_trial_params)
    final_pipeline = clone(base_pipeline)
    final_pipeline.steps[-1] = ('model', final_model)
    final_pipeline.fit(X, y)
    
    # Calculate optimal threshold on validation set
    y_scores = final_pipeline.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores)
    # Handle denominator to prevent division by zero
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    # thresholds array has len(f1_scores) - 1
    optimal_threshold = float(thresholds[f1_scores[:-1].argmax()])
    
    # Log and register best model
    with mlflow.start_run() as run:
        mlflow.set_tags({
            "dataset_hash": dataset_hash,
            "target_col": target_col,
            "model_class": best_model_info["name"]
        })
        mlflow.log_params(best_trial_params)
        mlflow.log_param("optuna_timeout_seconds", timeout)
        mlflow.log_metric("best_roc_auc", best_overall_score)
        mlflow.log_metric("optimal_threshold", optimal_threshold)
        
        # Save threshold artifact
        with tempfile.TemporaryDirectory() as tmp_dir:
            threshold_path = os.path.join(tmp_dir, "threshold.json")
            with open(threshold_path, "w") as f:
                json.dump({"optimal_threshold": optimal_threshold, "metric": "f1", "cv_fold": "last"}, f)
            mlflow.log_artifact(threshold_path)
        
        # Log model with signature and input example
        signature = infer_signature(X, final_pipeline.predict_proba(X))
        input_example = X.iloc[:5]
        
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=input_example
        )
        
        model_uri = f"runs:/{run.info.run_id}/model"
        
    return model_uri, schema_path