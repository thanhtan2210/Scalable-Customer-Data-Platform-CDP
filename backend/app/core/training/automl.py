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
from typing import List, Tuple, Optional
from mlflow.models.signature import infer_signature

from ..profiler.column_profile import ColumnProfile
from ..profiler.target_analysis import CompositeTargetConfig
from ..pipeline.builder import build_pipeline
from ..pipeline.schema_gen import generate_schema
from .model_router import route_models
from .mtl_trainer import is_mtl_available, MTLChurnModel
from .continual_trainer import ContinualMTLTrainer
from ..config import MODEL_NAME

def _hash_dataframe(df: pd.DataFrame) -> str:
    """Creates an MD5 hash of the dataframe for tracking."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def _cleanup_mlflow():
    try:
        import os
        from .mlflow_utils import cleanup_old_runs
        exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")
        keep_n = int(os.getenv("MLFLOW_KEEP_LAST_N_RUNS", "5"))
        cleanup_old_runs(exp_name, keep_n)
    except Exception as e:
        print(f"Failed to cleanup old MLflow runs: {e}")

def run_automl(
    df: pd.DataFrame,
    confirmed_profiles: List[ColumnProfile],
    target_col: str,
    dataset_id: str,
    composite_config: Optional[CompositeTargetConfig] = None,
    prior_model_uri: Optional[str] = None,
    random_state: int = 42
) -> Tuple[str, str]:
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

    # Calculate CPI column if not already in dataframe but composite_config is provided
    if composite_config and composite_config.cpi_column_name not in df.columns:
        from ..profiler.target_synthesizer import _weighted_synthesis, _pca_synthesis
        if composite_config.strategy == "PCA":
            _, cpi_series = _pca_synthesis(df, composite_config.source_columns, target_col)
            df[composite_config.cpi_column_name] = cpi_series
        elif composite_config.strategy == "WEIGHTED" and composite_config.weights:
            cpi_series = _weighted_synthesis(df, composite_config.weights)
            df[composite_config.cpi_column_name] = cpi_series

    # Generate schema
    schema_path = generate_schema(confirmed_profiles, dataset_id, target_col)
    
    # Determine auxiliary cols for standard pipeline build
    aux_cols = None
    if composite_config and not is_mtl_available():
        aux_cols = [composite_config.cpi_column_name]
        
    # Build preprocessor pipeline
    from ..exceptions import PipelineBuilderError
    try:
        base_pipeline = build_pipeline(confirmed_profiles, target_col, auxiliary_cols=aux_cols)
    except PipelineBuilderError as e:
        raise e
    except Exception as e:
        raise ValueError(f"Pipeline build failed: {e}")
        
    # Get feature columns
    feature_cols = [p.name for p in confirmed_profiles if p.name != target_col and p.inferred_role not in ["ID", "IGNORE", "TARGET"]]
    if aux_cols:
        for ac in aux_cols:
            if ac not in feature_cols:
                feature_cols.append(ac)

    # Continual Learning MTL training path
    if prior_model_uri and is_mtl_available():
        cpi_col = composite_config.cpi_column_name if composite_config else "cpi_score"
        if cpi_col not in df.columns:
            if composite_config:
                from ..profiler.target_synthesizer import _weighted_synthesis, _pca_synthesis
                if composite_config.strategy == "PCA":
                    _, cpi_series = _pca_synthesis(df, composite_config.source_columns, target_col)
                    df[cpi_col] = cpi_series
                elif composite_config.strategy == "WEIGHTED" and composite_config.weights:
                    cpi_series = _weighted_synthesis(df, composite_config.weights)
                    df[cpi_col] = cpi_series
            else:
                df[cpi_col] = 0.0

        trainer = ContinualMTLTrainer()
        model, final_pipeline, best_overall_score, optimal_threshold = trainer.train(
            prior_model_uri=prior_model_uri,
            dataset_id=dataset_id,
            df_new=df,
            feature_cols=feature_cols,
            target_col=target_col,
            cpi_col=cpi_col,
            random_state=random_state
        )
        
        dataset_hash = _hash_dataframe(df)
        
        with mlflow.start_run() as run:
            mlflow.set_tags({
                "dataset_hash": dataset_hash,
                "target_col": target_col,
                "model_class": "MTLChurnModel",
                "continual_learning": "True",
                "prior_model_uri": prior_model_uri
            })
            mlflow.log_metric("best_roc_auc", best_overall_score)
            mlflow.log_metric("optimal_threshold", optimal_threshold)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                threshold_path = os.path.join(tmp_dir, "threshold.json")
                with open(threshold_path, "w") as f:
                    json.dump({"optimal_threshold": optimal_threshold, "metric": "f1", "cv_fold": "last"}, f)
                mlflow.log_artifact(threshold_path)
            
            signature = infer_signature(df[feature_cols], final_pipeline.predict_proba(df[feature_cols]))
            input_example = df[feature_cols].iloc[:5]
            
            mlflow.sklearn.log_model(
                sk_model=final_pipeline,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                signature=signature,
                input_example=input_example
            )
            
            model_uri = f"runs:/{run.info.run_id}/model"
            
        _cleanup_mlflow()
        return model_uri, schema_path

    # MTL training path
    if composite_config and is_mtl_available():
        preprocessor = base_pipeline.named_steps['preprocessor']
        X_full_raw = df[feature_cols]
        y_full_bin = df[target_col]
        y_full_cpi = df[composite_config.cpi_column_name]
        
        # Fit preprocessor on full data
        preprocessor.fit(X_full_raw, y_full_bin)
        
        # Transform inputs
        X_full_trans = preprocessor.transform(X_full_raw)
        if hasattr(X_full_trans, "toarray"):
            X_full_trans = X_full_trans.toarray()
            
        # Encode target if object/categorical
        if not pd.api.types.is_numeric_dtype(y_full_bin):
            unique_classes = sorted(y_full_bin.dropna().unique())
            pos_label = unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
            y_full_bin_encoded = (y_full_bin == pos_label).astype(int)
        else:
            y_full_bin_encoded = y_full_bin

        # Train-test split
        X_train, X_val, y_train, y_val, y_cpi_train, y_cpi_val = train_test_split(
            X_full_trans, y_full_bin_encoded.values, y_full_cpi.values,
            test_size=0.2, stratify=y_full_bin_encoded.values, random_state=random_state
        )
        
        # Fit MTL model
        mtl_model = MTLChurnModel()
        mtl_model.fit(X_train, y_train, y_cpi_train, random_state=random_state)
        
        # Calculate optimal threshold on validation set
        y_scores = mtl_model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        optimal_threshold = float(thresholds[f1_scores[:-1].argmax()])
        
        from sklearn.metrics import roc_auc_score
        best_overall_score = float(roc_auc_score(y_val, y_scores))
        
        # Package preprocessor and mtl_model into the final pipeline
        final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', mtl_model)
        ])
        
        dataset_hash = _hash_dataframe(df)
        
        # Log and register best model
        with mlflow.start_run() as run:
            mlflow.set_tags({
                "dataset_hash": dataset_hash,
                "target_col": target_col,
                "model_class": "MTLChurnModel"
            })
            mlflow.log_metric("best_roc_auc", best_overall_score)
            mlflow.log_metric("optimal_threshold", optimal_threshold)
            
            # Save threshold artifact
            with tempfile.TemporaryDirectory() as tmp_dir:
                threshold_path = os.path.join(tmp_dir, "threshold.json")
                with open(threshold_path, "w") as f:
                    json.dump({"optimal_threshold": optimal_threshold, "metric": "f1", "cv_fold": "last"}, f)
                mlflow.log_artifact(threshold_path)
            
            # Log model with signature and input example
            signature = infer_signature(X_full_raw, final_pipeline.predict_proba(X_full_raw))
            input_example = X_full_raw.iloc[:5]
            
            mlflow.sklearn.log_model(
                sk_model=final_pipeline,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                signature=signature,
                input_example=input_example
            )
            
            model_uri = f"runs:/{run.info.run_id}/model"
            
        _cleanup_mlflow()
        return model_uri, schema_path

    
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
    
    # Encode y_val to binary 0/1 based on the classes of the fitted model
    if hasattr(final_pipeline, "classes_"):
        classes = final_pipeline.classes_
    elif hasattr(final_model, "classes_"):
        classes = final_model.classes_
    else:
        classes = None
        
    if classes is not None and len(classes) > 1 and not pd.api.types.is_numeric_dtype(y_val):
        pos_label = classes[1]
        y_val_bin = (y_val == pos_label).astype(int)
    else:
        y_val_bin = y_val
        
    precisions, recalls, thresholds = precision_recall_curve(y_val_bin, y_scores)
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
        
    _cleanup_mlflow()
    return model_uri, schema_path