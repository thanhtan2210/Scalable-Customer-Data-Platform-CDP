import optuna
import mlflow
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from .model_router import select_model
from ..pipeline.builder import build_pipeline

def run_automl(df: pd.DataFrame, profiles, target_col: str, n_trials: int = 30):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    model_name = select_model(profiles, len(df))
    base_pipeline = build_pipeline(profiles, target_col)
    
    def objective(trial):
        if model_name == "RandomForestClassifier":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
            }
            clf = RandomForestClassifier(**params, random_state=42)
        elif model_name == "XGBClassifier":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            }
            clf = XGBClassifier(**params, random_state=42)
        else: # LogisticRegression
            params = {"C": trial.suggest_float("C", 0.1, 10.0)}
            clf = LogisticRegression(**params, max_iter=1000)

        pipeline = Pipeline(base_pipeline.steps + [("model", clf)])
        score = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc").mean()
        return score

    study = optuna.create_study(direction="maximize", storage="sqlite:///optuna.db", load_if_exists=True)
    
    with mlflow.start_run(run_name=f"AutoML_{model_name}"):
        study.optimize(objective, n_trials=n_trials)
        
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_roc_auc", study.best_value)
        
        # Finalize and Register Best Model
        # ... logic to refit best model and log artifact ...
        
    return study.best_trial
