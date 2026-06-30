from __future__ import annotations
import os
import io
import json
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from ...core.storage import storage

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    TensorDataset = None
    DataLoader = None

class FisherCalculator:
    @staticmethod
    def calculate_fisher(
        model: nn.Module,
        X: np.ndarray,
        y_binary: np.ndarray,
        y_cpi: np.ndarray,
        alpha: float = 0.7,
        beta: float = 0.3
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate diagonal Fisher Information Matrix.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to calculate Fisher Matrix.")

        model.eval()
        fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_bin_t = torch.tensor(y_binary, dtype=torch.float32).unsqueeze(1)
        y_cpi_t = torch.tensor(y_cpi, dtype=torch.float32).unsqueeze(1)

        bce_loss_fn = nn.BCEWithLogitsLoss()
        mse_loss_fn = nn.MSELoss()

        n_samples = len(X)
        if n_samples == 0:
            return fisher

        for i in range(n_samples):
            model.zero_grad()
            bx = X_t[i:i+1]
            by_bin = y_bin_t[i:i+1]
            by_cpi = y_cpi_t[i:i+1]

            logits_a, out_b = model(bx)
            loss_a = bce_loss_fn(logits_a, by_bin)
            loss_b = mse_loss_fn(out_b, by_cpi)
            loss = alpha * loss_a + beta * loss_b
            
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += (param.grad.data.clone() ** 2) / n_samples

        return fisher

class ReplayBuffer:
    def __init__(self, max_size: Optional[int] = None):
        if max_size is None:
            max_size = int(os.getenv("REPLAY_BUFFER_MAX_SIZE", "1000"))
        self.max_size = max_size

    def get_path(self, dataset_id: str) -> str:
        return f"ml_artifacts/{dataset_id}/replay_buffer.parquet"

    def load_from_r2(self, dataset_id: str) -> Optional[pd.DataFrame]:
        path = self.get_path(dataset_id)
        try:
            content = storage.download_file(path)
            return pd.read_parquet(io.BytesIO(content))
        except Exception:
            return None

    def save_to_r2(self, dataset_id: str, df: pd.DataFrame):
        path = self.get_path(dataset_id)
        pq_buffer = io.BytesIO()
        df.to_parquet(pq_buffer, index=False)
        storage.upload_file(pq_buffer.getvalue(), path)

    def update(self, dataset_id: str, df_new: pd.DataFrame, target_col: str):
        df_old = self.load_from_r2(dataset_id)
        if df_old is not None:
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new

        if len(df_combined) <= self.max_size:
            self.save_to_r2(dataset_id, df_combined)
            return

        if target_col in df_combined.columns:
            counts = df_combined[target_col].value_counts(normalize=True)
            sampled_dfs = []
            for val, pct in counts.items():
                df_subset = df_combined[df_combined[target_col] == val]
                sub_size = max(1, int(round(pct * self.max_size)))
                n_to_sample = min(len(df_subset), sub_size)
                if n_to_sample > 0:
                    sampled_dfs.append(df_subset.sample(n=n_to_sample, random_state=42))
            
            df_sampled = pd.concat(sampled_dfs, ignore_index=True)
            if len(df_sampled) > self.max_size:
                df_sampled = df_sampled.sample(n=self.max_size, random_state=42)
            elif len(df_sampled) < self.max_size:
                remainder = df_combined[~df_combined.index.isin(df_sampled.index)]
                needed = self.max_size - len(df_sampled)
                if needed > 0 and len(remainder) > 0:
                    fill = remainder.sample(n=min(len(remainder), needed), random_state=42)
                    df_sampled = pd.concat([df_sampled, fill], ignore_index=True)
        else:
            df_sampled = df_combined.sample(n=self.max_size, random_state=42)

        self.save_to_r2(dataset_id, df_sampled)

class ContinualMTLTrainer:
    def __init__(self, lambda_ewc: Optional[float] = None, mixing_ratio: Optional[float] = None):
        if lambda_ewc is None:
            lambda_ewc = float(os.getenv("EWC_LAMBDA", "100.0"))
        self.lambda_ewc = lambda_ewc
        
        if mixing_ratio is None:
            mixing_ratio = float(os.getenv("REPLAY_BUFFER_RATIO", "0.2"))
        self.mixing_ratio = mixing_ratio
        
        self.replay_buffer = ReplayBuffer()

    def train(
        self,
        prior_model_uri: str,
        dataset_id: str,
        df_new: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        cpi_col: str,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64
    ) -> Tuple[nn.Module, object, float, float]:
        """
        Run Continual MTL training loop.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ContinualMTLTrainer.")

        import mlflow
        # 1. Load prior pipeline from MLflow
        pipeline = mlflow.sklearn.load_model(prior_model_uri)
        preprocessor = pipeline.named_steps['preprocessor']
        prior_mtl_model = pipeline.named_steps['model']
        
        # 2. Get baseline PyTorch model and weights
        model = prior_mtl_model.model
        param_old = {}
        for name, param in model.named_parameters():
            param_old[name] = param.data.clone()

        # 3. Load Replay Buffer
        df_replay = self.replay_buffer.load_from_r2(dataset_id)
        
        # 4. Trộn dữ liệu phân tầng
        if df_replay is not None and len(df_replay) > 0:
            # We want to mix based on mixing_ratio
            n_replay = int(len(df_new) * self.mixing_ratio / (1 - self.mixing_ratio))
            n_replay = min(len(df_replay), max(10, n_replay))
            
            # Stratified sample from replay
            if target_col in df_replay.columns:
                counts = df_replay[target_col].value_counts(normalize=True)
                sampled_dfs = []
                for val, pct in counts.items():
                    df_subset = df_replay[df_replay[target_col] == val]
                    sub_size = max(1, int(round(pct * n_replay)))
                    n_to_sample = min(len(df_subset), sub_size)
                    if n_to_sample > 0:
                        sampled_dfs.append(df_subset.sample(n=n_to_sample, random_state=42))
                df_replay_sampled = pd.concat(sampled_dfs, ignore_index=True)
            else:
                df_replay_sampled = df_replay.sample(n=n_replay, random_state=42)

            df_mixed = pd.concat([df_new, df_replay_sampled], ignore_index=True)
        else:
            df_mixed = df_new
            df_replay_sampled = df_new.sample(n=min(len(df_new), 200), random_state=42) # fallback for Fisher calculation

        # 5. Preprocess mixed data
        X_mixed_raw = df_mixed[feature_cols]
        y_mixed_bin = df_mixed[target_col]
        y_mixed_cpi = df_mixed[cpi_col]

        # Use preprocessor (which is already fitted) to transform data
        X_mixed_trans = preprocessor.transform(X_mixed_raw)
        if hasattr(X_mixed_trans, "toarray"):
            X_mixed_trans = X_mixed_trans.toarray()

        # Target encoding
        if not pd.api.types.is_numeric_dtype(y_mixed_bin):
            unique_classes = sorted(y_mixed_bin.dropna().unique())
            pos_label = unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
            y_mixed_bin_encoded = (y_mixed_bin == pos_label).astype(int)
        else:
            y_mixed_bin_encoded = y_mixed_bin

        # 6. Calculate Fisher Matrix using Replay Data as proxy
        X_replay_raw = df_replay_sampled[feature_cols]
        y_replay_bin = df_replay_sampled[target_col]
        y_replay_cpi = df_replay_sampled[cpi_col]

        X_replay_trans = preprocessor.transform(X_replay_raw)
        if hasattr(X_replay_trans, "toarray"):
            X_replay_trans = X_replay_trans.toarray()

        if not pd.api.types.is_numeric_dtype(y_replay_bin):
            unique_classes = sorted(y_replay_bin.dropna().unique())
            pos_label = unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
            y_replay_bin_encoded = (y_replay_bin == pos_label).astype(int)
        else:
            y_replay_bin_encoded = y_replay_bin

        fisher = FisherCalculator.calculate_fisher(
            model,
            X_replay_trans,
            y_replay_bin_encoded.values,
            y_replay_cpi.values
        )

        # 7. PyTorch Training Loop with EWC penalty
        model.train()
        X_t = torch.tensor(X_mixed_trans, dtype=torch.float32)
        y_bin_t = torch.tensor(y_mixed_bin_encoded.values, dtype=torch.float32).unsqueeze(1)
        y_cpi_t = torch.tensor(y_mixed_cpi.values, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_t, y_bin_t, y_cpi_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        mse_loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            for batch_x, batch_y_bin, batch_y_cpi in dataloader:
                if len(batch_x) <= 1:
                    continue
                optimizer.zero_grad()
                logits_a, out_b = model(batch_x)
                loss_a = bce_loss_fn(logits_a, batch_y_bin)
                loss_b = mse_loss_fn(out_b, batch_y_cpi)

                # Compute EWC loss
                ewc_loss = 0.0
                for name, param in model.named_parameters():
                    if name in fisher:
                        ewc_loss += (fisher[name] * (param - param_old[name]) ** 2).sum()

                # MTL Churn Model uses BCE weight 0.7, MSE weight 0.3
                loss = 0.7 * loss_a + 0.3 * loss_b + (self.lambda_ewc / 2.0) * ewc_loss
                loss.backward()
                optimizer.step()

        # Update Replay Buffer with new data
        self.replay_buffer.update(dataset_id, df_new, target_col)

        # Assign updated model back into prior_mtl_model
        prior_mtl_model.model = model
        
        # Create updated final pipeline
        final_pipeline = pipeline
        
        # Calculate optimal threshold on validation set of df_new
        # Get threshold using mtl_model.predict_proba
        X_new_raw = df_new[feature_cols]
        y_new_bin = df_new[target_col]
        
        X_new_trans = preprocessor.transform(X_new_raw)
        if hasattr(X_new_trans, "toarray"):
            X_new_trans = X_new_trans.toarray()
            
        if not pd.api.types.is_numeric_dtype(y_new_bin):
            unique_classes = sorted(y_new_bin.dropna().unique())
            pos_label = unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
            y_new_bin_encoded = (y_new_bin == pos_label).astype(int)
        else:
            y_new_bin_encoded = y_new_bin

        # Optimal Threshold
        y_scores = prior_mtl_model.predict_proba(X_new_trans)[:, 1]
        from sklearn.metrics import precision_recall_curve, roc_auc_score
        precisions, recalls, thresholds = precision_recall_curve(y_new_bin_encoded.values, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        optimal_threshold = float(thresholds[f1_scores[:-1].argmax()])
        best_roc_auc = float(roc_auc_score(y_new_bin_encoded.values, y_scores))

        return model, final_pipeline, best_roc_auc, optimal_threshold
