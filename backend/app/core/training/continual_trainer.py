from __future__ import annotations
import os
import io
import json
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from ...core.storage import storage
from .representation_analyzer import RepresentationAnalyzer
from .pcgrad_optimizer import project_conflicting_gradients_continual
from .coreset_sampler import sample_coreset

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
        beta: float = 0.3,
        max_samples: int = 500,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate diagonal Fisher Information Matrix efficiently.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to calculate Fisher Matrix.")

        model.eval()
        fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        n_samples = len(X)
        if n_samples == 0:
            return fisher

        if n_samples > max_samples:
            # Deterministic/stratified stride
            step = max(1, n_samples // max_samples)
            indices = list(range(0, n_samples, step))[:max_samples]
            X = X[indices]
            y_binary = y_binary[indices]
            y_cpi = y_cpi[indices]
            n_samples = len(X)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_bin_t = torch.tensor(y_binary, dtype=torch.float32).unsqueeze(1)
        y_cpi_t = torch.tensor(y_cpi, dtype=torch.float32).unsqueeze(1)

        bce_loss_fn = nn.BCEWithLogitsLoss()
        mse_loss_fn = nn.MSELoss()

        for i in range(n_samples):
            model.zero_grad()
            bx = X_t[i : i + 1]
            by_bin = y_bin_t[i : i + 1]
            by_cpi = y_cpi_t[i : i + 1]

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

    def update(
        self,
        dataset_id: str,
        df_new: pd.DataFrame,
        target_col: str,
        strategy: str = "stratified",
        feature_cols: Optional[List[str]] = None,
        model: Optional[nn.Module] = None,
        preprocessor: Optional[object] = None,
        random_state: int = 42,
    ) -> Dict[str, float]:
        df_old = self.load_from_r2(dataset_id)
        if df_old is not None:
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new

        if len(df_combined) <= self.max_size:
            self.save_to_r2(dataset_id, df_combined)
            return {"mean_shift": 0.0, "buffer_size": float(len(df_combined))}

        if strategy in ("herding", "kcenter") and feature_cols is not None:
            df_sampled, diag = sample_coreset(
                df_combined,
                feature_cols=feature_cols,
                target_col=target_col,
                coreset_size=self.max_size,
                strategy=strategy,
                model=model,
                preprocessor=preprocessor,
                random_state=random_state,
            )
        elif target_col in df_combined.columns:
            counts = df_combined[target_col].value_counts(normalize=True)
            sampled_dfs = []
            for val, pct in counts.items():
                df_subset = df_combined[df_combined[target_col] == val]
                sub_size = max(1, int(round(pct * self.max_size)))
                n_to_sample = min(len(df_subset), sub_size)
                if n_to_sample > 0:
                    sampled_dfs.append(
                        df_subset.sample(n=n_to_sample, random_state=random_state)
                    )

            df_sampled = pd.concat(sampled_dfs, ignore_index=True)
            if len(df_sampled) > self.max_size:
                df_sampled = df_sampled.sample(
                    n=self.max_size, random_state=random_state
                )
            elif len(df_sampled) < self.max_size:
                remainder = df_combined[~df_combined.index.isin(df_sampled.index)]
                needed = self.max_size - len(df_sampled)
                if needed > 0 and len(remainder) > 0:
                    fill = remainder.sample(
                        n=min(len(remainder), needed), random_state=random_state
                    )
                    df_sampled = pd.concat([df_sampled, fill], ignore_index=True)
            diag = {"mean_shift": 0.0, "buffer_size": float(len(df_sampled))}
        else:
            df_sampled = df_combined.sample(n=self.max_size, random_state=random_state)
            diag = {"mean_shift": 0.0, "buffer_size": float(len(df_sampled))}

        self.save_to_r2(dataset_id, df_sampled)
        return diag


class ContinualMTLTrainer:
    def __init__(
        self,
        lambda_ewc: Optional[float] = None,
        mixing_ratio: Optional[float] = None,
        use_pcgrad: bool = False,
        use_dynamic_fisher: bool = False,
        dynamic_fisher_freq: int = 10,
        coreset_strategy: str = "stratified",
    ):
        if lambda_ewc is None:
            lambda_ewc = float(os.getenv("EWC_LAMBDA", "100.0"))
        self.lambda_ewc = lambda_ewc

        if mixing_ratio is None:
            mixing_ratio = float(os.getenv("REPLAY_BUFFER_RATIO", "0.2"))
        self.mixing_ratio = mixing_ratio

        self.use_pcgrad = use_pcgrad
        self.use_dynamic_fisher = use_dynamic_fisher
        self.dynamic_fisher_freq = dynamic_fisher_freq
        self.coreset_strategy = coreset_strategy

        self.replay_buffer = ReplayBuffer()
        self.last_history_metrics: Optional[Dict[str, List[float]]] = None

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
        batch_size: int = 64,
        random_state: int = 42,
        df_test_a: Optional[pd.DataFrame] = None,
        return_history: bool = False,
    ) -> Union[
        Tuple[nn.Module, object, float, float],
        Tuple[nn.Module, object, float, float, Dict[str, List[float]]],
    ]:
        """
        Run Continual MTL training loop with train/val splitting, PCGrad gradient surgery,
        Dynamic Fisher recalculation, and diagnostic metrics tracking.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ContinualMTLTrainer.")

        import mlflow
        from sklearn.model_selection import train_test_split

        # 1. Load prior pipeline from MLflow
        pipeline = mlflow.sklearn.load_model(prior_model_uri)
        preprocessor = pipeline.named_steps["preprocessor"]
        prior_mtl_model = pipeline.named_steps["model"]

        # 2. Get baseline PyTorch model and weights
        model = prior_mtl_model.model
        param_old = {}
        for name, param in model.named_parameters():
            param_old[name] = param.data.clone()

        # 3. Split df_new into train and validation subsets for convergence tracking
        try:
            df_train_sub, df_val_sub = train_test_split(
                df_new,
                test_size=0.2,
                stratify=df_new[target_col],
                random_state=random_state,
            )
        except Exception:
            df_train_sub, df_val_sub = train_test_split(
                df_new, test_size=0.2, random_state=random_state
            )

        # Prepare validation data tensors
        X_val_raw = df_val_sub[feature_cols]
        y_val_bin = df_val_sub[target_col]
        y_val_cpi = df_val_sub[cpi_col]

        X_val_trans = preprocessor.transform(X_val_raw)
        if hasattr(X_val_trans, "toarray"):
            X_val_trans = X_val_trans.toarray()

        if not pd.api.types.is_numeric_dtype(y_val_bin):
            unique_classes = sorted(y_val_bin.dropna().unique())
            pos_label = (
                unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
            )
            y_val_bin_encoded = (y_val_bin == pos_label).astype(int)
        else:
            y_val_bin_encoded = y_val_bin

        X_val_t = torch.tensor(X_val_trans, dtype=torch.float32)
        y_val_bin_t = torch.tensor(
            y_val_bin_encoded.values, dtype=torch.float32
        ).unsqueeze(1)
        y_val_cpi_t = torch.tensor(y_val_cpi.values, dtype=torch.float32).unsqueeze(1)

        # 4. Load Replay Buffer
        df_replay = self.replay_buffer.load_from_r2(dataset_id)

        # 5. Mix stratified / coreset replay data with new train subset
        if df_replay is not None and len(df_replay) > 0:
            n_replay = int(
                len(df_train_sub) * self.mixing_ratio / (1 - self.mixing_ratio)
            )
            n_replay = min(len(df_replay), max(10, n_replay))

            if self.coreset_strategy in ("herding", "kcenter"):
                df_replay_sampled, _ = sample_coreset(
                    df_replay,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    coreset_size=n_replay,
                    strategy=self.coreset_strategy,
                    model=model,
                    preprocessor=preprocessor,
                    random_state=random_state,
                )
            elif target_col in df_replay.columns:
                counts = df_replay[target_col].value_counts(normalize=True)
                sampled_dfs = []
                for val, pct in counts.items():
                    df_subset = df_replay[df_replay[target_col] == val]
                    sub_size = max(1, int(round(pct * n_replay)))
                    n_to_sample = min(len(df_subset), sub_size)
                    if n_to_sample > 0:
                        sampled_dfs.append(
                            df_subset.sample(n=n_to_sample, random_state=random_state)
                        )
                df_replay_sampled = pd.concat(sampled_dfs, ignore_index=True)
            else:
                df_replay_sampled = df_replay.sample(
                    n=n_replay, random_state=random_state
                )

            df_mixed = pd.concat([df_train_sub, df_replay_sampled], ignore_index=True)
        else:
            df_mixed = df_train_sub
            df_replay_sampled = df_train_sub.sample(
                n=min(len(df_train_sub), 200), random_state=random_state
            )

        # Preprocess mixed data
        X_mixed_raw = df_mixed[feature_cols]
        y_mixed_bin = df_mixed[target_col]
        y_mixed_cpi = df_mixed[cpi_col]

        X_mixed_trans = preprocessor.transform(X_mixed_raw)
        if hasattr(X_mixed_trans, "toarray"):
            X_mixed_trans = X_mixed_trans.toarray()

        if not pd.api.types.is_numeric_dtype(y_mixed_bin):
            unique_classes = sorted(y_mixed_bin.dropna().unique())
            pos_label = (
                unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
            )
            y_mixed_bin_encoded = (y_mixed_bin == pos_label).astype(int)
        else:
            y_mixed_bin_encoded = y_mixed_bin

        # 6. Calculate Initial Fisher Matrix
        X_replay_raw = df_replay_sampled[feature_cols]
        y_replay_bin = df_replay_sampled[target_col]
        y_replay_cpi = df_replay_sampled[cpi_col]

        X_replay_trans = preprocessor.transform(X_replay_raw)
        if hasattr(X_replay_trans, "toarray"):
            X_replay_trans = X_replay_trans.toarray()

        if not pd.api.types.is_numeric_dtype(y_replay_bin):
            unique_classes = sorted(y_replay_bin.dropna().unique())
            pos_label = (
                unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
            )
            y_replay_bin_encoded = (y_replay_bin == pos_label).astype(int)
        else:
            y_replay_bin_encoded = y_replay_bin

        fisher = FisherCalculator.calculate_fisher(
            model, X_replay_trans, y_replay_bin_encoded.values, y_replay_cpi.values
        )

        # Optional Task A test tensor for generalization gap tracking
        X_test_a_t = None
        y_test_a_bin_t = None
        y_test_a_cpi_t = None
        if df_test_a is not None:
            X_ta = preprocessor.transform(df_test_a[feature_cols])
            if hasattr(X_ta, "toarray"):
                X_ta = X_ta.toarray()
            yta_bin = df_test_a[target_col]
            if not pd.api.types.is_numeric_dtype(yta_bin):
                unique_classes = sorted(yta_bin.dropna().unique())
                pos_label = unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
                yta_bin_enc = (yta_bin == pos_label).astype(int)
            else:
                yta_bin_enc = yta_bin
            X_test_a_t = torch.tensor(X_ta, dtype=torch.float32)
            y_test_a_bin_t = torch.tensor(yta_bin_enc.values, dtype=torch.float32).unsqueeze(1)
            y_test_a_cpi_t = torch.tensor(df_test_a[cpi_col].values, dtype=torch.float32).unsqueeze(1)

        # Replay batch tensor for Taylor residual & generalization gap
        X_rep_t = torch.tensor(X_replay_trans, dtype=torch.float32)
        y_rep_bin_t = torch.tensor(y_replay_bin_encoded.values, dtype=torch.float32).unsqueeze(1)
        y_rep_cpi_t = torch.tensor(y_replay_cpi.values, dtype=torch.float32).unsqueeze(1)

        # Initial true Task A loss at baseline theta_A^*
        model.eval()
        with torch.no_grad():
            l_init_a, o_init_b = model(X_rep_t)
            bce_fn = nn.BCEWithLogitsLoss()
            mse_fn = nn.MSELoss()
            loss_A_star = (0.7 * bce_fn(l_init_a, y_rep_bin_t) + 0.3 * mse_fn(o_init_b, y_rep_cpi_t)).item()

        # 7. PyTorch Training Loop
        model.train()
        X_t = torch.tensor(X_mixed_trans, dtype=torch.float32)
        y_bin_t = torch.tensor(
            y_mixed_bin_encoded.values, dtype=torch.float32
        ).unsqueeze(1)
        y_cpi_t = torch.tensor(y_mixed_cpi.values, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_t, y_bin_t, y_cpi_t)

        g = torch.Generator()
        g.manual_seed(random_state)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=g
        )

        optimizer = optim.Adam(model.parameters(), lr=lr)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        mse_loss_fn = nn.MSELoss()

        history_metrics: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "cosine_similarity": [],
            "conflict_rate": [],
            "effective_rank": [],
            "taylor_residual": [],
            "generalization_gap": [],
        }

        for epoch in range(epochs):
            # Dynamic Fisher Re-estimation
            if self.use_dynamic_fisher and epoch > 0 and epoch % self.dynamic_fisher_freq == 0:
                fisher = FisherCalculator.calculate_fisher(
                    model, X_replay_trans, y_replay_bin_encoded.values, y_replay_cpi.values
                )
                for name, param in model.named_parameters():
                    param_old[name] = param.data.clone()

            total_train_loss = 0.0
            batches_count = 0
            epoch_cos_sims = []
            epoch_conflicts = []

            for batch_x, batch_y_bin, batch_y_cpi in dataloader:
                if len(batch_x) <= 1:
                    continue

                logits_a, out_b = model(batch_x)
                loss_a = bce_loss_fn(logits_a, batch_y_bin)
                loss_b = mse_loss_fn(out_b, batch_y_cpi)
                loss_task = 0.7 * loss_a + 0.3 * loss_b

                # Compute EWC loss
                ewc_loss = 0.0
                for name, param in model.named_parameters():
                    if name in fisher:
                        ewc_loss += (
                            fisher[name] * (param - param_old[name]) ** 2
                        ).sum()
                loss_ewc_scaled = (self.lambda_ewc / 2.0) * ewc_loss

                if self.use_pcgrad and self.lambda_ewc > 0:
                    optimizer.zero_grad()
                    diag_pcgrad = project_conflicting_gradients_continual(
                        model, loss_task, loss_ewc_scaled, scope="shared"
                    )
                    optimizer.step()
                    epoch_cos_sims.append(diag_pcgrad["cosine_similarity"])
                    epoch_conflicts.append(diag_pcgrad["conflict_detected"])
                    total_train_loss += (loss_task.item() + loss_ewc_scaled.item())
                else:
                    optimizer.zero_grad()
                    loss = loss_task + loss_ewc_scaled
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()

                batches_count += 1

            epoch_train_loss = total_train_loss / max(1, batches_count)

            # Evaluate validation loss & representation metrics
            model.eval()
            with torch.no_grad():
                logits_val_a, out_val_b = model(X_val_t)
                val_loss_a = bce_loss_fn(logits_val_a, y_val_bin_t)
                val_loss_b = mse_loss_fn(out_val_b, y_val_cpi_t)
                val_ewc_loss = 0.0
                for name, param in model.named_parameters():
                    if name in fisher:
                        val_ewc_loss += (
                            fisher[name] * (param - param_old[name]) ** 2
                        ).sum()
                val_loss = (
                    0.7 * val_loss_a
                    + 0.3 * val_loss_b
                    + (self.lambda_ewc / 2.0) * val_ewc_loss
                )

                # 1. Effective Rank on validation hidden features
                hidden_acts = model.get_embeddings(X_val_t)
                erank = RepresentationAnalyzer.compute_effective_rank(hidden_acts)

                # 2. Taylor Residual Delta_t = |L_true(theta_t) - (L(theta*) + 0.5 * ewc_penalty)|
                logits_rep_a, out_rep_b = model(X_rep_t)
                current_true_L_A = (0.7 * bce_fn(logits_rep_a, y_rep_bin_t) + 0.3 * mse_fn(out_rep_b, y_rep_cpi_t)).item()
                taylor_approx_L_A = loss_A_star + val_ewc_loss.item()
                taylor_res = abs(current_true_L_A - taylor_approx_L_A)

                # 3. Generalization Gap: Loss(D_{A, test}) - Loss(M_A)
                if X_test_a_t is not None:
                    logits_test_a, out_test_b = model(X_test_a_t)
                    loss_test_a = (0.7 * bce_fn(logits_test_a, y_test_a_bin_t) + 0.3 * mse_fn(out_test_b, y_test_a_cpi_t)).item()
                    gen_gap = loss_test_a - current_true_L_A
                else:
                    gen_gap = 0.0

            model.train()

            avg_cos = float(np.mean(epoch_cos_sims)) if epoch_cos_sims else 0.0
            avg_conflict = float(np.mean(epoch_conflicts)) if epoch_conflicts else 0.0

            history_metrics["train_loss"].append(epoch_train_loss)
            history_metrics["val_loss"].append(float(val_loss.item()))
            history_metrics["cosine_similarity"].append(avg_cos)
            history_metrics["conflict_rate"].append(avg_conflict)
            history_metrics["effective_rank"].append(erank)
            history_metrics["taylor_residual"].append(taylor_res)
            history_metrics["generalization_gap"].append(gen_gap)

            if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:2d}: train_loss={epoch_train_loss:.4f}, val_loss={val_loss.item():.4f}, erank={erank:.2f}, cos={avg_cos:.3f}"
                )

        # Update Replay Buffer with chosen strategy
        self.replay_buffer.update(
            dataset_id,
            df_new,
            target_col,
            strategy=self.coreset_strategy,
            feature_cols=feature_cols,
            model=model,
            preprocessor=preprocessor,
            random_state=random_state,
        )

        prior_mtl_model.model = model
        final_pipeline = pipeline

        X_new_raw = df_new[feature_cols]
        y_new_bin = df_new[target_col]

        X_new_trans = preprocessor.transform(X_new_raw)
        if hasattr(X_new_trans, "toarray"):
            X_new_trans = X_new_trans.toarray()

        if not pd.api.types.is_numeric_dtype(y_new_bin):
            unique_classes = sorted(y_new_bin.dropna().unique())
            pos_label = (
                unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
            )
            y_new_bin_encoded = (y_new_bin == pos_label).astype(int)
        else:
            y_new_bin_encoded = y_new_bin

        self.last_history_metrics = history_metrics

        y_scores = prior_mtl_model.predict_proba(X_new_trans)[:, 1]
        from sklearn.metrics import precision_recall_curve, roc_auc_score

        precisions, recalls, thresholds = precision_recall_curve(
            y_new_bin_encoded.values, y_scores
        )
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        optimal_threshold = float(thresholds[f1_scores[:-1].argmax()])
        best_roc_auc = float(roc_auc_score(y_new_bin_encoded.values, y_scores))

        if return_history:
            return model, final_pipeline, best_roc_auc, optimal_threshold, history_metrics
        return model, final_pipeline, best_roc_auc, optimal_threshold
