import numpy as np
from typing import Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False
    # Fallback placeholders for static code analysis
    torch = None
    nn = None
    optim = None
    TensorDataset = None
    DataLoader = None

def is_mtl_available() -> bool:
    return _TORCH_AVAILABLE

if _TORCH_AVAILABLE:
    class _MTLPyTorchModel(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            # Shared encoder
            self.shared = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            # Head A (Binary classification)
            self.head_a = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            # Head B (Continuous CPI regression)
            self.head_b = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            shared_out = self.shared(x)
            logits_a = self.head_a(shared_out)
            out_b = self.head_b(shared_out)
            return logits_a, out_b

        def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
            return self.shared(x)
else:
    class _MTLPyTorchModel:
        def __init__(self, input_dim: int):
            pass

class MTLChurnModel:
    """Multi-Task Learning model with binary classification + regression heads."""
    _ALPHA = 0.7  # BCE weight (primary binary task)
    _BETA = 0.3   # MSE weight (CPI regression task)

    def __init__(self):
        self.model = None
        self.input_dim = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is not None:
            import io
            buf = io.BytesIO()
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_dim': self.input_dim
            }, buf)
            state['model_bytes'] = buf.getvalue()
            state['model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'model_bytes' in state:
            import io
            buf = io.BytesIO(state['model_bytes'])
            checkpoint = torch.load(buf, map_location='cpu')
            self.input_dim = checkpoint['input_dim']
            self.model = _MTLPyTorchModel(self.input_dim)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            del self.__dict__['model_bytes']

    def fit(self, X: np.ndarray, y_binary: np.ndarray, y_cpi: np.ndarray,
            epochs: int = 50, lr: float = 1e-3, batch_size: int = 64) -> "MTLChurnModel":
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to train MTLChurnModel.")

        X_t = torch.tensor(X, dtype=torch.float32)
        y_bin_t = torch.tensor(y_binary, dtype=torch.float32).unsqueeze(1)
        y_cpi_t = torch.tensor(y_cpi, dtype=torch.float32).unsqueeze(1)

        self.input_dim = X.shape[1]
        self.model = _MTLPyTorchModel(self.input_dim)
        self.model.train()

        dataset = TensorDataset(X_t, y_bin_t, y_cpi_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        bce_loss_fn = nn.BCEWithLogitsLoss()
        mse_loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            for batch_x, batch_y_bin, batch_y_cpi in dataloader:
                if len(batch_x) <= 1:
                    continue  # Avoid batch norm failure on batch size 1
                optimizer.zero_grad()
                logits_a, out_b = self.model(batch_x)
                loss_a = bce_loss_fn(logits_a, batch_y_bin)
                loss_b = mse_loss_fn(out_b, batch_y_cpi)
                loss = self._ALPHA * loss_a + self._BETA * loss_b
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MTLChurnModel inference.")
        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits_a, _ = self.model(X_t)
            prob_churn = torch.sigmoid(logits_a).cpu().numpy().ravel()

        prob_not_churn = 1.0 - prob_churn
        return np.column_stack([prob_not_churn, prob_churn])

    def get_shared_embeddings(self, X: np.ndarray) -> np.ndarray:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to extract embeddings.")
        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            embeddings = self.model.get_embeddings(X_t).cpu().numpy()
        return embeddings
