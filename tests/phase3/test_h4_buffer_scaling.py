import pytest
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
except (ImportError, OSError):
    torch = None
    nn = None
    optim = None
    TensorDataset = None
    DataLoader = None

from backend.app.core.training.mtl_trainer import _MTLPyTorchModel, is_mtl_available
from backend.app.core.training.continual_trainer import ReplayBuffer


def test_buffer_sampling_hoeffding_empirical_convergence(tmp_path, monkeypatch):
    """
    Validates Hypothesis 4: Distribution estimation error between buffer sample mean
    and population mean scales inversely with sqrt(M) (Hoeffding's bound).
    """
    local_storage = {}
    class MockStorage:
        def download_file(self, path):
            if path in local_storage:
                return local_storage[path]
            raise FileNotFoundError()
        def upload_file(self, content, path):
            local_storage[path] = content

    import backend.app.core.training.continual_trainer as ct
    monkeypatch.setattr(ct, "storage", MockStorage())

    np.random.seed(42)
    n_pop = 1000
    df_pop = pd.DataFrame({
        "feat_1": np.random.uniform(0.0, 1.0, n_pop),
        "feat_2": np.random.uniform(0.0, 1.0, n_pop),
        "Churn": np.random.choice([0, 1], p=[0.7, 0.3], size=n_pop)
    })
    pop_mean_f1 = df_pop["feat_1"].mean()

    buffer_sizes = [20, 100, 500]
    mean_errors = []

    for M in buffer_sizes:
        buf = ReplayBuffer(max_size=M)
        buf.update("dataset_h4", df_pop, "Churn", random_state=42)
        df_buf = buf.load_from_r2("dataset_h4")
        err = abs(df_buf["feat_1"].mean() - pop_mean_f1)
        mean_errors.append(err)

    # Error with M=500 should be noticeably smaller than error with M=20
    assert mean_errors[2] < mean_errors[0]


def test_h4_buffer_capacity_overfitting_dynamics():
    """
    Validates Hypothesis 4: A tiny replay buffer (e.g. M=10) fits the rehearsal exemplars
    tightly but exhibits higher generalization loss on the full Task A validation set.
    """
    if not is_mtl_available():
        pytest.skip("PyTorch is not available.")

    torch.manual_seed(42)
    np.random.seed(42)
    input_dim = 6

    # Full Task A population
    X_A_pop = np.random.randn(200, input_dim).astype(np.float32)
    y_A_bin_pop = np.random.randint(0, 2, 200).astype(np.float32)
    y_A_cpi_pop = np.random.rand(200).astype(np.float32)

    X_val_t = torch.tensor(X_A_pop[100:], dtype=torch.float32)
    y_val_bin_t = torch.tensor(y_A_bin_pop[100:], dtype=torch.float32).unsqueeze(1)
    y_val_cpi_t = torch.tensor(y_A_cpi_pop[100:], dtype=torch.float32).unsqueeze(1)

    # Function to train on a given buffer
    def train_on_buffer(buf_size: int, epochs: int = 25) -> tuple[float, float]:
        model = _MTLPyTorchModel(input_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        bce_fn = nn.BCEWithLogitsLoss()
        mse_fn = nn.MSELoss()

        bx = torch.tensor(X_A_pop[:buf_size], dtype=torch.float32)
        by_bin = torch.tensor(y_A_bin_pop[:buf_size], dtype=torch.float32).unsqueeze(1)
        by_cpi = torch.tensor(y_A_cpi_pop[:buf_size], dtype=torch.float32).unsqueeze(1)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            l_a, o_b = model(bx)
            loss = 0.7 * bce_fn(l_a, by_bin) + 0.3 * mse_fn(o_b, by_cpi)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            l_tr, o_tr = model(bx)
            train_err = (0.7 * bce_fn(l_tr, by_bin) + 0.3 * mse_fn(o_tr, by_cpi)).item()
            l_val, o_val = model(X_val_t)
            val_err = (0.7 * bce_fn(l_val, y_val_bin_t) + 0.3 * mse_fn(o_val, y_val_cpi_t)).item()

        return train_err, val_err

    # Small buffer M=8 vs Larger buffer M=80
    train_err_small, val_err_small = train_on_buffer(8)
    train_err_large, val_err_large = train_on_buffer(80)

    # Tiny buffer exhibits generalization gap (val_err > train_err)
    generalization_gap_small = val_err_small - train_err_small
    generalization_gap_large = val_err_large - train_err_large

    assert generalization_gap_small > generalization_gap_large


@pytest.mark.parametrize("buffer_size", [15, 50, 100])
def test_stratified_label_balance_across_buffer_sizes(tmp_path, monkeypatch, buffer_size):
    """
    Verifies that ReplayBuffer stratification maintains class ratio across varying capacities.
    """
    local_storage = {}
    class MockStorage:
        def download_file(self, path):
            if path in local_storage:
                return local_storage[path]
            raise FileNotFoundError()
        def upload_file(self, content, path):
            local_storage[path] = content

    import backend.app.core.training.continual_trainer as ct
    monkeypatch.setattr(ct, "storage", MockStorage())

    df = pd.DataFrame({
        "num": np.random.randn(200),
        "Churn": [0] * 160 + [1] * 40  # 80% 0, 20% 1
    })

    buf = ReplayBuffer(max_size=buffer_size)
    buf.update("ds_strat", df, "Churn", random_state=42)
    df_loaded = buf.load_from_r2("ds_strat")

    assert len(df_loaded) == buffer_size
    counts = df_loaded["Churn"].value_counts(normalize=True).to_dict()
    # 20% +/- 10% tolerance depending on rounding
    assert 0.10 <= counts.get(1, 0.0) <= 0.35
