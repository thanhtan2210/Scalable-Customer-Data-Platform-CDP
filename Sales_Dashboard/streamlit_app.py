import sys
from pathlib import Path
import os

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure project root in sys.path for src.config imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src import config as cfg
except Exception:
    # Fallback defaults if config not importable
    class cfg:
        MLFLOW_S3_ENDPOINT_URL = os.getenv(
            "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
        AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "admin")
        AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "password")
        PROCESSED_FEATURES_PATH = os.getenv(
            "PROCESSED_FEATURES_PATH", "s3://datalake/processed/features")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load processed features from MinIO/S3 or fallback to local sample.

    Cached for faster interactions.
    """
    s3_opts = {
        "key": cfg.AWS_ACCESS_KEY_ID,
        "secret": cfg.AWS_SECRET_ACCESS_KEY,
        "client_kwargs": {"endpoint_url": cfg.MLFLOW_S3_ENDPOINT_URL},
    }
    path = getattr(cfg, "PROCESSED_FEATURES_PATH",
                   "s3://datalake/processed/features")
    try:
        df = pd.read_parquet(path, storage_options=s3_opts)
        return df
    except Exception as e:
        st.warning(
            f"Cannot read from MinIO ({path}): {e}. Will try local data if available.")
        local_path = PROJECT_ROOT / "data" / "parquet" / "raw" / "telco_churn.parquet"
        if local_path.exists():
            try:
                return pd.read_parquet(local_path)
            except Exception as e2:
                st.error(f"Cannot read local data: {e2}")
        return pd.DataFrame()


def normalize_churn_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure churn column is numeric 0/1 if present."""
    out = df.copy()
    if "Churn" in out.columns:
        # Handle string Yes/No to 1/0
        if out["Churn"].dtype == object:
            out["Churn"] = out["Churn"].map({"Yes": 1, "No": 0})
        # Coerce to numeric
        out["Churn"] = pd.to_numeric(
            out["Churn"], errors="coerce").fillna(0).astype(int)
    return out


st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("Telco Churn Dashboard")

with st.sidebar:
    st.header("Filters")
    tenure_bin = st.multiselect("tenure_bin", options=None)
    monthly_bin = st.multiselect("monthly_bin", options=None)
    partner = st.selectbox("Partner", options=["All", 1, 0])
    dependents = st.selectbox("Dependents", options=["All", 1, 0])
    senior = st.selectbox("Senior Citizen", options=["All", 1, 0])
    st.divider()
    st.caption("Download filtered data")
    # placeholder for download button rendered after filtering

# Load and prepare data
_df = load_data()
if _df.empty:
    st.stop()

_df = normalize_churn_column(_df)

# Dynamic options for filters


def opts(col):
    return sorted([x for x in _df[col].dropna().unique().tolist()]) if col in _df.columns else []


if "tenure_bin" in _df.columns:
    tenure_bin = st.sidebar.multiselect(
        "tenure_bin", options=opts("tenure_bin"), default=[])
if "monthly_bin" in _df.columns:
    monthly_bin = st.sidebar.multiselect(
        "monthly_bin", options=opts("monthly_bin"), default=[])

# Apply filters
f = _df.copy()
if tenure_bin:
    f = f[f["tenure_bin"].isin(tenure_bin)]
if monthly_bin:
    f = f[f["monthly_bin"].isin(monthly_bin)]
if partner != "All" and "Partner" in f.columns:
    f = f[f["Partner"] == partner]
if dependents != "All" and "Dependents" in f.columns:
    f = f[f["Dependents"] == dependents]
if senior != "All" and "Senior Citizen" in f.columns:
    f = f[f["Senior Citizen"] == senior]

# KPIs
total = len(f)
churn_rate = f["Churn"].mean() if "Churn" in f.columns else float("nan")
col1, col2 = st.columns(2)
col1.metric("Total records", f"{total}")
col2.metric("Churn rate", f"{churn_rate:.2%}" if pd.notna(
    churn_rate) else "N/A")

# Breakdown by bins
st.subheader("Churn by tenure_bin")
if "tenure_bin" in f.columns and "Churn" in f.columns:
    tb = f.groupby("tenure_bin")["Churn"].mean().reset_index()
    tb = tb.sort_values("tenure_bin")
    st.bar_chart(tb.set_index("tenure_bin"))
else:
    st.info("Missing tenure_bin or Churn column.")

st.subheader("Churn by monthly_bin")
if "monthly_bin" in f.columns and "Churn" in f.columns:
    mb = f.groupby("monthly_bin")["Churn"].mean().reset_index()
    st.bar_chart(mb.set_index("monthly_bin"))
else:
    st.info("Missing monthly_bin or Churn column.")

st.subheader("Sample data")
st.dataframe(f.head(50))

# Additional analyses
st.subheader("Cost distribution (Monthly Charges / Total Charges)")
num_cols = [c for c in ["Monthly Charges", "Total Charges"] if c in f.columns]
if num_cols:
    for c in num_cols:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(f[c].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title(f"Distribution of {c}")
        st.pyplot(fig)
else:
    st.info("No valid cost columns found.")

st.subheader("Churn by Contract / Payment Method / Internet Service")
cat_cols = [
    c for c in ["Contract", "Payment Method", "Internet Service"] if c in f.columns
]
if "Churn" in f.columns and cat_cols:
    for c in cat_cols:
        g = f.groupby(c)["Churn"].mean().reset_index()
        g = g.sort_values("Churn", ascending=False)
        st.bar_chart(g.set_index(c))
else:
    st.info("Missing Contract/Payment/Internet Service or Churn column.")

# Data export
csv = f.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Download CSV",
    data=csv,
    file_name="filtered_telco.csv",
    mime="text/csv",
)

# Optional MLflow metrics panel
st.subheader("Model info (MLflow)")
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(
        getattr(cfg, "MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()
    # Try to find experiment by name
    exp = next((e for e in client.search_experiments(
        filter_string=f"name = 'CDP_Churn_Prediction'")), None)
    if exp:
        runs = client.search_runs([exp.experiment_id], order_by=[
                                  "attributes.start_time DESC"], max_results=1)
        if runs:
            r = runs[0]
            acc = r.data.metrics.get("accuracy")
            st.write({"run_id": r.info.run_id, "accuracy": acc,
                     "start_time": r.info.start_time})
        else:
            st.caption("No runs found for experiment 'CDP_Churn_Prediction'.")
    else:
        st.caption("Experiment 'CDP_Churn_Prediction' not found.")
except Exception as e:
    st.caption(f"Cannot fetch MLflow info: {e}")
