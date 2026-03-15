import sys
from pathlib import Path
import os

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure project root in sys.path for src.config imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    """Load processed features from local storage or MinIO/S3.

    Cached for faster interactions.
    """
    # 1. Try Local First (since Docker might be down)
    local_path = PROJECT_ROOT / "data" / "parquet" / "processed" / "cleaned_telco.parquet"
    if local_path.exists():
        try:
            return pd.read_parquet(local_path)
        except Exception as e:
            st.error(f"Cannot read local data: {e}")

    # 2. Try MinIO/S3
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
            f"Cannot read from MinIO ({path}): {e}. Local fallback failed too.")
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
st.title("📊 Telco Customer Churn Dashboard")

# Standardized column names from ETL
COL_MAP = {
    "MonthlyCharges": "Phí hàng tháng (USD)",
    "TotalCharges": "Tổng cước tích lũy (USD)",
    "SeniorCitizen": "Người cao tuổi (65+)",
    "InternetService": "Dịch vụ Internet",
    "tenure": "Số tháng sử dụng",
    "Churn": "Rời bỏ dịch vụ"
}

with st.sidebar:
    st.header("🔍 Bộ lọc dữ liệu")
    st.markdown("---")
    tenure_bin_sel = st.multiselect("Nhóm số tháng sử dụng (Tenure)", options=[])
    monthly_bin_sel = st.multiselect("Nhóm phí hàng tháng", options=[])
    partner_sel = st.selectbox("Có bạn đời (Partner)", options=["All", 1, 0])
    dependents_sel = st.selectbox("Có người phụ thuộc", options=["All", 1, 0])
    senior_sel = st.selectbox("Người cao tuổi", options=["All", 1, 0])
    st.divider()
    st.caption("📥 Tải dữ liệu đã lọc")

# Load and prepare data
_df = load_data()
if _df.empty:
    st.error("Không tìm thấy dữ liệu để hiển thị.")
    st.stop()

_df = normalize_churn_column(_df)

# Dictionary explanation
with st.expander("📖 Chú thích ý nghĩa các cột dữ liệu"):
    st.markdown("""
    - **Churn (Rời bỏ):** 1 = Khách hàng đã rời đi, 0 = Khách hàng ở lại.
    - **Tenure (Số tháng sử dụng):** Thời gian khách hàng đã gắn bó với công ty.
    - **Monthly Charges:** Số tiền khách hàng trả mỗi tháng.
    - **Total Charges:** Tổng số tiền khách hàng đã trả từ trước đến nay.
    - **Partner/Dependents:** Tình trạng gia đình của khách hàng.
    - **CLTV (Customer Lifetime Value):** Giá trị vòng đời khách hàng (dự đoán lợi nhuận).
    """)

# Dynamic options for filters
def opts(col):
    return sorted([x for x in _df[col].dropna().unique().tolist()]) if col in _df.columns else []

if "tenure_bin" in _df.columns:
    tenure_bin_sel = st.sidebar.multiselect(
        "Nhóm số tháng sử dụng (Tenure)", options=opts("tenure_bin"), key="tenure_sel")
if "monthly_bin" in _df.columns:
    monthly_bin_sel = st.sidebar.multiselect(
        "Nhóm phí hàng tháng", options=opts("monthly_bin"), key="monthly_sel")

# Apply filters
f = _df.copy()
if tenure_bin_sel:
    f = f[f["tenure_bin"].isin(tenure_bin_sel)]
if monthly_bin_sel:
    f = f[f["monthly_bin"].isin(monthly_bin_sel)]
if partner_sel != "All" and "Partner" in f.columns:
    f = f[f["Partner"] == partner_sel]
if dependents_sel != "All" and "Dependents" in f.columns:
    f = f[f["Dependents"] == dependents_sel]
if senior_sel != "All" and "SeniorCitizen" in f.columns:
    f = f[f["SeniorCitizen"] == senior_sel]

# KPIs
total = len(f)
churn_rate = f["Churn"].mean() if "Churn" in f.columns else float("nan")
col1, col2 = st.columns(2)
col1.metric("Tổng số khách hàng", f"{total}")
col2.metric("Tỷ lệ rời bỏ (Churn Rate)", f"{churn_rate:.2%}" if pd.notna(
    churn_rate) else "N/A", help="Tỷ lệ trung bình khách hàng rời bỏ trong nhóm này.")

st.markdown("---")

# Breakdown by bins
c1, c2 = st.columns(2)

with c1:
    st.subheader("📈 Tỷ lệ Churn theo thời gian gắn bó")
    st.info("Biểu đồ này cho biết nhóm khách hàng nào (mới dùng hay dùng lâu) có nguy cơ rời bỏ cao nhất.")
    if "tenure_bin" in f.columns and "Churn" in f.columns:
        tb = f.groupby("tenure_bin")["Churn"].mean().reset_index()
        tb = tb.sort_values("tenure_bin")
        st.bar_chart(tb.set_index("tenure_bin"))
    else:
        st.warning("Thiếu dữ liệu 'tenure_bin' hoặc 'Churn'.")

with c2:
    st.subheader("💰 Tỷ lệ Churn theo mức phí hàng tháng")
    st.info("Mức phí cao hay thấp ảnh hưởng thế nào đến quyết định rời bỏ của khách hàng?")
    if "monthly_bin" in f.columns and "Churn" in f.columns:
        mb = f.groupby("monthly_bin")["Churn"].mean().reset_index()
        st.bar_chart(mb.set_index("monthly_bin"))
    else:
        st.warning("Thiếu dữ liệu 'monthly_bin' hoặc 'Churn'.")

st.markdown("---")

# Additional analyses
st.subheader("📊 Phân bổ chi phí của khách hàng")
st.write("Biểu đồ mật độ cho thấy đa số khách hàng đang trả mức phí bao nhiêu.")
num_cols = [c for c in ["MonthlyCharges", "TotalCharges"] if c in f.columns]
if num_cols:
    cols = st.columns(len(num_cols))
    for idx, c in enumerate(num_cols):
        display_name = COL_MAP.get(c, c)
        with cols[idx]:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(f[c].dropna(), bins=30, kde=True, ax=ax, color="skyblue")
            ax.set_title(f"Phân bổ: {display_name}")
            st.pyplot(fig)
else:
    st.info("Không tìm thấy dữ liệu về chi phí.")

st.markdown("---")

st.subheader("🏢 Tỷ lệ Churn theo loại dịch vụ & Hợp đồng")
st.write("So sánh tỷ lệ rời bỏ giữa các nhóm đặc tính khác nhau.")
cat_cols = [
    c for c in ["Contract", "Payment Method", "InternetService"] if c in f.columns
]
if "Churn" in f.columns and cat_cols:
    for c in cat_cols:
        display_name = COL_MAP.get(c, c)
        g = f.groupby(c)["Churn"].mean().reset_index()
        g = g.sort_values("Churn", ascending=False)
        st.write(f"**Phân tích theo: {display_name}**")
        st.bar_chart(g.set_index(c))
else:
    st.info("Thiếu dữ liệu đặc tính hoặc dữ liệu Churn.")

st.markdown("---")

st.subheader("📋 Danh sách mẫu dữ liệu (Top 50)")
st.dataframe(f.head(50))

# Data export
csv = f.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="📥 Tải xuống CSV",
    data=csv,
    file_name="filtered_telco_churn.csv",
    mime="text/csv",
)

# Optional MLflow metrics panel
st.subheader("🤖 Thông tin Mô hình Dự đoán (MLflow)")
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(
        getattr(cfg, "MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()
    exp = next((e for e in client.search_experiments(
        filter_string="name = 'CDP_Churn_Prediction'")), None)
    if exp:
        runs = client.search_runs([exp.experiment_id], order_by=[
                                  "attributes.start_time DESC"], max_results=1)
        if runs:
            r = runs[0]
            acc = r.data.metrics.get("accuracy")
            st.success("Mô hình mới nhất đang hoạt động!")
            st.json({
                "Run ID": r.info.run_id,
                "Độ chính xác (Accuracy)": acc,
                "Thời gian huấn luyện": pd.to_datetime(r.info.start_time, unit='ms').strftime('%Y-%m-%d %H:%M')
            })
        else:
            st.caption("Chưa có lượt huấn luyện nào được ghi nhận.")
    else:
        st.caption("Không tìm thấy Experiment 'CDP_Churn_Prediction'.")
except Exception as e:
    st.caption(f"Không thể kết nối với MLflow: {e}")
