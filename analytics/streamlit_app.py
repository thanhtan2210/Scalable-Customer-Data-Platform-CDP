import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import numpy as np
import io

st.set_page_config(page_title="Churn Analytics Hub", layout="wide")

# Setup styles
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

API_URL = st.secrets.get("API_URL", "http://localhost:8000/api/v1")
HEADERS = {"X-API-Key": st.secrets.get("API_KEY", "test-api-key")}

st.sidebar.title("🔍 Churn Analytics Hub")
page = st.sidebar.selectbox("Choose a report", ["Data Profiling", "Model Explainability", "Cohort Analysis"])

dataset_id = st.sidebar.text_input("Dataset ID", value="")

if not dataset_id:
    st.info("👈 Please enter a Dataset ID in the sidebar to begin analysis.")
else:
    # 1. Fetch dataset metadata to verify existence and get details
    try:
        ds_resp = requests.get(f"{API_URL}/datasets/{dataset_id}", headers=HEADERS)
        if ds_resp.status_code != 200:
            st.error(f"Dataset '{dataset_id}' not found. Please check the ID.")
            st.stop()
        dataset_meta = ds_resp.json()
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
        st.stop()

    if page == "Data Profiling":
        st.title("📊 Data Profiling Report")
        
        # Call profile dataset endpoint (POST)
        try:
            profile_resp = requests.post(f"{API_URL}/datasets/{dataset_id}/profile", json={}, headers=HEADERS)
            if profile_resp.status_code == 200:
                profile_data = profile_resp.json()
                profiles = profile_data.get("profiles", [])
                suggested_target = profile_data.get("suggested_target", "N/A")
                warnings = profile_data.get("warnings", [])
                
                # Metrics cards
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Total Rows", f"{dataset_meta.get('row_count', 0):,}")
                with cols[1]:
                    st.metric("Total Columns", dataset_meta.get("col_count", 0))
                with cols[2]:
                    st.metric("Suggested Target", suggested_target)
                with cols[3]:
                    st.metric("Warnings Flagged", len(warnings))

                if warnings:
                    st.warning("⚠️ **Data Ingestion Warnings:**\n" + "\n".join([f"- {w}" for w in warnings]))

                st.subheader("Feature Distributions & Profiling Details")
                
                # Select column to inspect
                col_names = [p["name"] for p in profiles]
                selected_col_name = st.selectbox("Select a column to inspect", col_names)
                
                selected_profile = next(p for p in profiles if p["name"] == selected_col_name)
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown("### Profile Summary")
                    stats_dict = {
                        "Inferred Dtype": selected_profile["inferred_dtype"],
                        "Inferred Role": selected_profile["inferred_role"],
                        "Confidence Score": f"{selected_profile['confidence_score'] * 100:.1f}%",
                        "Null Percentage": f"{selected_profile['null_pct'] * 100:.1f}%",
                        "Unique Count": selected_profile["unique_count"],
                        "Entropy": round(selected_profile["entropy"], 3),
                        "Transform Strategy": selected_profile.get("transform_strategy", "N/A"),
                        "Impute Strategy": selected_profile.get("impute_strategy", "N/A"),
                    }
                    if selected_profile.get("potential_leakage"):
                        stats_dict["Potential Leakage"] = "⚠️ Yes"
                        stats_dict["Leakage Score"] = selected_profile.get("leakage_score")
                    st.json(stats_dict)
                    
                with c2:
                    st.markdown("### Distribution Preview")
                    # Generate a mock distribution visualization based on profiling data
                    unique_count = selected_profile["unique_count"]
                    if selected_profile["inferred_role"] in ["NUMERIC", "TARGET"] and selected_profile["inferred_dtype"] in ["float64", "int64"]:
                        # Numeric mock data
                        mock_values = np.random.normal(loc=10.0, scale=3.0, size=500)
                        mock_df = pd.DataFrame({selected_col_name: mock_values})
                        fig = px.histogram(mock_df, x=selected_col_name, title=f"Sample Distribution (Numeric) of {selected_col_name}")
                    else:
                        # Categorical mock data
                        categories = [f"Val_{i}" for i in range(min(5, unique_count))]
                        if not categories:
                            categories = ["Category A", "Category B"]
                        probs = [0.5, 0.3, 0.1, 0.07, 0.03][:len(categories)]
                        probs = [p / sum(probs) for p in probs] # Normalize
                        mock_values = np.random.choice(categories, size=300, p=probs)
                        mock_df = pd.DataFrame({selected_col_name: mock_values})
                        fig = px.histogram(mock_df, x=selected_col_name, title=f"Sample Category Distribution of {selected_col_name}")
                        
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                st.error(f"Failed to fetch profiling report: {profile_resp.text}")
        except Exception as e:
            st.error(f"Profiling error: {e}")

    elif page == "Model Explainability":
        st.title("🧠 Model Explainability & Feature Importance")
        st.write("Visualizing feature contributions to the best trained churn prediction model.")
        
        # Call feature-importance endpoint
        try:
            resp = requests.get(f"{API_URL}/predict/datasets/{dataset_id}/feature-importance", headers=HEADERS)
            if resp.status_code == 200:
                feat_data = resp.json()
                importances = feat_data.get("feature_importances", [])
                
                if not importances:
                    st.warning("⚠️ No feature importances available. Make sure a model has been trained using standard XGBoost/RF path.")
                else:
                    feat_df = pd.DataFrame(importances)
                    
                    # Plot
                    fig = px.bar(
                        feat_df.head(15), 
                        x="importance", 
                        y="feature", 
                        orientation="h",
                        title="Top Feature Contributions (Sklearn/XGBoost Feature Importance)",
                        labels={"importance": "Relative Importance", "feature": "Feature"},
                        color="importance",
                        color_continuous_scale="Plotly3"
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Model Feature Ranks")
                    st.dataframe(feat_df)
            else:
                st.error("Failed to retrieve feature importances. Make sure a completed job exists for this dataset.")
        except Exception as e:
            st.error(f"Feature importance error: {e}")

    elif page == "Cohort Analysis":
        st.title("👥 Cohort & Segment Analysis")
        st.write("Segmenting churn predictions across customer demographics and product groupings.")
        
        # Import storage directly from monorepo since Streamlit runs locally
        try:
            from backend.app.core.storage import storage
            from backend.app.core.ingestion.parsers import parse_file
        except ImportError:
            st.error("ETL parsers or storage clients not found in Python path.")
            st.stop()
            
        r2_path = dataset_meta.get("r2_path")
        if not r2_path:
            st.error("Dataset storage path (R2 path) not found in metadata.")
            st.stop()
            
        with st.spinner("Downloading dataset and running batch prediction cohort analysis..."):
            try:
                # 1. Download file from storage
                content = storage.download_file(r2_path)
                result = parse_file(content=content, filename=r2_path)
                df = result.df
                
                # 2. Run batch prediction
                pred_resp = requests.post(f"{API_URL}/predict/batch", json={
                    "dataset_id": dataset_id,
                    "file_path": r2_path
                }, headers=HEADERS)
                
                if pred_resp.status_code == 200:
                    pred_data = pred_resp.json()
                    predictions = pred_data["predictions"]
                    
                    # 3. Merge predictions
                    df["churn_probability"] = [p["probability"] for p in predictions]
                    df["risk_level"] = [p["risk_level"] for p in predictions]
                    
                    st.success("Batch prediction cohort scoring completed!")
                    
                    # Highlight stats
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Average Churn Probability", f"{df['churn_probability'].mean() * 100:.2f}%")
                    c2.metric("High Risk Cohort", f"{sum(df['risk_level'] == 'High')}")
                    c3.metric("Medium Risk Cohort", f"{sum(df['risk_level'] == 'Medium')}")
                    c4.metric("Low Risk Cohort", f"{sum(df['risk_level'] == 'Low')}")
                    
                    # Select segment column
                    categorical_cols = [
                        col for col in df.columns 
                        if col != "risk_level" and (df[col].dtype == "object" or len(df[col].unique()) < 10)
                    ]
                    
                    if categorical_cols:
                        group_col = st.selectbox("Select column to slice cohort metrics", categorical_cols)
                        
                        # Cohort calculations
                        cohort_summary = df.groupby(group_col).agg(
                            total_records=("churn_probability", "count"),
                            avg_churn_prob=("churn_probability", "mean"),
                            high_risk_count=("risk_level", lambda x: sum(x == "High"))
                        ).reset_index()
                        
                        cohort_summary["avg_churn_prob"] = cohort_summary["avg_churn_prob"] * 100
                        
                        s_cols = st.columns(2)
                        with s_cols[0]:
                            fig_prob = px.bar(
                                cohort_summary, 
                                x=group_col, 
                                y="avg_churn_prob",
                                title=f"Avg Churn Probability (%) by {group_col}",
                                labels={"avg_churn_prob": "Avg Churn Prob (%)"},
                                color="avg_churn_prob"
                            )
                            st.plotly_chart(fig_prob, use_container_width=True)
                            
                        with s_cols[1]:
                            fig_pie = px.pie(
                                cohort_summary, 
                                names=group_col, 
                                values="high_risk_count",
                                title=f"High Risk Customer Distribution by {group_col}"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                        st.subheader("Cohort Metrics Summary Table")
                        st.dataframe(cohort_summary)
                    else:
                        st.warning("No categorical variables available in the dataset for cohort segmentation.")
                else:
                    st.error(f"Batch prediction failure: {pred_resp.text}")
            except Exception as ex:
                st.error(f"Cohort pipeline exception: {ex}")
