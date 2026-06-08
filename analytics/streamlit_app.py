import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Churn Analytics Hub", layout="wide")

API_URL = st.secrets.get("API_URL", "http://localhost:8000/api/v1")
HEADERS = {"X-API-Key": st.secrets.get("API_KEY", "test-api-key")}

st.sidebar.title("🔍 Analytics Hub")
page = st.sidebar.selectbox("Choose a report", ["Data Profiling", "Model Explainability", "Cohort Analysis"])

dataset_id = st.sidebar.text_input("Dataset ID", value="")

if not dataset_id:
    st.info("👈 Please enter a Dataset ID in the sidebar to begin analysis.")
else:
    if page == "Data Profiling":
        st.title("📊 Data Profiling Report")
        
        # Fetch Profiling data from Backend
        try:
            response = requests.get(f"{API_URL}/datasets/{dataset_id}/profile", headers=HEADERS)
            if response.status_code == 200:
                data = response.json()
                profiles = data['profiles']
                
                cols = st.columns(3)
                cols[0].metric("Total Rows", "10,240") # Mock for now
                cols[1].metric("Suggested Target", data['suggested_target'])
                cols[2].metric("Columns Dropped", len(data['warnings']))

                st.subheader("Feature Distributions")
                for p in profiles:
                    with st.expander(f"Column: {p['name']} ({p['inferred_role']})"):
                        c1, c2 = st.columns([1, 2])
                        c1.json(p['stats'])
                        # Mock plot
                        mock_df = pd.DataFrame({"val": [1, 2, 2, 3, 3, 3, 4, 5]})
                        fig = px.histogram(mock_df, x="val", title=f"Distribution of {p['name']}")
                        c2.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to fetch profiling data.")
        except Exception as e:
            st.error(f"Error: {e}")

    elif page == "Model Explainability":
        st.title("🧠 Model Explainability (SHAP)")
        st.write("Visualizing how features contribute to churn prediction.")
        # SHAP integration placeholder
        st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_diagram.png", width=600)
        
    elif page == "Cohort Analysis":
        st.title("👥 Cohort & Segment Analysis")
        # Cohort analysis logic
