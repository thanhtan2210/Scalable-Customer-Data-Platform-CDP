# Smoke Test Checklist

## 1. Backend (Render)
- [ ] `GET /health` returns `{"status": "alive"}`
- [ ] `X-API-Key` is correctly enforced (403 if missing)
- [ ] Database connection is active (check logs for Supabase)

## 2. Frontend (Vercel)
- [ ] Page loads at root `/`
- [ ] Drag & Drop zone is visible
- [ ] `VITE_API_URL` points to Render URL

## 3. Analytics (Streamlit Cloud)
- [ ] Choose a report sidebar is functional
- [ ] Metric cards show data when Dataset ID is provided
- [ ] Plotly charts render distribution

## 4. Storage & Registry
- [ ] R2: Check Cloudflare dashboard for new files in `raw/`
- [ ] MLflow: Check DagsHub experiment tab for new runs
- [ ] Model Registry: Best model registered as `GenericChurnModel`

## 5. End-to-End
- [ ] Run `python tests/e2e_test.py` against Cloud URL
- [ ] All 3 datasets (Telco, Bank, HR) show `status: success`
