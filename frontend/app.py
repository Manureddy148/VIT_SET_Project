import os

import httpx
import streamlit as st

API = os.getenv("MEDICAL_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Medical AI Severity", layout="wide")
st.title("Medical AI — Severity (demo)")

query = st.text_area(
    "Clinical narrative",
    value="Patient with glucose 210, BMI 34, age 52, frequent urination",
)
clinical_json = st.text_input(
    "Optional JSON clinical_data",
    value="{}",
)

if st.button("Run /predict"):
    try:
        import json

        data = json.loads(clinical_json or "{}")
    except json.JSONDecodeError:
        st.error("clinical_data must be valid JSON")
        st.stop()
    payload = {"query_text": query, "clinical_data": data}
    try:
        r = httpx.post(f"{API}/predict", json=payload, timeout=120.0)
        r.raise_for_status()
        out = r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        st.stop()

    st.metric("Severity", f"{out['severity_score']:.1f} / 100", out["severity_label"])
    st.write("Domain:", out["disease_domain"])
    st.write("Top factors:", ", ".join(out["top_risk_factors"]))
    st.subheader("Explanation")
    st.write(out["explanation"])
    st.caption(out["disclaimer"])
