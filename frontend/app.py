"""
Medical AI Severity — Streamlit UI (Layer 7).
Run API first: python launch_api.py
Then: streamlit run frontend/app.py
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import httpx
import pandas as pd
import streamlit as st

API = os.getenv("MEDICAL_API_URL", "http://127.0.0.1:8000")

PRESETS: Dict[str, Dict[str, Any]] = {
    "Diabetes (high glucose)": {
        "query_text": "Patient with glucose 210 mg/dL, BMI 34, age 52, frequent urination and fatigue.",
        "clinical_data": {"glucose": 210, "bmi": 34, "age": 52},
    },
    "Heart (chest pain + labs)": {
        "query_text": "Chest pressure on exertion, history of high cholesterol.",
        "clinical_data": {"cholesterol": 280, "resting_bp": 152, "max_hr": 128, "age": 61},
    },
    "Pneumonia (hypoxia)": {
        "query_text": "Fever, productive cough, oxygen saturation dropped overnight.",
        "clinical_data": {
            "spo2": 89,
            "temperature_c": 38.9,
            "respiratory_rate": 26,
            "crp": 110,
            "age": 67,
        },
    },
    "Structured only (infer diabetes)": {
        "query_text": "",
        "clinical_data": {"glucose": 198, "bmi": 32.5, "age": 45},
    },
}


def _severity_color(label: str) -> str:
    return {
        "Low": "#10b981",
        "Moderate": "#f59e0b",
        "High": "#f97316",
        "Critical": "#ef4444",
    }.get(label, "#64748b")


def check_api(base: str) -> tuple[bool, str]:
    try:
        r = httpx.get(f"{base.rstrip('/')}/health", timeout=5.0)
        if r.status_code == 200 and r.json().get("status") == "ok":
            return True, "Connected"
        return False, f"HTTP {r.status_code}"
    except httpx.ConnectError:
        return False, "Connection refused — start the API: python launch_api.py"
    except Exception as e:
        return False, str(e)


st.set_page_config(
    page_title="Medical AI Severity",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 1.75rem; font-weight: 700; color: #0f172a; margin-bottom: 0.25rem; }
    .subtle { color: #64748b; font-size: 0.95rem; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    .disclaimer-box {
        background: #fffbeb; border-left: 4px solid #f59e0b;
        padding: 0.75rem 1rem; border-radius: 6px; margin-top: 1rem;
        font-size: 0.9rem; color: #78350f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### API")
    api_url = st.text_input("Base URL", value=API, help="FastAPI server root")
    ok, msg = check_api(api_url)
    if ok:
        st.success(msg)
    else:
        st.error(msg)
    st.markdown("---")
    st.markdown("### Quick load")
    preset = st.selectbox("Example case", list(PRESETS.keys()))
    st.caption("Research / demo only — not for clinical use.")

st.markdown('<p class="main-header">Medical AI — Severity & explanation</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtle">7-layer pipeline demo · routes to diabetes / heart / pneumonia models · SHAP-style drivers · optional RAG citations</p>',
    unsafe_allow_html=True,
)

p = PRESETS[preset]
query = st.text_area("Clinical narrative", value=p.get("query_text", ""), height=120)
clinical_json = st.text_area(
    "clinical_data (JSON)",
    value=json.dumps(p.get("clinical_data", {}), indent=2),
    height=140,
)

c1, c2, c3 = st.columns([1, 1, 1])
run = c1.button("Run prediction", type="primary", use_container_width=True)
if c2.button("Reset to preset", use_container_width=True):
    st.session_state.pop("last_result", None)
    st.rerun()

if run:
    try:
        clinical = json.loads(clinical_json or "{}")
        if not isinstance(clinical, dict):
            raise ValueError("clinical_data must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    payload = {"query_text": query, "clinical_data": clinical}
    try:
        with st.spinner("Calling /predict …"):
            r = httpx.post(f"{api_url.rstrip('/')}/predict", json=payload, timeout=120.0)
        if r.status_code == 422:
            st.warning("Could not infer disease domain. Add keywords (diabetes, heart, pneumonia, glucose, chest pain, cough, SpO2, etc.).")
            st.json(r.json())
            st.stop()
        r.raise_for_status()
        out = r.json()
    except httpx.ConnectError:
        st.error("Cannot reach API. In a terminal run: `python launch_api.py` and keep it open.")
        st.stop()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:500]}")
        st.stop()
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.session_state["last_result"] = out

if "last_result" in st.session_state:
    out = st.session_state["last_result"]
    label = out["severity_label"]
    score = float(out["severity_score"])

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Severity score", f"{score:.1f} / 100")
    m2.metric("Risk band", label)
    m3.metric("Domain", out["disease_domain"].replace("_", " "))
    m4.metric("Confidence", f"{out['confidence']:.2f}")

    hue = _severity_color(label)
    st.markdown(
        f"<p style='margin:0 0 0.5rem 0;font-weight:600;color:{hue};'>{label} risk band</p>",
        unsafe_allow_html=True,
    )
    st.progress(min(1.0, max(0.0, score / 100.0)))

    col_left, col_right = st.columns((1, 1))

    with col_left:
        st.subheader("Top risk drivers (model attribution)")
        shap = out.get("shap_values") or {}
        if shap:
            df = pd.DataFrame([{"feature": k, "SHAP": float(v)} for k, v in shap.items()])
            df["_abs"] = df["SHAP"].abs()
            df = df.sort_values("_abs", ascending=False).drop(columns=["_abs"])
            st.bar_chart(df.set_index("feature"))
        else:
            st.info("No SHAP values in response (stub model).")

        miss = out.get("missing_features") or []
        if miss:
            st.caption(f"Missing / imputed fields noted: {', '.join(miss)}")

    with col_right:
        st.subheader("Literature citations")
        cites = out.get("literature_citations") or []
        if not cites:
            st.caption("No citations (RAG off or empty store). Set `MEDICAL_AI_SKIP_RAG=0` and install `requirements-rag.txt` for Chroma.")
        else:
            for i, c in enumerate(cites[:6], 1):
                with st.expander(f"[{i}] {c.get('source', 'Source')}"):
                    st.write(c.get("text", "")[:1200])

    st.subheader("Clinical explanation")
    st.write(out.get("explanation", ""))

    st.markdown(
        f'<div class="disclaimer-box"><strong>Disclaimer</strong><br/>{out.get("disclaimer", "")}</div>',
        unsafe_allow_html=True,
    )

else:
    st.info("Choose an example or enter a case, then click **Run prediction**.")
