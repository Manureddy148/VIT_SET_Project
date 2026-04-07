import os
import json

import httpx
import plotly.graph_objects as go
import streamlit as st

API = os.getenv("MEDICAL_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Medical AI Severity", layout="wide")
st.title("Medical AI — Severity (demo)")


def render_result(out: dict) -> None:
    gauge_color = {
        "Low": "#16a34a",
        "Moderate": "#f59e0b",
        "High": "#f97316",
        "Critical": "#dc2626",
    }.get(out["severity_label"], "#2563eb")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=out["severity_score"],
                title={"text": f"Severity · {out['severity_label']}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 25], "color": "#dcfce7"},
                        {"range": [25, 50], "color": "#fef3c7"},
                        {"range": [50, 75], "color": "#fed7aa"},
                        {"range": [75, 100], "color": "#fecaca"},
                    ],
                },
            )
        )
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.write("Domain:", out["disease_domain"])
        st.write("Confidence:", f"{out['confidence']:.2f}")
        if out.get("audit_log_id"):
            st.caption(f"Audit log: {out['audit_log_id']}")

    with col2:
        shap_values = out.get("shap_values", {})
        if shap_values:
            ordered = sorted(shap_values.items(), key=lambda item: abs(item[1]), reverse=True)
            bar = go.Figure(
                go.Bar(
                    x=[item[1] for item in ordered],
                    y=[item[0] for item in ordered],
                    orientation="h",
                    marker_color=["#ef4444" if value > 0 else "#3b82f6" for _, value in ordered],
                )
            )
            bar.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), title="SHAP Risk Drivers")
            st.plotly_chart(bar, use_container_width=True)

    st.subheader("Clinical Report")
    st.write(out["explanation"])

    st.subheader("Safety")
    st.write("Flags:", ", ".join(out.get("safety_flags", [])) or "None")
    st.write("Faithfulness passed:", "Yes" if out.get("faithfulness_passed") else "No")
    if out.get("recommended_actions"):
        st.write("Recommended actions:")
        for item in out["recommended_actions"]:
            st.write(f"- {item}")
    if out.get("missing_feature_prompts"):
        st.write("Missing feature prompts:")
        for item in out["missing_feature_prompts"]:
            st.write(f"- {item}")

    st.subheader("Literature Citations")
    citations = out.get("literature_citations", [])
    if citations:
        for idx, citation in enumerate(citations, 1):
            label = citation.get("source", f"Citation {idx}")
            excerpt = citation.get("text", "")
            with st.expander(f"[{idx}] {label}"):
                st.write(excerpt)
    else:
        st.caption("No literature citations returned.")

    st.caption(out["disclaimer"])


def render_stream(query_text: str, clinical_data: dict) -> None:
    payload = {"query_text": query_text, "clinical_data": clinical_data}
    with httpx.stream("POST", f"{API}/predict/stream", json=payload, timeout=120.0) as response:
        response.raise_for_status()
        lines = []
        for line in response.iter_lines():
            if line:
                lines.append(line)
        if lines:
            st.code("\n".join(lines), language="text")

text_tab, image_tab, ecg_tab, audio_tab, genomics_tab = st.tabs(["Text", "Image", "ECG", "Audio", "Genomics"])

with text_tab:
    query = st.text_area(
        "Clinical narrative",
        value="Patient with glucose 210, BMI 34, age 52, frequent urination",
        key="text_query",
    )
    clinical_json = st.text_input(
        "Optional JSON clinical_data",
        value="{}",
        key="text_clinical_json",
    )

    if st.button("Run /predict"):
        try:
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
        render_result(out)
        with st.expander("Streamed report events"):
            render_stream(query, data)

with image_tab:
    image_query = st.text_input("Optional imaging note", value="Possible pneumonia chest x-ray", key="image_query")
    image_clinical_json = st.text_input("Optional JSON clinical_data", value='{"age": 67}', key="image_clinical_json")
    image_file = st.file_uploader("Upload image/DICOM file", type=["png", "jpg", "jpeg", "dcm"], key="image_file")

    if st.button("Run /predict/image"):
        if image_file is None:
            st.error("Upload an image file first")
            st.stop()
        try:
            json.loads(image_clinical_json or "{}")
        except json.JSONDecodeError:
            st.error("clinical_data must be valid JSON")
            st.stop()
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type or "application/octet-stream")}
        data = {"query_text": image_query, "clinical_json": image_clinical_json}
        try:
            r = httpx.post(f"{API}/predict/image", data=data, files=files, timeout=120.0)
            r.raise_for_status()
            out = r.json()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()
        render_result(out)

with ecg_tab:
    ecg_query = st.text_input("Optional ECG note", value="ECG waveform upload for heart triage", key="ecg_query")
    ecg_clinical_json = st.text_input("Optional JSON clinical_data", value='{"age": 61}', key="ecg_clinical_json")
    ecg_file = st.file_uploader("Upload ECG waveform file", type=["csv", "txt", "xml", "edf"], key="ecg_file")

    if st.button("Run /predict/ecg"):
        if ecg_file is None:
            st.error("Upload an ECG file first")
            st.stop()
        try:
            json.loads(ecg_clinical_json or "{}")
        except json.JSONDecodeError:
            st.error("clinical_data must be valid JSON")
            st.stop()
        files = {"file": (ecg_file.name, ecg_file.getvalue(), ecg_file.type or "text/plain")}
        data = {"query_text": ecg_query, "clinical_json": ecg_clinical_json}
        try:
            r = httpx.post(f"{API}/predict/ecg", data=data, files=files, timeout=120.0)
            r.raise_for_status()
            out = r.json()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()
        render_result(out)

with audio_tab:
    audio_query = st.text_input("Optional audio note", value="Lung sounds recording for respiratory triage", key="audio_query")
    audio_clinical_json = st.text_input("Optional JSON clinical_data", value='{"age": 59}', key="audio_clinical_json")
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3"], key="audio_file")

    if st.button("Run /predict/audio"):
        if audio_file is None:
            st.error("Upload an audio file first")
            st.stop()
        try:
            json.loads(audio_clinical_json or "{}")
        except json.JSONDecodeError:
            st.error("clinical_data must be valid JSON")
            st.stop()
        files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type or "audio/wav")}
        data = {"query_text": audio_query, "clinical_json": audio_clinical_json}
        try:
            r = httpx.post(f"{API}/predict/audio", data=data, files=files, timeout=120.0)
            r.raise_for_status()
            out = r.json()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()
        render_result(out)

with genomics_tab:
    genomics_query = st.text_input("Optional genomics note", value="VCF-based risk screening", key="genomics_query")
    genomics_clinical_json = st.text_input("Optional JSON clinical_data", value='{"age": 48}', key="genomics_clinical_json")
    genomics_file = st.file_uploader("Upload genomics file", type=["vcf", "fastq", "fq", "txt"], key="genomics_file")

    if st.button("Run /predict/genomics"):
        if genomics_file is None:
            st.error("Upload a genomics file first")
            st.stop()
        try:
            json.loads(genomics_clinical_json or "{}")
        except json.JSONDecodeError:
            st.error("clinical_data must be valid JSON")
            st.stop()
        files = {"file": (genomics_file.name, genomics_file.getvalue(), genomics_file.type or "text/plain")}
        data = {"query_text": genomics_query, "clinical_json": genomics_clinical_json}
        try:
            r = httpx.post(f"{API}/predict/genomics", data=data, files=files, timeout=120.0)
            r.raise_for_status()
            out = r.json()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()
        render_result(out)
