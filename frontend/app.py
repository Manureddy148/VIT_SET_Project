"""
Streamlit UI for the open-source medical AI pipeline (hybrid rules + HF + local RAG).
Run from repo root: streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("MODEL_REGISTRY_PATH", str(ROOT / "configs" / "model_registry.json"))
os.environ.setdefault("KNOWLEDGE_BASE_PATH", str(ROOT / "configs" / "knowledge_base.json"))

from medical_ai.core.pipeline import run_full_analyze
from medical_ai.core.schemas import AnalyzeRequest


def main() -> None:
    st.set_page_config(page_title="Medical AI (open-source inference)", layout="wide")
    st.title("Medical AI — hybrid pipeline")
    st.caption("Open-source inference only (no custom training). Not a medical device.")

    with st.sidebar:
        st.subheader("Registry")
        reg_path = st.text_input("model_registry.json", value=os.environ["MODEL_REGISTRY_PATH"])
        kb_path = st.text_input("knowledge_base.json", value=os.environ["KNOWLEDGE_BASE_PATH"])
        disable_hf = st.checkbox("Disable HF downloads (MEDAI_DISABLE_HF=1)", value=True)
        if disable_hf:
            os.environ["MEDAI_DISABLE_HF"] = "1"
        else:
            os.environ.pop("MEDAI_DISABLE_HF", None)

        try:
            payload = json.loads(Path(reg_path).read_text(encoding="utf-8"))
            st.json({"domains": list((payload.get("domains") or {}).keys())})
        except Exception as e:
            st.warning(str(e))

    query = st.text_area("Query / symptoms", height=120)
    domain_hint = st.text_input("Domain hint (optional)", "")
    image_path = st.text_input("Image path (optional, for imaging domains)", "")
    modality = st.text_input("Modality (optional: xray|mri|derm|...)", "")

    structured_json = st.text_area(
        'Structured input JSON (optional), e.g. {"glucose": 210, "bmi": 34}',
        value="{}",
        height=80,
    )

    if st.button("Run analyze"):
        try:
            structured = json.loads(structured_json or "{}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return

        if not query.strip():
            st.error("Enter a query.")
            return

        req = AnalyzeRequest(
            query_text=query,
            domain_hint=domain_hint.strip() or None,
            structured_input=structured,
            image_path=image_path.strip() or None,
            modality=modality.strip() or None,
        )
        try:
            out = run_full_analyze(req, registry_path=reg_path, knowledge_path=kb_path)
        except Exception as e:
            st.error(str(e))
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("Severity", f"{out['severity_score']}")
        c2.metric("Confidence", f"{out['confidence']}")
        c3.metric("Disagreement", f"{out['disagreement']}")

        st.subheader("Orchestrator")
        st.json(out.get("orchestrator", {}))

        st.subheader("Report")
        st.write(out["report"]["summary"])
        st.info(out["report"].get("grounding_note", ""))
        st.markdown(out["report"]["explanation"].replace("\n", "\n\n"))

        st.subheader("Citations")
        for c in out["report"].get("citations", []):
            st.write(f"**{c.get('title')}** ({c.get('source')}) — score {c.get('score')}")
            st.caption(c.get("snippet", ""))

        st.subheader("Model runs")
        st.json(out.get("runs", []))


if __name__ == "__main__":
    main()
