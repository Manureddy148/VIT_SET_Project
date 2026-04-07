import re
from io import BytesIO
from typing import Dict, Tuple

import numpy as np

from src.preprocessing.ecg_processor import process_ecg_signal


def detect_modality(filename: str, content_type: str | None = None) -> str:
    name = (filename or "").lower()
    ctype = (content_type or "").lower()
    if ctype.startswith("image/") or name.endswith((".png", ".jpg", ".jpeg", ".dcm")):
        return "image"
    if "ecg" in name or name.endswith((".csv", ".txt", ".edf", ".xml")):
        return "ecg"
    if "audio" in name or name.endswith((".wav", ".mp3")):
        return "audio"
    if "vcf" in name or name.endswith((".vcf", ".fastq", ".fq")):
        return "genomics"
    return "unknown"


def image_upload_to_clinical_features(
    filename: str,
    payload: bytes,
    query_text: str,
    base_clinical: Dict[str, float],
) -> Tuple[str, Dict[str, float], str]:
    if not payload:
        raise ValueError("Uploaded image file is empty.")

    values = np.frombuffer(payload[:4096], dtype=np.uint8)
    if filename.lower().endswith(".dcm"):
        try:
            import pydicom

            ds = pydicom.dcmread(BytesIO(payload))
            arr = ds.pixel_array.astype(np.float32)
            if arr.size:
                values = np.clip(arr.flatten(), 0, 255).astype(np.uint8)
        except Exception:
            pass

    mean_val = float(values.mean()) if len(values) else 127.0
    std_val = float(values.std()) if len(values) else 0.0
    opacity_score = max(0.0, min(1.0, (160.0 - mean_val) / 160.0))
    texture_score = max(0.0, min(1.0, std_val / 90.0))

    clinical = dict(base_clinical)
    clinical.setdefault("spo2", round(97.0 - opacity_score * 10.0 - texture_score * 2.0, 1))
    clinical.setdefault("temperature_c", round(37.1 + texture_score * 2.0, 1))
    clinical.setdefault("respiratory_rate", round(18.0 + opacity_score * 8.0 + texture_score * 3.0, 1))
    clinical.setdefault("crp", round(8.0 + opacity_score * 95.0 + texture_score * 20.0, 1))

    summary = (
        f"Imaging upload '{filename}' processed as chest imaging. "
        f"Derived features suggest opacity={opacity_score:.2f}, texture={texture_score:.2f}."
    )
    merged_query = " ".join(part for part in [query_text.strip(), "chest xray pneumonia imaging", summary] if part)
    return "pneumonia", clinical, merged_query


def ecg_upload_to_clinical_features(
    filename: str,
    payload: bytes,
    query_text: str,
    base_clinical: Dict[str, float],
) -> Tuple[str, Dict[str, float], str]:
    if not payload:
        raise ValueError("Uploaded ECG file is empty.")

    text = payload.decode("utf-8", errors="ignore")
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    signal = np.array([float(m) for m in matches[:5000]], dtype=np.float64)
    if signal.size == 0:
        signal = np.array([0.0], dtype=np.float64)

    ecg_features = process_ecg_signal(signal)
    mean_abs = ecg_features["signal_mean_abs"]
    std_val = ecg_features["signal_std"]
    peak_factor = ecg_features["signal_peak"]
    estimated_hr = int(ecg_features["heart_rate"])
    clinical = dict(base_clinical)
    clinical.setdefault("max_hr", float(estimated_hr))
    clinical.setdefault("resting_bp", round(120.0 + mean_abs * 12.0 + std_val * 4.0, 1))
    clinical.setdefault("cholesterol", round(210.0 + std_val * 18.0, 1))
    clinical.setdefault("st_depression", round(min(6.0, mean_abs * 0.06 + std_val * 0.03), 2))
    clinical.setdefault("major_vessels", float(1 if peak_factor > 3.0 else 0))

    summary = (
        f"ECG upload '{filename}' processed as waveform data. "
        f"Estimated heart rate={estimated_hr} bpm, variability={ecg_features['hrv']:.3f}."
    )
    merged_query = " ".join(part for part in [query_text.strip(), "heart ecg cardiac waveform", summary] if part)
    return "heart_disease", clinical, merged_query


def audio_upload_to_clinical_features(
    filename: str,
    payload: bytes,
    query_text: str,
    base_clinical: Dict[str, float],
) -> Tuple[str, Dict[str, float], str]:
    if not payload:
        raise ValueError("Uploaded audio file is empty.")
    values = np.frombuffer(payload[:8192], dtype=np.uint8)
    loudness = float(values.mean()) if len(values) else 0.0
    variability = float(values.std()) if len(values) else 0.0
    clinical = dict(base_clinical)
    clinical.setdefault("spo2", round(96.0 - min(8.0, variability / 30.0), 1))
    clinical.setdefault("respiratory_rate", round(18.0 + min(10.0, variability / 10.0), 1))
    clinical.setdefault("temperature_c", round(37.0 + min(2.0, loudness / 180.0), 1))
    clinical.setdefault("crp", round(10.0 + min(70.0, variability / 3.0), 1))
    summary = (
        f"Audio upload '{filename}' processed as lung-sound proxy; "
        f"loudness={loudness:.1f}, variability={variability:.1f}."
    )
    merged_query = " ".join(part for part in [query_text.strip(), "respiratory cough wheeze pneumonia audio", summary] if part)
    return "pneumonia", clinical, merged_query


def genomics_upload_to_clinical_features(
    filename: str,
    payload: bytes,
    query_text: str,
    base_clinical: Dict[str, float],
) -> Tuple[str, Dict[str, float], str]:
    if not payload:
        raise ValueError("Uploaded genomics file is empty.")
    text = payload.decode("utf-8", errors="ignore").lower()
    variant_hits = sum(token in text for token in ("pathogenic", "deleterious", "risk", "cyp2d6", "cyp2c19"))
    clinical = dict(base_clinical)
    clinical.setdefault("glucose", round(110.0 + 15.0 * variant_hits, 1))
    clinical.setdefault("bmi", round(27.0 + 1.5 * variant_hits, 1))
    clinical.setdefault("age", float(clinical.get("age", 50.0)))
    summary = f"Genomics upload '{filename}' processed with variant risk hits={variant_hits}."
    merged_query = " ".join(part for part in [query_text.strip(), "genomics metabolic diabetes risk", summary] if part)
    return "diabetes", clinical, merged_query