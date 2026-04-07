from typing import Dict

import numpy as np


def process_ecg_signal(signal: np.ndarray, sample_rate: int = 250) -> Dict[str, float]:
    """Extract ECG summary features with optional NeuroKit2 support."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        signal = np.array([0.0], dtype=np.float64)

    base = {
        "signal_mean_abs": float(np.mean(np.abs(signal))),
        "signal_std": float(np.std(signal)),
        "signal_peak": float(np.max(np.abs(signal))),
    }

    try:
        import neurokit2 as nk

        cleaned = nk.ecg_clean(signal, sampling_rate=sample_rate)
        peaks, _ = nk.ecg_peaks(cleaned, sampling_rate=sample_rate)
        peak_idx = peaks.get("ECG_R_Peaks", np.array([], dtype=int))
        if len(peak_idx) > 1:
            rr = np.diff(peak_idx) / float(sample_rate)
            hr = float(60.0 / np.mean(rr)) if np.mean(rr) > 0 else 75.0
            hrv = float(np.std(rr))
        else:
            hr = 75.0
            hrv = 0.0
        base.update({
            "heart_rate": float(np.clip(hr, 45.0, 210.0)),
            "hrv": float(max(0.0, hrv)),
            "neurokit_used": 1.0,
        })
        return base
    except Exception:
        approx_hr = float(np.clip(68.0 + base["signal_std"] * 18.0 + base["signal_peak"] * 5.0, 55.0, 190.0))
        base.update({
            "heart_rate": approx_hr,
            "hrv": float(base["signal_std"] * 0.05),
            "neurokit_used": 0.0,
        })
        return base
