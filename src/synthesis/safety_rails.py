MEDICAL_DISCLAIMER = (
    "This output is for research and educational purposes only. "
    "It is not a medical diagnosis. The 0-100 score is a disease-classification probability display, not validated clinical severity; confidence is not an uncertainty estimate. Never use it for medication dosing. Always consult a qualified clinician."
)


def build_missing_feature_prompts(missing_features: list[str]) -> list[str]:
    prompts = []
    for feature in missing_features:
        label = feature.replace("_", " ")
        prompts.append(f"Provide {label} if available to improve reliability.")
    return prompts


def build_safety_flags(severity_score: float, severity_label: str, faithfulness_passed: bool) -> list[str]:
    flags: list[str] = []
    if severity_label == "Critical" or severity_score >= 75:
        flags.append("urgent_review")
    elif severity_label == "High" or severity_score >= 50:
        flags.append("clinician_review")
    if not faithfulness_passed:
        flags.append("manual_evidence_review")
    return flags


def build_recommended_actions(severity_score: float, severity_label: str) -> list[str]:
    if severity_label == "Critical" or severity_score >= 75:
        return [
            "Escalate for urgent physician evaluation.",
            "Review vital signs and corroborating evidence immediately.",
        ]
    if severity_label == "High" or severity_score >= 50:
        return [
            "Arrange prompt clinician review.",
            "Confirm the main risk factors with structured measurements.",
        ]
    if severity_label == "Moderate" or severity_score >= 25:
        return [
            "Schedule follow-up assessment.",
            "Collect any missing structured measurements.",
        ]
    return [
        "Continue routine monitoring.",
        "Reassess if symptoms worsen or new findings appear.",
    ]
