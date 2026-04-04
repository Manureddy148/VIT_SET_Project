"""Layer 5: basic faithfulness check — retrieved text must appear in synthesis."""


def snippets_in_report(snippets: list[str], report: str) -> bool:
    report_l = report.lower()
    return all(s.lower()[:80] in report_l or s.lower() in report_l for s in snippets if s)
