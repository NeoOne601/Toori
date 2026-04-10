from __future__ import annotations

def test_proof_report_heading_is_provider_neutral(tmp_path):
    """
    Regression: the proof report section heading should not be hard-coded to Gemma 4.
    """
    # FPDF can compress content streams, so asserting on raw PDF bytes is flaky.
    # Instead, assert on the source string literal in the proof-report generator.
    source = (tmp_path / "proof_report.py")
    import inspect
    import cloud.runtime.proof_report as module

    source.write_text(inspect.getsource(module), encoding="utf-8")
    text = source.read_text(encoding="utf-8")

    assert "Analysis Report" in text
    assert "Gemma 4 Analysis Report" not in text
