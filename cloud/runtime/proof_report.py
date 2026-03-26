from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable

import numpy as np

from .models import JEPATick


def _html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _build_html(ticks: Iterable[JEPATick], session_id: str, chart_b64: str | None = None) -> str:
    ticks = list(ticks)
    mean_energy = [float(tick.mean_energy) for tick in ticks]
    sigreg = [float(tick.sigreg_loss) for tick in ticks]
    talker_events = [tick.talker_event for tick in ticks if tick.talker_event]
    recovered = sum(1 for tick in ticks for track in tick.entity_tracks if track.status == "re-identified")
    occluded = sum(1 for tick in ticks for track in tick.entity_tracks if track.status == "occluded")
    planning = [float(tick.planning_time_ms) for tick in ticks]
    fingerprint_dim = len(ticks[-1].session_fingerprint) if ticks else 0
    chart_html = ""
    if chart_b64:
        chart_html = f'<img alt="Chart" style="max-width:100%;border-radius:12px;" src="data:image/png;base64,{chart_b64}" />'
    return f"""
    <html>
      <body style="font-family:Arial,sans-serif;padding:24px;color:#101828;">
        <h1>Toori Proof Report</h1>
        <p><strong>Session:</strong> {_html_escape(session_id)}</p>
        <h2>Summary</h2>
        <p>Ticks captured: {len(ticks)} | Mean surprise: {np.mean(mean_energy) if mean_energy else 0.0:.3f}</p>
        <h2>SigReg Health</h2>
        <p>Mean SIGReg: {np.mean(sigreg) if sigreg else 0.0:.3f}</p>
        <h2>Talker Event Log</h2>
        <p>{_html_escape(', '.join(talker_events) if talker_events else 'No talker events recorded')}</p>
        <h2>Occlusion Recovery</h2>
        <p>Occluded tracks observed: {occluded} | Recoveries observed: {recovered}</p>
        <h2>Baseline Chart</h2>
        {chart_html or '<p>No external chart snapshot supplied.</p>'}
        <h2>Fingerprint</h2>
        <p>Fingerprint dimensionality: {fingerprint_dim}</p>
        <h2>Planning Speed</h2>
        <p>Average JEPA planning time: {np.mean(planning) if planning else 0.0:.2f} ms</p>
      </body>
    </html>
    """


def _fallback_pdf_bytes(html: str) -> bytes:
    text = html.replace("\n", " ").strip()[:4000]
    text = text.replace("(", "[").replace(")", "]")
    body = f"BT /F1 10 Tf 40 760 Td ({text}) Tj ET"
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Count 1 /Kids [3 0 R] >> endobj",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj",
        f"4 0 obj << /Length {len(body)} >> stream\n{body}\nendstream endobj",
        "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
    ]
    pdf = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf.encode("latin-1")))
        pdf += f"{obj}\n"
    xref_offset = len(pdf.encode("latin-1"))
    pdf += f"xref\n0 {len(objects) + 1}\n"
    pdf += "0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n"
    pdf += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF"
    return pdf.encode("latin-1", errors="ignore")


def generate_proof_report(ticks: list[JEPATick], session_id: str, chart_b64: str | None = None) -> Path:
    output_path = Path(f"/tmp/toori_proof_{session_id}.pdf")
    html = _build_html(ticks=ticks, session_id=session_id, chart_b64=chart_b64)
    try:
        from weasyprint import HTML

        HTML(string=html).write_pdf(str(output_path))
    except Exception:
        output_path.write_bytes(_fallback_pdf_bytes(html))
    return output_path
