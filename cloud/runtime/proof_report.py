from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Iterable

import numpy as np
from fpdf import FPDF

from .models import JEPATick


class ProofReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 18)
        self.cell(0, 10, "Toori Proof Report", align="C")
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def aggregate_stats(ticks: list[JEPATick]) -> dict:
    mean_energy = [float(tick.mean_energy) for tick in ticks]
    sigreg = [float(tick.sigreg_loss) for tick in ticks]
    talker_events = [tick.talker_event for tick in ticks if tick.talker_event]
    recovered = sum(1 for tick in ticks for track in tick.entity_tracks if track.status == "re-identified")
    occluded = sum(1 for tick in ticks for track in tick.entity_tracks if track.status == "occluded")
    planning = [float(tick.planning_time_ms) for tick in ticks]
    
    return {
        "total_ticks": len(ticks),
        "mean_surprise": np.mean(mean_energy) if mean_energy else 0.0,
        "mean_sigreg": np.mean(sigreg) if sigreg else 0.0,
        "talker_events": talker_events,
        "recoveries": recovered,
        "occlusions": occluded,
        "planning_latency": np.mean(planning) if planning else 0.0,
        "fingerprint_dim": len(ticks[-1].session_fingerprint) if ticks else 0
    }


def generate_proof_report(
    ticks: list[JEPATick],
    session_id: str,
    narration_text: str,
    chart_b64: str | None = None
) -> Path:
    stats = aggregate_stats(ticks)
    
    pdf = ProofReportPDF()
    pdf.add_page()
    
    # Session ID
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, f"Session ID: {session_id}")
    pdf.ln(12)
    
    # Gemma 4 Analysis Section
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, " Gemma 4 Analysis Report", fill=True)
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, narration_text)
    pdf.ln(10)
    
    # Mathematical Metrics Section
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, " Core Metrics Details", fill=True)
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"- Ticks captured: {stats['total_ticks']}")
    pdf.ln(6)
    pdf.cell(0, 8, f"- Mean spatial surprise: {stats['mean_surprise']:.3f}")
    pdf.ln(6)
    pdf.cell(0, 8, f"- Mean SIGReg health: {stats['mean_sigreg']:.3f}")
    pdf.ln(6)
    pdf.cell(0, 8, f"- Tracker occlusion recoveries: {stats['recoveries']} out of {stats['occlusions']} observed")
    pdf.ln(6)
    talkers = ", ".join(stats["talker_events"][:10]) if stats["talker_events"] else "None recorded"
    if len(stats["talker_events"]) > 10:
        talkers += "..."
    pdf.cell(0, 8, f"- Talker event log: {talkers}")
    pdf.ln(6)
    pdf.cell(0, 8, f"- Average JEPA planning time: {stats['planning_latency']:.2f} ms")
    pdf.ln(6)
    pdf.cell(0, 8, f"- Structural fingerprint dimensions: {stats['fingerprint_dim']}")
    pdf.ln(12)

    # Chart
    if chart_b64:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, " JEPA Rollout / Baseline Tracking Chart", fill=True)
        pdf.ln(15)
        try:
            chart_bytes = base64.b64decode(chart_b64)
            img = io.BytesIO(chart_bytes)
            # Center the image roughly
            pdf.image(img, x=15, w=180)
        except Exception:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 10, "Error rendering chart image.")

    output_path = Path(f"/tmp/toori_proof_{session_id}.pdf")
    pdf.output(str(output_path))
    return output_path
