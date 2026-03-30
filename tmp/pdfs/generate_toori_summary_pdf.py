from __future__ import annotations

from pathlib import Path

import fitz
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path("/Users/macuser/toori")
OUTPUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs"
RENDER_DIR = TMP_DIR / "rendered"
OUTPUT_PDF = OUTPUT_DIR / "toori-app-summary.pdf"

TITLE = "Toori"
SUBTITLE = "Repo-evidence summary"

MERMAID = """```mermaid
flowchart LR
  Client["Desktop / iOS / Android / SDK"] --> Runtime["FastAPI Runtime"]
  Runtime --> Perception["Primary local perception"]
  Perception --> Observation["Observation store"]
  Observation --> World["World-model layer"]
  World --> Proof["Living Lens / Challenges"]
  Runtime --> Smriti["Smriti memory"]
  Smriti --> Pipeline["TPDS -> SAG -> CWMA -> ECGD -> Setu-2"]
  World --> Events["WebSocket events"]
  Events --> Client
```"""


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="AppTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#12263A"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AppSubtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5C6B73"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=15,
            textColor=colors.HexColor("#0B4F6C"),
            spaceBefore=6,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=11.4,
            textColor=colors.HexColor("#1D2731"),
            alignment=TA_LEFT,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AppBullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.1,
            leading=11.1,
            leftIndent=10,
            firstLineIndent=-10,
            textColor=colors.HexColor("#1D2731"),
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AppTiny",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.1,
            leading=9.8,
            textColor=colors.HexColor("#425466"),
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AppCodeCaption",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=10,
            textColor=colors.HexColor("#425466"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AppCodeBody",
            fontName="Courier",
            fontSize=7.2,
            leading=8.5,
            textColor=colors.HexColor("#0F172A"),
            leftIndent=0,
            rightIndent=0,
            spaceAfter=0,
        )
    )
    return styles


def bullet(text: str, styles) -> Paragraph:
    return Paragraph(f"<font color='#0B4F6C'>-</font> {text}", styles["AppBullet"])


def section_header(text: str, styles):
    return [
        Paragraph(text, styles["Section"]),
        HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#D0D7DE"), spaceBefore=0, spaceAfter=6),
    ]


def info_box(lines: list[str], styles, *, bg: str = "#F8FAFC", border: str = "#D0D7DE"):
    flowables = [bullet(line, styles) for line in lines]
    table = Table([[flowables]], colWidths=[A4[0] - 72])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(border)),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_story():
    styles = build_styles()
    story = [
        Spacer(1, 6),
        Paragraph(TITLE, styles["AppTitle"]),
        Paragraph(SUBTITLE, styles["AppSubtitle"]),
    ]

    story.extend(section_header("What it is", styles))
    story.append(
        Paragraph(
            "Toori is a loopback-first, cross-platform JEPA proof surface and world-state runtime: clients send real camera frames or files to a local Python API that stores observations, computes local descriptors or embeddings, and serves desktop, mobile, and SDK consumers.",
            styles["BodySmall"],
        )
    )
    story.append(
        Paragraph(
            "The world model is the temporal layer above observations - <b>SceneState</b>, <b>EntityTrack</b>, <b>PredictionWindow</b>, continuity, surprise, and persistence - exposed through Living Lens rather than as one-shot captions.",
            styles["BodySmall"],
        )
    )
    story.append(
        Paragraph(
            "Smriti is the grounded memory system for media ingestion, guarded recall, journals, and clustered browsing; it is designed to stay tied to stored media and return uncertainty instead of invented descriptions.",
            styles["BodySmall"],
        )
    )
    story.append(
        Paragraph(
            "Setu-2 is the JEPA-to-language energy bridge in the Smriti pipeline that turns gated, world-model-aligned embeddings into grounded recall scores, template descriptions, and session-local feedback.",
            styles["BodySmall"],
        )
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph("Repo gaps", styles["AppCodeCaption"]))
    story.append(
        info_box(
            [
                'Why the names "Smriti" and "Setu-2": Not found in repo.',
                "Direct Apple Photos / Google Photos comparison: Not found in repo.",
                "Direct broad LLM / VLM / AI comparison: Not found in repo; the repo explicitly contrasts Toori with caption-only and retrieval-only systems and says reasoning is optional and secondary to proof.",
            ],
            styles,
            bg="#FFF8E8",
            border="#E6C068",
        )
    )
    story.append(Spacer(1, 10))

    story.extend(section_header("Who it's for", styles))
    story.extend(
        [
            bullet("Deep-tech teams evaluating JEPA-style world-model behavior in a real product.", styles),
            bullet("Engineers implementing perception, world-model, memory, and plugin-runtime systems.", styles),
            bullet("Tech products that need a reusable world-state runtime across desktop, mobile, assistive, robotics-adjacent, or ambient-intelligence workflows.", styles),
        ]
    )
    story.append(Spacer(1, 8))

    story.extend(section_header("What it does", styles))
    story.extend(
        [
            bullet("Captures live camera frames or file inputs and stores real observations in local runtime state.", styles),
            bullet("Computes local perception outputs plus JEPA proof signals for prediction consistency, temporal continuity, surprise, and persistence.", styles),
            bullet("Runs Living Lens and challenge flows that compare JEPA / Hybrid mode against captioning and retrieval baselines on the same scene.", styles),
            bullet("Ingests photo and video folders into Smriti through a background daemon and watch-folder workflow.", styles),
            bullet("Supports natural-language Smriti recall, cluster browsing, deepdive inspection, and person journals.", styles),
            bullet("Applies TPDS, SAG, CWMA, ECGD, and Setu-2 so recall stays grounded and can surface uncertainty.", styles),
            bullet("Exposes HTTP, WebSocket, and SDK interfaces so the same runtime can power first-party clients and host apps.", styles),
        ]
    )

    story.append(PageBreak())

    story.extend(section_header("How it works", styles))
    story.append(
        Paragraph(
            "Observed repo flow: client capture enters a FastAPI runtime that coordinates local perception, observation storage, JEPA worker processing, world-model state, Smriti services, and event publication.",
            styles["BodySmall"],
        )
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph("Compact architecture overview", styles["AppCodeCaption"]))
    mermaid_table = Table([[Preformatted(MERMAID, styles["AppCodeBody"])]], colWidths=[A4[0] - 72])
    mermaid_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F8FA")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#D0D7DE")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(mermaid_table)
    story.append(Spacer(1, 10))

    story.extend(section_header("Repo-backed component notes", styles))
    story.extend(
        [
            bullet("`RuntimeContainer` coordinates settings, provider health, observation storage, local similarity search, and event publication.", styles),
            bullet("`JEPAWorkerPool` keeps JEPA work off the FastAPI event loop and reports bounded queue/back-pressure metrics.", styles),
            bullet("`ObservationStore` persists observations; `SmetiDB` extends storage for Smriti media, recall, migrations, clusters, and person linking.", styles),
            bullet("`SmritiIngestionDaemon` watches folders and progressively indexes media through the JEPA + Setu-2 pipeline.", styles),
            bullet("Optional reasoning backends exist, but the repo says proof should remain readable without language output.", styles),
        ]
    )
    story.append(Spacer(1, 10))

    story.extend(section_header("Why it is different in repo terms", styles))
    story.extend(
        [
            bullet("It keeps a running scene state instead of treating every frame as a fresh caption problem.", styles),
            bullet("It exposes prediction consistency, continuity, surprise, and persistence as first-class evidence.", styles),
            bullet("It compares itself against caption-only and retrieval-only baselines on the exact same live session.", styles),
            bullet("It treats captions and reasoning as secondary explanations rather than the core proof.", styles),
        ]
    )

    story.append(PageBreak())

    story.extend(section_header("How to run", styles))
    story.extend(
        [
            bullet("Start the runtime: `TOORI_DATA_DIR=.toori python3 -m uvicorn cloud.api.main:app --host 127.0.0.1 --port 7777`", styles),
            bullet("Verify it: `curl http://127.0.0.1:7777/healthz` and `curl http://127.0.0.1:7777/v1/providers/health`", styles),
            bullet("Start the proof surface: `cd /Users/macuser/toori/desktop/electron && npm install && npm run web`", styles),
            bullet("Open `http://127.0.0.1:4173`; repo docs recommend browser mode as the default proof path during development.", styles),
            bullet("If needed, launch the Electron shell with `npm start`; repo docs note that a signed macOS app bundle is needed for reliable Camera privacy behavior.", styles),
        ]
    )
    story.append(Spacer(1, 12))

    story.extend(section_header("Evidence basis", styles))
    story.append(
        Paragraph(
            "Summary content and architecture wording were grounded in these repo files:",
            styles["BodySmall"],
        )
    )
    story.extend(
        [
            bullet("`README.md`", styles),
            bullet("`CLAUDE.md`", styles),
            bullet("`docs/system-design.md`", styles),
            bullet("`docs/user-manual.md`", styles),
            bullet("`cloud/runtime/app.py`", styles),
            bullet("`cloud/runtime/setu2.py`", styles),
            bullet("`cloud/runtime/smriti_ingestion.py`", styles),
        ]
    )
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "Output note: this PDF is a concise repo snapshot, not a product marketing brief; any item marked Not found in repo was left unresolved on purpose.",
            styles["AppTiny"],
        )
    )

    return story


def draw_page(canvas, doc):
    page_number = canvas.getPageNumber()
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D0D7DE"))
    canvas.setLineWidth(0.75)
    canvas.line(doc.leftMargin, height - 18 * mm, width - doc.rightMargin, height - 18 * mm)
    canvas.setFillColor(colors.HexColor("#5C6B73"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(doc.leftMargin, 10 * mm, "Toori repo-evidence summary")
    canvas.drawRightString(width - doc.rightMargin, 10 * mm, f"Page {page_number} / 3")
    canvas.restoreState()


def render_pages(pdf_path: Path) -> None:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    for index in range(len(doc)):
        page = doc.load_page(index)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
        pix.save(RENDER_DIR / f"toori-app-summary-page-{index + 1}.png")
    doc.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=26 * mm,
        bottomMargin=18 * mm,
        title="Toori App Summary",
        author="OpenAI Codex",
    )
    story = build_story()
    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    render_pages(OUTPUT_PDF)

    page_count = len(fitz.open(OUTPUT_PDF))
    print(f"PDF: {OUTPUT_PDF}")
    print(f"Pages: {page_count}")
    print(f"Renders: {RENDER_DIR}")


if __name__ == "__main__":
    main()
