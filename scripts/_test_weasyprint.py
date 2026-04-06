"""Quick test to see if WeasyPrint renders HTML to PDF correctly."""
try:
    from weasyprint import HTML
    pdf = HTML(string="<html><body><h1>Test</h1><p>Works!</p></body></html>").write_pdf()
    print(f"WeasyPrint OK: {len(pdf)} bytes, starts with %PDF: {pdf[:5]}")
except Exception as e:
    print(f"WeasyPrint FAILED: {type(e).__name__}: {e}")

# Also test the actual proof report fallback
import sys
sys.path.insert(0, ".")
from cloud.runtime.proof_report import _fallback_pdf_bytes, _render_pdf_bytes
html = "<html><body><h1>Test Report</h1><p>Hello world</p></body></html>"
pdf = _render_pdf_bytes(html)
print(f"_render_pdf_bytes: {len(pdf)} bytes, starts with: {pdf[:5]}")
fallback = _fallback_pdf_bytes(html)
print(f"_fallback_pdf_bytes: {len(fallback)} bytes, starts with: {fallback[:5]}")
