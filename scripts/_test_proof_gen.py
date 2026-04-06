import asyncio
from cloud.runtime.proof_report import generate_proof_report
from cloud.runtime.models import JEPATick

ticks = [
    JEPATick(
        timestamp=0, observation_id="ob1",
        session_fingerprint=[0.1, 0.2, 0.3],
        entity_tracks=[],
        mean_energy=0.3, surprise_score=0.2,
        sigreg_loss=0.1, talker_event=None,
        planning_time_ms=12.5
    )
]

pdf_path = generate_proof_report(ticks, "test_session_123", "Gemma 4 successfully narrated this session as stable and robust.", None)
print(f"PDF generated at: {pdf_path}")
with open(pdf_path, 'rb') as f:
    hdr = f.read(4)
    print(f"Valid PDF header: {hdr == b'%PDF'}")
