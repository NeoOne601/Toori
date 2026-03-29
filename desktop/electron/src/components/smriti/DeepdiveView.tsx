import { useEffect, useState } from "react";
import type { SmritiRecallResult } from "../../types";

type DeepdiveViewProps = {
  media: SmritiRecallResult;
  assetUrl: (filePath: string) => string;
  onClose: () => void;
  onTagPerson: (name: string) => Promise<void>;
};

function percentage(value: number) {
  return `${Math.round(value * 100)}%`;
}

export default function DeepdiveView({ media, assetUrl, onClose, onTagPerson }: DeepdiveViewProps) {
  const [personDraft, setPersonDraft] = useState(media.person_names[0] || "");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const previewPath = media.file_path || media.thumbnail_path;

  return (
    <div className="smriti-deepdive" role="dialog" aria-modal="true" aria-label="Memory deepdive view">
      <div className="smriti-deepdive__media">
        <img src={assetUrl(previewPath)} alt={media.primary_description} />
      </div>
      <aside className="smriti-deepdive__panel">
        <div className="smriti-deepdive__header">
          <div>
            <p className="eyebrow">Deepdive</p>
            <h3>{media.primary_description}</h3>
          </div>
          <button type="button" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="smriti-kpi-grid">
          <div className="smriti-kpi-card">
            <span>Hybrid score</span>
            <strong>{percentage(media.hybrid_score)}</strong>
          </div>
          <div className="smriti-kpi-card">
            <span>Setu score</span>
            <strong>{percentage(media.setu_score)}</strong>
          </div>
          <div className="smriti-kpi-card">
            <span>Hallucination risk</span>
            <strong>{percentage(media.hallucination_risk)}</strong>
          </div>
          <div className="smriti-kpi-card">
            <span>Depth</span>
            <strong>{media.depth_stratum}</strong>
          </div>
        </div>

        <div className="smriti-detail-stack">
          <section className="smriti-detail-card">
            <p className="eyebrow">Grounding</p>
            <p className="muted">Anchor basis: {media.anchor_basis}</p>
            <p className="muted">Created: {new Date(media.created_at).toLocaleString()}</p>
            <p className="muted">Source: {media.file_path}</p>
          </section>

          <section className="smriti-detail-card">
            <p className="eyebrow">Entity tagging</p>
            <form
              className="smriti-tag-form"
              onSubmit={async (event) => {
                event.preventDefault();
                if (!personDraft.trim()) {
                  return;
                }
                setBusy(true);
                try {
                  await onTagPerson(personDraft.trim());
                } finally {
                  setBusy(false);
                }
              }}
            >
              <input
                value={personDraft}
                onChange={(event) => setPersonDraft(event.target.value)}
                placeholder="Name a person to propagate"
              />
              <button type="submit" className="primary" disabled={busy || !personDraft.trim()}>
                {busy ? "Saving…" : "Tag and propagate"}
              </button>
            </form>
            <div className="smriti-chip-row">
              {media.person_names.length > 0 ? media.person_names.map((person) => (
                <span key={`${media.media_id}-${person}`} className="smriti-chip">
                  {person}
                </span>
              )) : <span className="muted">No confirmed people yet.</span>}
            </div>
          </section>
        </div>
      </aside>
    </div>
  );
}
