import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  SmritiAnchorMatch,
  SmritiDepthStrata,
  SmritiMediaNeighbor,
  SmritiRecallResult,
  SmritiSetuRecord,
} from "../../types";

type SmritiMediaDetail = {
  id: string;
  file_path: string;
  depth_strata: SmritiDepthStrata | null;
  anchor_matches: SmritiAnchorMatch[];
  setu_descriptions: SmritiSetuRecord[];
  hallucination_risk: number;
  alignment_loss?: number | null;
};

type DeepdiveViewProps = {
  media: SmritiRecallResult;
  assetUrl: (filePath: string) => string;
  runtimeRequest: <T>(path: string, method?: string, body?: unknown) => Promise<T>;
  sessionId: string;
  onClose: () => void;
  onTagPerson: (name: string) => Promise<void>;
};

type PatchSummary = {
  index: number;
  anchorName: string;
  depthStratum: string;
  confidence: number;
  hallucinationRisk: number;
  description: string;
};

function isTypingTarget(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null;
  if (!element) {
    return false;
  }
  const tagName = element.tagName.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select" || element.isContentEditable;
}

function patchStratum(patchIndex: number, depthStrata: SmritiDepthStrata | null): string {
  if (!depthStrata) {
    return "unknown";
  }
  const row = Math.floor(patchIndex / 14);
  const col = patchIndex % 14;
  if (depthStrata.foreground_mask?.[row]?.[col]) {
    return "foreground";
  }
  if (depthStrata.background_mask?.[row]?.[col]) {
    return "background";
  }
  if (depthStrata.midground_mask?.[row]?.[col]) {
    return "midground";
  }
  return "unknown";
}

function riskTone(value: number): "safe" | "watch" | "risk" {
  if (value <= 0.2) {
    return "safe";
  }
  if (value <= 0.45) {
    return "watch";
  }
  return "risk";
}

export default function DeepdiveView({
  media,
  assetUrl,
  runtimeRequest,
  sessionId,
  onClose,
  onTagPerson,
}: DeepdiveViewProps) {
  const modalRef = useRef<HTMLDivElement | null>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const [detail, setDetail] = useState<SmritiMediaDetail | null>(null);
  const [neighbors, setNeighbors] = useState<SmritiMediaNeighbor[]>([]);
  const [energyOverlayVisible, setEnergyOverlayVisible] = useState(false);
  const [activePatchIndex, setActivePatchIndex] = useState<number | null>(null);
  const [personName, setPersonName] = useState("");
  const [tagging, setTagging] = useState(false);
  const [fullscreenActive, setFullscreenActive] = useState(false);

  const depthStrata = detail?.depth_strata ?? media.depth_strata_data ?? null;
  const anchorMatches = detail?.anchor_matches ?? media.anchor_matches ?? [];
  const setuDescriptions = detail?.setu_descriptions ?? media.setu_descriptions ?? [];
  const imageUrl = assetUrl(media.file_path);

  const toggleFullscreen = useCallback(async () => {
    const modal = modalRef.current;
    if (!modal) {
      return;
    }
    if (document.fullscreenElement) {
      await document.exitFullscreen();
      return;
    }
    if (modal.requestFullscreen) {
      await modal.requestFullscreen();
    }
  }, []);

  useEffect(() => {
    previousFocusRef.current = document.activeElement as HTMLElement | null;
    const modal = modalRef.current;
    if (!modal) {
      return;
    }
    const timer = window.setTimeout(() => {
      modal.focus();
    }, 0);

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (!isTypingTarget(event.target)) {
        const lower = event.key.toLowerCase();
        if (lower === "e") {
          event.preventDefault();
          setEnergyOverlayVisible((current) => !current);
          return;
        }
        if (lower === "f") {
          event.preventDefault();
          void toggleFullscreen();
          return;
        }
      }
      if (event.key !== "Tab") {
        return;
      }
      const focusable = Array.from(
        modal.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => !element.hasAttribute("disabled") && element.getAttribute("aria-hidden") !== "true");
      if (focusable.length === 0) {
        event.preventDefault();
        modal.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    const handleFullscreen = () => {
      setFullscreenActive(Boolean(document.fullscreenElement));
    };

    modal.addEventListener("keydown", handleKeyDown);
    document.addEventListener("fullscreenchange", handleFullscreen);

    return () => {
      window.clearTimeout(timer);
      modal.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("fullscreenchange", handleFullscreen);
      previousFocusRef.current?.focus?.();
    };
  }, [onClose, toggleFullscreen]);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setNeighbors([]);

    runtimeRequest<SmritiMediaDetail>(`/v1/smriti/media/${media.media_id}`)
      .then((response) => {
        if (!cancelled) {
          setDetail(response);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setDetail(null);
        }
      });

    runtimeRequest<{ neighbors: SmritiMediaNeighbor[] }>(
      `/v1/smriti/media/${media.media_id}/neighbors?top_k=6`,
    )
      .then((response) => {
        if (!cancelled) {
          setNeighbors(response.neighbors || []);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setNeighbors([]);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [media.media_id, runtimeRequest]);

  const patchSummary = useMemo<PatchSummary | null>(() => {
    if (activePatchIndex == null) {
      return null;
    }
    const match = anchorMatches.find((item) => item.patch_indices.includes(activePatchIndex)) ?? null;
    const setuRecord =
      (match
        ? setuDescriptions.find(
            (item) =>
              item.description.anchor_basis === match.template_name ||
              item.gate?.anchor_name === match.template_name,
          )
        : null) ?? setuDescriptions[0];
    return {
      index: activePatchIndex,
      anchorName: match?.template_name ?? "unknown",
      depthStratum: match?.depth_stratum ?? patchStratum(activePatchIndex, depthStrata),
      confidence: match?.confidence ?? setuRecord?.description.confidence ?? depthStrata?.confidence ?? 0,
      hallucinationRisk:
        setuRecord?.description.hallucination_risk ??
        setuRecord?.gate?.estimated_hallucination_risk ??
        detail?.hallucination_risk ??
        media.hallucination_risk,
      description: setuRecord?.description.text ?? "No grounded description available for this patch.",
    };
  }, [activePatchIndex, anchorMatches, depthStrata, detail?.hallucination_risk, media.hallucination_risk, setuDescriptions]);

  const depthProxyMax = useMemo(() => {
    const values = depthStrata?.depth_proxy?.flat() ?? [];
    return values.length > 0 ? Math.max(...values, 1) : 1;
  }, [depthStrata]);

  async function handleTagSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextName = personName.trim();
    if (!nextName) {
      return;
    }
    setTagging(true);
    try {
      await onTagPerson(nextName);
      setPersonName("");
    } finally {
      setTagging(false);
    }
  }

  return (
    <div
      ref={modalRef}
      className="smriti-deepdive"
      role="dialog"
      aria-modal="true"
      aria-labelledby="smriti-deepdive-title"
      tabIndex={-1}
    >
      <section className="smriti-deepdive__media">
        <div className="smriti-deepdive__header">
          <div>
            <p className="eyebrow">Deepdive</p>
            <h3 id="smriti-deepdive-title">{media.primary_description}</h3>
          </div>
          <div className="smriti-deepdive__toolbar">
            <button type="button" onClick={() => setEnergyOverlayVisible((current) => !current)} aria-pressed={energyOverlayVisible}>
              {energyOverlayVisible ? "Hide Energy" : "Show Energy"}
            </button>
            <button type="button" onClick={() => void toggleFullscreen()} aria-pressed={fullscreenActive}>
              {fullscreenActive ? "Exit Fullscreen" : "Fullscreen"}
            </button>
            <button type="button" onClick={onClose} aria-label="Close deepdive modal">
              Close
            </button>
          </div>
        </div>

        <div className="smriti-deepdive__image-shell">
          <img src={imageUrl} alt={media.primary_description} />
          {energyOverlayVisible ? (
            depthStrata?.depth_proxy?.length ? (
              <div className="smriti-patch-grid" aria-label="JEPA energy patch grid">
                {Array.from({ length: 14 * 14 }, (_, patchIndex) => {
                  const row = Math.floor(patchIndex / 14);
                  const col = patchIndex % 14;
                  const proxyValue = depthStrata.depth_proxy?.[row]?.[col] ?? 0;
                  const alpha = Math.min(0.12 + proxyValue / depthProxyMax * 0.58, 0.72);
                  return (
                    <button
                      key={patchIndex}
                      type="button"
                      className={activePatchIndex === patchIndex ? "smriti-patch-cell is-active" : "smriti-patch-cell"}
                      style={{
                        left: `${col * (100 / 14)}%`,
                        top: `${row * (100 / 14)}%`,
                        width: `${100 / 14}%`,
                        height: `${100 / 14}%`,
                        background: `rgba(67,216,201,${alpha})`,
                      }}
                      onClick={() => setActivePatchIndex((current) => (current === patchIndex ? null : patchIndex))}
                      aria-label={`Patch ${patchIndex}`}
                    />
                  );
                })}
              </div>
            ) : (
              <div className="smriti-patch-grid-empty">Energy overlay unavailable for this item.</div>
            )
          ) : null}

          {patchSummary ? (
            <aside className="smriti-patch-popover" aria-live="polite">
              <p className="eyebrow">Patch {patchSummary.index}</p>
              <h4>{patchSummary.anchorName}</h4>
              <p className="muted">{patchSummary.description}</p>
              <div className="smriti-chip-row">
                <span className="smriti-chip">{patchSummary.depthStratum}</span>
                <span className={`smriti-pill smriti-pill--${riskTone(patchSummary.hallucinationRisk)}`}>
                  risk {(patchSummary.hallucinationRisk * 100).toFixed(0)}%
                </span>
              </div>
              <div className="smriti-score-row">
                <span>Confidence {(patchSummary.confidence * 100).toFixed(0)}%</span>
                <span>Session {sessionId}</span>
              </div>
            </aside>
          ) : null}
        </div>
      </section>

      <aside className="smriti-deepdive__panel">
        <div className="smriti-detail-stack">
          <section className="panel panel--comparison">
            <div className="smriti-panel-header">
              <div>
                <p className="eyebrow">Grounding</p>
                <h4>Anchor and Setu alignment</h4>
              </div>
              <span className={`smriti-pill smriti-pill--${riskTone(media.hallucination_risk)}`}>
                risk {(media.hallucination_risk * 100).toFixed(0)}%
              </span>
            </div>
            <div className="smriti-worker-metrics">
              <span>{media.anchor_basis}</span>
              <span>{media.depth_stratum}</span>
              <span>Hybrid {(media.hybrid_score * 100).toFixed(0)}</span>
              <span>Setu {(media.setu_score * 100).toFixed(0)}</span>
            </div>
            {setuDescriptions.length > 0 ? (
              <div className="smriti-side-stack">
                {setuDescriptions.slice(0, 4).map((record, index) => (
                  <div key={`${record.description.anchor_basis}-${index}`} className="smriti-detail-card">
                    <strong>{record.description.text}</strong>
                    <span className="muted">
                      {record.description.anchor_basis} · {record.description.depth_stratum}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">No patch-level Setu records were stored for this media item.</p>
            )}
          </section>

          <section className="panel panel--stable">
            <div className="smriti-panel-header">
              <div>
                <p className="eyebrow">People</p>
                <h4>Confirm person tags</h4>
              </div>
            </div>
            <div className="smriti-chip-row">
              {media.person_names.length > 0 ? (
                media.person_names.map((name) => (
                  <span key={name} className="smriti-chip">
                    {name}
                  </span>
                ))
              ) : (
                <span className="muted">No confirmed people yet.</span>
              )}
            </div>
            <form className="smriti-tag-form" onSubmit={(event) => void handleTagSubmit(event)}>
              <input
                value={personName}
                onChange={(event) => setPersonName(event.target.value)}
                placeholder="Tag a person and propagate"
                aria-label="Person name"
              />
              <button type="submit" className="primary" disabled={tagging || !personName.trim()}>
                {tagging ? "Saving…" : "Tag Person"}
              </button>
            </form>
          </section>

          {neighbors.length > 0 ? (
            <section className="panel panel--memory">
              <div className="smriti-panel-header">
                <div>
                  <p className="eyebrow">Semantic Neighbors</p>
                  <h4>Nearest Setu-2 matches</h4>
                </div>
              </div>
              <div className="semantic-neighbors-grid">
                {neighbors.slice(0, 6).map((neighbor) => (
                  <article key={neighbor.media_id} className="semantic-neighbor-card" title={`Setu score ${neighbor.setu_score.toFixed(3)}`}>
                    <img
                      src={assetUrl(neighbor.thumbnail_path || media.thumbnail_path || media.file_path)}
                      alt=""
                      loading="lazy"
                    />
                    <strong>{neighbor.media_id}</strong>
                    <span className="muted">Setu {neighbor.setu_score.toFixed(3)}</span>
                  </article>
                ))}
              </div>
            </section>
          ) : null}

          <section className="panel">
            <div className="smriti-panel-header">
              <div>
                <p className="eyebrow">Keyboard</p>
                <h4>Deepdive controls</h4>
              </div>
            </div>
            <div className="smriti-worker-metrics">
              <span className="smriti-kbd">E</span>
              <span>Toggle energy grid</span>
              <span className="smriti-kbd">F</span>
              <span>Fullscreen</span>
              <span className="smriti-kbd">Esc</span>
              <span>Close modal</span>
            </div>
            {detail?.alignment_loss != null ? (
              <p className="muted">Alignment loss {detail.alignment_loss.toFixed(4)}</p>
            ) : null}
          </section>
        </div>
      </aside>
    </div>
  );
}
