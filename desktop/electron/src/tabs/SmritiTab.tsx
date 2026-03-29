import { useState } from "react";
import MandalaView from "../components/smriti/MandalaView";
import DeepdiveView from "../components/smriti/DeepdiveView";
import PerformanceHUD from "../components/smriti/PerformanceHUD";
import PersonJournal from "../components/smriti/PersonJournal";
import RecallSurface from "../components/smriti/RecallSurface";
import { useSmritiState } from "../hooks/useSmritiState";
import { useDesktopApp } from "../state/DesktopAppContext";
import type { SmritiSection } from "../types";

const SECTION_LABELS: Array<{ id: Exclude<SmritiSection, "deepdive">; label: string }> = [
  { id: "mandala", label: "Mandala" },
  { id: "recall", label: "Recall" },
  { id: "journals", label: "Journals" },
  { id: "hud", label: "HUD" },
];

export default function SmritiTab() {
  const app = useDesktopApp();
  const smriti = useSmritiState({
    runtimeRequest: app.runtimeRequest,
    sessionId: app.sessionId,
  });
  const [selectedClusterId, setSelectedClusterId] = useState<number | null>(null);
  const [filePath, setFilePath] = useState("");
  const visibleSection = smriti.section === "deepdive" ? "recall" : smriti.section;

  const openCluster = (clusterId: number) => {
    setSelectedClusterId(clusterId);
    const label = smriti.mandalaData?.nodes.find((node) => node.id === clusterId)?.label || `cluster ${clusterId}`;
    smriti.setRecallQuery(label);
    smriti.setSection("recall");
  };

  return (
    <section className="smriti-shell">
      <a href="#smriti-main" className="skip-link">
        Skip to Smriti content
      </a>
      <div className="sr-live-region" aria-live="polite" aria-atomic="true">
        {smriti.ingestionStatus || smriti.status?.status || "Smriti ready"}
      </div>
      <div className="smriti-command-bar">
        <div>
          <p className="eyebrow">Smriti</p>
          <h3>Semantic memory orchestration</h3>
        </div>
        <div className="smriti-command-bar__actions">
          {SECTION_LABELS.map((section) => (
            <button
              key={section.id}
              type="button"
              className={visibleSection === section.id ? "smriti-section-button is-active" : "smriti-section-button"}
              onClick={() => smriti.setSection(section.id)}
            >
              {section.label}
            </button>
          ))}
        </div>
      </div>

      <div className="smriti-layout">
        <div className="smriti-main" id="smriti-main">
          {visibleSection === "mandala" ? (
            <section className="panel panel--memory smriti-surface">
              <div className="smriti-panel-header">
                <div>
                  <p className="eyebrow">Mandala View</p>
                  <h3>Clustered memory topology</h3>
                </div>
                <p className="muted">
                  {smriti.mandalaData
                    ? `Generated ${new Date(smriti.mandalaData.generated_at).toLocaleString()}`
                    : "Waiting for clustered memories"}
                </p>
              </div>
              <MandalaView
                data={smriti.mandalaData}
                selectedClusterId={selectedClusterId}
                onNodeSelect={(clusterId) => setSelectedClusterId(clusterId)}
                onNodeExpand={openCluster}
              />
            </section>
          ) : null}

          {visibleSection === "recall" ? (
            <RecallSurface
              query={smriti.recallQuery}
              onQueryChange={smriti.setRecallQuery}
              personFilter={smriti.personFilter}
              onPersonFilterChange={smriti.setPersonFilter}
              locationFilter={smriti.locationFilter}
              onLocationFilterChange={smriti.setLocationFilter}
              minConfidence={smriti.minConfidence}
              onMinConfidenceChange={smriti.setMinConfidence}
              timeRangeDays={smriti.timeRangeDays}
              onTimeRangeDaysChange={smriti.setTimeRangeDays}
              results={smriti.recallResults}
              busy={smriti.recallBusy}
              totalSearched={smriti.totalIndexed}
              onOpenMedia={smriti.openDeepdive}
              assetUrl={app.assetUrl}
              runtimeRequest={app.runtimeRequest}
            />
          ) : null}

          {visibleSection === "journals" ? (
            <PersonJournal
              personName={smriti.personName}
              onPersonNameChange={smriti.setPersonName}
              journal={smriti.personJournal}
              onLoadJournal={smriti.loadPersonJournal}
            />
          ) : null}

          {visibleSection === "hud" ? (
            <PerformanceHUD metrics={smriti.metrics} status={smriti.status} onRefresh={smriti.loadMetrics} />
          ) : null}
        </div>

        <aside className="smriti-sidebar">
          <section className="panel panel--stable">
            <div className="smriti-panel-header">
              <div>
                <p className="eyebrow">Ingestion</p>
                <h4>Index new media</h4>
              </div>
            </div>
            <div className="smriti-side-stack">
              <label className="smriti-field">
                <span>Folder path</span>
                <input
                  value={smriti.ingestionFolder}
                  onChange={(event) => smriti.setIngestionFolder(event.target.value)}
                  placeholder="/Users/you/Pictures"
                />
              </label>
              <button
                type="button"
                className="primary"
                disabled={smriti.ingestionBusy || !smriti.ingestionFolder.trim()}
                onClick={() => void smriti.ingestFolder(smriti.ingestionFolder)}
              >
                {smriti.ingestionBusy ? "Queueing…" : "Watch folder"}
              </button>
              <label className="smriti-field">
                <span>Single file</span>
                <input
                  value={filePath}
                  onChange={(event) => setFilePath(event.target.value)}
                  placeholder="/Users/you/Desktop/frame.png"
                />
              </label>
              <button
                type="button"
                disabled={smriti.ingestionBusy || !filePath.trim()}
                onClick={() => void smriti.ingestFile(filePath)}
              >
                Queue file
              </button>
              <p className="muted">{smriti.ingestionStatus}</p>
            </div>
          </section>

          <section className="panel panel--comparison">
            <div className="smriti-panel-header">
              <div>
                <p className="eyebrow">Snapshot</p>
                <h4>Runtime state</h4>
              </div>
            </div>
            <div className="smriti-kpi-grid smriti-kpi-grid--compact">
              <div className="smriti-kpi-card">
                <span>Processed</span>
                <strong>{smriti.status?.ingestion.processed || 0}</strong>
              </div>
              <div className="smriti-kpi-card">
                <span>Queued</span>
                <strong>{smriti.status?.ingestion.queue_depth || 0}</strong>
              </div>
              <div className="smriti-kpi-card">
                <span>Workers</span>
                <strong>{smriti.metrics?.workers.length || 0}</strong>
              </div>
              <div className="smriti-kpi-card">
                <span>Clusters</span>
                <strong>{smriti.mandalaData?.nodes.length || 0}</strong>
              </div>
            </div>
          </section>

          <section className="panel panel--memory">
            <div className="smriti-panel-header">
              <div>
                <p className="eyebrow">Hot clusters</p>
                <h4>Jump directly into recall</h4>
              </div>
            </div>
            <div className="smriti-side-stack">
              {(smriti.mandalaData?.nodes || []).slice(0, 6).map((node) => (
                <button
                  key={node.id}
                  type="button"
                  className="smriti-list-button"
                  onClick={() => openCluster(node.id)}
                >
                  <span>{node.label}</span>
                  <strong>{node.media_count}</strong>
                </button>
              ))}
              {(smriti.mandalaData?.nodes || []).length === 0 ? (
                <p className="muted">Clusters appear once ingestion produces completed media.</p>
              ) : null}
            </div>
          </section>
        </aside>
      </div>

      {smriti.section === "deepdive" && smriti.selectedMedia ? (
        <DeepdiveView
          media={smriti.selectedMedia}
          assetUrl={app.assetUrl}
          runtimeRequest={app.runtimeRequest}
          sessionId={app.sessionId}
          onClose={smriti.closeDeepdive}
          onTagPerson={async (name) => {
            await smriti.tagPerson(smriti.selectedMedia?.media_id || "", name);
            smriti.setPersonName(name);
            await smriti.loadPersonJournal(name);
          }}
        />
      ) : null}
    </section>
  );
}
