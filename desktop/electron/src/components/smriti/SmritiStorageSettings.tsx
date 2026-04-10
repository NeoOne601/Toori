import { useCallback, useEffect, useState } from "react";
import { pickFolderPath, runtimeRequest } from "../../hooks/useRuntimeBridge";
import type {
  SmritiMigrationRequest,
  SmritiMigrationResult,
  SmritiPruneRequest,
  SmritiPruneResult,
  SmritiStorageConfig,
  StorageUsageReport,
  WatchFolderStatus,
} from "../../types";

type SmritiStorageSettingsProps = {
  onStatusChange?: (message: string) => void;
};

function bytesHuman(bytes: number): string {
  if (bytes <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function watchPathToString(value: string | null): string {
  return value || "";
}

function isTransportFailure(error: unknown) {
  const message = typeof error === "string" ? error : (error as Error)?.message || "";
  return /runtime unreachable|failed to fetch|fetch failed|networkerror/i.test(message);
}

function UsageBar({
  percent,
  warning,
  critical,
}: {
  percent: number;
  warning: boolean;
  critical: boolean;
}) {
  const color = critical ? "var(--kpi-danger)" : warning ? "var(--kpi-watch)" : "var(--kpi-healthy)";
  return (
    <div style={{ height: 8, borderRadius: 999, background: "rgba(255,255,255,0.08)", overflow: "hidden" }}>
      <div style={{ width: `${Math.min(percent, 100)}%`, height: "100%", background: color }} />
    </div>
  );
}

export default function SmritiStorageSettings({ onStatusChange }: SmritiStorageSettingsProps) {
  const [config, setConfig] = useState<SmritiStorageConfig | null>(null);
  const [usage, setUsage] = useState<StorageUsageReport | null>(null);
  const [folders, setFolders] = useState<WatchFolderStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [pruning, setPruning] = useState(false);
  const [confirmClearAll, setConfirmClearAll] = useState("");
  const [migrationTarget, setMigrationTarget] = useState("");
  const [migrating, setMigrating] = useState(false);
  const [migrationResult, setMigrationResult] = useState<SmritiMigrationResult | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const hiddenFolderPattern = /(^|\/)\.[^/]+/;

  const loadAll = useCallback(async () => {
    setRefreshing(true);
    try {
      const [nextConfig, nextUsage, nextFolders] = await Promise.all([
        runtimeRequest<SmritiStorageConfig>("/v1/smriti/storage"),
        runtimeRequest<StorageUsageReport>("/v1/smriti/storage/usage"),
        runtimeRequest<WatchFolderStatus[]>("/v1/smriti/watch-folders"),
      ]);
      setConfig(nextConfig);
      setUsage(nextUsage);
      setFolders(nextFolders);
      setLoadError(null);
    } catch (error) {
      setLoadError((error as Error).message);
      onStatusChange?.((error as Error).message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [onStatusChange]);

  useEffect(() => {
    void loadAll();
    const timer = window.setInterval(() => {
      void loadAll();
    }, 15000);
    return () => window.clearInterval(timer);
  }, [loadAll]);

  const saveConfig = async () => {
    if (!config) {
      return;
    }
    setSaving(true);
    try {
      const nextConfig = await runtimeRequest<SmritiStorageConfig>("/v1/smriti/storage", "PUT", config);
      setConfig(nextConfig);
      onStatusChange?.("Smriti storage settings saved");
      await loadAll();
    } catch (error) {
      onStatusChange?.((error as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const choosePath = async (key: "data_dir" | "frames_dir" | "thumbs_dir") => {
    const nextPath = await pickFolderPath();
    if (!nextPath) {
      return;
    }
    setConfig((current) => (current ? { ...current, [key]: nextPath } : current));
  };

  const addWatchFolder = async () => {
    const nextPath = await pickFolderPath();
    if (!nextPath) {
      return;
    }
    try {
      await runtimeRequest<WatchFolderStatus>("/v1/smriti/watch-folders", "POST", { path: nextPath });
      onStatusChange?.(`Watching ${nextPath}`);
      await loadAll();
    } catch (error) {
      onStatusChange?.((error as Error).message);
    }
  };

  const removeWatchFolder = async (path: string) => {
    try {
      await runtimeRequest(`/v1/smriti/watch-folders?path=${encodeURIComponent(path)}`, "DELETE");
      onStatusChange?.(`Stopped watching ${path}`);
      await loadAll();
    } catch (error) {
      onStatusChange?.((error as Error).message);
    }
  };

  const pruneStorage = async (payload: SmritiPruneRequest) => {
    if (payload.clear_all && confirmClearAll !== "CONFIRM_CLEAR_ALL") {
      onStatusChange?.("Type CONFIRM_CLEAR_ALL to clear all Smriti data");
      return;
    }
    setPruning(true);
    try {
      const result = await runtimeRequest<SmritiPruneResult>("/v1/smriti/storage/prune", "POST", payload);
      onStatusChange?.(`Pruned ${result.removed_media_records} records and freed ${result.removed_bytes_human}`);
      setConfirmClearAll("");
      await loadAll();
    } catch (error) {
      onStatusChange?.((error as Error).message);
    } finally {
      setPruning(false);
    }
  };

  const runMigration = async (dryRun: boolean) => {
    if (!migrationTarget.trim()) {
      onStatusChange?.("Choose a target directory before running migration");
      return;
    }
    setMigrating(true);
    try {
      const payload: SmritiMigrationRequest = {
        target_data_dir: migrationTarget.trim(),
        dry_run: dryRun,
      };
      const result = await runtimeRequest<SmritiMigrationResult>("/v1/smriti/storage/migrate", "POST", payload);
      setMigrationResult(result);
      onStatusChange?.(
        result.success
          ? dryRun
            ? `Dry run complete: ${result.files_moved} files would be copied`
            : `Migration complete: ${result.bytes_moved_human} copied`
          : `Migration failed: ${result.errors[0] || "unknown error"}`,
      );
      if (result.success && !dryRun) {
        await loadAll();
      }
    } catch (error) {
      setMigrationResult(null);
      onStatusChange?.((error as Error).message);
    } finally {
      setMigrating(false);
    }
  };

  if (loading || (!config && !loadError) || (!usage && !loadError)) {
    return <div className="muted">Loading Smriti storage settings...</div>;
  }

  if (!config || !usage) {
    return (
      <div className="muted">
        {loadError && isTransportFailure(loadError)
          ? "Runtime unreachable. Smriti storage settings will refresh when the backend reconnects."
          : loadError || "Smriti storage settings are unavailable right now."}
      </div>
    );
  }

  const hasHiddenFolderPath = [config.data_dir, config.frames_dir, config.thumbs_dir].some((value) =>
    hiddenFolderPattern.test(value || ""),
  );

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <section className="panel panel--memory">
        <div className="panel-head">
          <h4>Storage Usage</h4>
          <span>{refreshing ? "Refreshing..." : usage.total_human}</span>
        </div>
        {loadError ? (
          <p className="field-hint" style={{ color: "var(--warning)", marginBottom: "0.75rem" }}>
            {isTransportFailure(loadError)
              ? "Runtime unreachable. Displayed storage values may be stale until the backend reconnects."
              : loadError}
          </p>
        ) : null}
        <div style={{ display: "grid", gap: "0.85rem" }}>
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem" }}>
              <span className="muted">Budget</span>
              <span>{usage.max_storage_gb > 0 ? `${usage.budget_pct.toFixed(1)}% used` : "Unlimited"}</span>
            </div>
            <UsageBar percent={usage.budget_pct} warning={usage.budget_warning} critical={usage.budget_critical} />
          </div>
          <div className="status-grid" style={{ gridTemplateColumns: "repeat(5, minmax(0, 1fr))" }}>
            {[
              ["Frames", usage.frames_bytes],
              ["Thumbs", usage.thumbs_bytes],
              ["Database", usage.smriti_db_bytes],
              ["FAISS", usage.faiss_index_bytes],
              ["Templates", usage.templates_bytes],
            ].map(([label, value]) => (
              <div key={label as string} className="status-metric">
                <span>{label as string}</span>
                <strong>{bytesHuman(value as number)}</strong>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h4>Storage Location</h4>
          <span>Where Smriti stores its heavy data</span>
        </div>
        {hasHiddenFolderPath ? (
          <div
            style={{
              background: "rgba(67, 216, 201, 0.08)",
              border: "1px solid rgba(67, 216, 201, 0.2)",
              borderRadius: 16,
              padding: "0.85rem 1rem",
              marginBottom: "0.9rem",
              color: "var(--muted)",
            }}
          >
            <strong style={{ color: "var(--accent-2)" }}>ⓘ Hidden folders</strong> The dot in `.toori` makes
            the folder hidden on macOS and Linux. Use Browse or press <kbd>Cmd</kbd> + <kbd>Shift</kbd> +{" "}
            <kbd>.</kbd> in Finder to reveal it.
          </div>
        ) : null}
        <div style={{ display: "grid", gap: "0.8rem" }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
              gap: "1rem",
            }}
          >
            <label className="field">
              <span>
                Primary Smriti data directory
                <span
                  title="This stores the Smriti database, frames, thumbnails, and learned templates. The dot in .toori makes the folder hidden on macOS and Linux."
                  style={{ cursor: "help", marginLeft: "0.35rem" }}
                >
                  ⓘ
                </span>
              </span>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <input
                  value={watchPathToString(config.data_dir)}
                  onChange={(event) =>
                    setConfig((current) => (current ? { ...current, data_dir: event.target.value || null } : current))
                  }
                  placeholder="Choose where Smriti stores indexes, frames, and thumbnails"
                />
                <button type="button" onClick={() => void choosePath("data_dir")}>
                  Browse
                </button>
              </div>
            </label>

            <label className="field">
              <span>
                Frames directory
                <span
                  title="Full-resolution frames are stored here. Point this at an external drive if the media library is large."
                  style={{ cursor: "help", marginLeft: "0.35rem" }}
                >
                  ⓘ
                </span>
              </span>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <input
                  value={watchPathToString(config.frames_dir)}
                  onChange={(event) =>
                    setConfig((current) => (current ? { ...current, frames_dir: event.target.value || null } : current))
                  }
                  placeholder="Full-resolution frame storage"
                />
                <button type="button" onClick={() => void choosePath("frames_dir")}>
                  Browse
                </button>
              </div>
            </label>

            <label className="field">
              <span>
                Thumbnails directory
                <span
                  title="Small preview images are stored here. These are safe to keep on the main drive and usually take far less space."
                  style={{ cursor: "help", marginLeft: "0.35rem" }}
                >
                  ⓘ
                </span>
              </span>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <input
                  value={watchPathToString(config.thumbs_dir)}
                  onChange={(event) =>
                    setConfig((current) => (current ? { ...current, thumbs_dir: event.target.value || null } : current))
                  }
                  placeholder="Preview image storage"
                />
                <button type="button" onClick={() => void choosePath("thumbs_dir")}>
                  Browse
                </button>
              </div>
            </label>
          </div>

          <label className="field checkbox">
            <input
              type="checkbox"
              checked={config.store_full_frames}
              onChange={(event) => setConfig((current) => (current ? { ...current, store_full_frames: event.target.checked } : current))}
            />
            <span>Store full-resolution frames</span>
          </label>

          <label className="field">
            <span>Storage budget (GB)</span>
            <input
              type="number"
              min="0"
              step="0.5"
              value={config.max_storage_gb}
              onChange={(event) => setConfig((current) => (current ? { ...current, max_storage_gb: Number(event.target.value) } : current))}
              style={{ maxWidth: 160 }}
            />
            <small className="field-hint">Set to `0` for unlimited.</small>
          </label>

          <label className="field">
            <span>Thumbnail max dimension</span>
            <input
              type="number"
              min="64"
              max="1920"
              value={config.thumbnail_max_dim}
              onChange={(event) => setConfig((current) => (current ? { ...current, thumbnail_max_dim: Number(event.target.value) } : current))}
              style={{ maxWidth: 160 }}
            />
          </label>

          <label className="field checkbox">
            <input
              type="checkbox"
              checked={config.auto_prune_missing}
              onChange={(event) => setConfig((current) => (current ? { ...current, auto_prune_missing: event.target.checked } : current))}
            />
            <span>Auto-prune missing files</span>
          </label>

          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <button type="button" className="primary" onClick={() => void saveConfig()} disabled={saving}>
              {saving ? "Saving..." : "Save Storage Settings"}
            </button>
            <button type="button" onClick={() => void loadAll()} disabled={refreshing}>
              Refresh
            </button>
          </div>
        </div>
      </section>

      <section className="panel migration-panel">
        <div className="panel-head">
          <h4>Data Migration</h4>
          <span>Copy Smriti data to a new location without deleting the source</span>
        </div>
        <div style={{ display: "grid", gap: "0.8rem" }}>
          <label className="field">
            <span>Target directory</span>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <input
                value={migrationTarget}
                onChange={(event) => setMigrationTarget(event.target.value)}
                placeholder="Choose a new Smriti data directory"
              />
              <button
                type="button"
                onClick={() =>
                  void pickFolderPath().then((path) => {
                    if (path) {
                      setMigrationTarget(path);
                    }
                  })
                }
              >
                Browse
              </button>
            </div>
            <small className="field-hint">
              Migration preserves the original data. Delete the old location manually after you verify the new one.
            </small>
          </label>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <button type="button" onClick={() => void runMigration(true)} disabled={migrating}>
              {migrating ? "Running…" : "Dry Run"}
            </button>
            <button type="button" className="primary" onClick={() => void runMigration(false)} disabled={migrating}>
              {migrating ? "Migrating…" : "Run Migration"}
            </button>
          </div>
          {migrationResult ? (
            <div className={migrationResult.success ? "migration-result success" : "migration-result failed"}>
              <strong>{migrationResult.success ? "Migration finished" : "Migration failed"}</strong>
              <span>
                {migrationResult.files_moved} files · {migrationResult.bytes_moved_human} · {migrationResult.dry_run ? "dry run" : "live"}
              </span>
              <span>Target: {migrationResult.new_data_dir}</span>
              {migrationResult.errors.length > 0 ? (
                <code style={{ whiteSpace: "pre-wrap" }}>{migrationResult.errors.join("\n")}</code>
              ) : null}
            </div>
          ) : null}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h4>Watch Folders</h4>
          <span>Live folder status and ingestion coverage</span>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.75rem" }}>
          <button type="button" className="primary" onClick={() => void addWatchFolder()}>
            Add Folder
          </button>
          <small className="field-hint" style={{ alignSelf: "center" }}>
            Adds a persistent watch entry and queues existing supported media.
          </small>
        </div>
        {folders.length === 0 ? (
          <p className="muted">No watched folders yet.</p>
        ) : (
          <div style={{ display: "grid", gap: "0.75rem" }}>
            {folders.map((folder) => (
              <div
                key={folder.path}
                style={{
                  border: "1px solid var(--line)",
                  borderRadius: 18,
                  padding: "0.9rem",
                  background: "rgba(255,255,255,0.03)",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem" }}>
                  <div style={{ minWidth: 0 }}>
                    <code style={{ wordBreak: "break-all" }}>{folder.path}</code>
                    <div className="muted" style={{ fontSize: "0.8rem", marginTop: "0.2rem" }}>
                      {folder.exists ? "Exists" : "Missing"} · {folder.is_accessible ? "Accessible" : "Inaccessible"} ·{" "}
                      {folder.watchdog_active ? "Watching" : "Not watching"}
                    </div>
                    {folder.error ? (
                      <div style={{ color: "var(--kpi-danger)", fontSize: "0.8rem", marginTop: "0.2rem" }}>
                        {folder.error}
                      </div>
                    ) : null}
                  </div>
                  <button type="button" onClick={() => void removeWatchFolder(folder.path)}>
                    Remove
                  </button>
                </div>
                <div className="status-grid" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))", marginTop: "0.8rem" }}>
                  <div className="status-metric">
                    <span>Total</span>
                    <strong>{folder.media_count_total.toLocaleString()}</strong>
                  </div>
                  <div className="status-metric">
                    <span>Indexed</span>
                    <strong>{folder.media_count_indexed.toLocaleString()}</strong>
                  </div>
                  <div className="status-metric">
                    <span>Pending</span>
                    <strong>{folder.media_count_pending.toLocaleString()}</strong>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-head">
          <h4>Prune Storage</h4>
          <span>Clean old, missing, or failed records</span>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <button
            type="button"
            onClick={() =>
              void pruneStorage({
                older_than_days: null,
                remove_missing_files: true,
                remove_failed: false,
                clear_all: false,
                confirm_clear_all: "",
              })
            }
            disabled={pruning}
          >
            Remove missing files
          </button>
          <button
            type="button"
            onClick={() =>
              void pruneStorage({
                older_than_days: null,
                remove_missing_files: false,
                remove_failed: true,
                clear_all: false,
                confirm_clear_all: "",
              })
            }
            disabled={pruning}
          >
            Remove failed records
          </button>
        </div>

        <div
          style={{
            marginTop: "0.9rem",
            padding: "0.9rem",
            borderRadius: 18,
            border: "1px solid rgba(230,57,70,0.25)",
            background: "rgba(230,57,70,0.05)",
          }}
        >
          <div className="field" style={{ marginBottom: "0.5rem" }}>
            <span>Type `CONFIRM_CLEAR_ALL` to wipe all Smriti storage</span>
            <input
              value={confirmClearAll}
              onChange={(event) => setConfirmClearAll(event.target.value)}
              placeholder="CONFIRM_CLEAR_ALL"
            />
          </div>
          <button
            type="button"
            onClick={() =>
              void pruneStorage({
                older_than_days: null,
                remove_missing_files: false,
                remove_failed: false,
                clear_all: true,
                confirm_clear_all: confirmClearAll,
              })
            }
            disabled={pruning || confirmClearAll !== "CONFIRM_CLEAR_ALL"}
            style={{ color: "var(--kpi-danger)" }}
          >
            {pruning ? "Pruning..." : "Clear All Smriti Data"}
          </button>
        </div>
      </section>
    </div>
  );
}
