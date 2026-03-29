import { useCallback, useEffect, useRef, useState } from "react";
import type {
  SmritiMandalaData,
  SmritiMetrics,
  SmritiPersonJournal,
  SmritiRecallRequest,
  SmritiRecallResult,
  SmritiSection,
  SmritiStatus,
} from "../types";

type UseSmritiStateOptions = {
  runtimeRequest: <T>(path: string, method?: string, body?: unknown) => Promise<T>;
  sessionId: string;
};

export function useSmritiState({ runtimeRequest, sessionId }: UseSmritiStateOptions) {
  const [section, setSection] = useState<SmritiSection>("mandala");
  const [mandalaData, setMandalaData] = useState<SmritiMandalaData | null>(null);
  const [recallResults, setRecallResults] = useState<SmritiRecallResult[]>([]);
  const [recallQuery, setRecallQuery] = useState("");
  const [recallBusy, setRecallBusy] = useState(false);
  const [selectedMedia, setSelectedMedia] = useState<SmritiRecallResult | null>(null);
  const [personFilter, setPersonFilter] = useState("");
  const [locationFilter, setLocationFilter] = useState("");
  const [minConfidence, setMinConfidence] = useState(0);
  const [timeRangeDays, setTimeRangeDays] = useState(0);
  const [personName, setPersonName] = useState("");
  const [personJournal, setPersonJournal] = useState<SmritiPersonJournal | null>(null);
  const [metrics, setMetrics] = useState<SmritiMetrics | null>(null);
  const [status, setStatus] = useState<SmritiStatus | null>(null);
  const [ingestionFolder, setIngestionFolder] = useState("");
  const [ingestionBusy, setIngestionBusy] = useState(false);
  const [ingestionStatus, setIngestionStatus] = useState("Ready to index");
  const [totalIndexed, setTotalIndexed] = useState(0);
  const metricsIntervalRef = useRef<number | null>(null);

  const loadMandala = useCallback(async () => {
    try {
      const data = await runtimeRequest<SmritiMandalaData>("/v1/smriti/clusters");
      setMandalaData(data);
    } catch {
      // Non-critical surface.
    }
  }, [runtimeRequest]);

  const runRecall = useCallback(
    async (request: SmritiRecallRequest) => {
      setRecallBusy(true);
      try {
        const response = await runtimeRequest<{
          query: string;
          results: SmritiRecallResult[];
          total_searched: number;
          setu_ms: number;
        }>("/v1/smriti/recall", "POST", {
          session_id: sessionId,
          top_k: 20,
          ...request,
        });
        setRecallResults(response.results);
        setTotalIndexed(response.total_searched);
      } catch (err) {
        setIngestionStatus((err as Error).message);
      } finally {
        setRecallBusy(false);
      }
    },
    [runtimeRequest, sessionId],
  );

  const openDeepdive = useCallback((media: SmritiRecallResult) => {
    setSelectedMedia(media);
    setSection("deepdive");
  }, []);

  const closeDeepdive = useCallback(() => {
    setSelectedMedia(null);
    setSection("recall");
  }, []);

  const loadPersonJournal = useCallback(
    async (name: string) => {
      if (!name.trim()) {
        return;
      }
      try {
        const journal = await runtimeRequest<SmritiPersonJournal>(
          `/v1/smriti/person/${encodeURIComponent(name.trim())}/journal`,
        );
        setPersonJournal(journal);
      } catch {
        setPersonJournal({ person_name: name, entries: [], count: 0 });
      }
    },
    [runtimeRequest],
  );

  const tagPerson = useCallback(
    async (mediaId: string, name: string) => {
      await runtimeRequest("/v1/smriti/tag/person", "POST", {
        media_id: mediaId,
        person_name: name,
        confirmed: true,
      });
      await loadPersonJournal(name);
    },
    [runtimeRequest, loadPersonJournal],
  );

  const ingestFolder = useCallback(
    async (folderPath: string) => {
      if (!folderPath.trim()) {
        return;
      }
      setIngestionBusy(true);
      setIngestionStatus(`Queuing folder: ${folderPath}`);
      try {
        const response = await runtimeRequest<{ queued: number; status: string }>(
          "/v1/smriti/ingest",
          "POST",
          { folder_path: folderPath.trim() },
        );
        setIngestionStatus(`Queued ${response.queued} files - ${response.status}`);
      } catch (err) {
        setIngestionStatus((err as Error).message);
      } finally {
        setIngestionBusy(false);
      }
    },
    [runtimeRequest],
  );

  const ingestFile = useCallback(
    async (filePath: string) => {
      if (!filePath.trim()) {
        return;
      }
      setIngestionBusy(true);
      try {
        const response = await runtimeRequest<{ queued: number; status: string }>(
          "/v1/smriti/ingest",
          "POST",
          { file_path: filePath.trim() },
        );
        setIngestionStatus(response.status === "duplicate" ? "Already indexed" : "Queued for indexing");
      } catch (err) {
        setIngestionStatus((err as Error).message);
      } finally {
        setIngestionBusy(false);
      }
    },
    [runtimeRequest],
  );

  const loadMetrics = useCallback(async () => {
    try {
      const [metricsResponse, statusResponse] = await Promise.all([
        runtimeRequest<SmritiMetrics>("/v1/smriti/metrics"),
        runtimeRequest<SmritiStatus>("/v1/smriti/status"),
      ]);
      setMetrics(metricsResponse);
      setStatus(statusResponse);
    } catch {
      // Non-critical HUD.
    }
  }, [runtimeRequest]);

  useEffect(() => {
    void loadMandala();
    void loadMetrics();
    metricsIntervalRef.current = window.setInterval(() => {
      void loadMetrics();
    }, 5000);
    return () => {
      if (metricsIntervalRef.current != null) {
        window.clearInterval(metricsIntervalRef.current);
      }
    };
  }, [loadMandala, loadMetrics]);

  useEffect(() => {
    if (!recallQuery.trim()) {
      setRecallResults([]);
      return;
    }
    const timeout = window.setTimeout(() => {
      const now = new Date();
      const timeStart =
        timeRangeDays > 0
          ? new Date(now.getTime() - timeRangeDays * 24 * 60 * 60 * 1000).toISOString()
          : null;
      void runRecall({
        query: recallQuery,
        person_filter: personFilter || null,
        location_filter: locationFilter || null,
        time_start: timeStart,
        min_confidence: minConfidence,
      });
    }, 350);
    return () => window.clearTimeout(timeout);
  }, [locationFilter, minConfidence, personFilter, recallQuery, runRecall, timeRangeDays]);

  return {
    section,
    setSection,
    mandalaData,
    recallResults,
    recallQuery,
    setRecallQuery,
    recallBusy,
    selectedMedia,
    personFilter,
    setPersonFilter,
    locationFilter,
    setLocationFilter,
    minConfidence,
    setMinConfidence,
    timeRangeDays,
    setTimeRangeDays,
    personName,
    setPersonName,
    personJournal,
    metrics,
    status,
    ingestionFolder,
    setIngestionFolder,
    ingestionBusy,
    ingestionStatus,
    totalIndexed,
    loadMandala,
    runRecall,
    openDeepdive,
    closeDeepdive,
    loadPersonJournal,
    tagPerson,
    ingestFolder,
    ingestFile,
    loadMetrics,
  };
}
