import { useEffect, useMemo, useState } from "react";
import { z } from "zod";
import type { TrainingAxis } from "../trainingContract";
import { runningTrainingStates as running } from "../trainingRunSelection";
import {
  detailSchema,
  trainingRequest,
  type RunDetail,
  type TrainingRun,
} from "../trainingApi";
const commandSchema = z.object({
  id: z.string(),
  artifact: z.string().optional(),
});
const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);
export function useRunDashboard(
  runId: string,
  detail: RunDetail | null,
  runs: TrainingRun[],
  setError: (error: string) => void,
  setNotice: (notice: string) => void,
) {
  const [reportBaseURL, setReportBaseURL] = useState(
    () => `http://localhost:${location.port || "8173"}`,
  );
  const [pauseRequested, setPauseRequested] = useState<string | null>(null);
  const [includeData, setIncludeData] = useState(false);
  const [dashboard, setDashboard] = useState<
    | "metrics"
    | "samples"
    | "hardware"
    | "logs"
    | "checkpoints"
    | "chat"
    | "artifacts"
  >("metrics");
  const [axis, setAxis] = useState<TrainingAxis>("step");
  const [smooth, setSmooth] = useState(false);
  const [comparisonId, setComparisonId] = useState("");
  const [comparison, setComparison] = useState<RunDetail | null>(null);
  useEffect(() => {
    if (detail?.reporting_url)
      setReportBaseURL(
        detail.reporting_url.replace(/\/api\/training\/reports\/[^/]+$/, ""),
      );
  }, [detail?.run.id, detail?.reporting_url]);

  useEffect(() => {
    if (!comparisonId) {
      setComparison(null);
      return;
    }
    const controller = new AbortController();
    void trainingRequest(
      `/runs/${encodeURIComponent(comparisonId)}`,
      detailSchema,
      undefined,
      undefined,
      controller.signal,
    )
      .then((value) => {
        if (!controller.signal.aborted) setComparison(value);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(message(e));
      });
    return () => controller.abort();
  }, [comparisonId]);

  async function command(kind: string) {
    const result = await trainingRequest(
      `/runs/${encodeURIComponent(runId)}/commands`,
      commandSchema,
      { kind, include_data: includeData },
    );
    if (kind === "stop") setPauseRequested(runId);
    setNotice(
      result.artifact
        ? "Snapshot saved. Open Checkpoints to download it."
        : kind === "snapshot"
          ? "Snapshot requested at the next safe training boundary. A custom script must support checkpoint requests."
          : kind === "stop"
            ? "Stop requested. Waiting for the next safe training boundary…"
            : "Force stop requested.",
    );
    if (kind === "snapshot") setDashboard("checkpoints");
  }
  const events = detail?.events;
  const checkpoints = useMemo(
    () => events?.filter((e) => e.kind === "checkpoint") ?? [],
    [events],
  );
  const scriptCapabilities =
    detail?.events
      .filter(
        (event) =>
          event.kind === "capabilities" &&
          (event.channel ?? "training") === "training",
      )
      .at(-1)?.payload ??
    detail?.script_capabilities ??
    {};
  const customRun = detail?.run.recipe === "custom" && !detail.run.code_first;
  const active = detail && running.has(detail.status.state);
  useEffect(() => {
    if (pauseRequested && (pauseRequested !== runId || (detail && !active)))
      setPauseRequested(null);
  }, [pauseRequested, runId, detail?.run.id, active]);
  const otherActiveRun =
    detail && !active
      ? runs.find(
          (run) =>
            run.id !== detail.run.id &&
            run.stage_id === detail.run.stage_id &&
            running.has(run.status?.state ?? ""),
        )
      : undefined;
  return {
    dashboard,
    setDashboard,
    axis,
    setAxis,
    smooth,
    setSmooth,
    comparisonId,
    setComparisonId,
    comparison,
    reportBaseURL,
    setReportBaseURL,
    pauseRequested,
    includeData,
    setIncludeData,
    command,
    checkpoints,
    scriptCapabilities,
    customRun,
    active,
    otherActiveRun,
  };
}
