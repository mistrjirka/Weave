import {
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import { z } from "zod";
import {
  detailSchema,
  monitorEventsSchema,
  monitorStatusSchema,
  runSchema,
  trainingRequest,
  TrainingRequestError,
  type RunDetail,
  type TrainingRun,
} from "../trainingApi";
import { mergeMonitorEvents } from "../trainingMonitor";
import { preferredTrainingRun } from "../trainingRunSelection";

const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

type Setter<T> = Dispatch<SetStateAction<T>>;

export function useRunMonitor(
  runId: string,
  authenticated: boolean | undefined,
  projectId: string,
  selected: string,
  setRuns: Setter<TrainingRun[]>,
  setRunId: Setter<string>,
  setSelected: Setter<string>,
  setSetupOpen: Setter<boolean>,
  setNotice: Setter<string>,
) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [monitorError, setMonitorError] = useState("");
  const [lastRefreshed, setLastRefreshed] = useState<number | null>(null);
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  useEffect(() => {
    if (!runId || !authenticated) {
      setDetail(null);
      setMonitorError("");
      setLastRefreshed(null);
      return;
    }
    const selectedAtStart = selectedRef.current;
    const controller = new AbortController();
    setDetail(null);
    setMonitorError("");
    setLastRefreshed(null);
    let pending = false;
    let cursor = "";
    let lastMetadata = 0;
    let lastState = "";
    const runEndpoint = `/runs/${encodeURIComponent(runId)}`;
    const recoverMissingRun = async (signal: AbortSignal) => {
      const list = await trainingRequest(
        `/runs?project_id=${encodeURIComponent(projectId)}`,
        z.array(runSchema),
        undefined,
        undefined,
        signal,
      );
      if (signal.aborted) return;
      const replacement = preferredTrainingRun(
        list.filter((run) => run.id !== runId),
        selectedAtStart,
      );
      setRuns(list);
      setMonitorError("");
      setRunId(replacement?.id ?? "");
      if (replacement) setSelected(replacement.stage_id);
      else setSetupOpen(true);
      setNotice(
        replacement
          ? "The selected run is unavailable. Opened the latest available run."
          : "The selected run is unavailable. No saved runs were found for this project.",
      );
    };
    const refresh = async () => {
      if (pending) return;
      pending = true;
      const request = new AbortController();
      const abortRequest = () => request.abort();
      controller.signal.addEventListener("abort", abortRequest, { once: true });
      const deadline = window.setTimeout(abortRequest, 10000);
      try {
        if (!lastMetadata) {
          const value = await trainingRequest(
            runEndpoint,
            detailSchema,
            undefined,
            undefined,
            request.signal,
          );
          if (request.signal.aborted)
            throw new Error("Monitor refresh timed out. Retrying.");
          cursor = value.events.at(-1)?.id ?? "";
          lastState = value.status.state;
          lastMetadata = Date.now();
          setDetail({
            ...value,
            first_event_time:
              value.first_event_time ?? value.events[0]?.time ?? null,
          });
          setSelected(value.run.stage_id);
        } else {
          const [status, page] = await Promise.all([
            trainingRequest(
              `${runEndpoint}/status?compact=true`,
              monitorStatusSchema,
              undefined,
              undefined,
              request.signal,
            ),
            trainingRequest(
              `${runEndpoint}/event-records?limit=500&after=${encodeURIComponent(cursor)}`,
              monitorEventsSchema,
              undefined,
              undefined,
              request.signal,
            ),
          ]);
          if (request.signal.aborted)
            throw new Error("Monitor refresh timed out. Retrying.");
          const changedState = lastState !== status.status.state;
          const relevant = page.events.some((event) =>
            ["checkpoint", "command", "configuration", "capabilities"].includes(
              event.kind,
            ),
          );
          cursor = page.next_after;
          lastState = status.status.state;
          setDetail((current) =>
            current
              ? {
                  ...current,
                  status: status.status,
                  events: mergeMonitorEvents(current.events, page.events),
                  first_event_time:
                    current.first_event_time ?? page.events[0]?.time ?? null,
                }
              : current,
          );
          setLastRefreshed(Date.now());
          if (changedState || relevant || Date.now() - lastMetadata >= 30000) {
            lastMetadata = Date.now();
            const [value, list] = await Promise.all([
              trainingRequest(
                runEndpoint,
                detailSchema,
                undefined,
                undefined,
                request.signal,
              ),
              trainingRequest(
                `/runs?project_id=${encodeURIComponent(projectId)}`,
                z.array(runSchema),
                undefined,
                undefined,
                request.signal,
              ),
            ]);
            if (request.signal.aborted)
              throw new Error("Monitor refresh timed out. Retrying.");
            setDetail((current) =>
              current
                ? {
                    ...value,
                    status: current.status,
                    events: current.events,
                    first_event_time:
                      current.first_event_time ??
                      value.first_event_time ??
                      null,
                  }
                : current,
            );
            setRuns(list);
          }
        }
        if (!controller.signal.aborted) {
          setMonitorError("");
          setLastRefreshed(Date.now());
        }
      } catch (e) {
        if (controller.signal.aborted) return;
        if (request.signal.aborted) {
          setMonitorError("Monitor refresh timed out. Retrying.");
        } else if (
          e instanceof TrainingRequestError &&
          e.status === 404 &&
          (e.path === runEndpoint || e.path.startsWith(`${runEndpoint}/`))
        ) {
          try {
            await recoverMissingRun(request.signal);
          } catch (recoveryError) {
            if (!controller.signal.aborted)
              setMonitorError(
                request.signal.aborted
                  ? "Monitor refresh timed out. Retrying."
                  : message(recoveryError),
              );
          }
        } else setMonitorError(message(e));
      } finally {
        window.clearTimeout(deadline);
        request.abort();
        controller.signal.removeEventListener("abort", abortRequest);
        pending = false;
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [
    runId,
    authenticated,
    projectId,
    setRuns,
    setRunId,
    setSelected,
    setSetupOpen,
    setNotice,
  ]);
  return { detail, setDetail, monitorError, lastRefreshed };
}
