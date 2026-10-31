import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { Activity, X } from "lucide-react";
import { z } from "zod";
import { Dialog } from "./ui";
import { trainingRequest } from "./trainingApi";
import "./activeTraining.css";

const activeRunsSchema = z.array(
  z.object({
    id: z.string(),
    project_id: z.string(),
    project_name: z.string(),
    stage_id: z.string(),
    name: z.string(),
    created: z.string(),
    state: z.enum(["starting", "preparing", "running", "evaluating"]),
  }),
);

export function ActiveTrainingRuns() {
  const [runs, setRuns] = useState<z.infer<typeof activeRunsSchema>>([]);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let disposed = false;
    let pending: AbortController | null = null;
    async function check() {
      if (pending || document.visibilityState === "hidden") return;
      const controller = new AbortController();
      pending = controller;
      const timeout = window.setTimeout(() => controller.abort(), 10000);
      try {
        const next = await trainingRequest(
          "/active-runs",
          activeRunsSchema,
          undefined,
          undefined,
          controller.signal,
        );
        if (!disposed) {
          setRuns(next);
          setError("");
        }
      } catch (failure) {
        if (!disposed)
          setError(
            failure instanceof Error ? failure.message : String(failure),
          );
      } finally {
        window.clearTimeout(timeout);
        pending = null;
      }
    }
    void check();
    const timer = window.setInterval(() => void check(), 5000);
    const focus = () => void check();
    window.addEventListener("focus", focus);
    document.addEventListener("visibilitychange", focus);
    return () => {
      disposed = true;
      pending?.abort();
      window.clearInterval(timer);
      window.removeEventListener("focus", focus);
      document.removeEventListener("visibilitychange", focus);
    };
  }, [refresh]);
  if (!runs.length && !open) return null;
  return (
    <>
      <button
        className="subtle-button active-training-button"
        aria-label={`Active training runs: ${runs.length}`}
        title={
          error
            ? "Training status may be stale"
            : `${runs.length} active training ${runs.length === 1 ? "run" : "runs"}`
        }
        aria-haspopup="dialog"
        onClick={() => {
          setOpen(true);
          setRefresh((value) => value + 1);
        }}
      >
        <Activity size={16} aria-hidden="true" />
        <span className="active-training-count">{runs.length}</span>
      </button>
      {open &&
        createPortal(
          <Dialog
            label="Active training runs"
            className="training-controls training-history-dialog active-training-dialog"
            onClose={() => setOpen(false)}
          >
            <div className="training-history-heading">
              <h2>Active training</h2>
              <button
                className="modal-close"
                aria-label="Close active training"
                onClick={() => setOpen(false)}
              >
                <X size={18} />
              </button>
            </div>
            <div className="training-history-content">
              <p className="training-muted">
                Jobs continue across projects. You can launch more than one run.
              </p>
              {error && (
                <p role="alert">
                  Updates unavailable; shown runs may be stale.{" "}
                  <button onClick={() => setRefresh((value) => value + 1)}>
                    Retry
                  </button>
                </p>
              )}
              {!runs.length && !error && (
                <p role="status">No active training runs.</p>
              )}
              <ul className="active-training-list">
                {runs.map((run) => (
                  <li key={run.id}>
                    <a
                      href={`?${new URLSearchParams({ model: run.project_id, view: "training", run: run.id, stage: run.stage_id })}`}
                    >
                      <strong>{run.project_name}</strong>
                      <span>
                        {run.name} · {run.state}
                      </span>
                      <small>
                        Started {new Date(run.created).toLocaleString()}
                      </small>
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </Dialog>,
          document.body,
        )}
    </>
  );
}
