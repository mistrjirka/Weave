import { useState } from "react";
import { createPortal } from "react-dom";
import { Dialog } from "./ui";
import {
  runSchema,
  trainingRequest,
  type RunDetail,
  type TrainingRun,
} from "./trainingApi";
import { runningTrainingStates } from "./trainingRunSelection";

export function CustomRunControls({
  detail,
  enabled,
  onRestart,
}: {
  detail: RunDetail;
  enabled: boolean;
  onRestart: (run: TrainingRun) => void;
}) {
  const [open, setOpen] = useState(false);
  const [source, setSource] = useState<"previous" | "edited">("previous");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  if (
    detail.run.execution !== "local" ||
    runningTrainingStates.has(detail.status.state) ||
    !detail.restart_context?.exact_resume
  )
    return null;
  async function resume() {
    setBusy(true);
    setError("");
    try {
      const run = await trainingRequest(
        `/runs/${encodeURIComponent(detail.run.id)}/restart`,
        runSchema,
        { script_source: source },
      );
      onRestart(run);
      setOpen(false);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <button
        disabled={!enabled}
        onClick={() => {
          setError("");
          setOpen(true);
        }}
      >
        Resume training
      </button>
      {open &&
        createPortal(
          <Dialog
            label="Resume custom training"
            className="training-controls"
            onClose={() => {
              if (!busy) setOpen(false);
            }}
          >
            <header className="training-section-heading">
              <h2>Resume training</h2>
              <button
                aria-label="Close dialog"
                disabled={busy}
                onClick={() => setOpen(false)}
              >
                ×
              </button>
            </header>
            <p>
              The script declares full resume support. Weave snapshots each
              restart attempt; your script restores its named components and
              data state.
            </p>
            <label>
              Resume script
              <select
                value={source}
                disabled={busy}
                onChange={(event) =>
                  setSource(
                    event.target.value === "edited" ? "edited" : "previous",
                  )
                }
              >
                <option value="previous">Recorded script</option>
                <option value="edited">Saved edited script</option>
              </select>
            </label>
            <p className="training-muted">
              {source === "edited"
                ? "Save Python editor changes first. Tensor shape mismatches reject the restore; you can fix the script and retry."
                : "Use the frozen script and helpers from this run."}
            </p>
            {error && <p role="alert">{error}</p>}
            <button
              className="primary"
              disabled={busy}
              onClick={() => void resume()}
            >
              {busy ? "Restoring…" : "Resume from checkpoint"}
            </button>
          </Dialog>,
          document.body,
        )}
    </>
  );
}
