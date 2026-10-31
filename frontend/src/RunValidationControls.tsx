import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Settings2, RotateCw, X } from "lucide-react";
import { z } from "zod";
import { Dialog } from "./ui";
import {
  detailSchema,
  runSchema,
  trainingRequest,
  type RunDetail,
  type TrainingGPU,
  type TrainingRun,
  type ValidationSettings,
} from "./trainingApi";

const activeStates = new Set([
  "starting",
  "preparing",
  "running",
  "evaluating",
]);
const commandResponse = z.object({ id: z.string() });

export function RunValidationControls({
  detail,
  gpus,
  enabled,
  onRestart,
}: {
  detail: RunDetail;
  gpus: TrainingGPU[];
  enabled: boolean;
  onRestart: (run: TrainingRun) => void;
}) {
  const [open, setOpen] = useState(false);
  const [settings, setSettings] = useState<ValidationSettings>({
    mode: "async",
    device: "cpu",
    eval_every: 50,
    eval_batches: 5,
  });
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [requested, setRequested] = useState<string | null>(null);
  const [confirmRestart, setConfirmRestart] = useState(false);
  const abort = useRef<AbortController | null>(null);
  useEffect(() => () => abort.current?.abort(), []);
  const control = detail.validation_control;
  const active = activeStates.has(detail.status.state);
  const checkpoints = detail.events.filter(
    (event) => event.kind === "checkpoint" && !event.payload.remote,
  );
  const latest = checkpoints.at(-1);
  const restorable = latest?.payload.metadata;
  const exactResume =
    typeof restorable === "object" &&
    restorable !== null &&
    "exact_resume" in restorable &&
    restorable.exact_resume === true;
  const canRestart =
    enabled &&
    detail.run.execution === "local" &&
    !!detail.run.code_first &&
    (active
      ? !latest || exactResume
      : exactResume && detail.status.state !== "completed");
  const applied =
    requested !== null && control?.applied_command_id === requested;
  const pending = !!control?.pending || (requested !== null && !applied);

  function show() {
    setSettings(control?.pending?.settings ?? control?.current ?? settings);
    setError("");
    setMessage("");
    setRequested(null);
    setConfirmRestart(false);
    setOpen(true);
  }

  async function apply() {
    setBusy(true);
    setError("");
    setRequested(null);
    setMessage("Sending validation change…");
    try {
      const result = await trainingRequest(
        `/runs/${encodeURIComponent(detail.run.id)}/validation`,
        commandResponse,
        settings,
        "PUT",
      );
      setRequested(result.id);
      setMessage(
        "Waiting for the trainer to apply this change between optimizer updates…",
      );
    } catch (failure) {
      setMessage("");
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }

  async function restart() {
    const controller = new AbortController();
    abort.current = controller;
    setBusy(true);
    setError("");
    const endpoint = `/runs/${encodeURIComponent(detail.run.id)}`;
    try {
      if (active) {
        setMessage(
          "Saving a checkpoint and stopping safely. Training progress remains visible behind this dialog.",
        );
        await trainingRequest(
          endpoint + "/commands",
          commandResponse,
          { kind: "stop" },
          "POST",
          controller.signal,
        );
        // A reload may interrupt this wait; the stopped run still offers Resume.
        while (!controller.signal.aborted) {
          const current = await trainingRequest(
            endpoint,
            detailSchema,
            undefined,
            undefined,
            controller.signal,
          );
          if (!activeStates.has(current.status.state)) break;
          await new Promise<void>((resolve, reject) => {
            const cancelled = () => {
              window.clearTimeout(timer);
              reject(new DOMException("Cancelled", "AbortError"));
            };
            const timer = window.setTimeout(() => {
              controller.signal.removeEventListener("abort", cancelled);
              resolve();
            }, 1500);
            controller.signal.addEventListener("abort", cancelled, {
              once: true,
            });
          });
        }
      }
      setMessage("Restoring the checkpoint in a new training process…");
      const run = await trainingRequest(
        endpoint + "/restart",
        runSchema,
        { validation: settings },
        "POST",
        controller.signal,
      );
      onRestart(run);
      setOpen(false);
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  return (
    <>
      <button onClick={show}>
        <Settings2 size={14} aria-hidden="true" /> Validation settings
      </button>
      {!active && canRestart && (
        <button
          onClick={() => {
            show();
            setConfirmRestart(true);
          }}
        >
          <RotateCw size={14} aria-hidden="true" /> Resume training
        </button>
      )}
      {open &&
        createPortal(
          <Dialog
            label="Run validation settings"
            onClose={() => {
              if (!busy) setOpen(false);
            }}
            className="training-controls training-validation-dialog"
          >
            <div className="training-history-heading">
              <h2>Validation settings</h2>
              <button
                className="modal-close"
                aria-label="Close validation settings"
                disabled={busy}
                onClick={() => setOpen(false)}
              >
                <X size={18} />
              </button>
            </div>
            <div className="training-validation-content">
              <p className="training-muted">
                For this run. Changes apply between optimizer updates. A
                validation job already running keeps its original device.
              </p>
              {!control?.supported && (
                <p role="status" className="training-alert">
                  This run needs a checkpoint restart to enable live changes.
                  Restart resumes its recorded script; edits in your stage are
                  not applied.
                </p>
              )}
              <fieldset
                disabled={busy || pending}
                className="training-validation-fields"
              >
                <label>
                  Validation execution
                  <select
                    aria-label="Run validation execution"
                    value={settings.mode}
                    onChange={(event) =>
                      setSettings((current) => ({
                        ...current,
                        mode: z
                          .enum(["async", "inline"])
                          .parse(event.target.value),
                      }))
                    }
                  >
                    <option value="async">
                      Asynchronous · continue training
                    </option>
                    <option value="inline">
                      Inline · pause for validation
                    </option>
                  </select>
                </label>
                <label>
                  Validation device
                  <select
                    aria-label="Run validation device"
                    value={
                      settings.mode === "inline" ? "training" : settings.device
                    }
                    disabled={settings.mode === "inline"}
                    onChange={(event) =>
                      setSettings((current) => ({
                        ...current,
                        device:
                          event.target.value === "cpu"
                            ? "cpu"
                            : Number(event.target.value),
                      }))
                    }
                  >
                    {settings.mode === "inline" && (
                      <option value="training">Training device</option>
                    )}
                    <option value="cpu">CPU</option>
                    {gpus.map((gpu) => (
                      <option key={gpu.uuid} value={gpu.index}>
                        GPU {gpu.index} · {gpu.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Validate every
                  <input
                    aria-label="Validation interval"
                    type="number"
                    min={1}
                    max={1000000}
                    value={settings.eval_every}
                    onChange={(event) =>
                      setSettings((current) => ({
                        ...current,
                        eval_every: event.target.valueAsNumber,
                      }))
                    }
                  />
                  <small>Successful optimizer updates</small>
                </label>
                <label>
                  Maximum validation batches
                  <input
                    aria-label="Maximum validation batches"
                    type="number"
                    min={1}
                    max={100000}
                    value={settings.eval_batches}
                    onChange={(event) =>
                      setSettings((current) => ({
                        ...current,
                        eval_batches: event.target.valueAsNumber,
                      }))
                    }
                  />
                  <small>Per validation pass</small>
                </label>
              </fieldset>
              <p className="training-muted">
                CPU validation uses FP32. A training GPU needs memory for
                another model copy and shares its compute with validation.
              </p>
              {error && (
                <p role="alert" className="training-alert">
                  {error}
                </p>
              )}
              {(message || pending || applied) && (
                <p role="status">
                  {applied
                    ? `Applied at optimizer update ${control?.applied_at_step}.`
                    : message ||
                      "Waiting for the trainer to apply the requested settings…"}
                </p>
              )}
              {confirmRestart ? (
                <div className="training-restart-confirmation">
                  <p>
                    {active
                      ? "Save a checkpoint, stop safely, and resume"
                      : "Resume"}{" "}
                    the frozen script with these validation settings. Optimizer,
                    scheduler, RNG and data cursor are restored. This does not
                    apply edits from your stage script.
                  </p>
                  <dl className="training-restart-details">
                    <div>
                      <dt>Frozen script</dt>
                      <dd>
                        <code>{detail.directory}/train.py</code>
                      </dd>
                    </div>
                    <div>
                      <dt>Checkpoint</dt>
                      <dd>
                        {active
                          ? "A new checkpoint saved at the next safe boundary"
                          : latest
                            ? `Update ${latest.step.toLocaleString()} · ${String(latest.payload.path)}`
                            : "No resumable checkpoint available"}
                      </dd>
                    </div>
                    <div>
                      <dt>Edited script</dt>
                      <dd>
                        Left unchanged. To train with edits, create a new stage.
                      </dd>
                    </div>
                  </dl>
                  <div className="training-actions">
                    <button
                      disabled={busy}
                      onClick={() => setConfirmRestart(false)}
                    >
                      Cancel restart
                    </button>
                    <button
                      className="primary"
                      disabled={
                        busy ||
                        !Number.isInteger(settings.eval_every) ||
                        settings.eval_every < 1 ||
                        !Number.isInteger(settings.eval_batches) ||
                        settings.eval_batches < 1
                      }
                      onClick={() => void restart()}
                    >
                      {busy
                        ? "Restarting…"
                        : active
                          ? "Save checkpoint & restart"
                          : "Resume from checkpoint"}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="training-actions">
                  {active && control?.supported && (
                    <button
                      className="primary"
                      disabled={
                        busy ||
                        pending ||
                        !Number.isInteger(settings.eval_every) ||
                        settings.eval_every < 1 ||
                        !Number.isInteger(settings.eval_batches) ||
                        settings.eval_batches < 1
                      }
                      onClick={() => void apply()}
                    >
                      {busy ? "Sending…" : "Apply to running training"}
                    </button>
                  )}
                  {canRestart && (
                    <button
                      title="Resume the recorded script and training state. Stage-script edits are not loaded."
                      disabled={busy}
                      onClick={() => setConfirmRestart(true)}
                    >
                      Restart from checkpoint…
                    </button>
                  )}
                  {!canRestart &&
                    !active &&
                    detail.status.state !== "completed" && (
                      <p className="training-muted">
                        This run has no checkpoint with a restorable data
                        cursor. Create a new stage from weights instead.
                      </p>
                    )}
                </div>
              )}
            </div>
          </Dialog>,
          document.body,
        )}
    </>
  );
}
