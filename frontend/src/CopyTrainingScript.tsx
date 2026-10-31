import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { Copy, X } from "lucide-react";
import { savedProjectsSchema, responseData } from "./api";
import { planSchema, trainingRequest, type TrainingStage } from "./trainingApi";
import { Dialog } from "./ui";

export function CopyTrainingScript({
  projectId,
  onCopy,
  disabled,
}: {
  projectId: string;
  onCopy: (script: string, source: string) => void;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button disabled={disabled} onClick={() => setOpen(true)}>
        <Copy size={14} aria-hidden="true" /> Copy script from project…
      </button>
      {open && (
        <ScriptSource
          projectId={projectId}
          onClose={() => setOpen(false)}
          onCopy={(script, source) => {
            onCopy(script, source);
            setOpen(false);
          }}
        />
      )}
    </>
  );
}

function ScriptSource({
  projectId,
  onClose,
  onCopy,
}: {
  projectId: string;
  onClose: () => void;
  onCopy: (script: string, source: string) => void;
}) {
  const [projects, setProjects] = useState<
    ReturnType<typeof savedProjectsSchema.parse>
  >([]);
  const [selectedProject, setSelectedProject] = useState("");
  const [stages, setStages] = useState<TrainingStage[]>([]);
  const [stageId, setStageId] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    let disposed = false;
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    void fetch("/api/projects", {
      cache: "no-store",
      signal: controller.signal,
    })
      .then(responseData)
      .then((value) => {
        if (controller.signal.aborted) return;
        const list = savedProjectsSchema
          .parse(value)
          .filter((project) => project.id !== projectId);
        setProjects(list);
        setSelectedProject(list[0]?.id ?? "");
        if (!list.length) setLoading(false);
      })
      .catch((failure) => {
        if (!controller.signal.aborted) {
          setError(
            failure instanceof Error ? failure.message : String(failure),
          );
          setLoading(false);
        } else if (!disposed) {
          setError(
            "Loading projects timed out. Close this dialog and try again.",
          );
          setLoading(false);
        }
      })
      .finally(() => window.clearTimeout(timeout));
    return () => {
      disposed = true;
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, [projectId]);
  useEffect(() => {
    if (!selectedProject) return;
    const controller = new AbortController();
    let disposed = false;
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    setLoading(true);
    setError("");
    void trainingRequest(
      `/plans/${encodeURIComponent(selectedProject)}`,
      planSchema,
      undefined,
      undefined,
      controller.signal,
    )
      .then((plan) => {
        if (!disposed) {
          setStages(plan.stages);
          setStageId(plan.stages[0]?.id ?? "");
        }
      })
      .catch((failure) => {
        if (!disposed)
          setError(
            controller.signal.aborted
              ? "Loading stages timed out. Choose the project again or reopen this dialog."
              : failure instanceof Error
                ? failure.message
                : String(failure),
          );
      })
      .finally(() => {
        window.clearTimeout(timeout);
        if (!disposed) setLoading(false);
      });
    return () => {
      disposed = true;
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, [selectedProject]);
  const stage = stages.find((item) => item.id === stageId);
  const project = projects.find((item) => item.id === selectedProject);
  return createPortal(
    <Dialog
      label="Copy training script"
      className="training-controls training-history-dialog training-script-copy-dialog"
      onClose={onClose}
    >
      <div className="training-history-heading">
        <h2>Copy training script</h2>
        <button
          className="modal-close"
          aria-label="Close script copy"
          onClick={onClose}
        >
          <X size={18} />
        </button>
      </div>
      <div className="training-history-content">
        <p className="training-muted">
          Copy only train.py into this editor. The current model and GUI launch
          settings are kept; dataset and helper files are not copied. Save when
          ready.
        </p>
        <label>
          Source project
          <select
            aria-label="Source project"
            value={selectedProject}
            onChange={(event) => {
              if (event.target.value === selectedProject) return;
              setStages([]);
              setStageId("");
              setSelectedProject(event.target.value);
            }}
          >
            {!projects.length && (
              <option value="">
                {loading
                  ? "Loading projects…"
                  : error
                    ? "Projects unavailable"
                    : "No other saved projects"}
              </option>
            )}
            {projects.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Source stage
          <select
            aria-label="Source stage"
            disabled={loading || !stages.length}
            value={stageId}
            onChange={(event) => setStageId(event.target.value)}
          >
            {!stages.length && (
              <option value="">
                {loading ? "Loading stages…" : "No training stages"}
              </option>
            )}
            {stages.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
        {loading && <p role="status">Loading saved scripts…</p>}
        {error && (
          <p role="alert" className="training-alert">
            {error}
          </p>
        )}
        {stage && !loading && (
          <>
            {stage.script_path && (
              <p className="training-script-copy-path">
                <code>{stage.script_path}</code>
              </p>
            )}
            <pre
              className="training-script-copy-preview"
              tabIndex={0}
              aria-label="Source training script"
            >
              {stage.script}
            </pre>
          </>
        )}
        <div className="training-actions">
          <button onClick={onClose}>Cancel</button>
          <button
            className="primary"
            disabled={loading || !!error || !stage || !project}
            onClick={() => {
              if (stage && project)
                onCopy(stage.script, `${project.name} / ${stage.name}`);
            }}
          >
            Replace editor script
          </button>
        </div>
        <p className="training-muted">
          Replaces the current editor text. Source files and running jobs are
          unchanged; copy any imported helper files separately.
        </p>
      </div>
    </Dialog>,
    document.body,
  );
}
