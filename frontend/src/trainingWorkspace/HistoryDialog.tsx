import {
  useRef,
  type ReactNode,
  type Dispatch,
  type SetStateAction,
} from "react";
import { createPortal } from "react-dom";
import { Upload, X } from "lucide-react";
import { z } from "zod";
import { Dialog } from "../ui";
import { runSchema, trainingRequest, type TrainingRun } from "../trainingApi";
type Props = {
  open: boolean;
  busy: boolean;
  projectId: string;
  runId: string;
  runs: TrainingRun[];
  children: ReactNode;
  onClose: () => void;
  selectRun: (run: TrainingRun) => void;
  task: (action: () => Promise<unknown>) => Promise<void>;
  setNotice: (notice: string) => void;
  setRunId: (id: string) => void;
  setRuns: Dispatch<SetStateAction<TrainingRun[]>>;
};
export function HistoryDialog({
  open: historyOpen,
  busy,
  projectId,
  runId,
  runs,
  children,
  onClose,
  selectRun,
  task,
  setNotice,
  setRunId,
  setRuns,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const runLabel = (run: TrainingRun) =>
    run.status?.state ??
    (run.execution === "monitor" ? "prepared" : "starting");
  return (
    <>
      {historyOpen &&
        createPortal(
          <Dialog
            label="Stages and run history"
            className="training-controls training-history-dialog"
            onClose={() => onClose()}
          >
            <header className="training-history-heading">
              <h2>Stages & run history</h2>
              <button
                className="modal-close"
                aria-label="Close dialog"
                onClick={() => onClose()}
              >
                <X size={18} aria-hidden="true" />
              </button>
            </header>
            <div className="training-workspace training-history-content">
              <aside className="training-stages" aria-label="Training stages">
                <input
                  hidden
                  ref={fileRef}
                  type="file"
                  accept=".zip"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    void task(async () => {
                      const form = new FormData();
                      form.append("file", file);
                      const imported = await trainingRequest(
                        "/import",
                        runSchema,
                        form,
                      );
                      if (imported.project_id !== projectId)
                        setNotice(
                          `Snapshot belongs to project ${imported.project_id}. Its recorded graph is available in the package.`,
                        );
                      setRunId(imported.id);
                      setRuns(
                        await trainingRequest(
                          `/runs?project_id=${encodeURIComponent(projectId)}`,
                          z.array(runSchema),
                        ),
                      );
                    });
                    e.target.value = "";
                  }}
                />

                <button
                  onClick={() => fileRef.current?.click()}
                  disabled={busy}
                >
                  <Upload aria-hidden="true" size={14} /> Import snapshot
                </button>
                {children}
                <h3>Run history</h3>
                {!runs.length && (
                  <p className="training-muted">
                    Your training runs will appear here.
                  </p>
                )}
                {runs.map((run) => (
                  <button
                    key={run.id}
                    className={`training-run ${runId === run.id ? "is-active" : ""}`}
                    onClick={() => {
                      selectRun(run);
                    }}
                  >
                    <strong>{run.name}</strong>
                    <small>
                      {runLabel(run)} · {new Date(run.created).toLocaleString()}
                    </small>
                    {run.parent && (
                      <small>
                        {run.revision
                          ? `Resumed · ${run.revision.script_source === "edited" ? "edited script" : "previous script"} · snapshot saved`
                          : "Continues a previous checkpoint"}
                      </small>
                    )}
                  </button>
                ))}
              </aside>
            </div>
          </Dialog>,
          document.body,
        )}
    </>
  );
}
