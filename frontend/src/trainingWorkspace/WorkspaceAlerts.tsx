import { Activity, CheckCircle2, Square } from "lucide-react";
import type { RunDetail, TrainingRun } from "../trainingApi";
type Props = {
  setupOpen: boolean;
  launching: boolean;
  detail: RunDetail | null;
  otherActiveRun: TrainingRun | undefined;
  error: string;
  monitorError: string;
  notice: string;
  runLink: (run: TrainingRun) => string;
  selectRun: (run: TrainingRun) => void;
};
export function WorkspaceAlerts({
  setupOpen,
  launching,
  detail,
  otherActiveRun,
  error,
  monitorError,
  notice,
  runLink,
  selectRun,
}: Props) {
  return (
    <>
      {!setupOpen && otherActiveRun && !launching && (
        <div
          className="training-notice training-current-run-notice"
          role="status"
        >
          <span>
            You’re viewing an earlier {detail?.status.state} run. Another run of
            this stage is running.
          </span>
          <a
            href={runLink(otherActiveRun)}
            onClick={(event) => {
              if (
                event.button !== 0 ||
                event.metaKey ||
                event.ctrlKey ||
                event.shiftKey ||
                event.altKey
              )
                return;
              event.preventDefault();
              selectRun(otherActiveRun);
            }}
          >
            <Activity size={14} aria-hidden="true" /> View running training
          </a>
        </div>
      )}
      {!setupOpen && detail?.status.state === "completed" && !launching && (
        <p
          className="training-completed-banner"
          role="status"
          aria-label="Training completed"
        >
          <CheckCircle2 size={18} aria-hidden="true" />
          <strong>Run finished</strong>
          {typeof detail.status.updated === "number" && (
            <span>
              Finished{" "}
              <time
                dateTime={new Date(detail.status.updated * 1000).toISOString()}
              >
                {new Date(detail.status.updated * 1000).toLocaleString()}
              </time>
            </span>
          )}
        </p>
      )}
      {!setupOpen &&
        detail?.status.state === "stopped" &&
        !otherActiveRun &&
        !launching && (
          <p
            className="training-stopped-banner"
            role="status"
            aria-label="Training stopped"
          >
            <Square size={16} aria-hidden="true" />
            <strong>Training stopped</strong>
            {detail.run.code_first && detail.restart_context?.exact_resume && (
              <span>
                Resume training restores the saved checkpoint without starting
                over.
              </span>
            )}
          </p>
        )}
      {error && (
        <p className="training-alert" role="alert">
          {error}
        </p>
      )}
      {detail?.resume_problem && (
        <div className="training-alert training-resume-problem" role="alert">
          <strong>Checkpoint resume failed.</strong> No training updates were
          applied. Use Retry resume to choose the previous script or try your
          corrected edits.
          <details>
            <summary>Compatibility details</summary>
            <pre>{detail.resume_problem.message}</pre>
            <p>{detail.resume_problem.recovery}</p>
          </details>
        </div>
      )}
      {!setupOpen && monitorError && (
        <p className="training-alert" role="alert">
          Live updates interrupted; displayed run data may be stale. Training
          can continue independently. Retrying automatically. {monitorError}
        </p>
      )}
      {notice && (
        <p className="training-notice" role="status">
          {notice}
        </p>
      )}
    </>
  );
}
