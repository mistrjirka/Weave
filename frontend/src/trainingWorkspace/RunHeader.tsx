import { Pause } from "lucide-react";
import { ExecutionStatus } from "../ExecutionStatus";
import { TrainingETA } from "../TrainingETA";
import { RunValidationControls } from "../RunValidationControls";
import { CustomRunControls } from "../CustomRunControls";
import type { RunDetail, TrainingGPU, TrainingRun } from "../trainingApi";
type Props = {
  detail: RunDetail;
  runs: TrainingRun[];
  runId: string;
  gpus: TrainingGPU[];
  enabled: boolean;
  active: boolean;
  busy: boolean;
  customRun: boolean;
  scriptCapabilities: Record<string, unknown>;
  pauseRequested: string | null;
  launchError: string;
  lastRefreshed: number | null;
  monitorError: string;
  selectRun: (run: TrainingRun) => void;
  onRestart: (run: TrainingRun, announce: boolean) => void;
  onCommand: (kind: "stop" | "kill") => void;
};
export function RunHeader({
  detail,
  runs,
  runId,
  gpus,
  enabled,
  active,
  busy,
  customRun,
  scriptCapabilities,
  pauseRequested,
  launchError,
  lastRefreshed,
  monitorError,
  selectRun,
  onRestart,
  onCommand,
}: Props) {
  const runLabel = (run: TrainingRun) =>
    run.status?.state ??
    (run.execution === "monitor" ? "prepared" : "starting");
  return (
    <div className="training-section-heading">
      <div>
        {launchError && (
          <p className="training-muted">Previous run · the new launch failed</p>
        )}
        <h3>
          {detail.run.name}{" "}
          <small className="training-run-state">{detail.status.state}</small>
        </h3>
        <ExecutionStatus detail={detail} />
        {lastRefreshed !== null && (
          <p className="training-muted" data-testid="run-freshness">
            {monitorError
              ? "Showing last successful update"
              : active
                ? "Live monitoring"
                : "Run data refreshed"}
            {" · "}
            <time dateTime={new Date(lastRefreshed).toISOString()}>
              {new Date(lastRefreshed).toLocaleTimeString()}
            </time>
            {!detail.events.some((event) => event.kind === "progress") && (
              <TrainingETA detail={detail} />
            )}
          </p>
        )}
      </div>
      <div className="training-actions">
        {runs.length > 1 && (
          <label className="training-run-picker">
            <span>Run</span>
            <select
              aria-label="Training run"
              value={runId}
              onChange={(event) => {
                const run = runs.find((item) => item.id === event.target.value);
                if (run) selectRun(run);
              }}
            >
              {runs.map((run) => (
                <option key={run.id} value={run.id}>
                  {runLabel(run)} ·{" "}
                  {new Date(run.created).toLocaleString(undefined, {
                    month: "short",
                    day: "numeric",
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </option>
              ))}
            </select>
          </label>
        )}
        {detail.run.code_first && (
          <RunValidationControls
            key={detail.run.id}
            detail={detail}
            gpus={gpus}
            enabled={enabled}
            onRestart={(run) => onRestart(run, true)}
          />
        )}
        {customRun &&
          (scriptCapabilities.resume === true ||
            detail.restart_context?.exact_resume) && (
            <CustomRunControls
              detail={detail}
              enabled={enabled}
              onRestart={(run) => onRestart(run, false)}
            />
          )}
        {active && (
          <>
            <button
              disabled={
                busy ||
                pauseRequested === detail.run.id ||
                (customRun && scriptCapabilities.commands !== true)
              }
              title={
                customRun && scriptCapabilities.commands !== true
                  ? "The custom script has not declared cooperative stop support."
                  : "Stop at the next safe training boundary. A checkpoint requires script support."
              }
              onClick={() => onCommand("stop")}
            >
              <Pause aria-hidden="true" size={14} />
              {pauseRequested === detail.run.id
                ? detail.run.code_first || scriptCapabilities.resume === true
                  ? "Pausing…"
                  : "Stopping…"
                : detail.run.code_first || scriptCapabilities.resume === true
                  ? "Pause training"
                  : "Stop safely"}
            </button>
            <button
              disabled={busy || detail.run.execution === "monitor"}
              onClick={() => onCommand("kill")}
            >
              Force stop
            </button>
          </>
        )}
      </div>
    </div>
  );
}
