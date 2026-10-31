import { Clock3 } from "lucide-react";
import type { RunDetail } from "./trainingApi";
import {
  estimateTrainingFinish,
  formatTimeRemaining,
} from "./trainingEstimates";

export function TrainingETA({ detail }: { detail: RunDetail }) {
  if (
    !["starting", "preparing", "running", "evaluating"].includes(
      detail.status.state,
    )
  )
    return null;
  const progress = detail.events
    .filter((event) => event.kind === "progress")
    .at(-1)?.payload;
  const done =
    typeof progress?.total === "number" &&
    typeof progress.completed === "number" &&
    progress.completed >= progress.total;
  const unbounded =
    progress &&
    ![progress.total, progress.max_epochs].some(
      (limit) =>
        typeof limit === "number" && Number.isFinite(limit) && limit > 0,
    );
  const estimate = estimateTrainingFinish(detail.events);
  const finish = estimate ? new Date(estimate.finishTime * 1000) : null;
  const remaining = estimate
    ? Math.max(0, estimate.finishTime - Date.now() / 1000)
    : 0;
  return (
    <span
      className="training-finish-estimate"
      aria-label="Training finish estimate"
      title="Estimate from the last five minutes of training progress. Variable sequence lengths can change the pace. Final validation and checkpoint writing are not included."
    >
      <Clock3 size={14} aria-hidden="true" />
      {done ? (
        "Training updates complete · finishing validation and checkpointing"
      ) : estimate && finish && remaining > 0 ? (
        <span>
          ~{formatTimeRemaining(remaining)} remaining · around{" "}
          <time dateTime={finish.toISOString()}>
            {finish.toLocaleString(undefined, {
              ...(finish.toDateString() !== new Date().toDateString()
                ? { month: "short" as const, day: "numeric" as const }
                : {}),
              hour: "2-digit",
              minute: "2-digit",
            })}
          </time>
        </span>
      ) : unbounded ? (
        "Finish estimate unavailable · no training limit reported"
      ) : (
        "Estimating finish time · waiting for measured progress"
      )}
    </span>
  );
}
