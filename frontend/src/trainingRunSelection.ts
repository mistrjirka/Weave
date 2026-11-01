import type { TrainingRun } from "./trainingApi";

export const runningTrainingStates = new Set([
  "starting",
  "running",
  "preparing",
  "evaluating",
]);

/** Run lists arrive newest first; prefer an active run within the linked stage. */
export function preferredTrainingRun(runs: TrainingRun[], stageId: string) {
  const stageRuns = runs.filter((run) => run.stage_id === stageId);
  const candidates = stageRuns.length ? stageRuns : runs;
  return (
    candidates.find((run) =>
      runningTrainingStates.has(run.status?.state ?? ""),
    ) ?? candidates[0]
  );
}
