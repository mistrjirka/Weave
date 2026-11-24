import { z } from "zod";
import type { RunDetail } from "../trainingApi";

export const commandResponse = z.object({ id: z.string() });

export const activeStates = new Set([
  "starting",
  "preparing",
  "running",
  "evaluating",
]);
export function validationPending(detail: RunDetail, requested: string | null) {
  const control = detail.validation_control;
  const applied =
    requested !== null && control?.applied_command_id === requested;
  return {
    applied,
    pending: !!control?.pending || (requested !== null && !applied),
  };
}

export function canRestartValidation(
  detail: RunDetail,
  enabled: boolean,
  active: boolean,
) {
  const checkpoints = detail.events.filter(
    (event) => event.kind === "checkpoint" && !event.payload.remote,
  );
  const latest = checkpoints.at(-1);
  const restorable = latest?.payload.metadata;
  const exactResume =
    detail.restart_context?.exact_resume ??
    (typeof restorable === "object" &&
      restorable !== null &&
      "exact_resume" in restorable &&
      restorable.exact_resume === true);
  const canRestart =
    enabled &&
    detail.run.execution === "local" &&
    !!detail.run.code_first &&
    (active
      ? !latest || exactResume
      : exactResume && detail.status.state !== "completed");
  return { canRestart, latest };
}
