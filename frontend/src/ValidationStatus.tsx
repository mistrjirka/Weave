import type { RunDetail } from "./trainingApi";

export function ValidationStatus({ events }: { events: RunDetail["events"] }) {
  const latest = events.filter((event) => event.kind === "evaluation").at(-1);
  const metric = events
    .filter(
      (event) =>
        event.kind === "metrics" &&
        typeof event.payload["eval/loss"] === "number",
    )
    .at(-1);
  const configuration = events
    .filter((event) => event.kind === "configuration")
    .at(-1);
  const interval = configuration?.payload.eval_every;
  const progress = events
    .filter(
      (event) =>
        event.kind === "evaluation_progress" &&
        (!latest || event.step === latest.payload.checkpoint_step),
    )
    .at(-1);
  if (!latest && !metric && typeof interval !== "number") return null;
  return (
    <section
      className="training-validation-status"
      role="status"
      aria-label="Background validation status"
    >
      {latest ? (
        <p>
          Background validation: {String(latest.payload.state)} · checkpoint
          update {String(latest.payload.checkpoint_step)} ·{" "}
          {latest.payload.device === "cpu"
            ? "CPU"
            : `GPU ${String(latest.payload.device)}`}
        </p>
      ) : typeof interval === "number" && !metric ? (
        <p>
          {interval > 0
            ? `Evaluation interval: every ${interval} successful optimizer updates. No validation result received yet.`
            : "Scheduled validation is disabled in this run."}
        </p>
      ) : null}
      {latest?.payload.state === "running" && progress && (
        <p>
          Validation batches: {String(progress.payload.completed)} / up to{" "}
          {String(progress.payload.total)}
          {typeof progress.payload.loss_running === "number" &&
            ` · Provisional loss: ${progress.payload.loss_running.toFixed(4)}`}
        </p>
      )}
      {!latest && metric && (
        <p>
          Latest validation loss:{" "}
          {Number(metric.payload["eval/loss"]).toFixed(4)} · checkpoint update{" "}
          {metric.step}
        </p>
      )}
    </section>
  );
}
