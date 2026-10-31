import type { RunDetail } from "./trainingApi";

export function FP16Status({ events }: { events: RunDetail["events"] }) {
  const overflows = events.filter(
    (event) =>
      event.kind === "warning" && event.payload.code === "amp-overflow",
  );
  const latest = overflows.at(-1);
  if (!latest) return null;
  const recovered = events.some(
    (event) =>
      (event.kind === "metrics" || event.kind === "progress") &&
      event.step > latest.step,
  );
  return (
    <section aria-label="FP16 loss scaling" className="training-fp16-status">
      <strong>
        FP16 loss scaling · {recovered ? "recovered" : "adjusting"}
      </strong>
      <p>
        {overflows.length} skipped updates recorded in the displayed history;
        latest at update {latest.step}.{" "}
        {recovered
          ? "Successful optimizer updates followed the last overflow. These are historical warnings."
          : "Waiting for a successful optimizer update after the latest overflow."}
      </p>
      <p className="training-muted">
        PyTorch reduces the loss scale and skips unsafe updates. A few skips at
        startup can be expected; frequent ongoing skips or nonfinite loss need
        investigation.
      </p>
    </section>
  );
}
