import type { RunDetail } from "./trainingApi";
import { formatMetric, metricPresentation } from "./trainingMetrics";

const headlineMetrics = [
  "train/loss",
  "eval/loss",
  "train/tokens_per_second",
  "lr/group_0",
] as const;

export function RunOverview({ detail }: { detail: RunDetail }) {
  const hasMetrics = detail.events.some((event) => event.kind === "metrics");
  if (!hasMetrics) return null;
  return (
    <dl className="training-overview" aria-label="Latest training metrics">
      {headlineMetrics.map((metric) => {
        const event = detail.events
          .filter(
            (item) =>
              item.kind === "metrics" &&
              typeof item.payload[metric] === "number",
          )
          .at(-1);
        const value = event?.payload[metric];
        return (
          typeof value === "number" && (
            <div key={metric}>
              <dt>{metricPresentation(metric).label}</dt>
              <dd title={typeof value === "number" ? String(value) : undefined}>
                {typeof value === "number" ? formatMetric(value) : "Pending"}
                <small>
                  {event
                    ? `Update ${event.step.toLocaleString()}`
                    : metric === "eval/loss"
                      ? "Waiting for evaluation"
                      : "Waiting for metrics"}
                </small>
              </dd>
            </div>
          )
        );
      })}
    </dl>
  );
}
