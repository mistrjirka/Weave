import { runContract } from "./trainingContract";
import type { RunDetail } from "./trainingApi";
import { formatMetric, metricPresentation } from "./trainingMetrics";

const headlineMetrics = [
  "train/loss",
  "eval/loss",
  "train/tokens_per_second",
  "lr/group_0",
] as const;

export function RunOverview({ detail }: { detail: RunDetail }) {
  const contract = runContract(detail.events);
  const reported = [
    ...new Set(
      detail.events
        .filter((event) => event.kind === "metrics")
        .flatMap((event) => Object.keys(event.payload)),
    ),
  ];
  const headlines =
    contract.headline_metrics ??
    (detail.run.recipe === "custom"
      ? [
          ...new Set([
            ...(contract.primary_metric ? [contract.primary_metric] : []),
            ...reported,
          ]),
        ].slice(0, 4)
      : headlineMetrics);
  const hasMetrics = detail.events.some((event) => event.kind === "metrics");
  if (!hasMetrics) return null;
  return (
    <dl className="training-overview" aria-label="Latest training metrics">
      {headlines.map((metric) => {
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
              <dt>{metricPresentation(metric, contract).label}</dt>
              <dd title={typeof value === "number" ? String(value) : undefined}>
                {typeof value === "number" ? formatMetric(value) : "Pending"}
                <small>
                  {event
                    ? `${event.step.toLocaleString()} ${contract.step_unit}`
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
