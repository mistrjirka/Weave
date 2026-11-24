import { useMemo } from "react";
import { runContract } from "./trainingContract";
import type { RunDetail } from "./trainingApi";
import {
  formatMetric,
  metricPresentation,
  metricScope,
} from "./trainingMetrics";

const headlineMetrics = [
  "train/loss",
  "eval/loss",
  "train/tokens_per_second",
  "lr/group_0",
] as const;

export function RunOverview({ detail }: { detail: RunDetail }) {
  const { events } = detail;
  const { contract, latest, reported } = useMemo(() => {
    const latest = new Map<string, (typeof events)[number]>();
    const reported = new Set<string>();
    for (const event of events) {
      if (event.kind !== "metrics") continue;
      for (const [key, value] of Object.entries(event.payload)) {
        reported.add(key);
        if (typeof value === "number") latest.set(key, event);
      }
    }
    return { contract: runContract(events), latest, reported };
  }, [events]);
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
  if (!latest.size) return null;
  return (
    <dl className="training-overview" aria-label="Latest training metrics">
      {headlines.map((metric) => {
        const event = latest.get(metric);
        const value = event?.payload[metric];
        return (
          typeof value === "number" && (
            <div key={metric}>
              <dt>{metricPresentation(metric, contract).label}</dt>
              <dd title={typeof value === "number" ? String(value) : undefined}>
                {typeof value === "number" ? formatMetric(value) : "Pending"}
                <small>
                  {metricScope(metric) && `${metricScope(metric)} · `}
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
