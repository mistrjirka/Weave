import type { RunContract } from "./trainingContract";
const compactUnits = [
  [1e12, "T"],
  [1e9, "B"],
  [1e6, "M"],
  [1e3, "k"],
] as const;

export function formatMetric(value: number, digits = 4): string {
  if (!Number.isFinite(value)) return "—";
  const unit = compactUnits.find(([scale]) => Math.abs(value) >= scale);
  if (unit)
    return (
      new Intl.NumberFormat("en", { maximumFractionDigits: 1 }).format(
        value / unit[0],
      ) + unit[1]
    );
  return new Intl.NumberFormat("en", {
    notation: "standard",
    maximumSignificantDigits: Math.max(1, Math.min(12, digits)),
  }).format(value);
}

type MetricPresentation = {
  label: string;
  color: string;
  min?: number;
  max?: number;
  empty?: string;
};
const metricDefinitions: Record<string, MetricPresentation> = {
  "train/loss": {
    label: "Training loss",
    color: "var(--metric-blue, #2563eb)",
  },
  "eval/loss": {
    label: "Validation loss",
    color: "var(--metric-purple, #9333ea)",
  },
  "train/tokens_per_second": {
    min: 0,
    label: "Training speed · tokens/s",
    color: "var(--metric-teal, #0d9488)",
  },
  "train/learning_rate": {
    label: "Learning rate",
    min: 0,
    color: "var(--metric-pink, #be185d)",
  },
  "train/gradient_norm": {
    min: 0,
    label: "Gradient norm",
    color: "var(--metric-orange, #c2410c)",
  },
};
export function metricPresentation(
  metric: string,
  contract?: RunContract,
): MetricPresentation {
  const custom = contract?.metrics[metric];
  if (custom)
    return {
      ...metricPresentation(metric),
      ...custom,
      color: [
        "var(--metric-blue, #2563eb)",
        "var(--metric-purple, #9333ea)",
        "var(--metric-teal, #0d9488)",
        "var(--metric-orange, #c2410c)",
      ][
        [...metric].reduce(
          (hash, letter) => (hash * 31 + letter.charCodeAt(0)) >>> 0,
          0,
        ) % 4
      ],
    };
  const utilization = /^gpu\/(\d+)\/utilization$/.exec(metric);
  if (utilization)
    return {
      label: `GPU ${utilization[1]} utilization`,
      color: "var(--metric-teal, #0d9488)",
      min: 0,
      max: 100,
      empty: "No utilization readings received for this GPU.",
    };
  return (
    metricDefinitions[metric] ?? {
      label:
        metric === "lr/group_0"
          ? "Learning rate"
          : metric.startsWith("lr/")
            ? "Learning rate · " + metric.slice(3)
            : metric,
      min: metric.startsWith("lr/") ? 0 : undefined,
      color: metric.startsWith("lr/")
        ? "var(--metric-pink, #be185d)"
        : "var(--metric-blue, #2563eb)",
    }
  );
}

export function metricDomain(
  metric: string,
  minimum: number,
  maximum: number,
  contract?: RunContract,
): [number, number] {
  const presentation = metricPresentation(metric, contract);
  const padding =
    maximum === minimum
      ? Math.max(Math.abs(minimum) * 0.03, 1e-6)
      : (maximum - minimum) * 0.08;
  return [
    presentation.min ?? minimum - padding,
    presentation.max ?? maximum + padding,
  ];
}

export function metricsForDisplay(
  events: readonly { kind: string; payload: Record<string, unknown> }[],
  codeFirst: boolean,
  contract?: RunContract,
  custom = false,
): string[] {
  const reported = events
    .filter((event) => event.kind === "metrics")
    .flatMap((event) =>
      Object.entries(event.payload)
        .filter(
          ([, value]) => typeof value === "number" && Number.isFinite(value),
        )
        .map(([key]) => key),
    );
  if (custom)
    return [
      ...new Set([
        ...(contract?.primary_metric ? [contract.primary_metric] : []),
        ...Object.keys(contract?.metrics ?? {}),
        ...reported,
      ]),
    ];
  const learningRate =
    ["lr/group_0", "train/learning_rate"].find((key) =>
      reported.includes(key),
    ) ?? (codeFirst ? "lr/group_0" : "train/learning_rate");
  return [
    ...new Set([
      "train/loss",
      "eval/loss",
      "train/tokens_per_second",
      learningRate,
      ...reported,
    ]),
  ];
}
