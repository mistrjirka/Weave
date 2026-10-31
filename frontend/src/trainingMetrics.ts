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

const metricDefinitions: Record<string, { label: string; color: string }> = {
  "train/loss": {
    label: "Training loss",
    color: "var(--metric-blue, #2563eb)",
  },
  "eval/loss": {
    label: "Validation loss",
    color: "var(--metric-purple, #9333ea)",
  },
  "train/tokens_per_second": {
    label: "Training speed · tokens/s",
    color: "var(--metric-teal, #0d9488)",
  },
  "train/gradient_norm": {
    label: "Gradient norm",
    color: "var(--metric-orange, #c2410c)",
  },
};
export function metricPresentation(metric: string) {
  return (
    metricDefinitions[metric] ?? {
      label:
        metric === "lr/group_0"
          ? "Learning rate"
          : metric.startsWith("lr/")
            ? "Learning rate · " + metric.slice(3)
            : metric,
      color: metric.startsWith("lr/")
        ? "var(--metric-pink, #be185d)"
        : "var(--metric-blue, #2563eb)",
    }
  );
}
