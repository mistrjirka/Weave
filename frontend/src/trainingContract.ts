import { z } from "zod";
import type { TrainingEvent } from "./trainingApi";

const metricDefinition = z.object({
  label: z.string().optional(),
  direction: z.enum(["min", "max"]).optional(),
  min: z.number().finite().optional(),
  max: z.number().finite().optional(),
});
const contractSchema = z.object({
  metrics: z.record(z.string(), metricDefinition).default({}),
  primary_metric: z.string().nullable().default(null),
  headline_metrics: z.array(z.string()).nullable().default(null),
  step_unit: z.string().default("optimizer updates"),
});
export type RunContract = z.infer<typeof contractSchema>;
export type TrainingAxis = "step" | "tokens" | "time" | `counter:${string}`;
export function isTrainingAxis(value: string): value is TrainingAxis {
  return (
    ["step", "tokens", "time"].includes(value) || value.startsWith("counter:")
  );
}
export function runContract(
  events: readonly Pick<TrainingEvent, "kind" | "payload" | "channel">[],
): RunContract {
  const configuration = events
    .filter(
      (event) =>
        event.kind === "configuration" &&
        (event.channel ?? "training") === "training",
    )
    .at(-1)?.payload;
  const parsed = contractSchema.safeParse(configuration?.reporting ?? {});
  return parsed.success ? parsed.data : contractSchema.parse({});
}
export function namedCounters(
  events: readonly TrainingEvent[],
): Record<string, number> {
  return events
    .filter((event) => (event.channel ?? "training") === "training")
    .reduce<Record<string, number>>(
      (values, event) => ({ ...values, ...event.counters }),
      {},
    );
}
export function counterLabel(name: string) {
  return name
    .replaceAll("_", " ")
    .replace(/^./, (letter) => letter.toUpperCase());
}
