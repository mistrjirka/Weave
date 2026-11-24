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
  let configuration: TrainingEvent["payload"] | undefined;
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i];
    if (
      event.kind === "configuration" &&
      (event.channel ?? "training") === "training"
    ) {
      configuration = event.payload;
      break;
    }
  }
  const parsed = contractSchema.safeParse(configuration?.reporting ?? {});
  return parsed.success ? parsed.data : contractSchema.parse({});
}
export function namedCounters(
  events: readonly TrainingEvent[],
): Record<string, number> {
  const counters = new Map<string, number>();
  for (const event of events) {
    if ((event.channel ?? "training") !== "training" || !event.counters)
      continue;
    for (const [name, value] of Object.entries(event.counters))
      counters.set(name, value);
  }
  return Object.fromEntries(counters);
}
export function counterLabel(name: string) {
  return name
    .replaceAll("_", " ")
    .replace(/^./, (letter) => letter.toUpperCase());
}
