import type { TrainingEvent } from "./trainingApi";

export function groupTrainingSamples(events: readonly TrainingEvent[]) {
  const groups = new Map<
    string,
    { key: string; step: number; checkpoint: string; events: TrainingEvent[] }
  >();
  for (const event of events) {
    if (event.kind !== "sample") continue;
    const checkpoint = String(event.payload.checkpoint ?? "Unspecified");
    const key = JSON.stringify([event.step, checkpoint]);
    const group = groups.get(key) ?? {
      key,
      step: event.step,
      checkpoint,
      events: [],
    };
    group.events.push(event);
    groups.set(key, group);
  }
  return [...groups.values()].sort(
    (a, b) =>
      b.step - a.step ||
      (b.events.at(-1)?.time ?? 0) - (a.events.at(-1)?.time ?? 0),
  );
}
