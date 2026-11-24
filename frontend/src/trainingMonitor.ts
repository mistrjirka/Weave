import type { TrainingEvent } from "./trainingApi";

// Keep the same array when a poll contains no new records, so derived charts
// do not recalculate for status-only updates.
export function mergeMonitorEvents(
  current: TrainingEvent[],
  incoming: TrainingEvent[],
): TrainingEvent[] {
  if (!incoming.length) return current;
  const ids = new Set(current.map((event) => event.id));
  const added = incoming.filter((event) => {
    if (ids.has(event.id)) return false;
    ids.add(event.id);
    return true;
  });
  return added.length ? [...current, ...added].slice(-10000) : current;
}
