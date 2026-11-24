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
  if (!added.length) return current;
  const all = [...current, ...added];
  // Metadata is a snapshot, not part of the rolling metric window. Keep the
  // latest training declaration of each kind ahead of the chronological tail.
  const metadata = new Map<string, TrainingEvent>();
  for (const event of all) {
    if (
      (event.channel ?? "training") === "training" &&
      (event.kind === "configuration" || event.kind === "capabilities")
    ) {
      metadata.set(event.kind, event);
    }
  }
  const tail = all.slice(-10000);
  const tailIds = new Set(tail.map((event) => event.id));
  return [...metadata.values()]
    .filter((event) => !tailIds.has(event.id))
    .concat(tail);
}
