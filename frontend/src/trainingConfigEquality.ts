import type { TrainingConfig } from "./trainingApi";

/** Compare JSON config values without treating object insertion order as a change. */
export function trainingConfigsEqual(
  a: TrainingConfig,
  b: TrainingConfig,
): boolean {
  return stableJSON(a) === stableJSON(b);
}

export function stableJSON(value: unknown): string {
  return JSON.stringify(value, (_key: string, item: unknown) => {
    if (item === null || typeof item !== "object" || Array.isArray(item))
      return item;
    return Object.fromEntries(
      Object.entries(item).sort(([a], [b]) => a.localeCompare(b)),
    );
  });
}
