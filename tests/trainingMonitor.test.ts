import { test } from "node:test";
import { strict as assert } from "node:assert";
import { mergeMonitorEvents } from "../frontend/src/trainingMonitor.ts";
import type { TrainingEvent } from "../frontend/src/trainingApi.ts";

function event(id: string): TrainingEvent {
  return {
    version: 1,
    id,
    time: 1,
    kind: "metrics",
    step: 1,
    tokens: 1,
    payload: {},
  };
}

test("monitor retains event identity on empty and duplicate pages", () => {
  const events = [event("a")];
  assert.equal(mergeMonitorEvents(events, []), events);
  assert.equal(mergeMonitorEvents(events, [event("a")]), events);
});

test("monitor appends unique records in order and bounds retained records", () => {
  const events = Array.from({ length: 10000 }, (_, i) => event(String(i)));
  const result = mergeMonitorEvents(events, [
    event("9999"),
    event("new"),
    event("new"),
  ]);
  assert.equal(result.length, 10000);
  assert.equal(result[0].id, "1");
  assert.equal(result.at(-1)?.id, "new");
});
