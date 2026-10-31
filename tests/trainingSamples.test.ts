import { test } from "node:test";
import assert from "node:assert/strict";
import { groupTrainingSamples } from "../frontend/src/trainingSamples.ts";
import type { TrainingEvent } from "../frontend/src/trainingApi.ts";

const sample = (
  step: number,
  time: number,
  checkpoint: string | null = null,
): TrainingEvent => ({
  version: 1,
  id: `${step}-${time}`,
  time,
  step,
  tokens: 0,
  kind: "sample",
  channel: "evaluation",
  payload: { checkpoint, output: "example" },
});
test("latest generated examples follow checkpoint step, including out-of-order evaluation reports", () => {
  const groups = groupTrainingSamples([
    sample(1000, 10),
    sample(500, 30),
    sample(1000, 11),
  ]);
  assert.deepEqual(
    groups.map((group) => group.step),
    [1000, 500],
  );
  assert.equal(groups[0].events.length, 2);
  const next = groupTrainingSamples([
    ...groups.flatMap((group) => group.events),
    sample(1500, 35),
  ]);
  assert.equal(next[0].step, 1500);
  assert.notEqual(groups[0].key, groups[1].key);
  assert.equal(groupTrainingSamples([]).length, 0);
});
