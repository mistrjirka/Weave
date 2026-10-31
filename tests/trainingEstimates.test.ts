import { test } from "node:test";
import assert from "node:assert/strict";
import {
  estimateTrainingFinish,
  formatTimeRemaining,
} from "../frontend/src/trainingEstimates.ts";
import type { TrainingEvent } from "../frontend/src/trainingApi.ts";

const progress = (
  step: number,
  time: number,
  total: number | null = 101,
): TrainingEvent => ({
  version: 1,
  id: String(step),
  kind: "progress",
  time,
  step,
  tokens: 0,
  payload: { completed: step, total },
});

test("finish estimates use measured recent progress, including resumed runs", () => {
  assert.deepEqual(
    estimateTrainingFinish([progress(1, 100), progress(11, 160)]),
    { seconds: 540, finishTime: 700 },
  );
  assert.deepEqual(
    estimateTrainingFinish([progress(91, 100), progress(96, 160)]),
    { seconds: 60, finishTime: 220 },
  );
  assert.equal(estimateTrainingFinish([progress(1, 100)]), null);
  assert.equal(
    estimateTrainingFinish([progress(1, 100), progress(1, 120)]),
    null,
  );
  assert.equal(
    estimateTrainingFinish([progress(1, 100), progress(101, 160)]),
    null,
  );
  assert.equal(
    estimateTrainingFinish([progress(1, 100, null), progress(11, 160, null)]),
    null,
  );
  assert.deepEqual(
    estimateTrainingFinish([
      progress(0, -10000),
      progress(1, 100),
      progress(11, 160),
    ]),
    { seconds: 540, finishTime: 700 },
  );
});

test("finish estimates support epoch budgets with batch progress", () => {
  const first = progress(1, 100, null),
    last = progress(2, 160, null);
  first.payload = {
    max_epochs: 2,
    epochs_completed: 0,
    batches_in_epoch: 2,
    batches_per_epoch: 10,
  };
  last.payload = {
    max_epochs: 2,
    epochs_completed: 0,
    batches_in_epoch: 4,
    batches_per_epoch: 10,
  };
  assert.deepEqual(estimateTrainingFinish([first, last]), {
    seconds: 480,
    finishTime: 640,
  });
  assert.equal(formatTimeRemaining(30), "<1m");
  assert.equal(formatTimeRemaining(540), "9m");
  assert.equal(formatTimeRemaining(8100), "2h 15m");
  assert.equal(formatTimeRemaining(90000), "1d 1h");
});
