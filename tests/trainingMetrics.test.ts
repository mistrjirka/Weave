import { test } from "node:test";
import assert from "node:assert/strict";
import {
  formatMetric,
  metricPresentation,
} from "../frontend/src/trainingMetrics.ts";

test("training metrics use readable units without scientific notation", () => {
  for (const [value, expected] of [
    [2000, "2k"],
    [23500, "23.5k"],
    [1200000, "1.2M"],
    [0.0003, "0.0003"],
    [6.1234, "6.123"],
    [Infinity, "—"],
  ] as const)
    assert.equal(formatMetric(value), expected);
  assert.notEqual(
    metricPresentation("train/loss").color,
    metricPresentation("eval/loss").color,
  );
  assert.equal(metricPresentation("eval/loss").label, "Validation loss");
});
