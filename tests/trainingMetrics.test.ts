import { test } from "node:test";
import assert from "node:assert/strict";
import {
  formatMetric,
  metricPresentation,
  metricDomain,
  metricsForDisplay,
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

test("utilization always has a 0–100 percent axis, including idle and saturated GPUs", () => {
  for (const [min, max] of [
    [0, 0],
    [100, 100],
    [0, 100],
    [20, 80],
  ])
    assert.deepEqual(metricDomain("gpu/1/utilization", min, max), [0, 100]);
  assert.equal(metricDomain("train/tokens_per_second", 0, 10)[0], 0);
  assert.ok(metricDomain("custom/signed", -5, 5)[0] < -5);
});

test("metric charts exist before reporting and retain placeholders after the first point", () => {
  const initial = metricsForDisplay([], true);
  assert.deepEqual(initial, [
    "train/loss",
    "eval/loss",
    "train/tokens_per_second",
    "lr/group_0",
  ]);
  assert.deepEqual(
    metricsForDisplay(
      [{ kind: "metrics", payload: { "train/loss": 2.5 } }],
      true,
    ),
    initial,
  );
  const legacy = metricsForDisplay(
    [
      {
        kind: "metrics",
        payload: { "train/learning_rate": 0.001, "custom/accuracy": 0.5 },
      },
    ],
    false,
  );
  assert.ok(legacy.includes("custom/accuracy"));
  assert.equal(
    legacy.filter(
      (key) => key.includes("learning_rate") || key.startsWith("lr/"),
    ).length,
    1,
  );
});
