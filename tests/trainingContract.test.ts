import { test } from "node:test";
import assert from "node:assert/strict";
import {
  runContract,
  namedCounters,
} from "../frontend/src/trainingContract.ts";
import {
  metricsForDisplay,
  metricDomain,
  metricPresentation,
} from "../frontend/src/trainingMetrics.ts";

const configuration = {
  metrics: {
    "eval/success": {
      label: "Tool success",
      direction: "max" as const,
      min: 0,
      max: 1,
    },
  },
  primary_metric: "eval/success",
  headline_metrics: ["eval/success"],
  step_unit: "policy updates",
};
const event = (kind: string, payload: Record<string, unknown>, extra = {}) => ({
  version: 1 as const,
  id: kind,
  time: 1,
  kind,
  step: 3,
  tokens: 0,
  payload,
  ...extra,
});
test("custom metric contracts choose task labels and avoid unrelated LM placeholders", () => {
  const events = [
    event("configuration", { reporting: configuration }),
    event("metrics", { "train/reward": 0.5 }),
  ];
  const contract = runContract(events);
  assert.equal(contract.step_unit, "policy updates");
  assert.equal(
    metricPresentation("eval/success", contract).label,
    "Tool success",
  );
  assert.deepEqual(metricDomain("eval/success", 0.4, 0.7, contract), [0, 1]);
  assert.deepEqual(metricsForDisplay(events, false, contract, true), [
    "eval/success",
    "train/reward",
  ]);
});
test("late evaluation never replaces the current training counters or configuration", () => {
  const events = [
    event("configuration", { reporting: configuration }),
    event("metrics", {}, { counters: { episodes: 100 }, channel: "training" }),
    event(
      "configuration",
      { reporting: { step_unit: "wrong" } },
      { channel: "evaluation" },
    ),
    event("metrics", {}, { counters: { episodes: 20 }, channel: "evaluation" }),
  ];
  assert.deepEqual(namedCounters(events), { episodes: 100 });
  assert.equal(runContract(events).step_unit, "policy updates");
});
test("older runs and malformed metadata retain a useful default presentation", () => {
  assert.equal(runContract([]).step_unit, "optimizer updates");
  assert.equal(
    runContract([event("configuration", { reporting: { metrics: 42 } })])
      .primary_metric,
    null,
  );
});
