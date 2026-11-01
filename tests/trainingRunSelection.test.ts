import test from "node:test";
import assert from "node:assert/strict";
import { preferredTrainingRun } from "../frontend/src/trainingRunSelection.ts";
import type { TrainingRun } from "../frontend/src/trainingApi.ts";

function run(id: string, stage_id: string, state: string): TrainingRun {
  return {
    id,
    stage_id,
    project_id: "project-test",
    name: id,
    created: "2026-11-01T00:00:00Z",
    recipe: "causal",
    strategy: "single",
    execution: "monitor",
    parent: null,
    warnings: [],
    model: { modules: [], split_supported: false, split_reasons: [] },
    status: { state },
  };
}

test("missing links prefer an active run in the linked stage", () => {
  const newest = run("newest", "other", "running");
  const stopped = run("stopped", "selected", "stopped");
  const active = run("active", "selected", "running");
  assert.equal(
    preferredTrainingRun([newest, stopped, active], "selected"),
    active,
  );
});

test("fallback uses newest saved run, or another stage when needed", () => {
  const newest = run("newest", "selected", "stopped");
  const older = run("older", "selected", "completed");
  assert.equal(preferredTrainingRun([newest, older], "selected"), newest);
  assert.equal(preferredTrainingRun([newest, older], "missing-stage"), newest);
  assert.equal(preferredTrainingRun([], "selected"), undefined);
});
