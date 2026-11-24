import { test } from "node:test";
import { strict as assert } from "node:assert";
import {
  matchesSavedRevision,
  parseScheduleValue,
  shouldRecoverProject,
  withProjectIdentity,
} from "../frontend/src/projectWorkflow.ts";
import type { Project } from "../frontend/src/types.ts";

const project = {
  id: "example",
  name: "First",
  graph: { nodes: [], edges: [] },
} as Project;

test("only dirty or legacy compatible drafts recover", () => {
  assert.equal(
    shouldRecoverProject("p", { id: "p", dirty: false }, true),
    false,
  );
  assert.equal(shouldRecoverProject("p", { id: "p", dirty: true }, true), true);
  assert.equal(shouldRecoverProject("p", null, false), true);
  assert.equal(shouldRecoverProject("p", null, true), false);
  assert.equal(
    shouldRecoverProject("p", { id: "other", dirty: true }, true),
    false,
  );
});

test("schedule controls reject bindings and non-numbers", () => {
  assert.equal(parseScheduleValue(3), 3);
  assert.equal(parseScheduleValue("$step"), null);
  assert.equal(parseScheduleValue(Infinity), null);
});

test("saved identity survives undo and redo revisions", () => {
  const before = withProjectIdentity(project, "project-1");
  const saved = withProjectIdentity({ ...project, name: "Saved" }, "project-1");
  const snapshot = JSON.stringify(saved);
  assert.equal(before.id, saved.id);
  assert.equal(matchesSavedRevision(before, snapshot), false);
  assert.equal(matchesSavedRevision(saved, snapshot), true);
  assert.equal(matchesSavedRevision(saved, null), false);
});
