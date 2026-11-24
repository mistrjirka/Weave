import { test } from "node:test";
import { strict as assert } from "node:assert";
import {
  matchesSavedRevision,
  withProjectIdentity,
} from "../frontend/src/projectWorkflow.ts";
import type { Project } from "../frontend/src/types.ts";

const project = {
  id: "example",
  name: "First",
  graph: { nodes: [], edges: [] },
} as Project;

test("saved identity survives undo and redo revisions", () => {
  const before = withProjectIdentity(project, "project-1");
  const saved = withProjectIdentity({ ...project, name: "Saved" }, "project-1");
  const snapshot = JSON.stringify(saved);
  assert.equal(before.id, saved.id);
  assert.equal(matchesSavedRevision(before, snapshot), false);
  assert.equal(matchesSavedRevision(saved, snapshot), true);
  assert.equal(matchesSavedRevision(saved, null), false);
});
