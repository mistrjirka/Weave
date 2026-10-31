import { test } from "node:test";
import assert from "node:assert/strict";
import { connectionFeedback } from "../frontend/src/connectionFeedback.ts";
import type { Project, ValidationReport } from "../frontend/src/types.ts";

const project: Project = {
  version: 1,
  id: "test",
  name: "Test",
  description: "",
  config: {},
  graph: {
    nodes: [],
    edges: [{ id: "edge", source: "ids", target: "target", targetHandle: "b" }],
  },
};
const report: ValidationReport = {
  valid: false,
  errors: [],
  warnings: [],
  shapes: {},
  counts: {},
  parameters: 0,
  outputs: [],
};
const attempt = { projectId: "test", path: [], edgeId: "edge" };

test("only the connected input's errors are shown", () => {
  const errors = [
    { node: "target", port: "a", message: "Input a is unconnected." },
    { node: "other", message: "Unrelated graph issue" },
  ];
  assert.equal(
    connectionFeedback(project, { ...report, errors }, attempt),
    null,
  );
  const feedback = connectionFeedback(
    project,
    {
      ...report,
      errors: [
        ...errors,
        { node: "target", port: "b", message: "Mask dimensions must match." },
      ],
    },
    attempt,
  );
  assert.deepEqual(feedback?.messages, ["Mask dimensions must match."]);
});

test("feedback clears on removal, project change, valid reports, and deleted scopes", () => {
  assert.equal(connectionFeedback(project, report, attempt), null);
  assert.equal(
    connectionFeedback(project, report, { ...attempt, projectId: "other" }),
    null,
  );
  assert.equal(
    connectionFeedback(project, report, { ...attempt, edgeId: "deleted" }),
    null,
  );
  assert.equal(
    connectionFeedback(project, report, { ...attempt, path: ["deleted"] }),
    null,
  );
});

test("cycles are actionable, unrelated graph-wide warnings are excluded", () => {
  assert.equal(
    connectionFeedback(
      project,
      {
        ...report,
        errors: [{ node: null, message: "Graph needs an Output node." }],
      },
      attempt,
    ),
    null,
  );
  assert.deepEqual(
    connectionFeedback(
      project,
      {
        ...report,
        errors: [
          {
            node: null,
            message:
              "A direct cycle was found. Put feedback inside a Loop container.",
          },
        ],
      },
      attempt,
    )?.messages,
    ["A direct cycle was found. Put feedback inside a Loop container."],
  );
});
