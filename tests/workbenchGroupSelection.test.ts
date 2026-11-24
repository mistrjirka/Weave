import { test } from "node:test";
import { strict as assert } from "node:assert";
import { groupSelection } from "../frontend/src/workbench/groupSelection.ts";
import type { Graph } from "../frontend/src/types.ts";

function makeIdFactory() {
  let next = 0;
  return () => `generated-${next++}`;
}

test("group selection preserves external connections and builds a reusable block", () => {
  const graph: Graph = {
    nodes: [
      {
        id: "in",
        type: "architecture",
        position: { x: 0, y: 0 },
        data: { kind: "input", label: "Input", params: {} },
      },
      {
        id: "a",
        type: "architecture",
        position: { x: 100, y: 20 },
        selected: true,
        data: { kind: "linear", label: "Linear", params: {} },
      },
      {
        id: "out",
        type: "architecture",
        position: { x: 200, y: 0 },
        data: { kind: "output", label: "Output", params: {} },
      },
    ],
    edges: [
      {
        id: "e1",
        source: "in",
        target: "a",
        sourceHandle: "out",
        targetHandle: "a",
      },
      {
        id: "e2",
        source: "a",
        target: "out",
        sourceHandle: "out",
        targetHandle: "a",
      },
    ],
  };
  let selected = "";
  let message = "";
  groupSelection(
    graph,
    null,
    null,
    "",
    makeIdFactory(),
    (change) => change(graph),
    (id) => {
      selected = id;
    },
    (text) => {
      message = text;
    },
  );
  const block = graph.nodes.find((node) => node.id === selected);
  assert.equal(block?.data.kind, "block");
  assert.equal(block?.data.body?.nodes.length, 3);
  assert.equal(graph.edges.find((edge) => edge.id === "e2")?.source, selected);
  assert.equal(
    graph.edges.find((edge) => edge.source === "in")?.target,
    selected,
  );
  assert.match(message, /Reusable block created/);
});

test("group selection rejects interface nodes without changing the graph", () => {
  const graph: Graph = {
    nodes: [
      {
        id: "in",
        type: "architecture",
        position: { x: 0, y: 0 },
        selected: true,
        data: { kind: "input", label: "Input", params: {} },
      },
    ],
    edges: [],
  };
  let message = "";
  groupSelection(
    graph,
    null,
    null,
    "",
    makeIdFactory(),
    (change) => change(graph),
    () => {},
    (text) => {
      message = text;
    },
  );
  assert.match(message, /Select computation nodes/);
  assert.equal(graph.nodes.length, 1);
});
