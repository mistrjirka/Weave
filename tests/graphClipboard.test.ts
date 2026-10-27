import test from "node:test";
import assert from "node:assert/strict";
import {
  serializeSelection,
  parseSelection,
  pasteSelection,
} from "../frontend/src/graphClipboard.ts";
import type { Graph, GraphNode } from "../frontend/src/types.ts";
const node = (id: string, kind = "linear"): GraphNode => ({
  id,
  type: "architecture",
  position: { x: 10, y: 20 },
  selected: true,
  measured: { width: 280, height: 200 },
  data: { kind, label: id, params: {} },
});
const edge = (source: string, target: string) => ({
  id: source + target,
  source,
  target,
  sourceHandle: "out",
  targetHandle: "a",
});
let counter = 0;
const id = () => `new${++counter}`;

test("copy contains only the selection and internal connections, without view state", () => {
  const a = node("a"),
    b = node("b"),
    c = node("c");
  c.selected = false;
  a.data._focused = true;
  const source: Graph = {
    nodes: [a, b, c],
    edges: [edge("a", "b"), edge("b", "c")],
  };
  const text = serializeSelection(source, null)!;
  const result = parseSelection(text)!;
  assert.deepEqual(
    result.nodes.map((n) => n.id),
    ["a", "b"],
  );
  assert.equal(result.edges.length, 1);
  assert.equal(result.nodes[0].measured, undefined);
  assert.equal(result.nodes[0].selected, undefined);
  assert.equal(result.nodes[0].data._focused, undefined);
  assert.equal(serializeSelection({ nodes: [], edges: [] }, null), null);
  assert.equal(parseSelection("ordinary text"), null);
  assert.equal(parseSelection(JSON.stringify(source)), null);
  assert.equal(
    parseSelection(
      JSON.stringify({
        format: "weave.graph-selection",
        version: 1,
        graph: { ...source, edges: [edge("a", "missing")] },
      }),
    ),
    null,
  );
});

test("paste remaps edges, ties and shared weights, while preserving nested definitions", () => {
  const embedding = node("embedding", "embedding"),
    head = node("head", "lm_head"),
    block = node("block", "block");
  embedding.data.share = "weights";
  head.data.params.tie = "embedding";
  block.data.share = "outer";
  const nested = node("nested");
  nested.data.share = "weights";
  nested.data.params.tie = "local";
  block.data.body = {
    nodes: [nested, node("local", "embedding")],
    edges: [edge("local", "nested")],
  };
  const source: Graph = {
    nodes: [embedding, head, block],
    edges: [edge("embedding", "head")],
  };
  const original = structuredClone(source);
  const destination: Graph = { nodes: [node("existing")], edges: [] };
  const pasted = pasteSelection(destination, source, { x: 100, y: 200 }, id);
  assert.equal(destination.nodes[0].selected, false);
  assert(pasted.every((n) => n.selected));
  assert.deepEqual(pasted[0].position, { x: 100, y: 200 });
  assert.notEqual(pasted[0].id, "embedding");
  assert.equal(pasted[1].data.params.tie, pasted[0].id);
  assert.equal(destination.edges[0].source, pasted[0].id);
  assert.equal(destination.edges[0].target, pasted[1].id);
  assert.notEqual(pasted[0].data.share, "weights");
  assert.equal(pasted[2].data.body!.nodes[0].data.share, pasted[0].data.share);
  assert.equal(pasted[2].data.body!.nodes[0].data.params.tie, "local");
  assert.deepEqual(source, original);
  const again = pasteSelection(destination, source, { x: 150, y: 250 }, id);
  assert.notEqual(again[0].id, pasted[0].id);
  assert.notEqual(again[0].data.share, pasted[0].data.share);
  assert.equal(pasted[0].selected, false);
});

test("copying a head alone removes its external embedding tie and supports focused nodes", () => {
  const head = node("head", "lm_head");
  head.selected = false;
  head.data.params.tie = "outside";
  const source = parseSelection(
    serializeSelection({ nodes: [head], edges: [] }, "head")!,
  )!;
  const result = pasteSelection(
    { nodes: [], edges: [] },
    source,
    { x: 0, y: 0 },
    id,
  );
  assert.equal(result[0].data.params.tie, undefined);
});

test("reference copies capture external dimensions while retaining local variables and input vocabulary", () => {
  const block = node("block", "block");
  block.data.params.heads = 4;
  const input = node("x", "input");
  input.data.params = { shape: ["B", "T", "$width"], dtype: "float32" };
  const attention = node("a", "attention");
  attention.data.params = {
    heads: "$heads",
    kv_heads: "$kv",
    window: "$window",
  };
  const head = node("head", "head");
  head.data.params = { out: "$vocab" };
  block.data.body = {
    nodes: [input, attention, head],
    edges: [],
    variables: { window: 16 },
  };
  const graph = { nodes: [block], edges: [] };
  const before = structuredClone(graph);
  const copied = parseSelection(
    serializeSelection(graph, null, {
      width: 448,
      heads: 8,
      kv: 2,
      window: 0,
      vocab: 1024,
    })!,
  )!;
  const body = copied.nodes[0].data.body!;
  assert.deepEqual(body.nodes[0].data.params.shape, ["B", "T", 448]);
  assert.deepEqual(body.nodes[1].data.params, {
    heads: "$heads",
    kv_heads: 2,
    window: "$window",
  });
  assert.equal(body.nodes[2].data.params.out, "$vocab");
  assert.deepEqual(body.variables, { window: 16 });
  assert.deepEqual(graph, before);
});
