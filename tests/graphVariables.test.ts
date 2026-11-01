import test from "node:test";
import assert from "node:assert/strict";
import {
  graphVariableScope,
  graphVariableUses,
  bindingValue,
  previewParameters,
} from "../frontend/src/graphVariables.ts";
import { localizeLayerDefaults } from "../frontend/src/parameterScopes.ts";
import { compilationInput } from "../frontend/src/compilationInput.ts";
import {
  parseSelection,
  serializeSelection,
  pasteSelection,
} from "../frontend/src/graphClipboard.ts";
import type {
  Project,
  GraphNode,
  TokenizerInfo,
} from "../frontend/src/types.ts";
const info: TokenizerInfo = {
  id: "12345678901234567890",
  class: "Test",
  capacity: 8,
  size: 8,
  base_size: 8,
  special_tokens: {},
};
const node = (id: string, dtype = "int64"): GraphNode => ({
  id,
  type: "architecture",
  position: { x: 0, y: 0 },
  data: { kind: "input", label: id, params: { dtype, shape: ["B", "T"] } },
});
const project = (): Project => ({
  version: 1,
  id: "test",
  name: "Test",
  description: "",
  config: { width: 16, hidden: 32 },
  graph: {
    nodes: [node("a"), node("b"), node("features", "float32")],
    edges: [],
  },
});

test("legacy tokenizer metadata moves onto eligible external inputs exactly once", () => {
  const source = project();
  source.tokenizer = info;
  source.graph.nodes[1].data.tokenizer = { ...info, id: "another" };
  const next = localizeLayerDefaults(source);
  assert.equal(next.tokenizer, undefined);
  assert.deepEqual(next.graph.nodes[0].data.tokenizer, info);
  assert.equal(next.graph.nodes[1].data.tokenizer?.id, "another");
  assert.equal(next.graph.nodes[2].data.tokenizer, undefined);
  assert.deepEqual(localizeLayerDefaults(next), next);
  assert.deepEqual(source.tokenizer, info);
  assert.equal(source.graph.nodes[0].data.tokenizer, undefined);
});

test("graph variables inherit block parameters and support local overrides", () => {
  const source = project();
  source.graph.variables = { width: 12 };
  source.graph.nodes.push({
    id: "body",
    type: "architecture",
    position: { x: 0, y: 0 },
    data: {
      kind: "block",
      label: "Body",
      params: { hidden: "$width" },
      body: { variables: { hidden: 3 }, nodes: [], edges: [] },
    },
  });
  const root = graphVariableScope(source, []);
  assert.equal(root.effective.width, 12);
  const nested = graphVariableScope(source, ["body"]);
  assert.equal(nested.inherited.hidden, 12);
  assert.deepEqual(nested.own, { hidden: 3 });
  assert.equal(nested.effective.width, 12);
  assert.equal(nested.effective.hidden, 3);
  assert.equal(bindingValue("$hidden", nested.effective), 3);
  assert.equal(bindingValue("$missing", nested.effective), undefined);
  assert.equal(bindingValue("$sqrt_width", { width: 16 }), 4);
});

test("compilation and clipboard preserve scoped variables and input metadata", () => {
  const source = project();
  source.graph.variables = { custom: 7 };
  source.graph.nodes[0].data.tokenizer = info;
  source.graph.nodes[0].selected = true;
  const compilation = JSON.parse(compilationInput(source));
  assert.deepEqual(compilation.graph.variables, { custom: 7 });
  assert.equal(compilation.graph.nodes[0].data.tokenizer.capacity, 8);
  assert.equal(compilation.graph.nodes[0].selected, undefined);
  const copied = parseSelection(serializeSelection(source.graph, null)!)!;
  const pasted = pasteSelection(
    { nodes: [], edges: [] },
    copied,
    { x: 0, y: 0 },
    () => "fresh",
  );
  assert.deepEqual(pasted[0].data.tokenizer, info);
});

test("diagram bindings show resolved heads and do not treat a zero window as enabled", () => {
  const params = {
    heads: "$heads",
    window: "$window",
    shape: ["B", "T", "$width"],
  };
  const before = structuredClone(params);
  assert.deepEqual(
    previewParameters(params, { heads: 4, window: 0, width: 32 }),
    { heads: 4, window: 0, shape: ["B", "T", 32] },
  );
  assert.deepEqual(params, before);
});

test("typed container bindings match backend scope for booleans, strings and numeric values", () => {
  const p = project();
  p.graph.nodes.push({
    id: "body",
    type: "architecture",
    position: { x: 0, y: 0 },
    data: {
      kind: "block",
      label: "Body",
      params: { causal: false, backend: "auto", heads: 4 },
      body: { nodes: [], edges: [] },
    },
  });
  const scope = graphVariableScope(p, ["body"]);
  assert.equal(scope.bindings.causal, false);
  assert.equal(scope.bindings.backend, "auto");
  assert.deepEqual(
    previewParameters(
      { causal: "$causal", backend: "$backend", heads: "$heads" },
      scope.bindings,
    ),
    { causal: false, backend: "auto", heads: 4 },
  );
});

test("loading and reloading preserve explicit root FFN links instead of freezing numbers", () => {
  const p = project();
  p.config.ffn_width = 2176;
  p.graph.nodes.push({
    id: "layer",
    type: "architecture",
    position: { x: 0, y: 0 },
    data: {
      kind: "block",
      label: "Layer",
      params: { ffn_width: "$ffn_width" },
      body: {
        nodes: [
          {
            id: "ffn",
            type: "architecture",
            position: { x: 0, y: 0 },
            data: {
              kind: "ffn",
              label: "FFN",
              params: { hidden: "$ffn_width" },
            },
          },
        ],
        edges: [],
      },
    },
  });
  const loaded = localizeLayerDefaults(p);
  assert.equal(loaded.graph.nodes.at(-1)!.data.params.ffn_width, "$ffn_width");
  assert.equal(graphVariableScope(loaded, ["layer"]).effective.ffn_width, 2176);
  loaded.config.ffn_width = 4096;
  const reloaded = localizeLayerDefaults(loaded);
  assert.equal(
    reloaded.graph.nodes.at(-1)!.data.params.ffn_width,
    "$ffn_width",
  );
  assert.equal(
    graphVariableScope(reloaded, ["layer"]).effective.ffn_width,
    4096,
  );
});

test("variable uses belong to the nearest owner and report nested links", () => {
  const p = project();
  p.graph.nodes.push({
    id: "layer",
    type: "architecture",
    position: { x: 0, y: 0 },
    data: {
      kind: "block",
      label: "Layer",
      params: { heads: "$heads" },
      body: {
        variables: { ffn_width: 384 },
        nodes: [
          {
            id: "ffn",
            type: "architecture",
            position: { x: 0, y: 0 },
            data: {
              kind: "ffn",
              label: "FFN",
              params: { hidden: "$ffn_width" },
            },
          },
        ],
        edges: [],
      },
    },
  });
  assert.deepEqual(graphVariableUses(p.graph).heads, ["Layer · heads"]);
  assert.equal(graphVariableUses(p.graph).ffn_width, undefined);
  assert.deepEqual(
    graphVariableUses(p.graph.nodes.at(-1)!.data.body!).ffn_width,
    ["FFN · hidden"],
  );
});
