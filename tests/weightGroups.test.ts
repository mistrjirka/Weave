import test from "node:test";
import assert from "node:assert/strict";
import {
  sameWeightType,
  weightGroupChoices,
  supportsWeightSharing,
} from "../frontend/src/weightGroups.ts";
import type { NodeData, NodeLocation } from "../frontend/src/types.ts";
const data = (kind: string, params: NodeData["params"] = {}): NodeData => ({
  kind,
  label: kind,
  params,
});
const location = (id: string, data: NodeData): NodeLocation => ({
  path: id,
  node: { id, type: "architecture", position: { x: 0, y: 0 }, data },
});

test("weight groups only offer matching operators and retain stable badge labels", () => {
  const linear = data("linear", { out: 16 });
  linear.share = "linear";
  const norm = data("norm");
  norm.share = "norm";
  const choices = weightGroupChoices(
    [location("a", norm), location("b", linear)],
    data("linear", { out: 32 }),
  );
  assert.equal(choices.length, 1);
  assert.equal(choices[0].id, "linear");
  assert.match(choices[0].label, /θ2/);
  assert.equal(sameWeightType(linear, norm), false);
  assert.equal(supportsWeightSharing(data("input")), false);
  assert.equal(supportsWeightSharing(linear), true);
});

test("container and custom block types are distinguished without requiring equal settings", () => {
  const block = (operation: string): NodeData => ({
    ...data("block"),
    body: {
      nodes: [
        location(
          "x",
          data("input", { dtype: "float32", shape: ["B", "T", 16] }),
        ).node,
        location("op", data(operation)).node,
        location("out", data("output")).node,
      ],
      edges: [],
    },
  });
  assert.equal(sameWeightType(block("attention"), block("ffn")), false);
  const first = block("attention"),
    second = structuredClone(first);
  second.label = "Different label";
  second.params.heads = 4;
  second.body!.nodes.reverse();
  assert.equal(sameWeightType(first, second), true);
  second.body!.nodes.find((node) => node.id === "x")!.data.params.dtype =
    "int64";
  assert.equal(sameWeightType(first, second), false);
  assert.equal(
    sameWeightType(
      data("custom", { source: "class A: pass" }),
      data("custom", { source: "class B: pass" }),
    ),
    false,
  );
});
