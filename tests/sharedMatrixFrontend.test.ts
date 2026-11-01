import test from "node:test";
import assert from "node:assert/strict";
import { localizeLayerDefaults } from "../frontend/src/parameterScopes.ts";
import {
  compatibleWeightGroupType,
  weightGroupChoices,
} from "../frontend/src/weightGroups.ts";
import type { GraphNode, Project } from "../frontend/src/types.ts";

const node = (
  id: string,
  kind: string,
  params: GraphNode["data"]["params"] = {},
  share?: string,
): GraphNode => ({
  id,
  type: "architecture",
  position: { x: 0, y: 0 },
  data: { kind, label: id, params, ...(share ? { share } : {}) },
});
const project = (...nodes: GraphNode[]): Project => ({
  version: 1,
  id: "p",
  name: "P",
  description: "",
  config: { width: 8, vocab: 16 },
  graph: { nodes, edges: [] },
});

test("embedding and head are compatible group choices", () => {
  const before = project(
    node("emb", "embedding", { width: 8, vocab: 16 }, "matrix"),
    node("head", "head", { out: 16 }),
  );
  assert.equal(
    compatibleWeightGroupType(
      before.graph.nodes[0].data,
      before.graph.nodes[1].data,
    ),
    true,
  );
  assert.equal(
    weightGroupChoices(
      before.graph.nodes.map((n) => ({ node: n, path: n.id })),
      before.graph.nodes[1].data,
    )[0].id,
    "matrix",
  );
});

test("legacy tie migrates when safe, preserving an existing embedding group", () => {
  const original = project(
    node("emb", "embedding", {}, "already"),
    node("head", "head", { tie: "emb", out: 16 }),
  );
  const loaded = localizeLayerDefaults(original);
  assert.equal(loaded.graph.nodes[1].data.share, "already");
  assert.equal(loaded.graph.nodes[1].data.params.tie, undefined);
  assert.equal(original.graph.nodes[1].data.params.tie, "emb");
  assert.deepEqual(localizeLayerDefaults(loaded), loaded);
});

test("legacy ties inside independent loops and multiple heads retain their semantics", () => {
  const nested = project(
    node("emb", "embedding"),
    node("head", "head", { tie: "emb" }),
  );
  const loop = node("loop", "loop", { shared: false, iterations: 2 });
  loop.data.body = nested.graph;
  const loadedLoop = localizeLayerDefaults(project(loop));
  assert.equal(
    loadedLoop.graph.nodes[0].data.body!.nodes[1].data.params.tie,
    "emb",
  );
  assert.equal(
    loadedLoop.graph.nodes[0].data.body!.nodes[0].data.share,
    undefined,
  );
  const multi = localizeLayerDefaults(
    project(
      node("emb", "embedding"),
      node("h1", "head", { tie: "emb" }),
      node("h2", "head", { tie: "emb" }),
    ),
  );
  assert.equal(multi.graph.nodes[1].data.params.tie, "emb");
  assert.equal(multi.graph.nodes[2].data.params.tie, "emb");
});
