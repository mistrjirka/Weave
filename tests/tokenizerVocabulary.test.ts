import test from "node:test";
import assert from "node:assert/strict";
import {
  tokenizerOwnsVocabulary,
  automaticVocabulary,
} from "../frontend/src/tokenizerVocabulary.ts";
import type { Project, TokenizerInfo } from "../frontend/src/types.ts";
const tok: TokenizerInfo = {
  id: "test",
  capacity: 16000,
  size: 16000,
  base_size: 16000,
  special_tokens: {},
};
const project = (): Project => ({
  version: 1,
  id: "test",
  name: "Test",
  description: "",
  config: { width: 16, vocab: 1024 },
  graph: {
    nodes: [
      {
        id: "ids",
        type: "architecture",
        position: { x: 0, y: 0 },
        data: {
          kind: "input",
          label: "IDs",
          params: { shape: ["B", "T"], dtype: "int64" },
          tokenizer: tok,
        },
      },
      {
        id: "emb",
        type: "architecture",
        position: { x: 0, y: 0 },
        data: {
          kind: "embedding",
          label: "Embedding",
          params: { vocab: "$vocab", width: 16 },
        },
      },
    ],
    edges: [],
  },
});

test("tokenizer-owned vocabulary is hidden without changing legacy fallback data", () => {
  const p = project(),
    before = structuredClone(p);
  assert.equal(tokenizerOwnsVocabulary(p, [], { ids: tok, emb: tok }), true);
  assert.equal(automaticVocabulary(p.graph.nodes[1].data, tok), true);
  assert.deepEqual(p, before);
});

test("manual branches keep their fallback and missing metadata never hides needed controls", () => {
  const p = project();
  assert.equal(tokenizerOwnsVocabulary(p, []), false);
  assert.equal(tokenizerOwnsVocabulary(p, [], { ids: tok }), false);
  p.graph.nodes[1].data.params.vocab = 20000;
  assert.equal(automaticVocabulary(p.graph.nodes[1].data, tok), false);
  assert.equal(tokenizerOwnsVocabulary(p, [], { ids: tok }), true);
});

test("independent tokenizer capacities do not require a single global vocabulary", () => {
  const p = project();
  p.graph.nodes.push({
    id: "other",
    type: "architecture",
    position: { x: 0, y: 0 },
    data: { kind: "head", label: "Other head", params: { out: "$vocab" } },
  });
  assert.equal(
    tokenizerOwnsVocabulary(p, [], {
      ids: tok,
      emb: tok,
      other: { ...tok, id: "other", capacity: 8000 },
    }),
    true,
  );
  assert.equal(automaticVocabulary(p.graph.nodes[2].data, tok), true);
});
