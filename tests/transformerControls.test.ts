import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  attentionVariants,
  attentionVariant,
  exposeTransformerSettings,
  setAttentionVariant,
  setTransformerSetting,
} from "../frontend/src/transformerControls.ts";
import type { Project } from "../frontend/src/types.ts";

function python(script: string, input?: string) {
  const result = spawnSync(".venv/bin/python", ["-c", script], {
    input,
    encoding: "utf8",
    env: { ...process.env, OMP_NUM_THREADS: "1", MKL_NUM_THREADS: "1" },
    timeout: 60000,
  });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  return result.stdout;
}
const template: Project = JSON.parse(
  python(`
import json
from backend.templates import templates
print(json.dumps(next(p for p in templates() if p['id']=='transformer')))
`),
);
let id = 0;
const uid = () => `pattern_${++id}`;

test("outer transformer settings bind inner operations and preserve independent layers", () => {
  const project = structuredClone(template);
  const data = project.graph.nodes.find(
    (node) => node.data.kind === "loop",
  )!.data;
  exposeTransformerSettings(data);
  setTransformerSetting(data, "heads", 4);
  setTransformerSetting(data, "ffn_width", 48);
  setTransformerSetting(data, "causal", false);
  const attention = data.body!.nodes.find(
    (node) => node.data.kind === "attention",
  )!;
  assert.equal(attention.data.params.heads, "$heads");
  assert.equal(attention.data.params.causal, false);
  assert.equal(data.params.heads, 4);
  assert.equal(
    data.body!.nodes.find((node) => node.data.kind === "ffn")!.data.params
      .hidden,
    "$ffn_width",
  );
  const original = data.body!.nodes.map((node) => node.id);
  setAttentionVariant(data, "sparse", project.config, uid);
  assert.equal(attentionVariant(data, project.config), "sparse");
  assert.equal(data.body!.nodes.length, original.length + 3);
  setAttentionVariant(data, "sparse", project.config, uid);
  assert.equal(data.body!.nodes.length, original.length + 3);
  setAttentionVariant(data, "mha", project.config, uid);
  assert.deepEqual(
    data.body!.nodes.map((node) => node.id),
    original,
  );
  assert.equal(attentionVariant(data, project.config), "mha");
  setTransformerSetting(data, "heads", 8);
  assert.equal(data.params.kv_heads, 8);
});

test("all attention presets validate, export, run forward and backward with outer loop settings", () => {
  const projects = attentionVariants.map(({ id: variant }) => {
    const project = structuredClone(template);
    project.config = {
      ...project.config,
      width: 16,
      heads: 4,
      ffn_width: 32,
      vocab: 32,
    };
    const data = project.graph.nodes.find(
      (node) => node.data.kind === "loop",
    )!.data;
    data.params = { iterations: 2, shared: true, heads: 4, ffn_width: 32 };
    exposeTransformerSettings(data);
    setAttentionVariant(data, variant, project.config, uid);
    assert.equal(attentionVariant(data, project.config), variant);
    return project;
  });
  python(
    `
import json, sys, torch
from backend import runtime, export_helpers, sparse_runtime
from backend.compiler import compile_project
from backend.core import validate_project
for name, module in [('runtime',runtime),('export_helpers',export_helpers),('sparse_runtime',sparse_runtime)]:
    sys.modules[name] = module
torch.set_num_threads(1)
for project in json.load(sys.stdin):
    report = validate_project(project)
    assert report['valid'], report['errors']
    namespace = {}
    exec(compile_project(project)['model.py'], namespace)
    model = namespace['Model']()
    output = model(torch.randint(0,32,(2,5)))
    assert list(output.shape) == [2,5,32]
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == report['parameters']
`,
    JSON.stringify(projects),
  );
});
