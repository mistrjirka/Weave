import { projectSchema } from "../frontend/src/api.ts";
import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  attentionVariants,
  attentionVariant,
  attentionFieldDisabledReason,
  transformerSettings,
  exposeTransformerSettings,
  wrapLoopTransformer,
  inheritTransformerDefaults,
  setAttentionVariant,
  setTransformerSetting,
  transformerResidual,
  setTransformerResidual,
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
  assert.equal(attention.data.params.causal, "$causal");
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
  assert.equal(attention.data.label, "Multi-head attention (MHA)");
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

test("attention controls preserve derived head values and sync literal settings", () => {
  const project = structuredClone(template);
  const data = project.graph.nodes.find(
    (node) => node.data.kind === "loop",
  )!.data;
  data.params = { ...data.params, heads: "$query_heads" };
  const scope = { ...project.config, query_heads: 8 };
  setAttentionVariant(data, "mha", scope, uid);
  assert.equal(data.params.kv_heads, "$query_heads");
  assert.equal(attentionVariant(data, { ...scope, query_heads: 12 }), "mha");
  setTransformerSetting(data, "heads", 4, scope);
  assert.equal(data.params.kv_heads, 4);
  setAttentionVariant(data, "mqa", scope, uid);
  assert.equal(data.params.kv_heads, 1);
  setTransformerSetting(data, "heads", 8, scope);
  assert.equal(data.params.kv_heads, 1);
  setAttentionVariant(data, "gqa", scope, uid);
  assert.equal(attentionVariant(data, scope), "gqa");
  const attention = data.body!.nodes.find(
    (node) => node.data.kind === "attention",
  )!;
  setTransformerSetting(data, "qkv_bias", true, scope);
  setTransformerSetting(data, "rope_scaling", "yarn", scope);
  setTransformerSetting(data, "rope_factor", 4, scope);
  assert.equal(attention.data.params.qkv_bias, "$qkv_bias");
  assert.equal(attention.data.params.rope_scaling, "$rope_scaling");
  assert.equal(attention.data.params.rope_factor, "$rope_factor");
  assert.equal(data.params.rope_factor, 4);
});

test("dependent fields are disabled only when their controlling feature is inactive", () => {
  assert.match(attentionFieldDisabledReason("kv_heads", {}, "mha")!, /MHA/);
  assert.match(attentionFieldDisabledReason("kv_heads", {}, "mqa")!, /MQA/);
  assert.equal(attentionFieldDisabledReason("kv_heads", {}, "gqa"), undefined);
  assert.equal(
    attentionFieldDisabledReason("kv_heads", {}, "local"),
    undefined,
  );
  assert.match(
    attentionFieldDisabledReason("theta", { rope: false })!,
    /Enable rotary/,
  );
  assert.match(
    attentionFieldDisabledReason("rope_factor", {
      rope: true,
      rope_scaling: "none",
    })!,
    /YaRN/,
  );
  assert.equal(
    attentionFieldDisabledReason("rope_factor", {
      rope: true,
      rope_scaling: "yarn",
    }),
    undefined,
  );
  assert.match(
    attentionFieldDisabledReason("qk_norm_eps", { qk_norm: false })!,
    /normalization/,
  );
});

test("sliding and sparse masks preserve an existing multi-head grouping", () => {
  const data = structuredClone(template).graph.nodes.find(
    (node) => node.data.kind === "loop",
  )!.data;
  const scope = template.config;
  setAttentionVariant(data, "mha", scope, uid);
  setAttentionVariant(data, "local", scope, uid);
  assert.equal(
    transformerSettings(data)?.kv_heads,
    transformerSettings(data)?.heads,
  );
  setAttentionVariant(data, "sparse", scope, uid);
  assert.equal(
    transformerSettings(data)?.kv_heads,
    transformerSettings(data)?.heads,
  );
});

test("attention-only containers do not offer an unused FFN width", () => {
  const data = structuredClone(template).graph.nodes.find(
    (node) => node.data.kind === "loop",
  )!.data;
  data.body!.nodes = data.body!.nodes.filter(
    (node) => node.data.kind !== "ffn",
  );
  data.params.ffn_width = 64;
  assert.equal(transformerSettings(data)?.ffn_width, undefined);
});

test("convolution-window option is inactive only when a required part is absent", () => {
  assert.match(
    attentionFieldDisabledReason("conv_respects_window", {
      conv_taps: 0,
      window: 32,
    })!,
    /convolution/,
  );
  assert.match(
    attentionFieldDisabledReason("conv_respects_window", {
      conv_taps: 3,
      window: 0,
    })!,
    /window/,
  );
  assert.equal(
    attentionFieldDisabledReason("conv_respects_window", {
      conv_taps: 3,
      window: 32,
    }),
    undefined,
  );
});

test("a loop can contain one editable transformer without changing forward or backward", () => {
  const original = structuredClone(template);
  original.config = {
    ...original.config,
    width: 16,
    heads: 4,
    ffn_width: 32,
    vocab: 32,
  };
  const originalLoop = original.graph.nodes.find(
    (node) => node.data.kind === "loop",
  )!.data;
  originalLoop.role = "transformer";
  originalLoop.params = {
    iterations: 3,
    shared: true,
    heads: 4,
    ffn_width: 32,
  };
  exposeTransformerSettings(originalLoop);
  const wrapped = structuredClone(original);
  const loop = wrapped.graph.nodes.find(
    (node) => node.data.kind === "loop",
  )!.data;
  wrapLoopTransformer(loop);
  assert.equal(projectSchema.safeParse(wrapped).success, true);
  assert.deepEqual(
    loop.body!.nodes.map((node) => node.data.kind),
    ["input", "block", "output"],
  );
  const layer = loop.body!.nodes.find(
    (node) => node.data.kind === "block",
  )!.data;
  assert.equal(layer.role, "transformer");
  assert.equal(transformerSettings(loop)?.ffn_width, 32);
  python(
    `
import json, sys, torch
from backend import runtime, export_helpers, sparse_runtime
from backend.compiler import compile_project
from backend.core import validate_project
for name, module in [('runtime',runtime),('export_helpers',export_helpers),('sparse_runtime',sparse_runtime)]:
    sys.modules[name] = module
torch.set_num_threads(1)
models=[]
for project in json.load(sys.stdin):
    report=validate_project(project)
    assert report['valid'], report['errors']
    namespace={}
    exec(compile_project(project)['model.py'], namespace)
    torch.manual_seed(42)
    model=namespace['Model']()
    assert sum(p.numel() for p in model.parameters()) == report['parameters']
    models.append(model)
a,b=models
assert [p.shape for p in a.parameters()] == [p.shape for p in b.parameters()]
for x,y in zip(a.parameters(),b.parameters()):
    torch.testing.assert_close(x,y)
ids=torch.tensor([[1,2,3,4]])
x,y=a(ids),b(ids)
torch.testing.assert_close(x,y,rtol=0,atol=0)
x.square().mean().backward();y.square().mean().backward()
for p,q in zip(a.parameters(),b.parameters()):
    assert p.grad is not None and q.grad is not None
    torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
`,
    JSON.stringify([original, wrapped]),
  );
  setTransformerSetting(loop, "heads", 8, wrapped.config);
  assert.equal(transformerSettings(loop)?.heads, 8);
  assert.equal(layer.params.heads, 8);
  setAttentionVariant(loop, "sparse", wrapped.config, uid);
  assert.equal(attentionVariant(loop, wrapped.config), "sparse");
  assert.equal(
    layer.body!.nodes.filter((node) => node.data.role === "attention-layout")
      .length,
    3,
  );
  assert.equal(loop.body!.nodes.length, 3);
  setAttentionVariant(loop, "mqa", wrapped.config, uid);
  assert.equal(attentionVariant(loop, wrapped.config), "mqa");
  assert.equal(
    layer.body!.nodes.filter((node) => node.data.role === "attention-layout")
      .length,
    0,
  );
});

test("two independent transformer layers in a shared loop execute A B A B A B", () => {
  const project = structuredClone(template);
  project.config = {
    ...project.config,
    width: 16,
    heads: 4,
    ffn_width: 32,
    vocab: 32,
  };
  const loop = project.graph.nodes.find(
    (node) => node.data.kind === "loop",
  )!.data;
  loop.role = "transformer";
  loop.params = { iterations: 3, shared: true, heads: 4, ffn_width: 32 };
  exposeTransformerSettings(loop);
  wrapLoopTransformer(loop);
  const graph = loop.body!;
  const a = graph.nodes.find((node) => node.data.kind === "block")!;
  const b = structuredClone(a);
  b.id = "transformer_b";
  b.data.label = "Transformer B";
  graph.nodes.push(b);
  const outgoing = graph.edges.find((edge) => edge.source === a.id)!;
  outgoing.source = b.id;
  graph.edges.push({
    id: "a-b",
    source: a.id,
    sourceHandle: "out",
    target: b.id,
    targetHandle: "a",
  });
  python(
    `
import json, sys, torch
from backend import runtime, export_helpers, sparse_runtime
from backend.compiler import compile_project
for name, module in [('runtime',runtime),('export_helpers',export_helpers),('sparse_runtime',sparse_runtime)]:
    sys.modules[name] = module
torch.set_num_threads(1)
namespace={}
exec(compile_project(json.load(sys.stdin))['model.py'], namespace)
model=namespace['Model']()
a=next(module for name,module in model.named_modules() if name.endswith('.transformer_layer'))
b=next(module for name,module in model.named_modules() if name.endswith('.transformer_b'))
assert set(map(id,a.parameters())).isdisjoint(set(map(id,b.parameters())))
visited=[]
a.register_forward_hook(lambda *_: visited.append('A'))
b.register_forward_hook(lambda *_: visited.append('B'))
out=model(torch.tensor([[1,2,3,4]]))
assert visited == ['A','B','A','B','A','B'], visited
out.square().mean().backward()
assert all(p.grad is not None for p in [*a.parameters(),*b.parameters()])
`,
    JSON.stringify(project),
  );
});

test("editing an outer setting preserves intentional inner boolean, enum and numeric overrides", () => {
  const data = structuredClone(template).graph.nodes.find(
    (node) => node.data.kind === "loop",
  )!.data;
  exposeTransformerSettings(data);
  const attention = data.body!.nodes.find(
    (node) => node.data.kind === "attention",
  )!.data;
  attention.params.causal = false;
  attention.params.backend = "auto";
  attention.params.heads = 4;
  assert.equal(transformerSettings(data)?.causal, false);
  assert.equal(transformerSettings(data)?.backend, "auto");
  assert.equal(transformerSettings(data)?.heads, 4);
  setTransformerSetting(data, "ffn_width", 48, template.config);
  assert.equal(transformerSettings(data)?.causal, false);
  assert.equal(transformerSettings(data)?.backend, "auto");
  assert.equal(transformerSettings(data)?.heads, 4);
  assert.equal(attention.params.causal, false);
  assert.equal(attention.params.backend, "auto");
});

for (const [key, fallback] of Object.entries({
  causal: true,
  rope: true,
  qk_norm: false,
  gated: false,
  qkv_bias: false,
  output_bias: false,
  conv_respects_window: false,
})) {
  test(`omitted inner ${key} shows its effective default and wrapper edits bind it`, () => {
    const data = structuredClone(template).graph.nodes.find(
      (node) => node.data.kind === "loop",
    )!.data;
    const attention = data.body!.nodes.find(
      (node) => node.data.kind === "attention",
    )!.data;
    data.params[key] = !fallback;
    delete attention.params[key];
    assert.equal(transformerSettings(data)?.[key], fallback);
    setTransformerSetting(data, key, !fallback);
    assert.equal(transformerSettings(data)?.[key], !fallback);
    assert.equal(attention.params[key], `$${key}`);
    setTransformerSetting(data, key, fallback);
    assert.equal(transformerSettings(data)?.[key], fallback);
    assert.equal(data.params[key], fallback);
    assert.equal(attention.params[key], `$${key}`);
  });
}

test("loop and inner transformer edit one attention owner in both directions", () => {
  const data = structuredClone(template).graph.nodes.find(
    (node) => node.data.kind === "loop",
  )!.data;
  data.role = "transformer";
  exposeTransformerSettings(data);
  wrapLoopTransformer(data);
  const inner = data.body!.nodes.find(
    (node) => node.data.kind === "block",
  )!.data;
  setAttentionVariant(data, "mqa", template.config, uid);
  assert.equal(
    attentionVariant(inner, { ...template.config, ...data.params }),
    "mqa",
  );
  setAttentionVariant(
    inner,
    "mha",
    { ...template.config, ...data.params },
    uid,
  );
  assert.equal(attentionVariant(data, template.config), "mha");
  setTransformerSetting(data, "heads", 4, template.config);
  assert.equal(transformerSettings(data)?.heads, 4);
  assert.equal(transformerSettings(inner)?.heads, 4);
  assert.equal(inner.params.kv_heads, 4);
});

test("heterogeneous loop bodies and multiple attention branches have no whole-container attention control", () => {
  const data = structuredClone(template).graph.nodes.find(
    (node) => node.data.kind === "loop",
  )!.data;
  data.role = "transformer";
  exposeTransformerSettings(data);
  wrapLoopTransformer(data);
  const a = data.body!.nodes.find((node) => node.data.kind === "block")!;
  const b = structuredClone(a);
  b.id = "b";
  data.body!.nodes.push(b);
  assert.equal(transformerSettings(data), null);
  assert.equal(attentionVariant(data, template.config), "custom");
  const flat = structuredClone(a.data);
  flat.body!.nodes.push({
    ...structuredClone(
      flat.body!.nodes.find((node) => node.data.kind === "attention")!,
    ),
    id: "second_attention",
  });
  assert.equal(transformerSettings(flat), null);
});

test("a body variable is reported as effective and cannot be silently edited from its parent", () => {
  const data = structuredClone(template).graph.nodes.find(
    (node) => node.data.kind === "loop",
  )!.data;
  exposeTransformerSettings(data);
  data.body!.variables = { heads: 4 };
  assert.equal(transformerSettings(data)?.heads, 4);
  assert.throws(
    () => setTransformerSetting(data, "heads", 8, template.config),
    /local graph variable/,
  );
});

test("new transformer FFN width is local while heads follow the parent", () => {
  const data = structuredClone(template).graph.nodes.find(
    (node) => node.data.kind === "loop",
  )!.data;
  data.role = "transformer";
  inheritTransformerDefaults(data, { heads: 12, ffn_width: 2176 });
  exposeTransformerSettings(data);
  wrapLoopTransformer(data);
  const inner = data.body!.nodes.find(
    (node) => node.data.role === "transformer",
  )!.data;
  assert.equal(inner.body!.variables?.ffn_width, 2176);
  assert.equal(inner.params.ffn_width, undefined);
  assert.equal(inner.params.heads, "$heads");
  const ffn = inner.body!.nodes.find((node) => node.data.kind === "ffn")!;
  ffn.data.params.hidden = 384;
  setTransformerSetting(inner, "ffn_width", 512, { ffn_width: 2176 });
  assert.equal(inner.body!.variables?.ffn_width, 512);
  assert.equal(ffn.data.params.hidden, "$ffn_width");
  assert.equal(transformerSettings(inner)?.ffn_width, 512);
});

test("attention output gate and residual type are independent choices", () => {
  const data = structuredClone(template).graph.nodes.find(
    (node) => node.data.kind === "loop",
  )!.data;
  const residual = data.body!.nodes.find((node) => node.id === "r1")!;
  assert.equal(transformerResidual(data), "add");
  setTransformerResidual(data, "gate");
  assert.equal(transformerResidual(data), "gate");
  assert.equal(
    data.body!.nodes.find((node) => node.id === "attn")!.data.params.gated,
    false,
  );
  residual.data.share = "residual_gate";
  setTransformerResidual(data, "gate");
  assert.equal(residual.data.share, "residual_gate");
  setTransformerResidual(data, "add");
  assert.equal(transformerResidual(data), "add");
  assert.equal(residual.data.share, undefined);
});
