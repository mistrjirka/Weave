import { newId as uid } from "../ui";
import { makeIndependent, adapterBody } from "../graph";
import { graphVariableScope } from "../graphVariables";
import { inheritTransformerDefaults, exposeTransformerSettings, wrapLoopTransformer } from "../transformerControls";
import type { Project, Graph, NodeData, JsonValue, CatalogEntry } from "../types";

const copy = <T,>(value: T): T => structuredClone(value);

export function transformerBody(templates: Project[]): Graph {
  const body = templates
    .find((t) => t.id === "transformer")
    ?.graph.nodes.find((n) => n.data.kind === "loop")?.data.body;
  if (!body)
    throw new Error("The transformer block definition is unavailable");
  return copy(body);
}
function moeBody(templates: Project[]): Graph {
  const body = templates
    .find((t) => t.id === "moe-transformer")
    ?.graph.nodes.find((n) => n.data.kind === "moe")?.data.body;
  if (!body)
    throw new Error("The mixture of experts definition is unavailable");
  return copy(body);
}
export function resizeExperts(data: NodeData, value: JsonValue) {
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value < 1 ||
    value > 256
  )
    throw new Error("Expert count must be an integer between 1 and 256.");
  const body = data.body;
  if (!body) throw new Error("The MoE graph is missing.");
  const input = body.nodes.find((n) => n.data.kind === "input");
  const router = body.nodes.find((n) => n.data.kind === "moe_router");
  const combine = body.nodes.find((n) => n.data.kind === "moe_combine");
  const experts = body.nodes.filter((n) => n.data.role === "moe-expert");
  if (!input || !router || !combine || !experts.length)
    throw new Error("The MoE scaffold is incomplete.");
  for (const expert of experts.slice(value)) {
    body.nodes = body.nodes.filter((n) => n.id !== expert.id);
    body.edges = body.edges.filter(
      (edge) => edge.source !== expert.id && edge.target !== expert.id,
    );
  }
  for (let i = experts.length; i < value; i++) {
    const expert = copy(experts[0]);
    makeIndependent(expert);
    expert.id = uid();
    expert.data.label = `Expert ${i + 1}`;
    expert.position = { x: experts[0].position.x, y: i * 230 };
    body.nodes.push(expert);
    body.edges.push(
      {
        id: uid(),
        source: input.id,
        target: expert.id,
        sourceHandle: "out",
        targetHandle: "a",
        type: "smoothstep",
      },
      {
        id: uid(),
        source: expert.id,
        target: combine.id,
        sourceHandle: "out",
        targetHandle: `e${i}`,
        type: "smoothstep",
      },
    );
  }
  router.data.params.experts = value;
  combine.data.params.experts = value;
  data.params.experts = value;
  if (typeof data.params.top_k === "number" && data.params.top_k > value)
    data.params.top_k = value;
}
export function catalogNodeData(entry: CatalogEntry, templates: Project[], project: Project, path: string[]): NodeData {
  const { kind, label, params: p } = entry;
  let body: Graph | undefined;
  if (["loop", "block"].includes(kind)) {
    body = {
      nodes: [
        {
          id: "state",
          type: "architecture",
          position: { x: 0, y: 100 },
          data: {
            kind: "input",
            label: "State input",
            params: {
              index: 0,
              shape: ["B", "T", "$width"],
              dtype: "float32",
            },
          },
        },
        {
          id: "out",
          type: "architecture",
          position: { x: 350, y: 100 },
          data: {
            kind: "output",
            label: "State output",
            params: { index: 0 },
          },
        },
      ],
      edges: [
        {
          id: "pass",
          source: "state",
          target: "out",
          sourceHandle: "out",
          targetHandle: "a",
          type: "smoothstep",
        },
      ],
    };
  }
  if (kind === "moe") body = moeBody(templates);
  if (entry.role === "moe-shared-expert") {
    const expert = moeBody(templates).nodes.find((n) => n.data.role === "moe-expert")
      ?.data.body;
    if (!expert) throw new Error("Expert definition is unavailable");
    body = copy(expert);
  }
  if (entry.role === "transformer") body = transformerBody(templates);
  if (entry.role === "adapter") body = adapterBody();
  const data: NodeData = {
    kind,
    label,
    params: copy(p),
    ...(body ? { body } : {}),
    ...(entry.role ? { role: entry.role } : {}),
  };
  if (entry.role === "transformer") {
    inheritTransformerDefaults(
      data,
      graphVariableScope(project, path).bindings,
    );
    exposeTransformerSettings(data);
  }
  if (
    entry.role === "transformer" ||
    ["attention", "cross_attention", "rope"].includes(kind)
  ) {
    const theta = graphVariableScope(project, path).effective.theta;
    data.params.theta = typeof theta === "number" ? "$theta" : 1_000_000;
  }
  wrapLoopTransformer(data);
  return data;
}
