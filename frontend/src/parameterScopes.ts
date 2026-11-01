import type { Graph, JsonValue, Parameters, Project } from "./types";

const layerDefaults = new Set(["heads", "ffn_width"]);

function bindings(value: JsonValue | undefined): string[] {
  if (typeof value === "string" && /^\$\w+$/.test(value))
    return [value.slice(1)];
  if (Array.isArray(value)) return value.flatMap(bindings);
  if (value && typeof value === "object")
    return Object.values(value).flatMap(bindings);
  return [];
}

/** Preserve explicit variable links while migrating legacy input tokenizers.
 * Missing container defaults inherit through bindings, never frozen copies.
 */
export function localizeLayerDefaults(project: Project): Project {
  const next = structuredClone(project);
  if (next.tokenizer) {
    for (const node of next.graph.nodes)
      if (
        node.data.kind === "input" &&
        (node.data.params.dtype ?? "int64") === "int64" &&
        (node.data.params.shape ?? ["B", "T"]).length === 2 &&
        !node.data.tokenizer
      )
        node.data.tokenizer = structuredClone(next.tokenizer);
  }
  delete next.tokenizer;
  function visit(graph: Graph, inherited: Parameters) {
    inherited = { ...inherited, ...graph.variables };
    for (const node of graph.nodes) {
      const params = { ...node.data.params };
      const body = node.data.body;
      if (body) {
        const used = new Set(
          body.nodes.flatMap((child) =>
            Object.values(child.data.params).flatMap(bindings),
          ),
        );
        for (const key of layerDefaults)
          if (
            used.has(key) &&
            params[key] === undefined &&
            inherited[key] !== undefined
          )
            params[key] = "$" + key;
      }
      node.data.params = params;
      if (body) {
        const values = Object.fromEntries(
          Object.entries(params).map(([key, value]) => [
            key,
            typeof value === "string" && value.startsWith("$")
              ? (inherited[value.slice(1)] ?? value)
              : value,
          ]),
        );
        visit(body, { ...inherited, ...values });
      }
    }
  }
  visit(next.graph, next.config);
  return next;
}
