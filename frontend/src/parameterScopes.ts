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

function resolve(
  value: JsonValue | undefined,
  variables: Parameters,
  local: Set<string>,
): JsonValue | undefined {
  if (
    typeof value === "string" &&
    value.startsWith("$") &&
    layerDefaults.has(value.slice(1)) &&
    !local.has(value.slice(1))
  )
    return variables[value.slice(1)] ?? value;
  if (Array.isArray(value))
    return value.map((item) => resolve(item, variables, local) ?? null);
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        resolve(item, variables, local) ?? null,
      ]),
    );
  return value;
}

/** Materialize legacy architecture defaults on their owning blocks. Internal
 * bindings remain local to the container; editing one independent block cannot
 * silently reconfigure the rest of the model. Project input dimensions survive.
 */
export function localizeLayerDefaults(project: Project): Project {
  const next = structuredClone(project);
  function visit(graph: Graph, inherited: Parameters, local: Set<string>) {
    for (const node of graph.nodes) {
      const params = Object.fromEntries(
        Object.entries(node.data.params).map(([key, value]) => [
          key,
          resolve(value, inherited, local),
        ]),
      );
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
            params[key] = inherited[key];
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
        visit(
          body,
          { ...inherited, ...values },
          new Set([...local, ...Object.keys(params)]),
        );
      }
    }
  }
  visit(next.graph, next.config, new Set());
  return next;
}
