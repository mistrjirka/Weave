import type { JsonValue, Parameters, Project } from "./types";

/** Effective values at the current graph, including enclosing block parameters. */
export function graphVariableScope(project: Project, path: string[]) {
  let graph = project.graph;
  let inherited: Record<string, number> = {};
  let values = { ...project.config, ...graph.variables };
  for (const id of path) {
    const node = graph.nodes.find((node) => node.id === id);
    if (!node?.data.body) break;
    const resolved = Object.fromEntries(
      Object.entries(node.data.params).flatMap(([key, value]) => {
        const effective =
          typeof value === "string" && value.startsWith("$")
            ? values[value.slice(1)]
            : value;
        return typeof effective === "number" ? [[key, effective]] : [];
      }),
    );
    inherited = { ...values, ...resolved };
    graph = node.data.body;
    values = { ...inherited, ...graph.variables };
  }
  return {
    inherited,
    own: path.length
      ? { ...graph.variables }
      : { ...project.config, ...graph.variables },
    effective: values,
  };
}

export function bindingValue(
  value: string,
  variables: Parameters,
): number | undefined {
  if (value === "$sqrt_width" && typeof variables.width === "number")
    return Math.sqrt(variables.width);
  const resolved = variables[value.slice(1)];
  return typeof resolved === "number" ? resolved : undefined;
}

/** Resolve numeric bindings for diagrams without changing the editable document. */
export function previewParameters(
  params: Parameters,
  variables: Parameters,
): Parameters {
  function resolve(value: JsonValue): JsonValue {
    if (typeof value === "string" && value.startsWith("$"))
      return bindingValue(value, variables) ?? value;
    if (Array.isArray(value)) return value.map(resolve);
    if (value && typeof value === "object")
      return Object.fromEntries(
        Object.entries(value).map(([key, item]) => [key, resolve(item)]),
      );
    return value;
  }
  return Object.fromEntries(
    Object.entries(params).map(([key, value]) => [
      key,
      value === undefined ? null : resolve(value),
    ]),
  );
}
