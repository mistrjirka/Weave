import type { Graph, JsonValue, Parameters, Project } from "./types";

/** Find links reached from this graph. A child parameter or variable owns its
 * names inside that child, so those uses do not belong to the parent scope. */
export function graphVariableUses(graph: Graph): Record<string, string[]> {
  const uses: Record<string, string[]> = {};
  function collect(
    value: JsonValue | undefined,
    location: string,
    blocked: Set<string>,
  ) {
    if (typeof value === "string" && /^\$[A-Za-z_]\w*$/.test(value)) {
      const name = value === "$sqrt_width" ? "width" : value.slice(1);
      if (!blocked.has(name)) (uses[name] ??= []).push(location);
    } else if (Array.isArray(value)) {
      value.forEach((item, index) =>
        collect(item, `${location}[${index}]`, blocked),
      );
    } else if (value && typeof value === "object") {
      Object.entries(value).forEach(([key, item]) =>
        collect(item, `${location}.${key}`, blocked),
      );
    }
  }
  function visit(current: Graph, blocked: Set<string>, prefix: string) {
    for (const node of current.nodes) {
      const location = prefix + node.data.label;
      Object.entries(node.data.params).forEach(([key, value]) =>
        collect(value, `${location} · ${key}`, blocked),
      );
      if (node.data.body) {
        const childBlocked = new Set([
          ...blocked,
          ...Object.keys(node.data.params),
          ...Object.keys(node.data.body.variables ?? {}),
        ]);
        visit(node.data.body, childBlocked, `${location} / `);
      }
    }
  }
  visit(graph, new Set(), "");
  return uses;
}

/** Effective values at the current graph, including enclosing block parameters. */
export function graphVariableScope(project: Project, path: string[]) {
  let graph = project.graph;
  let inherited: Record<string, number> = {};
  let bindings: Parameters = { ...project.config, ...graph.variables };
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
    bindings = {
      ...bindings,
      ...previewParameters(node.data.params, bindings),
      ...node.data.body.variables,
    };
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
    bindings,
  };
}

export function bindingValue(
  value: string,
  variables: Parameters,
): JsonValue | undefined {
  if (value === "$sqrt_width" && typeof variables.width === "number")
    return Math.sqrt(variables.width);
  const resolved = variables[value.slice(1)];
  return resolved;
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
