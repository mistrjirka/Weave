import type { Parameters, Project } from "./types";

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
