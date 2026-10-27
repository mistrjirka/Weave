import type { Graph, Project } from "./types";

function computationGraph(graph: Graph): Graph {
  return {
    ...(graph.variables ? { variables: graph.variables } : {}),
    nodes: graph.nodes.map((node) => ({
      id: node.id,
      type: node.type,
      position: { x: 0, y: 0 },
      data: {
        ...node.data,
        ...(node.data.body ? { body: computationGraph(node.data.body) } : {}),
      },
    })),
    edges: graph.edges.map(
      ({ id, source, target, sourceHandle, targetHandle }) => ({
        id,
        source,
        target,
        sourceHandle,
        targetHandle,
      }),
    ),
  };
}

/** Geometry/selection must not restart validation or cancel an in-flight compile.
 * Keep the actual document for persistence and exports; normalize only this API input.
 */
export function compilationInput(project: Project): string {
  return JSON.stringify({ ...project, graph: computationGraph(project.graph) });
}
