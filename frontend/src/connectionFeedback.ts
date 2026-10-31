import type { Project, ValidationReport } from "./types";

export interface ConnectionAttempt {
  projectId: string;
  path: string[];
  edgeId: string;
}

/** Surface the receiving block's errors without unrelated unfinished inputs. */
export function connectionFeedback(
  project: Project,
  report: ValidationReport,
  attempt: ConnectionAttempt,
) {
  if (project.id !== attempt.projectId) return null;
  let graph = project.graph;
  for (const id of attempt.path) {
    const body = graph.nodes.find((node) => node.id === id)?.data.body;
    if (!body) return null;
    graph = body;
  }
  const edge = graph.edges.find((item) => item.id === attempt.edgeId);
  if (!edge) return null;
  const prefix = attempt.path.length ? attempt.path.join("/") + "/" : "";
  const target = prefix + edge.target;
  const port = edge.targetHandle || "a";
  const issues = report.errors.filter(
    (issue) =>
      (issue.node === target && (!issue.port || issue.port === port)) ||
      issue.node?.startsWith(target + "/") ||
      ((issue.node ?? "") === prefix.slice(0, -1) &&
        issue.message.includes("direct cycle")),
  );
  if (!issues.length) return null;
  const label = (id: string) =>
    graph.nodes.find((node) => node.id === id)?.data.label || id;
  return {
    title: `Connection error: ${label(edge.source)} → ${label(edge.target)} (input ${port})`,
    messages: [...new Set(issues.map((issue) => issue.message))],
  };
}
