import type { Project, Graph } from "../types";

export function download(
  name: string,
  content: Blob | string,
  type = "application/json",
) {
  const url = URL.createObjectURL(
    typeof content === "string" ? new Blob([content], { type }) : content,
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function graphAt(project: Project, path: string[]) {
  let g = project.graph;
  for (const id of path) {
    const nested = g.nodes.find((n) => n.id === id)?.data.body;
    if (!nested) return project.graph;
    g = nested;
  }
  return g;
}
export function replaceGraph(project: Project, path: string[], graph: Graph) {
  if (!path.length) {
    project.graph = graph;
    return;
  }
  const parent = graphAt(project, path.slice(0, -1));
  const target = parent.nodes.find((n) => n.id === path.at(-1));
  if (!target?.data.body)
    throw new Error("The containing block no longer exists");
  target.data.body = graph;
}
