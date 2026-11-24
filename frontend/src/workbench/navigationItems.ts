import { catalog } from "../catalog";
import type { GraphNode, Project } from "../types";

export function workbenchNavigationItems(project: Project, path: string[], search: string) {
  let crumbs: string[] = [];
  let g = project.graph;
  for (const id of path) {
    const n = g.nodes.find((n: GraphNode) => n.id === id);
    if (!n) break;
    crumbs.push(n.data.label);
    if (!n.data.body) break;
    g = n.data.body;
  }
  const groups = [
    ...new Set(
      catalog
        .filter(
          (e) =>
            e.label.toLowerCase().includes(search.toLowerCase()) ||
            e.description.toLowerCase().includes(search.toLowerCase()),
        )
        .map((e) => e.category),
    ),
  ];
  return { crumbs, groups };
}
