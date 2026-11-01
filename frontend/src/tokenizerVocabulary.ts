import type { Graph, NodeData, Project, TokenizerInfo } from "./types";

const usesVocabulary = (value: unknown): boolean => {
  if (value === "$vocab") return true;
  if (Array.isArray(value)) return value.some(usesVocabulary);
  return (
    !!value &&
    typeof value === "object" &&
    Object.values(value).some(usesVocabulary)
  );
};

/** A tokenizer-owned value is not an editable graph variable. Keep the legacy
 * fallback visible only when a consumer in this scope still needs it. */
export function tokenizerOwnsVocabulary(
  project: Project,
  path: string[],
  domains: Record<string, TokenizerInfo> = {},
): boolean {
  let graph = project.graph;
  for (const id of path) {
    const body = graph.nodes.find((node) => node.id === id)?.data.body;
    if (!body) return false;
    graph = body;
  }
  const prefix = path.length ? path.join("/") + "/" : "";
  let hasTokenizer = Object.keys(domains).some((key) => key.startsWith(prefix));
  let needsFallback = false;
  function visit(graph: Graph, scope: string) {
    for (const node of graph.nodes) {
      const tokenizer = domains[scope + node.id] ?? node.data.tokenizer;
      hasTokenizer ||= !!tokenizer;
      if (usesVocabulary(node.data.params) && !tokenizer) needsFallback = true;
      if (node.data.body) visit(node.data.body, scope + node.id + "/");
    }
  }
  visit(graph, prefix);
  return hasTokenizer && !needsFallback;
}

export function vocabularyField(data: NodeData): "vocab" | "out" | undefined {
  return data.kind === "embedding"
    ? "vocab"
    : data.kind === "head"
      ? "out"
      : undefined;
}

export function automaticVocabulary(
  data: NodeData,
  tokenizer?: TokenizerInfo | null,
) {
  const field = vocabularyField(data);
  return (
    !!tokenizer &&
    !!field &&
    (data.params[field] === "$vocab" || data.params[field] === undefined)
  );
}
