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
  migrateLegacyEmbeddingTies(next);
  return next;
}

/** Translate a legacy head-to-embedding ID into the existing weight-group model.
 * Multiple heads can have distinct settings and biases, so keep their legacy
 * ties until those semantics can be represented without joining the heads.
 */
export function migrateLegacyEmbeddingTies(project: Project): void {
  const used = new Set<string>();
  const headGroups = new Map<string, number>();
  function collect(graph: Graph) {
    for (const node of graph.nodes) {
      if (node.data.share) {
        used.add(node.data.share);
        if (node.data.kind === "head")
          headGroups.set(
            node.data.share,
            (headGroups.get(node.data.share) ?? 0) + 1,
          );
      }
      if (node.data.body) collect(node.data.body);
    }
  }
  collect(project.graph);
  function visit(graph: Graph, path: string, insideIndependentLoop = false) {
    const heads = graph.nodes.filter(
      (node) =>
        node.data.kind === "head" && typeof node.data.params.tie === "string",
    );
    const byEmbedding = new Map<string, typeof heads>();
    for (const head of heads) {
      const id = String(head.data.params.tie);
      byEmbedding.set(id, [...(byEmbedding.get(id) ?? []), head]);
    }
    for (const [id, tied] of byEmbedding) {
      const embedding = graph.nodes.find(
        (node) => node.id === id && node.data.kind === "embedding",
      );
      if (!embedding || tied.length !== 1 || insideIndependentLoop) continue;
      const head = tied[0];
      if (
        embedding.data.share &&
        (headGroups.get(embedding.data.share) ?? 0) >
          (head.data.share === embedding.data.share ? 1 : 0)
      )
        continue;
      if (head.data.share && head.data.share !== embedding.data.share) continue;
      if (!embedding.data.share) {
        const base = `legacy_tie_${encodeURIComponent(path + "/" + id)}`;
        let group = base;
        for (let suffix = 2; used.has(group); suffix++)
          group = `${base}_${suffix}`;
        embedding.data.share = group;
        used.add(group);
      }
      const alreadyGrouped = head.data.share === embedding.data.share;
      head.data.share = embedding.data.share;
      if (!alreadyGrouped)
        headGroups.set(
          embedding.data.share,
          (headGroups.get(embedding.data.share) ?? 0) + 1,
        );
      delete head.data.params.tie;
    }
    for (const node of graph.nodes)
      if (node.data.body)
        visit(
          node.data.body,
          path + "/" + node.id,
          insideIndependentLoop ||
            (node.data.kind === "loop" &&
              node.data.params.shared !== true &&
              node.data.params.shared !== undefined),
        );
  }
  visit(project.graph, "");
}
