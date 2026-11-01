import type { Graph } from "./types";

export function EmbeddingTiePicker({
  graph,
  value,
  onChange,
}: {
  graph: Graph;
  value: string;
  onChange: (value: string) => void;
}) {
  const embeddings = graph.nodes.filter(
    (node) => node.data.kind === "embedding",
  );
  const missing = value && !embeddings.some((node) => node.id === value);
  return (
    <section aria-label="Output embedding weights">
      <label className="property-field">
        <span>Output weights</span>
        <select
          aria-label="Output weights"
          value={value}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">Independent output weights</option>
          {missing && (
            <option value={value}>Missing embedding ({value})</option>
          )}
          {embeddings.map((node) => (
            <option key={node.id} value={node.id}>
              Share with {node.data.label}
            </option>
          ))}
        </select>
      </label>
      <p className="field-help">
        Share the input embedding’s trainable matrix to produce token scores.
        The hidden width and vocabulary must match; graph validation checks
        this.
        {embeddings.length === 0 &&
          " Add an embedding in this graph to share its weights."}
      </p>
    </section>
  );
}
