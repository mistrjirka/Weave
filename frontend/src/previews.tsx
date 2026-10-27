import { Fragment } from "react";
import type { ReactNode } from "react";
import { ArrowRight, RotateCcw, GitBranch } from "lucide-react";
import type { Graph, Parameters } from "./types";
import { previewAllowed } from "./patterns";

interface PreviewProps {
  kind: string;
  p: Parameters;
  body?: Graph;
  role?: string;
}
type Renderer = (props: PreviewProps) => ReactNode;
export function PatternGrid({ p }: { p: Parameters }) {
  const size = 8,
    kind = p.pattern || "local";
  return (
    <div
      className="pattern-grid"
      role="img"
      aria-label={`${kind.replaceAll("_", " ")} attention pattern schematic`}
    >
      {Array.from({ length: size * size }, (_, i) => (
        <i
          key={i}
          className={
            previewAllowed(Math.floor(i / size), i % size, p) ? "allowed" : ""
          }
        />
      ))}
    </div>
  );
}
function ReturnPath() {
  return (
    <svg className="return-path" viewBox="0 0 280 30">
      <path d="M263 0 V12 Q263 22 251 22 H24 Q12 22 12 10 V4" />
      <path d="M6 10 L12 4 L18 10" />
    </svg>
  );
}
const stageLabels: Record<string, string> = {
  norm: "Norm",
  attention: "Attention",
  cross_attention: "Cross attn",
  ffn: "FFN",
  linear: "Linear",
  block: "Block",
  loop: "Loop",
};
function TransformerMotif({ body }: { body: Graph }) {
  const stages = body.nodes.filter((n) =>
    Object.hasOwn(stageLabels, n.data.kind),
  );
  const residual = body.nodes.some((n) =>
    ["add", "gate"].includes(n.data.kind),
  );
  const visibleStages = stages.slice(0, 5);
  return (
    <div className="transformer-motif">
      {residual && (
        <div className="transformer-bypass">
          residual stream <span>↷</span>
        </div>
      )}
      <div className="transformer-stages">
        {visibleStages.map((n, i) => (
          <span className="transformer-step" key={n.id}>
            {i > 0 && <ArrowRight size={12} />}
            <span
              className={
                "transformer-stage " +
                (n.data.kind.includes("attention") ? "attention-stage" : "")
              }
            >
              {stageLabels[n.data.kind]}
            </span>
            {residual && i === visibleStages.length - 1 && (
              <b className="transformer-sum">+</b>
            )}
          </span>
        ))}
      </div>
      <small>
        {stages.length > 5 ? `${stages.length} stages · ` : ""}
        {
          body.nodes.filter((n) => !["input", "output"].includes(n.data.kind))
            .length
        }{" "}
        editable operations
      </small>
    </div>
  );
}
const pattern: Renderer = ({ p }) => (
  <div className="pattern-preview">
    <PatternGrid p={p} />
    <div>
      <b>{String(p.pattern || "local").replaceAll("_", " ")}</b>
      <span>query × key positions</span>
      <small>
        {p.pattern === "content_topk"
          ? "input-dependent selection"
          : "64-position schematic"}
      </small>
    </div>
  </div>
);
const combine: Renderer = ({ kind }) => (
  <div className="pattern-combine">
    <div className="pattern-operand">
      <b>A</b>
      <small>input pattern</small>
    </div>
    <b>{kind === "pattern_union" ? "∪" : "∩"}</b>
    <div className="pattern-operand">
      <b>B</b>
      <small>input pattern</small>
    </div>
  </div>
);
const denoising: Renderer = ({ kind, p }) => (
  <div className="denoise-preview">
    <div className="denoise-states">
      {[
        ["◇", "◇", "◇"],
        ["a", "◇", "◇"],
        ["a", "b", "c"],
      ].map((tokens, i) => (
        <Fragment key={i}>
          {i > 0 && <ArrowRight size={12} />}
          <div>
            {tokens.map((t, j) => (
              <span className={t === "◇" ? "unresolved" : ""} key={j}>
                {t}
              </span>
            ))}
          </div>
        </Fragment>
      ))}
    </div>
    <div className="denoise-cycle">
      <span>predict → select → update state</span>
      {kind === "loop" ? <RotateCcw size={15} /> : <ArrowRight size={15} />}
    </div>
    {kind === "loop" && <ReturnPath />}
    <small>
      {kind === "loop" ? `${p.iterations || 1} steps` : "one application"} ·
      editable operations
    </small>
  </div>
);
const transformer: Renderer = ({ kind, p, body }) =>
  body ? (
    <div
      className={
        "transformer-preview " + (kind === "loop" ? "recurrent-preview" : "")
      }
    >
      <TransformerMotif body={body} />
      {kind === "loop" && (
        <>
          <ReturnPath />
          <span className="repeat-caption">
            {p.iterations || 1} applications ·{" "}
            {p.shared === false ? "independent parameters" : "same parameters"}
          </span>
        </>
      )}
    </div>
  ) : null;
const adapter: Renderer = ({ body }) => (
  <div className="adapter-preview">
    <span>input</span>
    <ArrowRight size={13} />
    <div>
      <b>Learned update</b>
      <small>
        {body
          ? body.nodes.some((n) => ["add", "gate"].includes(n.data.kind))
            ? "residual adapter"
            : "learned module"
          : "independent projection"}
      </small>
    </div>
    <ArrowRight size={13} />
    <span>output</span>
  </div>
);
const predictor: Renderer = ({ body }) => (
  <div className="predictor-preview">
    <div>
      {body?.nodes
        .filter((n) =>
          [
            "embedding",
            "ngram",
            "engram_hash",
            "lookup",
            "engram_lookup",
          ].includes(n.data.kind),
        )
        .slice(0, 3)
        .map((n) => (
          <span key={n.id}>{n.data.label}</span>
        ))}
    </div>
    <GitBranch size={22} />
    <b>Editable neural graph</b>
  </div>
);
const kinds: Record<string, Renderer> = {
  attention_pattern: pattern,
  pattern_union: combine,
  pattern_intersection: combine,
};
const roles: Record<string, Renderer> = {
  denoising,
  transformer,
  adapter,
  predictor,
};
export function renderPreview(props: PreviewProps): ReactNode {
  return (kinds[props.kind] || roles[props.role || ""])?.(props) ?? null;
}

export function MaskMatrixEditor({
  matrix,
  onChange,
}: {
  matrix: boolean[][];
  onChange: (matrix: boolean[][]) => void;
}) {
  const shown = matrix.slice(0, 16);
  return (
    <div className="mask-matrix-editor">
      <span>Allowed query → key pairs</span>
      <div
        className="mask-matrix-cells"
        style={{
          gridTemplateColumns: `repeat(${Math.min(matrix[0]?.length || 0, 16)},1fr)`,
        }}
      >
        {shown.flatMap((row, q) =>
          row.slice(0, 16).map((allowed, k) => (
            <button
              key={`${q}:${k}`}
              aria-label={`Query ${q}, key ${k}`}
              aria-pressed={allowed}
              title={`Query ${q} → key ${k}: ${allowed ? "allowed" : "blocked"}`}
              className={allowed ? "allowed" : ""}
              onClick={() => {
                const next = structuredClone(matrix);
                next[q][k] = !allowed;
                onChange(next);
              }}
            />
          )),
        )}
      </div>
      <small>
        {matrix.length} × {matrix[0]?.length || 0} fixed positions. Click a cell
        to toggle.
        {matrix.length > 16 || (matrix[0]?.length || 0) > 16
          ? " Showing the first 16 positions; use the matrix field for all positions."
          : ""}
      </small>
    </div>
  );
}
