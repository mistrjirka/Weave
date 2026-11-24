import { ArrowDownToLine, ArrowRight, Braces, RotateCcw } from "lucide-react";
import { multiOutputLanes } from "../lanePorts";
import { renderPreview } from "../previews";
import type { Graph, GraphNode, Parameters } from "../types";
import { family } from "./presentation";

export function Diagram({
  kind,
  p,
  body,
  role,
}: {
  kind: string;
  p: Parameters;
  body?: Graph;
  role?: string;
}) {
  if (multiOutputLanes.has(kind) || kind === "lane_join") {
    const count =
      typeof p.lanes === "number" ? Math.min(8, Math.max(1, p.lanes)) : 4;
    const split = kind !== "lane_join";
    return (
      <div className="explicit-lane-preview">
        <svg viewBox={`0 0 100 ${count * 20}`} aria-hidden="true">
          {Array.from({ length: count }, (_, i) => (
            <path
              key={i}
              d={
                split
                  ? `M5 ${count * 10} H30 L70 ${10 + i * 20} H95`
                  : `M5 ${10 + i * 20} H30 L70 ${count * 10} H95`
              }
            />
          ))}
        </svg>
        <span>
          {kind === "lane_fork"
            ? "identical copies"
            : kind === "lane_split"
              ? "unpack lanes"
              : p.mode === "mean"
                ? "average"
                : p.mode === "sum"
                  ? "add lanes"
                  : "stack lanes"}
        </span>
      </div>
    );
  }
  const preview = renderPreview({ kind, p, body, role });
  if (preview) return preview;
  if (kind.startsWith("lane_") || kind === "routing")
    return (
      <div className="lane-preview">
        <svg viewBox="0 0 240 62">
          {[0, 1, 2, 3].map((v) => (
            <g key={v}>
              <path
                d={
                  kind === "lane_reduce"
                    ? `M12 ${10 + v * 14} H110 L175 31 H228`
                    : kind === "lane_expand"
                      ? `M12 31 H60 L125 ${10 + v * 14} H228`
                      : `M12 ${10 + v * 14} H70 L170 ${52 - v * 14} H228`
                }
              />
              <circle
                cx={kind === "routing" ? 120 : 170}
                cy={10 + v * 14}
                r="3"
              />
            </g>
          ))}
        </svg>
        <span>
          {kind === "routing"
            ? "input-dependent routing · Sinkhorn"
            : `${p.lanes || 4} parallel residual streams`}
        </span>
      </div>
    );
  if (kind === "engram_hash") return <Diagram kind="ngram" p={p} />;
  if (kind === "engram_lookup") return <Diagram kind="lookup" p={p} />;
  if (kind === "loop")
    return (
      <div className="loop-preview">
        <div className="loop-rail">
          <span className="mini-tag">state in</span>
          <span className="loop-label">
            {p.shared === false ? "independent layers" : "shared weights"}
            <RotateCcw size={12} />
          </span>
        </div>
        <div className="loop-body">
          {(body?.nodes || [])
            .filter(
              (n: GraphNode) => !["input", "output"].includes(n.data.kind),
            )
            .slice(0, 3)
            .map((n: GraphNode, i: number) => (
              <span className="loop-stage" key={n.id}>
                {i > 0 && <ArrowRight size={13} />}
                <span className={"mini-block " + family(n.data.kind)}>
                  {n.data.kind === "block"
                    ? "predictor"
                    : n.data.kind === "norm"
                      ? "norm"
                      : n.data.kind === "select"
                        ? "select"
                        : n.data.kind === "attention"
                          ? "attention"
                          : n.data.kind === "ffn"
                            ? "SwiGLU"
                            : n.data.label.slice(0, 12)}
                </span>
              </span>
            ))}
        </div>
        <svg className="return-path" viewBox="0 0 280 30">
          <path d="M263 0 V12 Q263 22 251 22 H24 Q12 22 12 10 V4" />
          <path d="M6 10 L12 4 L18 10" />
        </svg>
        <span className="repeat-caption">
          {p.iterations || 1}{" "}
          {p.shared === false ? "distinct layers" : "iterations"} · editable
          body
        </span>
      </div>
    );
  if (kind === "embedding" || kind === "lookup")
    return (
      <div className="embedding-preview">
        <div className="token-column">
          {["42", "17", "93"].map((v) => (
            <span key={v}>{v}</span>
          ))}
        </div>
        <ArrowRight size={16} />
        <div className="embedding-grid">
          {Array.from({ length: 24 }, (_, i) => (
            <i key={i} style={{ opacity: 0.2 + ((i * 7) % 9) / 12 }} />
          ))}
        </div>
        <span className="vector-caption">
          {kind === "lookup" ? "memory tables" : "embedding table"}
          <br />
          <b>→ vectors</b>
        </span>
      </div>
    );
  if (kind === "ngram")
    return (
      <div className="ngram-preview">
        <div className="token-row">
          {["the", "cat", "sat", "down"].map((t, i) => (
            <span key={t} className={"t" + i}>
              {t}
            </span>
          ))}
        </div>
        <div className="window w1" />
        <div className="window w2" />
        <div className="ngram-label">
          n = {(p.orders || [2, 3]).join(", ")}
          <span>overlap → hash</span>
        </div>
      </div>
    );
  if (kind === "attention" || kind === "cross_attention" || kind === "sdpa")
    return (
      <div className="attention-preview">
        <div className="qkv">
          {["Q", "K", "V"].map((t) => (
            <span key={t}>
              {t}
              <i />
            </span>
          ))}
        </div>
        <div
          className={
            "attention-matrix " + (p.causal !== false ? "causal" : "full")
          }
        >
          {Array.from({ length: 25 }, (_, i) => (
            <i
              key={i}
              className={
                p.causal === false || i % 5 <= Math.floor(i / 5) ? "filled" : ""
              }
            />
          ))}
        </div>
        <span className="vector-caption">
          {String(p.heads || "$heads")} / {p.kv_heads || "H"}
          <br />
          <b>
            {p.causal === false ? "bidirectional" : "causal"}
            {p.window ? " · windowed" : ""}
          </b>
        </span>
      </div>
    );
  if (kind === "mask" || kind === "select")
    return (
      <div className="sampling-preview">
        <div className="token-row">
          {(kind === "mask"
            ? ["the", "cat", "◇", "◇"]
            : ["◇", "◇", "◇", "◇"]
          ).map((t, i) => (
            <span className={t === "◇" ? "masked" : ""} key={i}>
              {t}
            </span>
          ))}
        </div>
        {kind === "select" && (
          <>
            <ArrowDownToLine size={14} />
            <div className="token-row">
              <span>the</span>
              <span className="masked">◇</span>
              <span>sat</span>
              <span className="masked">◇</span>
            </div>
          </>
        )}
        <span className="preview-label">
          {kind === "mask"
            ? "preserve prompt · append masks"
            : "illustrative unmasking step"}
        </span>
      </div>
    );
  if (kind === "block")
    return (
      <div className="block-preview">
        {(body?.nodes || [])
          .filter((n: GraphNode) => !["input", "output"].includes(n.data.kind))
          .slice(0, 5)
          .map((n: GraphNode) => (
            <span className={family(n.data.kind)} key={n.id}>
              {n.data.kind === "embedding"
                ? "embed"
                : n.data.kind === "loop"
                  ? "↻ loop"
                  : n.data.kind === "ngram"
                    ? "n-grams"
                    : n.data.kind}
            </span>
          ))}
        <span className="block-count">{body?.nodes?.length || 0} nodes</span>
      </div>
    );
  if (kind === "input" || kind === "output")
    return (
      <div className="interface-preview">
        <Braces size={22} />
        <span>{kind === "input" ? "tensor input" : "tensor output"}</span>
        <span className="dtype-tag">{p.dtype || "tensor"}</span>
      </div>
    );
  if (kind === "ffn")
    return (
      <div className="ffn-preview">
        <span>gate · SiLU</span>
        <span>×</span>
        <span>up</span>
        <ArrowRight size={14} />
        <span>down</span>
      </div>
    );
  if (kind === "add" || kind === "gate")
    return (
      <div className="residual-preview">
        <svg viewBox="0 0 240 45">
          <path d="M10 10 H210 V30 H230 M10 30 H230" />
          <circle cx="210" cy="30" r="10" />
          <text x="210" y="34">
            +
          </text>
        </svg>
        <span>
          {kind === "gate" ? "learned residual gate" : "residual addition"}
        </span>
      </div>
    );
  return (
    <div className="operation-preview">
      <span>
        {kind === "norm"
          ? "x / RMS(x) × γ"
          : kind === "head"
            ? "[B, T, D] → [B, T, V]"
            : kind === "custom"
              ? "class CustomBlock(nn.Module)"
              : kind === "reference"
                ? "Reference · implementation required"
                : kind === "linear"
                  ? "xWᵀ + b"
                  : kind === "hadamard"
                    ? "H · σ(Hx)"
                    : "tensor operation"}
      </span>
    </div>
  );
}
