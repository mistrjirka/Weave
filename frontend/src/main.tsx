import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createRoot } from "react-dom/client";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Handle,
  Position,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  useReactFlow,
  Panel,
  useViewport,
} from "@xyflow/react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  Box,
  Braces,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  Code2,
  Copy,
  Database,
  Download,
  ExternalLink,
  FileJson,
  FolderOpen,
  GitBranch,
  Layers3,
  LayoutGrid,
  Columns2,
  Tags,
  Maximize2,
  Plus,
  Redo2,
  RotateCcw,
  Save,
  Search,
  Settings2,
  Shapes,
  Sparkles,
  Trash2,
  Undo2,
  Upload,
  Workflow,
  X,
} from "lucide-react";
import "@xyflow/react/dist/style.css";
import "./style.css";
import { newId, copyText, retainedPath, Dialog } from "./ui";
import { visiblePatternField, changePattern } from "./patterns";
import {
  addSharedApplication,
  insertOnEdge,
  buildLayer,
  isInsertionKind,
} from "./graphOperations";
import type { InsertionKind } from "./graphOperations";
import { renderPreview, MaskMatrixEditor } from "./previews";
import { arrangeGraph } from "./layout";
import { layoutSignature } from "./layout-core";
import { measureLabels } from "./labels";
import { GraphNavigator } from "./GraphNavigator";
import { RoutingProvider, edgeTypes } from "./edges";
import {
  parameterGroups,
  presentNodes,
  presentEdges,
} from "./graphPresentation";
import { ReferencePane } from "./ReferencePane";
import { SplitDivider } from "./SplitDivider";
import { readWorkspaceLocation, workspaceURL } from "./workspaceLocation";
import type { ReferenceLocation } from "./workspaceLocation";
import { catalog } from "./catalog";
import {
  allNodes,
  synchronizeDefinitions,
  makeIndependent,
  layoutGraph,
  arrangeExample,
  computationOrder,
  inferRole,
  adapterBody,
} from "./graph";

import {
  api,
  describeError,
  parametersSchema,
  responseData,
  projectSchema,
  projectsSchema,
  savedProjectsSchema,
  graphSchema,
  tokenizerSchema,
  previewSchema,
  unrollSchema,
  codeSchema,
} from "./api";
import type { NodeProps, Edge } from "@xyflow/react";
import type { LucideIcon } from "lucide-react";
import type {
  JsonValue,
  Parameters,
  Project,
  Graph,
  GraphNode,
  NodeData,
  DisplayNode,
  CatalogEntry,
  ValidationReport,
  ValidationIssue,
  TokenPreview,
} from "./types";
const copy = <T,>(v: T): T => structuredClone(v);
const uid = newId;
const motion = (duration: number) =>
  window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : duration;
const format = (n: number) =>
  n >= 1e9
    ? (n / 1e9).toFixed(2) + "B"
    : n >= 1e6
      ? (n / 1e6).toFixed(2) + "M"
      : n >= 1e3
        ? (n / 1e3).toFixed(1) + "K"
        : String(n || 0);
const icons: Record<string, LucideIcon> = {
  input: Braces,
  output: ArrowRight,
  embedding: Database,
  attention: GitBranch,
  cross_attention: GitBranch,
  ffn: Layers3,
  ngram: Workflow,
  lookup: Database,
  loop: RotateCcw,
  block: Box,
  norm: Activity,
  head: LayoutGrid,
  mask: Shapes,
  select: Sparkles,
  custom: Code2,
  reference: CircleAlert,
  attention_pattern: LayoutGrid,
  pattern_union: GitBranch,
  pattern_intersection: GitBranch,
};
const family = (kind: string) =>
  ["embedding", "lookup", "engram_lookup", "memory_fuse"].includes(kind)
    ? "mint"
    : [
          "ngram",
          "hadamard",
          "engram_hash",
          "attention_pattern",
          "pattern_union",
          "pattern_intersection",
        ].includes(kind)
      ? "amber"
      : [
            "attention",
            "cross_attention",
            "ffn",
            "block",
            "monarch",
            "sdpa",
            "rope",
            "split_heads",
            "merge_heads",
            "repeat_heads",
            "causal_conv",
          ].includes(kind)
        ? "violet"
        : [
              "loop",
              "lane_expand",
              "lane_reduce",
              "routing",
              "lane_mix",
              "lane_merge",
            ].includes(kind)
          ? "blue"
          : ["mask", "select"].includes(kind)
            ? "coral"
            : ["reference"].includes(kind)
              ? "red"
              : "neutral";

function download(
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
function graphAt(project: Project, path: string[]) {
  let g = project.graph;
  for (const id of path) {
    const nested = g.nodes.find((n) => n.id === id)?.data.body;
    if (!nested) return project.graph;
    g = nested;
  }
  return g;
}
function replaceGraph(project: Project, path: string[], graph: Graph) {
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

function Diagram({
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
              <React.Fragment key={n.id}>
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
              </React.Fragment>
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
          {kind === "lookup" ? "hash tables" : "lookup table"}
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

function ArchitectureNode({ id, data, selected }: NodeProps<DisplayNode>) {
  const kind = data.kind;
  const Icon = icons[kind] || Shapes;
  const p = data.params || {};
  const expanded = ["loop", "block"].includes(kind);
  const role = data._role || inferRole(data);
  const color = role === "denoising" ? "coral" : family(kind);
  const shape = data._shape;
  return (
    <div
      className={`arch-node ${color} role-${role} ${expanded ? "container-node" : ""} ${selected || data._focused ? "selected" : ""} ${data._error ? "has-error" : ""}`}
      data-visual-role={role}
      aria-label={`${data.label} node`}
    >
      {data._ports.map((port: string, i: number) => (
        <React.Fragment key={port}>
          <Handle
            type="target"
            position={Position.Left}
            id={port}
            title={
              data._optional.includes(port)
                ? `${port}: optional attention pattern / invariant`
                : `Input ${port}`
            }
            className={data._optional.includes(port) ? "optional-port" : ""}
            style={{ top: 68 + i * 28 }}
          />
          <span className="port-name" style={{ top: 60 + i * 28 }}>
            {port}
          </span>
        </React.Fragment>
      ))}
      {kind !== "output" && (
        <Handle
          type="source"
          position={Position.Right}
          id="out"
          style={{ top: 68 }}
        />
      )}
      {data.share && (
        <>
          <Handle
            type="source"
            position={Position.Top}
            id="weights-out"
            isConnectable={false}
            style={{ left: "55%" }}
            className="weight-handle"
          />
          <Handle
            type="target"
            position={Position.Top}
            id="weights-in"
            isConnectable={false}
            style={{ left: "45%" }}
            className="weight-handle"
          />
        </>
      )}
      <div className="node-top">
        <span className="node-icon">
          <Icon size={16} />
        </span>
        <span className="node-kind">
          {kind === "loop"
            ? role === "denoising"
              ? "DENOISING LOOP"
              : "REPEAT / LOOP"
            : kind === "block" && role !== "composition"
              ? role.toUpperCase()
              : kind.replace("_", " ").toUpperCase()}
        </span>
        {expanded && (
          <span className="node-badge">
            {kind === "loop" ? `${p.iterations || 1}×` : "{}"}
          </span>
        )}
        {data._error && <CircleAlert size={15} className="error-icon" />}
      </div>
      <div className="node-title">{data.label}</div>
      {data.share && (
        <div className="weight-badge" title={data.share}>
          ⛓ {data._groupLabel} · same weights{" "}
          {data.iteration && (
            <span>application {data.iteration.index + 1}</span>
          )}
        </div>
      )}
      <Diagram kind={kind} p={p} body={data.body} role={role} />
      <div className="node-footer">
        <code>
          {shape
            ? `[${shape.shape.join(", ")}]`
            : data._error
              ? "shape unresolved"
              : "connect input"}
        </code>
        <span>
          {data._count ? format(data._count) + " params" : shape?.dtype || ""}
        </span>
      </div>
      {expanded && (
        <button
          className="open-block nodrag"
          onClick={(e) => {
            e.stopPropagation();
            data._open(id);
          }}
        >
          Open {kind === "loop" ? "loop body" : "block"}
          <ArrowUpRight size={14} />
        </button>
      )}
    </div>
  );
}
const nodeTypes = { architecture: ArchitectureNode };
function ZoomReadout() {
  const { zoom } = useViewport();
  const flow = useReactFlow();
  return (
    <button
      className="zoom-readout"
      aria-label="Zoom to 100 percent"
      title="Zoom to 100% for readable node details"
      onClick={() => flow.zoomTo(1, { duration: motion(200) })}
    >
      {Math.round(zoom * 100)}%
    </button>
  );
}

function Field<T extends JsonValue>({
  name,
  value,
  onChange,
}: {
  name: string;
  value: T;
  onChange: (value: T) => void;
}) {
  const [error, setError] = useState("");
  const errorId = React.useId();
  const [draft, setDraft] = useState(
    typeof value === "object" ? JSON.stringify(value) : String(value ?? ""),
  );
  useEffect(() => {
    setError("");
    setDraft(
      typeof value === "object" ? JSON.stringify(value) : String(value ?? ""),
    );
  }, [value]);
  const label = name.replaceAll("_", " ");
  if (typeof value === "boolean")
    return (
      <label className="property-row">
        <span>{label}</span>
        <button
          className={"toggle " + (value ? "on" : "")}
          role="switch"
          aria-checked={value}
          aria-label={label}
          onClick={() => onChange(!value as T)}
        >
          <i />
        </button>
      </label>
    );
  const selects: Record<string, string[]> = {
    dtype: ["int64", "float32", "bool", "attention_pattern"],
    output_dtype: ["float32", "int64", "bool", "attention_pattern"],
    strategy: ["confidence", "random"],
    pattern: [
      "local",
      "global",
      "block",
      "random",
      "causal",
      "full",
      "content_topk",
      "custom",
    ],
    backend: ["dense", "auto", "flex"],
  };
  if (selects[name])
    return (
      <label className="property-field">
        <span>{label}</span>
        <select
          name={name}
          value={String(value)}
          onChange={(e) => onChange(e.target.value as T)}
        >
          {selects[name].map((v: string) => (
            <option key={v}>{v}</option>
          ))}
        </select>
      </label>
    );
  const commit = () => {
    if (
      draft ===
      (typeof value === "object" ? JSON.stringify(value) : String(value))
    ) {
      setError("");
      return;
    }
    let next: JsonValue = draft;
    if (
      typeof value === "number" ||
      (typeof value === "string" && value.startsWith("$"))
    ) {
      if (/^\$[a-zA-Z_][a-zA-Z0-9_]*$/.test(draft.trim())) next = draft.trim();
      else if (draft.trim() && Number.isFinite(Number(draft)))
        next = Number(draft);
      else {
        setError("Not applied. Enter a number or a binding such as $width.");
        return;
      }
    } else if (typeof value === "object") {
      try {
        next = JSON.parse(draft);
        if (
          Array.isArray(value)
            ? !Array.isArray(next)
            : !next || typeof next !== "object" || Array.isArray(next)
        )
          throw new Error();
      } catch {
        setError(
          Array.isArray(value)
            ? "Not applied. Enter a valid JSON array, such as [2, 3]."
            : "Not applied. Enter a valid JSON object.",
        );
        return;
      }
    }
    setError("");
    onChange(next as T);
  };
  return (
    <label className="property-field">
      <span>{label}</span>
      {name === "source" ? (
        <textarea
          className="source-editor"
          aria-label="Python source"
          name={name}
          aria-invalid={!!error}
          aria-describedby={error ? errorId : undefined}
          value={draft}
          spellCheck={false}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
        />
      ) : (
        <input
          aria-label={label}
          name={name}
          autoComplete="off"
          spellCheck={false}
          aria-invalid={!!error}
          aria-describedby={error ? errorId : undefined}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
          }}
        />
      )}
      {error && (
        <span id={errorId} className="field-error" role="alert">
          {error}
        </span>
      )}
    </label>
  );
}

type SavedProject = {
  id: string;
  name: string;
  description: string;
  updated?: string;
};
function Bootstrap() {
  const [state, setState] = useState<
    | { status: "loading" }
    | { status: "error"; message: string }
    | {
        status: "ready";
        templates: Project[];
        saved: SavedProject[];
        initial: Project;
        arrange: boolean;
      }
  >({ status: "loading" });
  useEffect(() => {
    let active = true;
    Promise.all([api("/templates"), api("/projects")])
      .then(async ([ts, ps]) => {
        const templates = projectsSchema.parse(ts),
          saved = savedProjectsSchema.parse(ps);
        const location = readWorkspaceLocation(window.location.search);
        let initial = templates[0];
        let arrange = true;
        if (!initial) throw new Error("No initial project is available");
        let recovered: unknown = null;
        try {
          recovered = JSON.parse(
            localStorage.getItem("weave.project") || "null",
          );
        } catch {}
        const restored = projectSchema.safeParse(recovered);
        if (
          restored.success &&
          (!location.model || restored.data.id === location.model)
        ) {
          initial = restored.data;
          arrange = false;
        } else if (location.model) {
          const template = templates.find((p) => p.id === location.model);
          if (template) initial = template;
          else
            initial = projectSchema.parse(
              await api("/projects/" + encodeURIComponent(location.model)),
            );
          arrange = !!template;
        }
        if (active)
          setState({ status: "ready", templates, saved, initial, arrange });
      })
      .catch((e) => {
        if (active) setState({ status: "error", message: String(e) });
      });
    return () => {
      active = false;
    };
  }, []);
  if (state.status !== "ready")
    return (
      <div className="loading">
        <Workflow size={30} />
        <h2>Opening Weave</h2>
        <p>
          {state.status === "error"
            ? state.message
            : "Loading the architecture workbench…"}
        </p>
        {state.status === "error" && (
          <button
            onClick={() => {
              const url = new URL(window.location.href);
              for (const key of ["model", "block", "node"])
                url.searchParams.delete(key);
              window.location.assign(url);
            }}
          >
            Open default workspace
          </button>
        )}
      </div>
    );
  return (
    <Workbench
      initial={state.initial}
      initialTemplates={state.templates}
      initialSaved={state.saved}
      initiallyArrange={state.arrange}
    />
  );
}
function Workbench({
  initial,
  initialTemplates,
  initialSaved,
  initiallyArrange,
}: {
  initial: Project;
  initialTemplates: Project[];
  initialSaved: SavedProject[];
  initiallyArrange: boolean;
}) {
  const [project, setProject] = useState<Project>(initial),
    [templates] = useState(initialTemplates),
    [saved, setSaved] = useState(initialSaved);
  const [initialLocation] = useState(() =>
    readWorkspaceLocation(window.location.search),
  );
  const [path, setPath] = useState<string[]>(() =>
      retainedPath(initial, initialLocation.blocks),
    ),
    [selection, setSelection] = useState<string | null>(initialLocation.node),
    [tab, setTab] = useState(initialLocation.tab),
    [left, setLeft] = useState(initialLocation.library),
    [search, setSearch] = useState(""),
    [report, setReport] = useState<ValidationReport | null>(null),
    [source, setSource] = useState(""),
    [modal, setModal] = useState<string | null>(null),
    [toast, setToast] = useState(""),
    [busy, setBusy] = useState(false),
    [dirty, setDirty] = useState(false);
  const [history, setHistory] = useState<Project[]>([]),
    [future, setFuture] = useState<Project[]>([]),
    [customBlocks, setCustomBlocks] = useState<NodeData[]>([]),
    [checksOpen, setChecksOpen] = useState(false);
  const [mobilePanel, setMobilePanel] = useState<string | null>(
    initialLocation.panel,
  );
  const [splitView, setSplitView] = useState(initialLocation.split);
  const [splitRatio, setSplitRatio] = useState(initialLocation.ratio);
  const [referenceChoice, setReferenceChoice] = useState(
    initialLocation.reference ||
      "example:" +
        (initialTemplates.find((p) => p.id !== initial.id)?.id || initial.id),
  );
  const [referenceLocation, setReferenceLocation] = useState<ReferenceLocation>(
    initialLocation.referenceLocation,
  );
  const splitButton = useRef<HTMLButtonElement>(null);
  const comparisonStyle: React.CSSProperties & {
    "--editable-pane-size": string;
  } = { "--editable-pane-size": `${splitRatio}%` };
  function closeSplitView() {
    setSplitView(false);
    requestAnimationFrame(() => splitButton.current?.focus());
  }
  useEffect(() => {
    const panel =
      mobilePanel === "library" || mobilePanel === "properties"
        ? mobilePanel
        : null;
    const url = workspaceURL(window.location.href, {
      model: project.id,
      blocks: path,
      node: selection,
      tab,
      library: left,
      panel,
      split: splitView,
      reference: referenceChoice,
      referenceLocation,
      ratio: splitRatio,
    });
    if (url !== window.location.href)
      window.history.replaceState(window.history.state, "", url);
  }, [
    project.id,
    path,
    selection,
    tab,
    left,
    mobilePanel,
    splitView,
    referenceChoice,
    referenceLocation,
    splitRatio,
  ]);
  const [arranging, setArranging] = useState(false),
    [showLabels, setShowLabels] = useState(false);
  const arrangeRequest = useRef(0);
  const editRevision = useRef(0);
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const [insertKind, setInsertKind] = useState<InsertionKind>("linear");
  const [pendingLoad, setPendingLoad] = useState<{
    project: Project | { id: string; name: string; description: string };
    isSaved: boolean;
    arrange: boolean;
  } | null>(null);
  const [saving, setSaving] = useState(false);
  const [deletedProject, setDeletedProject] = useState<Project | null>(null);
  const [tokSource, setTokSource] = useState(""),
    [tokRevision, setTokRevision] = useState(""),
    [tokText, setTokText] = useState("The cat sat on the mat."),
    [tokPreview, setTokPreview] = useState<TokenPreview | null>(null),
    [tokError, setTokError] = useState("");
  const viewPath = useRef(path),
    focusedNode = useRef(selection);
  viewPath.current = path;
  focusedNode.current = selection;
  const pendingArrangement = useRef<Set<string>>(
    new Set(
      initiallyArrange
        ? [
            "",
            ...allNodes(initial)
              .filter((v) => v.node.data.body)
              .map((v) => v.path),
          ]
        : [],
    ),
  );
  const reactFlow = useReactFlow();
  const canvasRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null),
    tokenFiles = useRef<HTMLInputElement>(null);
  const ref = useRef<Project>(project);
  ref.current = project;
  const notify = (s: string) => {
    setToast(s);
    setTimeout(() => setToast(""), 4500);
  };
  useEffect(() => {
    if (!dirty) return;
    const guard = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", guard);
    return () => window.removeEventListener("beforeunload", guard);
  }, [dirty]);
  useEffect(() => {
    if (!project || !canvasRef.current || tab !== "graph") return;
    let timer: ReturnType<typeof setTimeout>;
    const observer = new ResizeObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(() => fitCanvas(), 150);
    });
    observer.observe(canvasRef.current);
    return () => {
      observer.disconnect();
      clearTimeout(timer);
    };
  }, [tab, path, splitView]);
  useEffect(() => {
    try {
      const parsed: unknown = JSON.parse(
        localStorage.getItem("weave.blocks") || "[]",
      );
      const result = graphSchema.safeParse({
        nodes: Array.isArray(parsed)
          ? parsed.map((data, i) => ({
              id: String(i),
              type: "architecture",
              position: { x: 0, y: 0 },
              data,
            }))
          : [],
        edges: [],
      });
      if (result.success) setCustomBlocks(result.data.nodes.map((n) => n.data));
    } catch {}
  }, []);
  useEffect(() => {
    if (!project) return;
    try {
      localStorage.setItem("weave.project", JSON.stringify(project));
    } catch {}
    const controller = new AbortController();
    const timer = setTimeout(() => {
      fetch("/api/code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(project),
        signal: controller.signal,
      })
        .then(responseData)
        .then((d) => {
          const parsed = codeSchema.parse(d);
          if (parsed.report) {
            setReport(parsed.report);
            setSource(parsed.source);
          }
        })
        .catch((e) => {
          if (e.name !== "AbortError")
            notify("Validation unavailable: " + e.message);
        });
    }, 220);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [project]);
  useEffect(() => {
    try {
      localStorage.setItem("weave.blocks", JSON.stringify(customBlocks));
    } catch {}
  }, [customBlocks]);
  const graph = project ? graphAt(project, path) : { nodes: [], edges: [] };
  const change = useCallback(
    (fn: (p: Project) => void, record = true, markDirty = true) => {
      const before = ref.current;
      if (!before) return;
      const next = copy(before);
      try {
        fn(next);
        synchronizeDefinitions(before, next);
      } catch (e) {
        notify(describeError(e));
        return;
      }
      if (markDirty) {
        editRevision.current++;
        pendingArrangement.current.delete(viewPath.current.join("/"));
      }
      if (record) {
        setHistory((h) => [...h.slice(-39), copy(before)]);
        setFuture([]);
      }
      ref.current = next;
      setProject(next);
      if (markDirty) setDirty(true);
    },
    [],
  );
  const changeGraph = (
    fn: (g: Graph) => void,
    record = true,
    markDirty = true,
  ) => change((p) => fn(graphAt(p, path)), record, markDirty);
  useEffect(() => {
    const scope = path.join("/");
    if (
      !pendingArrangement.current.has(scope) ||
      !graph.nodes.length ||
      !graph.nodes.every((n) => n.measured?.width && n.measured?.height) ||
      !report
    )
      return;
    pendingArrangement.current.delete(scope);
    void arrangeScope(path, false);
  }, [graph, path, report]);
  function fitCanvas() {
    if (window.innerWidth > 760 && !splitView) {
      reactFlow.fitView({ padding: 0.16, duration: motion(250) });
      return;
    }
    const current = graphAt(ref.current, viewPath.current);
    const first =
      current.nodes.find((n) => n.id === focusedNode.current) ||
      computationOrder(current)[0];
    if (first)
      reactFlow.fitView({
        nodes: [{ id: first.id }],
        padding: splitView
          ? { top: "16px", bottom: "76px", left: "24px", right: "24px" }
          : 0.4,
        minZoom: splitView ? 0.3 : 0.8,
        maxZoom: 0.95,
        duration: motion(200),
      });
  }
  const open = (id: string) => {
    setPath((p) => [...p, id]);
    setSelection(null);
    setSelectedEdge(null);
    setTimeout(() => fitCanvas(), 80);
  };
  const prefix = path.length ? path.join("/") + "/" : "";
  const sharedGroups = useMemo(() => parameterGroups(project), [project]);
  const displayNodes = useMemo(
    () =>
      presentNodes(graph, {
        report,
        prefix,
        groups: sharedGroups,
        selection,
        onOpen: open,
      }),
    [graph, report, prefix, sharedGroups, selection],
  );
  const displayEdges = useMemo<Edge[]>(
    () =>
      presentEdges(graph, { report, prefix, groups: sharedGroups, selection }),
    [graph, report, prefix, sharedGroups, selection],
  );
  const selected = graph.nodes.find((n: GraphNode) => n.id === selection);
  const connection = graph.edges.find((e) => e.id === selectedEdge);
  const connectionShape = connection
    ? report?.shapes[prefix + connection.source]
    : null;
  const updateNode = (fn: (n: GraphNode) => void) =>
    changeGraph((g) => {
      const n = g.nodes.find((n: GraphNode) => n.id === selection);
      if (n) fn(n);
    });
  const undo = () => {
    const prev = history.at(-1);
    if (!prev) return;
    setFuture((f) => [copy(ref.current), ...f]);
    setHistory((h) => h.slice(0, -1));
    setProject(prev);
    ref.current = prev;
    setPath((p) => retainedPath(prev, p));
    setSelection(null);
    setDirty(true);
  };
  const redo = () => {
    if (!future.length) return;
    const next = future[0];
    setHistory((h) => [...h, copy(ref.current)]);
    setFuture((f) => f.slice(1));
    setProject(next);
    ref.current = next;
    setPath((p) => retainedPath(next, p));
    setSelection(null);
    setDirty(true);
  };
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        modal ||
        (e.target instanceof Element &&
          e.target.closest("[data-reference-pane]"))
      )
        return;
      if (e.key === "Escape") {
        setMobilePanel(null);
        return;
      }
      if (
        (e.target as HTMLElement).closest(
          "input,textarea,select,[contenteditable]",
        )
      )
        return;
      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault();
        e.shiftKey ? redo() : undo();
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        saveProject();
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "g") {
        e.preventDefault();
        groupSelection();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [history, future, project, path, modal]);
  function transformerBody(): Graph {
    const body = templates
      .find((t) => t.id === "transformer")
      ?.graph.nodes.find((n) => n.data.kind === "loop")?.data.body;
    if (!body)
      throw new Error("The transformer block definition is unavailable");
    return copy(body);
  }
  function addNode(
    entry: CatalogEntry,
    screenPosition?: { x: number; y: number },
  ) {
    const { kind, label, params: p } = entry;
    let body: Graph | undefined;
    if (["loop", "block"].includes(kind)) {
      body = {
        nodes: [
          {
            id: "state",
            type: "architecture",
            position: { x: 0, y: 100 },
            data: {
              kind: "input",
              label: "State input",
              params: {
                index: 0,
                shape: ["B", "T", "$width"],
                dtype: "float32",
              },
            },
          },
          {
            id: "out",
            type: "architecture",
            position: { x: 350, y: 100 },
            data: {
              kind: "output",
              label: "State output",
              params: { index: 0 },
            },
          },
        ],
        edges: [
          {
            id: "pass",
            source: "state",
            target: "out",
            sourceHandle: "out",
            targetHandle: "a",
            type: "smoothstep",
          },
        ],
      };
    }
    if (entry.role === "transformer") body = transformerBody();
    if (entry.role === "adapter") body = adapterBody();
    const id = uid();
    const position = reactFlow.screenToFlowPosition(
      screenPosition || {
        x: window.innerWidth / 2 - 70,
        y: window.innerHeight / 2,
      },
    );
    changeGraph((g) =>
      g.nodes.push({
        id,
        type: "architecture",
        position,
        data: {
          kind,
          label,
          params: copy(p),
          ...(body ? { body } : {}),
          ...(entry.role ? { role: entry.role } : {}),
        },
      }),
    );
    setSelection(id);
    setMobilePanel(window.innerWidth <= 760 ? "properties" : null);
    notify(label + " added · connect its ports");
  }
  async function arrangeScope(scope: string[], record = true) {
    const request = ++arrangeRequest.current,
      revision = editRevision.current,
      id = ref.current.id,
      snapshot = copy(graphAt(ref.current, scope)),
      signature = layoutSignature(snapshot);
    const key = scope.length ? scope.join("/") + "/" : "";
    const texts = snapshot.edges
      .map((edge) => ({
        id: edge.id,
        text:
          report?.shapes[key + edge.source]?.shape.map(String).join(" × ") ||
          "",
      }))
      .filter((label) => label.text);
    const sizes = measureLabels(texts),
      labels = new Map(
        texts.map((label) => [
          label.id,
          { ...sizes.get(label.id)!, text: label.text },
        ]),
      );
    setArranging(true);
    try {
      const result = await arrangeGraph(snapshot, labels);
      if (ref.current.id !== id) return;
      if (
        editRevision.current !== revision ||
        layoutSignature(graphAt(ref.current, scope)) !== signature
      ) {
        if (record)
          notify("The graph changed while arranging. Try Auto arrange again.");
        return;
      }
      change(
        (project) => {
          const graph = graphAt(project, scope),
            positions = new Map(result.nodes.map((n) => [n.id, n.position]));
          for (const node of graph.nodes) {
            const position = positions.get(node.id);
            if (position) node.position = position;
          }
          graph.layout = result.layout;
        },
        record,
        record,
      );
      const fitRevision = editRevision.current;
      if (viewPath.current.join("/") === scope.join("/"))
        setTimeout(() => {
          if (
            editRevision.current === fitRevision &&
            viewPath.current.join("/") === scope.join("/")
          )
            fitCanvas();
        }, 80);
      if (record)
        notify(
          "Graph arranged with ELK: port-aware routes, crossing reduction, and space for labels.",
        );
    } catch (error) {
      notify("Auto arrange failed: " + describeError(error));
    } finally {
      if (request === arrangeRequest.current) setArranging(false);
    }
  }
  function autoLayout() {
    void arrangeScope(path);
  }
  async function saveProject() {
    if (!ref.current || saving) return false;
    setSaving(true);
    try {
      const before = ref.current;
      const id = ref.current.id?.startsWith("project-")
        ? ref.current.id
        : "project-" + uid();
      const next = { ...ref.current, id };
      await api("/projects/" + id, next, "PUT");
      if (ref.current === before) {
        setProject(next);
        ref.current = next;
        setDirty(false);
      } else {
        const current = { ...ref.current, id };
        setProject(current);
        ref.current = current;
      }
      setSaved(savedProjectsSchema.parse(await api("/projects")));
      notify("Project saved on server");
      return true;
    } catch (e) {
      notify(String(e));
      return false;
    } finally {
      setSaving(false);
    }
  }
  async function loadProject(
    p: Project | { id: string; name: string; description: string },
    isSaved = false,
    force = false,
    arrange = false,
  ) {
    if (dirty && !force) {
      setPendingLoad({ project: p, isSaved, arrange });
      setModal("unsaved");
      return false;
    }
    try {
      const next = isSaved
        ? projectSchema.parse(await api("/projects/" + p.id))
        : projectSchema.parse(p);
      pendingArrangement.current = new Set(arrange ? arrangeExample(next) : []);
      setProject(next);
      ref.current = next;
      setPath([]);
      setSelection(null);
      setSelectedEdge(null);
      setHistory([]);
      setFuture([]);
      setDirty(false);
      setModal(null);
      setPendingLoad(null);
      setMobilePanel(null);
      setReport(null);
      setSource("");
      setTab("graph");
      setTimeout(() => fitCanvas(), 100);
      return true;
    } catch (e) {
      notify(String(e));
      return false;
    }
  }
  async function deleteSavedProject(p: { id: string }) {
    try {
      const doc = projectSchema.parse(await api("/projects/" + p.id));
      await api("/projects/" + p.id, undefined, "DELETE");
      setSaved(savedProjectsSchema.parse(await api("/projects")));
      setDeletedProject(doc);
      notify("Project deleted. You can undo this action.");
    } catch (e) {
      notify(String(e));
    }
  }
  async function restoreDeletedProject() {
    if (!deletedProject) return;
    try {
      await api("/projects/" + deletedProject.id, deletedProject, "PUT");
      setSaved(savedProjectsSchema.parse(await api("/projects")));
      setDeletedProject(null);
      notify("Project restored");
    } catch (e) {
      notify(String(e));
    }
  }
  async function exportCode() {
    setBusy(true);
    try {
      const r = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(project),
      });
      if (!r.ok) throw new Error("Resolve graph errors before exporting.");
      download(project.name.replaceAll(" ", "_") + ".zip", await r.blob());
      notify("PyTorch package downloaded");
    } catch (e) {
      notify(String(e));
      setChecksOpen(true);
    } finally {
      setBusy(false);
    }
  }
  function removeSelected() {
    if (!selection) return;
    changeGraph((g) => {
      g.nodes = g.nodes.filter((n: GraphNode) => n.id !== selection);
      g.edges = g.edges.filter(
        (e) => e.source !== selection && e.target !== selection,
      );
    });
    setSelection(null);
  }
  function duplicate() {
    if (!selected) return;
    changeGraph((g) => {
      const n = copy(selected);
      n.id = uid();
      n.position = { x: n.position.x + 45, y: n.position.y + 65 };
      n.data.label += " copy";
      makeIndependent(n);
      n.selected = false;
      g.nodes.push(n);
    });
  }
  function duplicateShared() {
    if (!selected) return;
    const input = graph.edges.find(
      (e) => e.target === selected.id && (e.targetHandle || "a") === "a",
    );
    changeGraph((g) => {
      const next = addSharedApplication(
        g,
        selected.id,
        input ? report?.shapes[prefix + input.source] : null,
        report?.shapes[prefix + selected.id],
      );
      setSelection(next.id);
    });
    notify("Shared application added. Both instances use the same parameters.");
    setTimeout(
      () => reactFlow.fitView({ padding: 0.12, duration: motion(250) }),
      100,
    );
  }
  async function unrollSelected() {
    if (!selected) return;
    try {
      const result = unrollSchema.parse(
        await api("/graph/unroll", { project, path, node_id: selected.id }),
      );
      change((p) => {
        layoutGraph(result.graph);
        replaceGraph(p, path, result.graph);
      });
      setSelection(result.first_id);
      setSelectedEdge(null);
      notify(
        "Loop expanded into editable applications. Dashed links identify shared weights.",
      );
      setTimeout(
        () => reactFlow.fitView({ padding: 0.12, duration: motion(250) }),
        100,
      );
    } catch (e) {
      notify(String(e));
    }
  }
  function insertOnConnection() {
    if (!connection || connectionShape?.dtype !== "float32") return;
    const width = connectionShape.shape.at(-1);
    if (typeof width !== "number") return;
    const data = buildLayer(insertKind, {
      width,
      heads: project.config.heads,
      transformerBody,
    });
    changeGraph((g) => {
      const next = insertOnEdge(g, connection.id, data);
      setSelection(next.id);
    });
    setSelectedEdge(null);
    notify(
      "Independent learned layer inserted. Shared transformer weights remain tied.",
    );
    setTimeout(
      () => reactFlow.fitView({ padding: 0.12, duration: motion(250) }),
      100,
    );
  }

  function groupSelection() {
    const ids = new Set(
      graph.nodes
        .filter((n: GraphNode) => n.selected)
        .map((n: GraphNode) => n.id),
    );
    if (!ids.size && selection) ids.add(selection);
    if (!ids.size) {
      notify("Select nodes with Shift + click, then create a block");
      return;
    }
    const outgoing = graph.edges.filter(
      (e) => ids.has(e.source) && !ids.has(e.target),
    );
    const outputIds = [...new Set(outgoing.map((e) => e.source))];
    if (outputIds.length > 1) {
      notify(
        "A reusable block needs one output stream. Select nodes with a single outgoing stream.",
      );
      return;
    }
    const incoming = graph.edges.filter(
      (e) => !ids.has(e.source) && ids.has(e.target),
    );
    const selectedNodes = graph.nodes.filter((n: GraphNode) => ids.has(n.id));
    if (
      selectedNodes.some((n: GraphNode) =>
        ["input", "output"].includes(n.data.kind),
      )
    ) {
      notify(
        "Select computation nodes; keep graph Input and Output nodes outside the block.",
      );
      return;
    }
    const dangling = selectedNodes.filter(
      (n: GraphNode) =>
        !graph.edges.some((e) => e.source === n.id && ids.has(e.target)),
    );
    const output = outputIds[0] || dangling[0]?.id;
    if (!output) {
      notify("Select a subgraph with an output");
      return;
    }
    const id = uid();
    const minX = Math.min(...selectedNodes.map((n: GraphNode) => n.position.x)),
      minY = Math.min(...selectedNodes.map((n: GraphNode) => n.position.y));
    const body: Graph = {
      nodes: selectedNodes.map((n: GraphNode) => ({
        ...copy(n),
        selected: false,
        position: { x: n.position.x - minX + 300, y: n.position.y - minY + 80 },
      })),
      edges: copy(
        graph.edges.filter((e) => ids.has(e.source) && ids.has(e.target)),
      ),
    };
    const external = new Map<string, number>();
    const newIn: Edge[] = [];
    incoming.forEach((e) => {
      const key = e.source;
      if (!external.has(key)) {
        const i = external.size;
        external.set(key, i);
        const shape = report?.shapes[prefix + e.source];
        body.nodes.push({
          id: "in" + i,
          type: "architecture",
          position: { x: 0, y: i * 230 },
          data: {
            kind: "input",
            label: "Input " + String.fromCharCode(97 + i),
            params: {
              index: i,
              shape: shape?.shape || ["B", "T", "$width"],
              dtype: shape?.dtype || "float32",
            },
          },
        });
        newIn.push({
          ...e,
          id: uid(),
          target: id,
          targetHandle: String.fromCharCode(97 + i),
        });
      }
      body.edges.push({
        ...e,
        id: uid(),
        source: "in" + external.get(key),
        sourceHandle: "out",
      });
    });
    body.nodes.push({
      id: "result",
      type: "architecture",
      position: {
        x: Math.max(...body.nodes.map((n: GraphNode) => n.position.x)) + 350,
        y: 120,
      },
      data: { kind: "output", label: "Block output", params: { index: 0 } },
    });
    body.edges.push({
      id: uid(),
      source: output,
      target: "result",
      targetHandle: "a",
      sourceHandle: "out",
      type: "smoothstep",
    });
    const block: GraphNode = {
      id,
      type: "architecture",
      position: { x: minX, y: minY },
      data: { kind: "block", label: "Custom block", params: {}, body },
    };
    changeGraph((g) => {
      g.nodes = [...g.nodes.filter((n: GraphNode) => !ids.has(n.id)), block];
      g.edges = [
        ...g.edges.filter((e) => !ids.has(e.source) && !ids.has(e.target)),
        ...newIn,
        ...outgoing.map((e) => ({ ...e, source: id })),
      ];
    });
    setSelection(id);
    notify("Reusable block created · open to edit its internals");
  }
  async function inspectTokenizer() {
    setBusy(true);
    setTokError("");
    try {
      const info = tokenizerSchema.parse(
        await api("/tokenizers/inspect", {
          source: tokSource,
          revision: tokRevision || null,
        }),
      );
      change((p) => {
        p.tokenizer = info;
        p.config.vocab = info.capacity;
        if (info.mask_id != null) p.config.mask_id = info.mask_id;
      });
      notify("Tokenizer loaded · linked vocabulary dimensions updated");
    } catch (e) {
      setTokError(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function uploadTokenizer(files: FileList | null) {
    if (!files?.length) return;
    setBusy(true);
    setTokError("");
    try {
      const form = new FormData();
      Array.from(files).forEach((f) => form.append("files", f));
      const r = await fetch("/api/tokenizers/upload", {
        method: "POST",
        body: form,
      });
      const info = tokenizerSchema.parse(await responseData(r));
      change((p) => {
        p.tokenizer = info;
        p.config.vocab = info.capacity;
        if (info.mask_id != null) p.config.mask_id = info.mask_id;
      });
      notify("Tokenizer bundle inspected");
    } catch (e) {
      setTokError(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function previewTokenizer() {
    if (!project.tokenizer) return;
    try {
      setTokPreview(
        previewSchema.parse(
          await api("/tokenizers/preview", {
            id: project.tokenizer.id,
            text: tokText,
          }),
        ),
      );
      setTokError("");
    } catch (e) {
      setTokError(String(e));
    }
  }
  function focusError(e: ValidationIssue) {
    if (!e.node) return;
    const parts = e.node.split("/");
    const id = parts.pop();
    if (!id) return;
    setPath(parts);
    setSelection(id);
    setTab("graph");
    setTimeout(
      () =>
        reactFlow.fitView({
          nodes: [{ id }],
          padding: 0.8,
          duration: motion(250),
        }),
      100,
    );
  }
  if (!project)
    return (
      <div className="loading">
        <Workflow size={30} />
        <h2>Opening Weave</h2>
        <p>{toast || "Loading the architecture workbench…"}</p>
      </div>
    );
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
  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace">
        Skip to editor
      </a>
      {mobilePanel && (
        <button
          className="drawer-backdrop"
          aria-label="Close side panel"
          onClick={() => setMobilePanel(null)}
        />
      )}
      <aside
        className={"sidebar " + (mobilePanel === "library" ? "is-open" : "")}
        aria-label="Component library"
      >
        <button
          className="drawer-close"
          aria-label="Close library"
          onClick={() => setMobilePanel(null)}
        >
          <X size={20} />
        </button>
        <a className="brand" href="#" onClick={(e) => e.preventDefault()}>
          <span className="brand-icon">
            <Workflow size={22} />
          </span>
          <span>
            weave<span className="brand-dot">.</span>
          </span>
          <span className="alpha">STUDIO</span>
        </a>
        <button
          className="workspace-button"
          onClick={() => setModal("projects")}
        >
          <span className="workspace-avatar">N</span>
          <span>
            Neural workspace<small>Self-hosted · personal</small>
          </span>
          <ChevronDown size={14} />
        </button>
        <div className="sidebar-tabs">
          <button
            className={left === "blocks" ? "active" : ""}
            onClick={() => {
              setLeft("blocks");
              setSearch("");
            }}
          >
            <Shapes size={14} />
            Blocks
          </button>
          <button
            className={left === "models" ? "active" : ""}
            onClick={() => {
              setLeft("models");
              setSearch("");
            }}
          >
            <FolderOpen size={14} />
            Models
          </button>
        </div>
        <div className="library-search">
          <Search size={14} />
          <input
            aria-label="Search blocks"
            name="library-search"
            autoComplete="off"
            placeholder={
              left === "blocks"
                ? "Find a building block…"
                : "Find an architecture…"
            }
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <span>⌕</span>
        </div>
        <div className="library-scroll">
          {search && !groups.length && left === "blocks" && (
            <p className="library-empty" role="status">
              No blocks match “{search}”. Try attention, memory, or loop.
            </p>
          )}
          {left === "blocks" ? (
            <>
              {groups.map((group) => (
                <div className="library-group" key={group}>
                  <h3>{group}</h3>
                  {catalog
                    .filter(
                      (e) =>
                        e.category === group &&
                        (e.label + " " + e.description)
                          .toLowerCase()
                          .includes(search.toLowerCase()),
                    )
                    .map((entry) => {
                      const Icon = icons[entry.kind] || Shapes;
                      return (
                        <button
                          className="library-item"
                          key={entry.label}
                          title={entry.description}
                          onClick={() => addNode(entry)}
                          draggable
                          onDragStart={(e) =>
                            e.dataTransfer.setData(
                              "application/weave",
                              entry.label,
                            )
                          }
                        >
                          <span
                            className={"library-icon " + family(entry.kind)}
                          >
                            <Icon size={15} />
                          </span>
                          <span>
                            {entry.label}
                            <small>{entry.description}</small>
                          </span>
                          <Plus size={12} className="add-hint" />
                        </button>
                      );
                    })}
                </div>
              ))}
              {customBlocks.length > 0 && (
                <div className="library-group">
                  <h3>Your blocks</h3>
                  {customBlocks.map((b, i) => (
                    <button
                      className="library-item"
                      key={i}
                      onClick={() => {
                        changeGraph((g) =>
                          g.nodes.push({
                            id: uid(),
                            type: "architecture",
                            position: reactFlow.screenToFlowPosition({
                              x: window.innerWidth / 2,
                              y: window.innerHeight / 2,
                            }),
                            data: copy(b),
                          }),
                        );
                        notify("Saved block added");
                      }}
                    >
                      <span className="library-icon violet">
                        <Box size={15} />
                      </span>
                      <span>
                        {b.label}
                        <small>Reusable subgraph</small>
                      </span>
                      <Plus size={12} />
                    </button>
                  ))}
                </div>
              )}
            </>
          ) : (
            <>
              <div className="library-group">
                <h3>Reference architectures</h3>
                {templates
                  .filter((t) =>
                    t.name.toLowerCase().includes(search.toLowerCase()),
                  )
                  .map((t) => (
                    <button
                      className={
                        "model-item " + (project.id === t.id ? "active" : "")
                      }
                      key={t.id}
                      onClick={() => loadProject(t, false, false, true)}
                    >
                      <span
                        className={
                          "model-symbol " +
                          (t.id === "combined" ? "mint" : "violet")
                        }
                      >
                        {t.id.startsWith("needle")
                          ? "N" + t.id.slice(-1)
                          : t.id === "thimble"
                            ? "T"
                            : t.id === "diffusion"
                              ? "◇"
                              : t.id === "looped"
                                ? "↻"
                                : "⌘"}
                      </span>
                      <span>
                        {t.name}
                        <small>
                          {t.id.startsWith("needle")
                            ? "Source-linked reference"
                            : t.id === "combined"
                              ? "Composable example"
                              : "Editable architecture"}
                        </small>
                      </span>
                    </button>
                  ))}
              </div>
              {saved.length > 0 && (
                <div className="library-group">
                  <h3>Saved projects</h3>
                  {saved.map((p) => (
                    <button
                      className="model-item"
                      key={p.id}
                      onClick={() => loadProject(p, true)}
                    >
                      <FileJson size={16} />
                      <span>
                        {p.name}
                        <small>{p.updated}</small>
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
        <div className="sidebar-bottom">
          <span className="online-dot" />
          <span>
            Local graph engine<small>PyTorch export ready</small>
          </span>
          <Settings2 size={16} />
        </div>
      </aside>
      <main className="main-workspace" id="workspace" tabIndex={-1}>
        <header className="topbar">
          <div className="project-heading">
            <span className="project-icon">
              <GitBranch size={18} />
            </span>
            <div>
              <button
                className="project-title"
                title={project.name}
                onClick={() => setModal("settings")}
              >
                {project.name}
                <ChevronDown size={14} />
              </button>
              <span className="project-meta">
                NEURAL GRAPH STUDIO <span> / </span>{" "}
                {dirty ? "Unsaved changes" : "Local workspace"}
              </span>
            </div>
          </div>
          <div className="topbar-actions">
            <button
              className="subtle-button"
              aria-label="Import project JSON"
              onClick={() => fileRef.current?.click()}
              title="Import project JSON"
            >
              <Upload size={15} />
              <span>Import</span>
            </button>
            <button
              className="subtle-button"
              aria-label={saving ? "Saving project" : "Save"}
              onClick={saveProject}
              disabled={saving}
            >
              <Save size={15} />
              <span>{saving ? "Saving…" : "Save"}</span>
            </button>
            <button
              className="export-button"
              aria-label="Export PyTorch"
              onClick={exportCode}
              disabled={busy}
            >
              <Download size={15} />
              {busy ? "Working…" : "Export PyTorch"}
              <ChevronDown size={13} />
            </button>
          </div>
        </header>
        <nav className="viewbar">
          <div className="view-tabs">
            <button
              className={tab === "graph" ? "active" : ""}
              aria-label="Architecture"
              aria-current={tab === "graph" ? "page" : undefined}
              onClick={() => setTab("graph")}
            >
              <Workflow size={15} />
              Architecture
            </button>
            <button
              className={tab === "code" ? "active" : ""}
              aria-label="PyTorch code"
              aria-current={tab === "code" ? "page" : undefined}
              onClick={() => setTab("code")}
            >
              <Code2 size={15} />
              PyTorch code
            </button>
            <button
              className={tab === "sources" ? "active" : ""}
              aria-label="References"
              aria-current={tab === "sources" ? "page" : undefined}
              onClick={() => setTab("sources")}
            >
              <ExternalLink size={14} />
              References
            </button>
          </div>
          <div className="view-actions">
            <button
              ref={splitButton}
              className="split-view-button"
              aria-label="Split view"
              aria-pressed={splitView}
              onClick={() => {
                setSplitView((value) => !value);
                if (!splitView) setTab("graph");
              }}
            >
              <Columns2 size={15} aria-hidden="true" />
              <span>Split view</span>
            </button>
            <button
              className="tokenizer-chip"
              onClick={() => setModal("tokenizer")}
            >
              <Database size={13} />
              {project.tokenizer
                ? project.tokenizer.repo || "Local tokenizer"
                : "Select tokenizer"}
              <span>{format(project.config.vocab)} vocab</span>
              <ChevronDown size={12} />
            </button>
          </div>
        </nav>
        <div className={"editor-area " + (splitView ? "has-comparison" : "")}>
          <div
            className={"canvas-comparison " + (splitView ? "is-split" : "")}
            style={comparisonStyle}
          >
            <section className="canvas-section" aria-label="Editable graph">
              {splitView && (
                <div className="editable-pane-heading">
                  <span>Editing</span>
                  <strong>{project.name}</strong>
                </div>
              )}
              <div className="canvas-toolbar">
                <div className="breadcrumbs">
                  <button
                    onClick={() => {
                      setPath([]);
                      setSelection(null);
                      setTimeout(
                        () => reactFlow.fitView({ padding: 0.15 }),
                        80,
                      );
                    }}
                  >
                    <Box size={14} />
                    Model
                  </button>
                  {crumbs.map((c, i) => (
                    <React.Fragment key={i}>
                      <ChevronRight size={12} />
                      <button
                        onClick={() => {
                          setPath(path.slice(0, i + 1));
                          setSelection(null);
                          setTimeout(
                            () => reactFlow.fitView({ padding: 0.15 }),
                            80,
                          );
                        }}
                      >
                        {c}
                      </button>
                    </React.Fragment>
                  ))}
                </div>
                <div className="graph-actions">
                  <button
                    aria-label="Undo (Ctrl+Z)"
                    title="Undo (Ctrl+Z)"
                    disabled={!history.length}
                    onClick={undo}
                  >
                    <Undo2 size={15} />
                  </button>
                  <button
                    aria-label="Redo"
                    title="Redo"
                    disabled={!future.length}
                    onClick={redo}
                  >
                    <Redo2 size={15} />
                  </button>
                  <i />
                  <button
                    aria-label="Create reusable block (Ctrl+G)"
                    title="Create reusable block (Ctrl+G)"
                    onClick={groupSelection}
                  >
                    <Box size={15} />
                  </button>
                  <button
                    aria-label="Auto arrange"
                    title="Auto arrange: reduce crossings and reserve label space"
                    className="auto-arrange-button"
                    disabled={arranging}
                    onClick={autoLayout}
                  >
                    <LayoutGrid size={15} />
                    <span>{arranging ? "Arranging…" : "Auto arrange"}</span>
                  </button>
                  <button
                    aria-label={
                      showLabels ? "Hide edge labels" : "Show edge labels"
                    }
                    aria-pressed={showLabels}
                    title="Show all edge labels"
                    onClick={() => setShowLabels((value) => !value)}
                  >
                    <Tags size={15} />
                  </button>
                  <button
                    aria-label="Fit graph"
                    title="Fit graph"
                    onClick={() =>
                      reactFlow.fitView({
                        padding: 0.15,
                        duration: motion(250),
                      })
                    }
                  >
                    <Maximize2 size={15} />
                  </button>
                  <ZoomReadout />
                </div>
              </div>
              {tab === "graph" ? (
                <div
                  ref={canvasRef}
                  className="flow-wrap"
                  onDragOver={(e) => {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = "copy";
                  }}
                  onDrop={(e) => {
                    e.preventDefault();
                    const kind = e.dataTransfer.getData("application/weave");
                    const entry = catalog.find((c) => c.label === kind);
                    if (entry) addNode(entry, { x: e.clientX, y: e.clientY });
                  }}
                >
                  <RoutingProvider
                    nodes={displayNodes}
                    edges={graph.edges}
                    layout={graph.layout}
                    showLabels={showLabels}
                  >
                    <ReactFlow
                      id="editable-flow"
                      nodes={displayNodes}
                      edges={displayEdges}
                      nodeTypes={nodeTypes}
                      edgeTypes={edgeTypes}
                      onNodesChange={(changes) => {
                        changeGraph(
                          (g) => {
                            g.nodes = applyNodeChanges(changes, g.nodes);
                          },
                          changes.some((c) => c.type === "remove"),
                          changes.some((c) =>
                            ["remove", "add", "position"].includes(c.type),
                          ),
                        );
                      }}
                      onEdgesChange={(changes) =>
                        changeGraph(
                          (g) => {
                            g.edges = applyEdgeChanges(
                              changes.filter(
                                (c) =>
                                  !("id" in c) || !c.id.startsWith("weights:"),
                              ),
                              g.edges,
                            );
                          },
                          changes.some((c) => c.type === "remove"),
                          changes.some((c) =>
                            ["remove", "add", "replace"].includes(c.type),
                          ),
                        )
                      }
                      onConnect={(connection) =>
                        changeGraph((g) => {
                          g.edges = addEdge(
                            { ...connection, id: uid(), type: "smoothstep" },
                            g.edges,
                          );
                        })
                      }
                      onNodeClick={(_, n) => {
                        setSelection(n.id);
                        setSelectedEdge(null);
                        if (window.innerWidth <= 760)
                          setMobilePanel("properties");
                      }}
                      onNodeDoubleClick={(_, n) => {
                        if (n.data.body) open(n.id);
                      }}
                      onEdgeClick={(_, edge) => {
                        if (edge.id.startsWith("weights:")) return;
                        setSelection(null);
                        setSelectedEdge(edge.id);
                        if (window.innerWidth <= 760)
                          setMobilePanel("properties");
                      }}
                      onPaneClick={() => {
                        setSelection(null);
                        setSelectedEdge(null);
                      }}
                      onNodeDragStart={() => {
                        setHistory((h) => [...h.slice(-39), copy(project)]);
                        setFuture([]);
                      }}
                      fitView={window.innerWidth > 760 && !splitView}
                      fitViewOptions={{ padding: 0.15 }}
                      minZoom={0.04}
                      maxZoom={1.7}
                      deleteKeyCode={["Backspace", "Delete"]}
                      selectionKeyCode="Shift"
                      multiSelectionKeyCode="Shift"
                      proOptions={{ hideAttribution: true }}
                    >
                      <Background
                        variant={BackgroundVariant.Dots}
                        gap={22}
                        size={1}
                        color="#ced6cc"
                      />
                      <Controls showInteractive={false} />
                      <MiniMap
                        nodeColor={(n) =>
                          ({
                            mint: "#8ab7a2",
                            violet: "#afa1ce",
                            blue: "#8cabc8",
                            amber: "#ceb47c",
                            coral: "#d49a8a",
                            neutral: "#b2beb3",
                            red: "#b86767",
                          })[family(n.data.kind as string)] || "#b2beb3"
                        }
                        maskColor="rgba(247,249,245,.7)"
                        pannable
                        zoomable
                      />
                      <Panel position="top-left">
                        <div className="canvas-caption">
                          <span className="eyebrow">
                            {path.length
                              ? "INSIDE THE BLOCK"
                              : "MODEL OVERVIEW"}
                          </span>
                          <h1>{path.length ? crumbs.at(-1) : project.name}</h1>
                          <p>
                            {path.length
                              ? "Editable computation · changes flow through to the model"
                              : project.description}
                          </p>
                        </div>
                      </Panel>
                      <Panel position="bottom-center">
                        <GraphNavigator
                          graph={graph}
                          selection={selection}
                          onFocus={setSelection}
                          compact={splitView}
                        />
                        <div className="legend">
                          <span>
                            <i className="mint" />
                            Representation
                          </span>
                          <span>
                            <i className="violet" />
                            Neural layers
                          </span>
                          <span>
                            <i className="amber" />
                            Memory
                          </span>
                          <span>
                            <i className="blue" />
                            Composition
                          </span>
                          <span>
                            <i className="coral" />
                            Sampling
                          </span>
                        </div>
                      </Panel>
                    </ReactFlow>
                  </RoutingProvider>
                </div>
              ) : tab === "code" ? (
                <div className="code-view">
                  <div className="code-heading">
                    <span>
                      <span className="file-dot" />
                      model.py <small>Generated from this graph</small>
                    </span>
                    <button
                      className="subtle-button"
                      onClick={() => {
                        copyText(source)
                          .then(() => notify("Code copied"))
                          .catch(() =>
                            notify("Use Export to download the code"),
                          );
                      }}
                    >
                      <Copy size={14} />
                      Copy code
                    </button>
                  </div>
                  <pre>
                    {source.split("\n").map((line, i) => (
                      <div key={i}>
                        <span className="line-number">{i + 1}</span>
                        <code
                          className={
                            line.trim().startsWith("#") ? "comment" : ""
                          }
                        >
                          {line || " "}
                        </code>
                      </div>
                    ))}
                  </pre>
                  <div className="code-note">
                    Export includes runtime operations, tokenizer files, project
                    JSON, dependencies, and a runnable example.
                  </div>
                </div>
              ) : (
                <div className="references-view">
                  <span className="eyebrow">ARCHITECTURE PROVENANCE</span>
                  <h1>Sources & implementation notes</h1>
                  <p>
                    Reference examples retain their published settings. Read the
                    notes for computational differences and unavailable
                    operations.
                  </p>
                  {project.references?.length ? (
                    project.references.map((r, i: number) => (
                      <article key={i}>
                        <ExternalLink size={20} />
                        <div>
                          <h3>{r.title}</h3>
                          <a href={r.url} target="_blank" rel="noreferrer">
                            View upstream source <ArrowUpRight size={13} />
                          </a>
                          <code>{r.revision}</code>
                          <p>{r.notes}</p>
                          {r.config && (
                            <details>
                              <summary>Published configuration</summary>
                              <pre>{JSON.stringify(r.config, null, 2)}</pre>
                            </details>
                          )}
                        </div>
                      </article>
                    ))
                  ) : (
                    <div className="empty-reference">
                      <GitBranch size={30} />
                      <h3>Built from composable primitives</h3>
                      <p>
                        This is a design example rather than a reproduction of a
                        published checkpoint. All weights are randomly
                        initialized on export.
                      </p>
                    </div>
                  )}
                </div>
              )}
              <div
                className="validation-bar"
                aria-live="polite"
                aria-atomic="true"
              >
                <button
                  onClick={() => setChecksOpen(!checksOpen)}
                  className={report?.valid ? "valid" : "invalid"}
                >
                  {report?.valid ? (
                    <CheckCircle2 size={14} />
                  ) : (
                    <CircleAlert size={14} />
                  )}{" "}
                  {report
                    ? report.valid
                      ? "Shapes compatible"
                      : report.errors.length + " graph issues"
                    : "Checking graph…"}
                  <ChevronDown size={12} />
                </button>
                <span>
                  {graph.nodes.length} nodes <i>·</i> {graph.edges.length}{" "}
                  connections
                </span>
                <span className="parameters-total">
                  {format(report?.parameters || 0)} parameters <i>·</i> float32
                </span>
              </div>
              {checksOpen && (
                <div className="checks-drawer">
                  <div>
                    <strong>Graph validation</strong>
                    <button
                      aria-label="Close validation issues"
                      onClick={() => setChecksOpen(false)}
                    >
                      <X size={15} />
                    </button>
                  </div>
                  {report?.valid && (
                    <p className="valid">
                      <CheckCircle2 size={15} />
                      All connected tensor shapes are compatible.
                    </p>
                  )}
                  {report?.errors.map((e: ValidationIssue, i: number) => (
                    <button
                      className="check-error"
                      key={i}
                      onClick={() => focusError(e)}
                    >
                      <CircleAlert size={15} />
                      <span>
                        <b>{e.node || "Model"}</b>
                        {e.message}
                      </span>
                      <ArrowUpRight size={14} />
                    </button>
                  ))}
                  {report?.warnings.map((w: ValidationIssue, i: number) => (
                    <p key={i}>{w.message}</p>
                  ))}
                </div>
              )}
            </section>
            {splitView && (
              <>
                <SplitDivider value={splitRatio} onChange={setSplitRatio} />
                <ReferencePane
                  templates={templates}
                  saved={saved}
                  choice={referenceChoice}
                  onChoose={(value) => {
                    setReferenceChoice(value);
                    setReferenceLocation({ blocks: [], node: null });
                  }}
                  location={referenceLocation}
                  onNavigate={setReferenceLocation}
                  nodeTypes={nodeTypes}
                  onClose={closeSplitView}
                />
              </>
            )}
          </div>
          <aside
            className={
              "inspector " + (mobilePanel === "properties" ? "is-open" : "")
            }
            aria-label="Property inspector"
          >
            <div className="inspector-heading">
              <span>
                <Settings2 size={15} />
                {selected
                  ? "Block properties"
                  : connection
                    ? "Connection properties"
                    : "Model properties"}
              </span>
              <button
                className="drawer-close"
                aria-label="Close properties"
                onClick={() => setMobilePanel(null)}
              >
                <X size={20} />
              </button>
              {selected && (
                <button
                  aria-label="Close inspector selection"
                  title="Close inspector selection"
                  onClick={() => setSelection(null)}
                >
                  <X size={14} />
                </button>
              )}
            </div>
            <div className="inspector-scroll">
              {selected ? (
                <>
                  <div
                    className={
                      "selected-block-icon " + family(selected.data.kind)
                    }
                  >
                    {React.createElement(icons[selected.data.kind] || Shapes, {
                      size: 24,
                    })}
                  </div>
                  <span className="eyebrow">
                    {selected.data.kind.replaceAll("_", " ")}
                  </span>
                  <input
                    className="inspector-name"
                    aria-label="Block name"
                    value={selected.data.label}
                    onChange={(e) =>
                      updateNode((n) => {
                        n.data.label = e.target.value;
                      })
                    }
                  />
                  <p className="inspector-description">
                    {selected.data.description ||
                      catalog.find((c) => c.kind === selected.data.kind)
                        ?.description ||
                      "Editable composite computation."}
                  </p>
                  {["attention", "cross_attention"].includes(
                    selected.data.kind,
                  ) && (
                    <button
                      className="inspect-open"
                      onClick={async () => {
                        try {
                          const body = graphSchema.parse(
                            await api("/attention/expand", {
                              project,
                              path,
                              node_id: selected.id,
                            }),
                          );
                          updateNode((n) => {
                            n.data.kind = "block";
                            n.data.body = body;
                            n.data.params = {};
                            n.data.description =
                              "Editable Q/K/V projections, head transforms, positions and attention computation.";
                          });
                          open(selected.id);
                        } catch (e) {
                          notify(String(e));
                        }
                      }}
                    >
                      Edit attention internals
                      <ArrowUpRight size={15} />
                    </button>
                  )}
                  {selected.data.body && (
                    <button
                      className="inspect-open"
                      onClick={() => open(selected.id)}
                    >
                      Open internal graph
                      <ArrowUpRight size={15} />
                    </button>
                  )}
                  {selected.data.kind === "loop" && (
                    <button className="inspect-open" onClick={unrollSelected}>
                      Expand into editable applications <GitBranch size={15} />
                    </button>
                  )}
                  {![
                    "input",
                    "output",
                    "attention_pattern",
                    "pattern_union",
                    "pattern_intersection",
                    "reference",
                  ].includes(selected.data.kind) && (
                    <button className="inspect-open" onClick={duplicateShared}>
                      Add application with shared weights <Copy size={15} />
                    </button>
                  )}
                  {selected.data.share && (
                    <p className="inspector-description">
                      {sharedGroups.get(selected.data.share)?.label}:{" "}
                      {sharedGroups.get(selected.data.share)?.members.length}{" "}
                      applications share parameters. Editing internals updates
                      every application.{" "}
                      <button
                        className="save-library"
                        onClick={() => updateNode(makeIndependent)}
                      >
                        Make this application independent
                      </button>
                    </p>
                  )}
                  <button
                    className="save-library"
                    onClick={() =>
                      reactFlow.fitView({
                        nodes: [{ id: selected.id }],
                        padding: 0.6,
                        duration: motion(250),
                      })
                    }
                  >
                    <Maximize2 size={13} /> Zoom to this node
                  </button>
                  <div className="section-label">Configuration</div>
                  {selected.data.body && (
                    <button
                      className="save-library"
                      onClick={() =>
                        updateNode((n) => {
                          n.data.params = {
                            ...n.data.params,
                            heads: "$heads",
                            ffn_width: "$ffn_width",
                          };
                        })
                      }
                    >
                      <Settings2 size={13} />
                      Expose attention & FFN settings
                    </button>
                  )}
                  {selected.data.kind === "attention_pattern" &&
                    selected.data.params.pattern === "custom" &&
                    selected.data.params.matrix && (
                      <MaskMatrixEditor
                        matrix={selected.data.params.matrix}
                        onChange={(matrix) =>
                          updateNode((n) => {
                            n.data.params.matrix = matrix;
                          })
                        }
                      />
                    )}
                  {Object.entries(selected.data.params || {})
                    .filter(
                      ([k]) =>
                        k !== "source" &&
                        (selected.data.kind !== "attention_pattern" ||
                          visiblePatternField(selected.data.params, k)) &&
                        (!selected.data.iteration ||
                          !["iterations", "shared"].includes(k)),
                    )
                    .map(([k, v]) => (
                      <Field
                        key={k}
                        name={k}
                        value={v ?? null}
                        onChange={(next) =>
                          updateNode((n) => {
                            n.data.params = parametersSchema.parse({
                              ...n.data.params,
                              [k]: next,
                            });
                            if (k === "pattern")
                              n.data.params = changePattern(
                                n.data.params,
                                next,
                              );
                          })
                        }
                      />
                    ))}
                  {selected.data.kind === "attention_pattern" && (
                    <p className="field-help">
                      Connect features to a, and optionally key features to b.
                      The output is an attention pattern. Connect it to
                      attention’s optional mask port; combine patterns with
                      union or intersection. Custom matrices require fixed
                      sequence lengths. Content top-k selects from features and
                      computes dense routing scores.
                    </p>
                  )}
                  {["attention", "cross_attention", "sdpa"].includes(
                    selected.data.kind,
                  ) && (
                    <p className="field-help">
                      Dense uses PyTorch SDPA on CPU or GPU. Auto uses compiled
                      FlexAttention on CUDA; flex requires a compatible GPU. A
                      pattern defines permitted positions independently of the
                      backend.
                    </p>
                  )}
                  {selected.data.kind === "head" && (
                    <Field
                      name="tie"
                      value={selected.data.params.tie || ""}
                      onChange={(v) =>
                        updateNode((n) => {
                          if (v) n.data.params.tie = v;
                          else delete n.data.params.tie;
                        })
                      }
                    />
                  )}
                  {selected.data.iteration && (
                    <>
                      <div className="section-label">Application schedule</div>
                      <p className="field-help">
                        Zero-based step and total are passed to sampling
                        operations. Shared applications may use different
                        schedules.
                      </p>
                      {(["index", "total"] as const).map((key) => (
                        <Field
                          key={key}
                          name={"schedule " + key}
                          value={selected.data.iteration![key]}
                          onChange={(value) =>
                            updateNode((n) => {
                              if (n.data.iteration)
                                n.data.iteration[key] = value;
                            })
                          }
                        />
                      ))}
                    </>
                  )}
                  <Field
                    name="weight group"
                    value={selected.data.share || ""}
                    onChange={(value: string) =>
                      updateNode((n) => {
                        if (value.trim()) n.data.share = value.trim();
                        else makeIndependent(n);
                      })
                    }
                  />
                  <p className="field-help">
                    Use the same group to reuse a module definition and its
                    parameters. Matching a group adopts its existing definition.
                  </p>
                  {selected.data.kind === "custom" && (
                    <Field
                      name="source"
                      value={selected.data.params.source || ""}
                      onChange={(v) =>
                        updateNode((n) => {
                          n.data.params.source = v;
                        })
                      }
                    />
                  )}
                  <div className="section-label">Output contract</div>
                  <div className="shape-card">
                    <span>Tensor shape</span>
                    <code>
                      {report?.shapes[prefix + selected.id]
                        ? `[${report.shapes[prefix + selected.id]?.shape.join(", ")}]`
                        : "Unresolved"}
                    </code>
                    <small>
                      {report?.shapes[prefix + selected.id]?.dtype ||
                        "Connect valid inputs to infer shape"}
                    </small>
                  </div>
                  <div className="stat-row">
                    <span>Parameters</span>
                    <b>{format(report?.counts[prefix + selected.id] || 0)}</b>
                  </div>
                  {report?.errors
                    .filter((e) => e.node === prefix + selected.id)
                    .map((e: ValidationIssue, i: number) => (
                      <div className="inline-error" key={i}>
                        <CircleAlert size={15} />
                        {e.message}
                      </div>
                    ))}
                  <div className="node-actions">
                    <button onClick={duplicate}>
                      <Copy size={14} />
                      Duplicate
                    </button>
                    <button className="danger" onClick={removeSelected}>
                      <Trash2 size={14} />
                      Delete
                    </button>
                  </div>
                  {selected.data.body && (
                    <button
                      className="save-library"
                      onClick={() => {
                        setCustomBlocks((bs) => [...bs, copy(selected.data)]);
                        notify("Block saved to your library");
                      }}
                    >
                      <Plus size={14} />
                      Save to block library
                    </button>
                  )}
                </>
              ) : connection ? (
                <div className="connection-editor">
                  <span className="eyebrow">EDIT THE COMPUTATION PATH</span>
                  <h2>Insert between nodes</h2>
                  <p>
                    {
                      graph.nodes.find((n) => n.id === connection.source)?.data
                        .label
                    }{" "}
                    →{" "}
                    {
                      graph.nodes.find((n) => n.id === connection.target)?.data
                        .label
                    }
                  </p>
                  <div className="shape-card">
                    <span>Connection contract</span>
                    <code>
                      {connectionShape
                        ? `[${connectionShape.shape.join(", ")}] · ${connectionShape.dtype}`
                        : "Unresolved"}
                    </code>
                  </div>
                  <p className="inspector-description">
                    An inserted layer has independent parameters. Existing
                    shared applications keep their weight group.
                  </p>
                  {connectionShape?.dtype === "float32" &&
                  typeof connectionShape.shape.at(-1) === "number" ? (
                    <>
                      <label className="property-field">
                        <span>Layer to insert</span>
                        <select
                          aria-label="Layer to insert"
                          value={insertKind}
                          onChange={(e) => {
                            if (isInsertionKind(e.target.value))
                              setInsertKind(e.target.value);
                          }}
                        >
                          <option value="linear">Learned linear layer</option>
                          <option value="adapter">Residual adapter</option>
                          <option value="ffn">Feed-forward network</option>
                          <option value="norm">RMSNorm</option>
                          <option value="transformer">Transformer layer</option>
                        </select>
                      </label>
                      <button
                        className="inspect-open"
                        onClick={insertOnConnection}
                      >
                        Insert independent layer <Plus size={15} />
                      </button>
                    </>
                  ) : (
                    <p>
                      Connect floating point features with a known width to
                      insert a learned layer here. Token IDs need an embedding
                      first.
                    </p>
                  )}
                </div>
              ) : (
                <>
                  <div className="model-summary-icon">
                    <Workflow size={28} />
                  </div>
                  <span className="eyebrow">COMPOSABLE ARCHITECTURE</span>
                  <h2>{project.name}</h2>
                  <p className="inspector-description">
                    Select a node to inspect properties, or a connection to
                    insert a layer. Open a block to edit the computation inside.
                  </p>
                  <div className="section-label">Model configuration</div>
                  {Object.entries(project.config).map(([k, v]) => (
                    <Field
                      key={k}
                      name={k}
                      value={v}
                      onChange={(next) =>
                        change((p) => {
                          p.config[k] = next;
                        })
                      }
                    />
                  ))}
                  <div className="binding-note">
                    <GitBranch size={14} />
                    <span>
                      Values such as <code>$width</code> link block properties
                      to this configuration.
                    </span>
                  </div>
                  <div className="section-label">Tokenizer</div>
                  <button
                    className="tokenizer-card"
                    onClick={() => setModal("tokenizer")}
                  >
                    <Database size={20} />
                    <span>
                      {project.tokenizer?.repo ||
                        project.tokenizer?.class ||
                        "Choose a tokenizer"}
                      <small>
                        {project.tokenizer
                          ? "Vocabulary detected automatically"
                          : "Hugging Face or local files"}
                      </small>
                    </span>
                    <ChevronRight size={15} />
                  </button>
                  <div className="section-label">Architecture at a glance</div>
                  <div className="stat-row">
                    <span>Trainable parameters</span>
                    <b>{format(report?.parameters || 0)}</b>
                  </div>
                  <div className="stat-row">
                    <span>Precision</span>
                    <b>float32</b>
                  </div>
                  <div className="stat-row">
                    <span>Graph validation</span>
                    <b className={report?.valid ? "valid" : "invalid"}>
                      {report?.valid ? "Compatible" : "Needs attention"}
                    </b>
                  </div>
                  <div className="inspector-tip">
                    <span className="tip-icon">
                      <Box size={17} />
                    </span>
                    <h4>Build at any depth</h4>
                    <p>
                      Layers, memory, loops, and sampling all share one canvas.
                      Combine them, then package your own blocks.
                    </p>
                    <button onClick={() => setModal("projects")}>
                      Explore architectures
                      <ArrowRight size={13} />
                    </button>
                  </div>
                </>
              )}
            </div>
          </aside>
        </div>
      </main>
      <nav className="mobile-tools" aria-label="Editor panels">
        <button
          aria-expanded={mobilePanel === "library" && left === "blocks"}
          onClick={() => {
            setLeft("blocks");
            setSearch("");
            setMobilePanel(
              mobilePanel === "library" && left === "blocks" ? null : "library",
            );
          }}
        >
          <Plus size={18} />
          Add blocks
        </button>
        <button
          aria-expanded={mobilePanel === "library" && left === "models"}
          onClick={() => {
            setLeft("models");
            setSearch("");
            setMobilePanel(
              mobilePanel === "library" && left === "models" ? null : "library",
            );
          }}
        >
          <FolderOpen size={18} />
          Models
        </button>
        <button
          aria-expanded={mobilePanel === "properties"}
          onClick={() =>
            setMobilePanel(mobilePanel === "properties" ? null : "properties")
          }
        >
          <Settings2 size={18} />
          Properties
        </button>
      </nav>
      <input
        hidden
        ref={fileRef}
        type="file"
        accept=".json"
        onChange={async (e) => {
          try {
            const f = e.target.files?.[0];
            if (!f) return;
            const p = projectSchema.parse(JSON.parse(await f.text()));
            await loadProject(p);
            notify("Project imported");
          } catch (e) {
            notify(String(e));
          } finally {
            if (fileRef.current) fileRef.current.value = "";
          }
        }}
      />
      {modal && (
        <Dialog
          className={modal === "projects" ? "wide" : ""}
          label={
            modal === "tokenizer"
              ? "Tokenizer configuration"
              : modal === "settings"
                ? "Project settings"
                : modal === "unsaved"
                  ? "Unsaved changes"
                  : "Open architecture"
          }
          onClose={() => {
            setModal(null);
            setPendingLoad(null);
          }}
        >
          <div className="modal-heading">
            <div>
              <span className="eyebrow">
                {modal === "tokenizer"
                  ? "MODEL INPUT"
                  : modal === "projects"
                    ? "YOUR WORKSPACE"
                    : "PROJECT"}
              </span>
              <h2>
                {modal === "tokenizer"
                  ? "Connect a tokenizer"
                  : modal === "projects"
                    ? "Open an architecture"
                    : modal === "unsaved"
                      ? "Keep your changes?"
                      : "Project settings"}
              </h2>
            </div>
            <button onClick={() => setModal(null)} aria-label="Close dialog">
              <X size={20} />
            </button>
          </div>
          {modal === "projects" && deletedProject && (
            <div className="undo-banner" aria-live="polite">
              <span>Deleted {deletedProject.name}</span>
              <button onClick={restoreDeletedProject}>Undo</button>
            </div>
          )}
          {modal === "unsaved" ? (
            <>
              <p className="unsaved-description">
                You have unsaved changes in <strong>{project.name}</strong>.
                Save them before opening {pendingLoad?.project.name}.
              </p>
              <div className="unsaved-actions">
                <button
                  className="primary"
                  disabled={saving}
                  onClick={async () => {
                    const next = pendingLoad;
                    if (next && (await saveProject()))
                      await loadProject(
                        next.project,
                        next.isSaved,
                        true,
                        next.arrange,
                      );
                  }}
                >
                  Save & open
                </button>
                <button
                  onClick={() => {
                    const next = pendingLoad;
                    if (next)
                      loadProject(
                        next.project,
                        next.isSaved,
                        true,
                        next.arrange,
                      );
                  }}
                >
                  Discard & open
                </button>
                <button
                  onClick={() => {
                    setModal(null);
                    setPendingLoad(null);
                  }}
                >
                  Keep editing
                </button>
              </div>
            </>
          ) : modal === "tokenizer" ? (
            <>
              <p>
                Inspect vocabulary, special tokens, and token shapes. Embeddings
                and output heads linked to <code>$vocab</code> update
                automatically.
              </p>
              <label className="property-field">
                <span>Hugging Face repository or server-local path</span>
                <input
                  placeholder="e.g. gpt2 or /path/to/tokenizer"
                  value={tokSource}
                  onChange={(e) => setTokSource(e.target.value)}
                />
              </label>
              <label className="property-field">
                <span>
                  Revision <small>optional commit or tag</small>
                </span>
                <input
                  placeholder="main"
                  value={tokRevision}
                  onChange={(e) => setTokRevision(e.target.value)}
                />
              </label>
              <div className="modal-actions">
                <button
                  className="primary"
                  disabled={busy || !tokSource.trim()}
                  onClick={inspectTokenizer}
                >
                  {busy ? "Inspecting…" : "Load & inspect"}
                  <ArrowRight size={14} />
                </button>
                <button onClick={() => tokenFiles.current?.click()}>
                  <Upload size={14} />
                  Upload bundle
                </button>
              </div>
              <input
                ref={tokenFiles}
                hidden
                multiple
                type="file"
                accept=".json,.model,.txt,.vocab,.merges"
                onChange={(e) => uploadTokenizer(e.target.files)}
              />
              {tokError && <div className="inline-error">{tokError}</div>}
              {project.tokenizer && (
                <div className="tokenizer-result">
                  <div className="result-heading">
                    <CheckCircle2 size={17} />
                    <b>{project.tokenizer.repo || project.tokenizer.class}</b>
                    <span>Loaded</span>
                  </div>
                  <div className="token-stats">
                    <div>
                      <span>Vocabulary</span>
                      <b>{project.tokenizer.size.toLocaleString()}</b>
                    </div>
                    <div>
                      <span>Embedding capacity</span>
                      <b>{project.tokenizer.capacity.toLocaleString()}</b>
                    </div>
                    <div>
                      <span>Base vocabulary</span>
                      <b>{project.tokenizer.base_size.toLocaleString()}</b>
                    </div>
                  </div>
                  <div className="special-tokens">
                    {Object.entries(project.tokenizer.special_tokens).map(
                      ([k, v]) => (
                        <span key={k}>
                          {k.replace("_token", "")}{" "}
                          <code>
                            {v.token} · {v.id}
                          </code>
                        </span>
                      ),
                    )}
                  </div>
                  <label className="property-field">
                    <span>Tokenization preview</span>
                    <textarea
                      value={tokText}
                      onChange={(e) => setTokText(e.target.value)}
                    />
                  </label>
                  <button className="small-button" onClick={previewTokenizer}>
                    Tokenize sample
                    <ArrowRight size={13} />
                  </button>
                  {tokPreview && (
                    <>
                      <div className="token-pills">
                        {tokPreview.tokens.map((t: string, i: number) => (
                          <span key={i}>
                            {t}
                            <small>{tokPreview.ids[i]}</small>
                          </span>
                        ))}
                      </div>
                      <p className="preview-shape">
                        Token IDs: [{tokPreview.shape.join(", ")}] · int64
                      </p>
                    </>
                  )}
                </div>
              )}
            </>
          ) : modal === "projects" ? (
            <>
              <p>
                Every example is an editable graph. Mix components across
                architectures.
              </p>
              <div className="project-grid">
                {templates.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => loadProject(t, false, false, true)}
                  >
                    <span
                      className={
                        "project-template-icon " +
                        (t.id === "combined" ? "mint" : "violet")
                      }
                    >
                      {t.id.startsWith("needle")
                        ? "N" + t.id.at(-1)
                        : t.id === "looped"
                          ? "↻"
                          : t.id === "diffusion"
                            ? "◇"
                            : "⌘"}
                    </span>
                    <h3>{t.name}</h3>
                    <p>{t.description}</p>
                    <span>
                      Open graph <ArrowUpRight size={13} />
                    </span>
                  </button>
                ))}
              </div>
              {saved.length > 0 && (
                <>
                  <div className="section-label">Saved on this server</div>
                  {saved.map((p) => (
                    <div className="saved-project" key={p.id}>
                      <button onClick={() => loadProject(p, true)}>
                        <FileJson size={15} />
                        {p.name}
                      </button>
                      <button
                        title="Delete saved project"
                        aria-label={"Delete " + p.name}
                        onClick={() => deleteSavedProject(p)}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  ))}
                </>
              )}
            </>
          ) : (
            <>
              <Field
                name="Project name"
                value={project.name}
                onChange={(v) =>
                  change((p) => {
                    p.name = v;
                  })
                }
              />
              <Field
                name="Description"
                value={project.description}
                onChange={(v) =>
                  change((p) => {
                    p.description = v;
                  })
                }
              />
              <button
                className="primary"
                onClick={() =>
                  download(
                    project.name + ".json",
                    JSON.stringify(project, null, 2),
                  )
                }
              >
                <Download size={15} />
                Download project JSON
              </button>
              <p>
                Project files include the complete graph, reusable block
                definitions, tokenizer metadata, and source references.
              </p>
            </>
          )}
        </Dialog>
      )}
      {toast && (
        <div className="toast" role="status">
          <CheckCircle2 size={16} />
          {toast}
          {deletedProject && !modal && toast.startsWith("Project deleted") && (
            <button className="toast-undo" onClick={restoreDeletedProject}>
              Undo
            </button>
          )}
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <ReactFlowProvider>
    <Bootstrap />
  </ReactFlowProvider>,
);
