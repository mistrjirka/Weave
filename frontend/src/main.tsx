import {
  inputSettingOwnership,
  containerParameterShadowReason,
} from "./settingOwnership";
import {
  tokenizerOwnsVocabulary,
  vocabularyField,
  automaticVocabulary,
} from "./tokenizerVocabulary";
import {
  outputPorts,
  portLabel,
  edgeContract,
  multiOutputLanes,
  portTop,
} from "./lanePorts";
import { TokenizerPanel, tokenizerLabel } from "./TokenizerPanel";
import {
  connectionFeedback,
  type ConnectionAttempt,
} from "./connectionFeedback";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { GPUTuning } from "./GPUTuning";
import { TrainingWorkspace } from "./TrainingWorkspace";
import { UIUpdateNotice } from "./UIUpdateNotice";
import { ActiveTrainingRuns } from "./ActiveTrainingRuns";
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
  useUpdateNodeInternals,
  type FitViewOptions,
  Panel,
  useViewport,
  SelectionMode,
} from "@xyflow/react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  ArrowUp,
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
  FilePlus2,
  FolderOpen,
  GitBranch,
  Layers3,
  MessageSquareText,
  Menu,
  LayoutGrid,
  Columns2,
  Tags,
  Maximize2,
  Plus,
  Pencil,
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
import "./theme.css";
import { ThemePicker } from "./ThemePicker";
import { settingHelp, generalFieldDisabledReason } from "./settingHelp";
import { compilationInput } from "./compilationInput";
import {
  graphClipboardType,
  serializeSelection,
  parseSelection,
  pasteSelection,
} from "./graphClipboard";
import {
  attentionVariants,
  attentionVariant,
  attentionFieldDisabledReason,
  isAttentionVariant,
  transformerSettings,
  transformerSettingKeys,
  exposeTransformerSettings,
  wrapLoopTransformer,
  wrappedTransformer,
  inheritTransformerDefaults,
  transformerParentScope,
  transformerResidual,
  setTransformerResidual,
  setTransformerSetting,
  setAttentionVariant,
} from "./transformerControls";
import { ConnectionPicker } from "./ConnectionPicker";
import type { ConnectionBlockChoice } from "./ConnectionPicker";
import { WeightGroupPicker } from "./WeightGroupPicker";
import { supportsWeightSharing, weightGroupChoices } from "./weightGroups";
import { canvasFitPadding, canvasFitInsets } from "./canvasFitPadding";
import { GraphVariableCard } from "./GraphVariableCard";
import { GraphAnnotations } from "./GraphAnnotations";
import { GraphVariables } from "./GraphVariables";
import {
  graphVariableScope,
  graphVariableUses,
  bindingValue,
  previewParameters,
} from "./graphVariables";
import { localizeLayerDefaults } from "./parameterScopes";
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
  clearGraphSelection,
  nodePorts,
} from "./graphPresentation";
import { ReferencePane } from "./ReferencePane";
import { ReferenceInspector, isGraphEditShortcut } from "./ReferenceInspector";
import type { ReferenceInspection } from "./ReferenceInspector";
import { SplitDivider } from "./SplitDivider";
import { DesignAdvice } from "./DesignAdvice";
import {
  ProjectBrowser,
  NewProjectForm,
  ReferenceCard,
} from "./ProjectBrowser";
import { createProject } from "./projectWorkflow";
import type {
  OpenProjectOptions,
  ProjectTab,
  SavedProject,
} from "./projectWorkflow";
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
  unrollSchema,
  codeSchema,
  recoveryStatusSchema,
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
  TokenizerInfo,
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
              "lane_fork",
              "lane_split",
              "lane_join",
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

function ArchitectureNode({ id, data, selected }: NodeProps<DisplayNode>) {
  const kind = data.kind;
  const Icon = icons[kind] || Shapes;
  const p = data._previewParams ?? data.params;
  const expanded = ["loop", "block", "moe"].includes(kind);
  const role = data._role || inferRole(data);
  const color = role === "denoising" ? "coral" : family(kind);
  const shape = data._shape;
  const updateInternals = useUpdateNodeInternals();
  const portSignature = JSON.stringify([data._ports, data._outputs]);
  useEffect(() => {
    updateInternals(id);
  }, [id, portSignature, updateInternals]);
  const laneNode = multiOutputLanes.has(kind) || kind === "lane_join";
  return (
    <div
      className={`arch-node ${color} role-${role} ${expanded ? "container-node" : ""} ${selected || data._focused ? "selected" : ""} ${data._error ? "has-error" : ""} ${laneNode ? "explicit-lane-node" : ""}`}
      style={
        kind === "moe_combine" || laneNode
          ? {
              minHeight:
                105 + Math.max(data._ports.length, data._outputs.length) * 28,
            }
          : undefined
      }
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
              data._portErrors?.[port]
                ? `Input ${port}: ${data._portErrors[port]}`
                : data._optional.includes(port)
                  ? `${port}: optional attention pattern / invariant`
                  : `Input ${port}`
            }
            className={`${data._optional.includes(port) ? "optional-port" : ""} ${data._portErrors?.[port] ? "port-error" : ""}`}
            style={{ top: portTop(kind) + i * 28 }}
          />
          <span
            className={`port-name ${data._portErrors?.[port] ? "port-error-label" : ""}`}
            style={{ top: portTop(kind) - 8 + i * 28 }}
          >
            {portLabel(port)}
          </span>
        </React.Fragment>
      ))}
      {data._outputs.map((port, i) => (
        <React.Fragment key={port}>
          <Handle
            type="source"
            position={Position.Right}
            id={port}
            title={`Output ${portLabel(port)}`}
            style={{ top: portTop(kind) + i * 28 }}
          />
          {port !== "out" && (
            <span
              className="port-name output-port-name"
              style={{ top: portTop(kind) - 8 + i * 28 }}
            >
              {portLabel(port)}
            </span>
          )}
        </React.Fragment>
      ))}
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
      <div className="node-title">
        {data.label}{" "}
        {data.annotation && (
          <span title="Has a design note" aria-label="Has a design note">
            ✎
          </span>
        )}
      </div>
      {data.share && (
        <div className="weight-badge" title={data.share}>
          ⛓ {data._groupLabel} · same weights{" "}
          {data.iteration && (
            <span>application {data.iteration.index + 1}</span>
          )}
        </div>
      )}
      <div
        className="node-body"
        style={
          laneNode
            ? {
                minHeight:
                  50 + Math.max(data._ports.length, data._outputs.length) * 28,
              }
            : undefined
        }
      >
        <Diagram kind={kind} p={p} body={data.body} role={role} />
        {kind === "input" && data._tokenizer && (
          <div className="input-tokenizer-summary">
            <Database size={12} />
            <span>{tokenizerLabel(data._tokenizer)}</span>
            <small>{format(data._tokenizer.capacity)} IDs</small>
          </div>
        )}
      </div>
      <div className="node-footer">
        <code>
          {shape
            ? `${data._outputs.length > 1 ? data._outputs.length + " × " : ""}[${shape.shape.join(", ")}]`
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
          Open{" "}
          {kind === "loop" ? "loop body" : kind === "moe" ? "experts" : "block"}
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
      aria-label={`Zoom to 100 percent · current ${Math.round(zoom * 100)}%`}
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
  disabledReason,
  help,
  options,
  variables,
  displayName,
}: {
  name: string;
  options?: string[];
  variables?: Parameters;
  displayName?: string;
  value: T;
  onChange: (value: T) => void;
  disabledReason?: string;
  help?: string;
}) {
  const [helpOpen, setHelpOpen] = useState(false);
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
  const label = displayName ?? name.replaceAll("_", " ");
  const helpId = `${errorId}-explanation`;
  const heading = (
    <span className="setting-heading">
      <span>{label}</span>
      {help && (
        <button
          type="button"
          className="setting-help-button"
          aria-label={`Explain ${label}`}
          aria-expanded={helpOpen}
          aria-controls={helpId}
          onClick={() => setHelpOpen((open) => !open)}
        >
          ?
        </button>
      )}
    </span>
  );
  const explanation =
    help && helpOpen ? (
      <p id={helpId} className="setting-explanation">
        {help}
      </p>
    ) : null;
  const inheritedValue =
    typeof value === "string" && value.startsWith("$")
      ? bindingValue(value, variables ?? {})
      : undefined;
  if (inheritedValue !== undefined && typeof inheritedValue !== "number")
    return (
      <div className="property-field">
        {heading}
        <div className="stat-row">
          <code>{String(value)}</code>
          <b>
            {typeof inheritedValue === "object"
              ? JSON.stringify(inheritedValue)
              : String(inheritedValue)}
          </b>
        </div>
        <small className="field-help">
          Linked to the enclosing parameter. Change it there, or set an
          independent value here.
        </small>
        <button
          className="small-button"
          disabled={!!disabledReason}
          onClick={() => onChange(inheritedValue as T)}
        >
          Set independently
        </button>
        {disabledReason && (
          <small className="field-help">{disabledReason}</small>
        )}
        {explanation}
      </div>
    );
  if (typeof value === "boolean")
    return (
      <div className="setting-boolean">
        <div className="property-row">
          {heading}
          <button
            className={"toggle " + (value ? "on" : "")}
            role="switch"
            aria-checked={value}
            aria-label={label}
            aria-describedby={
              disabledReason ? `${errorId}-help` : helpOpen ? helpId : undefined
            }
            disabled={!!disabledReason}
            onClick={() => onChange(!value as T)}
          >
            <i />
          </button>
        </div>
        {disabledReason && (
          <small id={`${errorId}-help`} className="field-help">
            {disabledReason}
          </small>
        )}
        {explanation}
      </div>
    );
  const selects: Record<string, string[]> = {
    variant: ["two_stage", "three_stage"],
    rope_scaling: ["none", "yarn"],
    approximate: ["none", "tanh"],
    mode: ["max", "average"],
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
  const choices = options ?? selects[name];
  if (choices)
    return (
      <div className="property-field">
        {heading}
        <select
          aria-label={label}
          name={name}
          disabled={!!disabledReason}
          aria-describedby={
            disabledReason ? `${errorId}-help` : helpOpen ? helpId : undefined
          }
          value={String(value)}
          onChange={(e) => onChange(e.target.value as T)}
        >
          {choices.map((v: string) => (
            <option key={v}>{v}</option>
          ))}
        </select>
        {disabledReason && (
          <small id={`${errorId}-help`} className="field-help">
            {disabledReason}
          </small>
        )}
        {explanation}
      </div>
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
    <div className="property-field">
      {heading}
      {name === "source" ? (
        <textarea
          className="source-editor"
          aria-label="Python source"
          disabled={!!disabledReason}
          name={name}
          aria-invalid={!!error}
          aria-describedby={
            error
              ? errorId
              : disabledReason
                ? `${errorId}-help`
                : helpOpen
                  ? helpId
                  : undefined
          }
          value={draft}
          spellCheck={false}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
        />
      ) : (
        <input
          aria-label={label}
          disabled={!!disabledReason}
          name={name}
          autoComplete="off"
          spellCheck={false}
          aria-invalid={!!error}
          aria-describedby={
            error
              ? errorId
              : disabledReason
                ? `${errorId}-help`
                : helpOpen
                  ? helpId
                  : undefined
          }
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
          }}
        />
      )}
      {disabledReason && (
        <small id={`${errorId}-help`} className="field-help">
          {disabledReason}
        </small>
      )}
      {explanation}
      {error && (
        <span id={errorId} className="field-error" role="alert">
          {error}
        </span>
      )}
    </div>
  );
}

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
        const templates = projectsSchema.parse(ts).map(localizeLayerDefaults),
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
          setState({
            status: "ready",
            templates,
            saved,
            initial: localizeLayerDefaults(initial),
            arrange,
          });
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
  const [connectionAttempt, setConnectionAttempt] =
    useState<ConnectionAttempt | null>(null);
  const [validatedInput, setValidatedInput] = useState("");
  const [initialLocation] = useState(() =>
    readWorkspaceLocation(window.location.search),
  );
  const [path, setPath] = useState<string[]>(() =>
      retainedPath(initial, initialLocation.blocks),
    ),
    [selection, setSelection] = useState<string | null>(
      initialLocation.split && initialLocation.referenceLocation.node
        ? null
        : initialLocation.node,
    ),
    [tab, setTab] = useState(initialLocation.tab),
    [left, setLeft] = useState(initialLocation.library),
    [search, setSearch] = useState(""),
    [report, setReport] = useState<ValidationReport | null>(null),
    [source, setSource] = useState(""),
    [modal, setModal] = useState<string | null>(
      initialLocation.workspace === "new"
        ? "new-project"
        : initialLocation.workspace
          ? "projects"
          : null,
    ),
    [toast, setToast] = useState(""),
    [busy, setBusy] = useState(false),
    [dirty, setDirty] = useState(() => {
      try {
        const status = recoveryStatusSchema.safeParse(
          JSON.parse(localStorage.getItem("weave.project.status") || "null"),
        );
        if (status.success && status.data.id === initial.id)
          return status.data.dirty;
      } catch {}
      return (
        initial.id.startsWith("project-") &&
        !initialSaved.some((p) => p.id === initial.id)
      );
    });
  const librarySearchRef = useRef<HTMLInputElement>(null);
  const [projectTab, setProjectTab] = useState<ProjectTab>(
    initialLocation.workspace === "examples" ? "examples" : "projects",
  );
  const [starterId, setStarterId] = useState(
    initialTemplates.some((p) => p.id === initialLocation.starter)
      ? initialLocation.starter || "blank"
      : "blank",
  );
  const [history, setHistory] = useState<Project[]>([]),
    [future, setFuture] = useState<Project[]>([]),
    [customBlocks, setCustomBlocks] = useState<NodeData[]>([]),
    [checksOpen, setChecksOpen] = useState(false);
  const [mobilePanel, setMobilePanel] = useState<string | null>(
    initialLocation.panel,
  );
  const sidebarMenuButton = useRef<HTMLButtonElement>(null);
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
  const [referenceInspection, setReferenceInspection] =
    useState<ReferenceInspection | null>(null);
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
      workspace:
        modal === "new-project"
          ? "new"
          : modal === "projects"
            ? projectTab
            : null,
      starter: modal === "new-project" ? starterId : null,
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
    modal,
    projectTab,
    starterId,
  ]);
  const [arranging, setArranging] = useState(false),
    [showLabels, setShowLabels] = useState(false),
    [showAnnotations, setShowAnnotations] = useState(true);
  const arrangeRequest = useRef(0);
  const editRevision = useRef(0);
  const [connectionPicker, setConnectionPicker] = useState<{
    projectId: string;
    scope: string;
    source: string;
    sourceHandle: string;
    position: { x: number; y: number };
    bounds: { left: number; top: number; right: number; bottom: number };
  } | null>(null);
  useEffect(() => setConnectionPicker(null), [project.id, path, tab, modal]);
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  useEffect(() => {
    if (selection || selectedEdge)
      setReferenceLocation((previous) =>
        previous.node ? { ...previous, node: null } : previous,
      );
  }, [selection, selectedEdge]);
  const [insertKind, setInsertKind] = useState<InsertionKind>("linear");
  const [pendingLoad, setPendingLoad] = useState<{
    project: Project | SavedProject;
    options: OpenProjectOptions;
  } | null>(null);
  const [saving, setSaving] = useState(false);
  const [deletedProject, setDeletedProject] = useState<Project | null>(null);
  const [tokenizerTarget, setTokenizerTarget] = useState<string | null>(null);
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
  const pasteSequence = useRef({ key: "", count: 0 });
  const fileRef = useRef<HTMLInputElement>(null);
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
    try {
      localStorage.setItem(
        "weave.project.status",
        JSON.stringify({ id: project.id, dirty }),
      );
    } catch {}
  }, [project.id, dirty]);
  useEffect(() => {
    if (!project) return;
    try {
      localStorage.setItem("weave.project", JSON.stringify(project));
    } catch {}
  }, [project]);
  const compilationBody = useMemo(() => compilationInput(project), [project]);
  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(() => {
      fetch("/api/code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: compilationBody,
        signal: controller.signal,
      })
        .then(responseData)
        .then((d) => {
          const parsed = codeSchema.parse(d);
          if (parsed.report) {
            setReport(parsed.report);
            setValidatedInput(compilationBody);
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
  }, [compilationBody]);
  const connectionError =
    connectionAttempt && report && validatedInput === compilationBody
      ? connectionFeedback(project, report, connectionAttempt)
      : null;
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
  const navigateReference = useCallback<
    React.Dispatch<React.SetStateAction<ReferenceLocation>>
  >(
    (next) => {
      setSelection(null);
      setSelectedEdge(null);
      change((project) => clearGraphSelection(project.graph), false, false);
      setReferenceLocation(next);
    },
    [change],
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
  // Keep the always-visible variables card clear of fitted nodes and ports.
  function fitGraph(options: Omit<FitViewOptions, "padding"> = {}) {
    return reactFlow.fitView({
      ...options,
      padding: canvasFitPadding(canvasRef.current),
    });
  }
  function fitCanvas() {
    if (
      window.innerWidth > 760 &&
      (!splitView || (canvasRef.current?.clientWidth ?? 0) >= 720)
    ) {
      fitGraph({ duration: motion(250) });
      return;
    }
    const current = graphAt(ref.current, viewPath.current);
    const first =
      current.nodes.find((n) => n.id === focusedNode.current) ||
      computationOrder(current)[0];
    if (first)
      fitGraph({
        nodes: [{ id: first.id }],
        minZoom: 0.3,
        maxZoom: 0.95,
        duration: motion(200),
      });
  }
  const goUp = () => {
    const container = path.at(-1);
    if (!container) return;
    setPath(path.slice(0, -1));
    setSelection(container);
    setSelectedEdge(null);
    setMobilePanel(null);
    setTimeout(() => fitCanvas(), 80);
  };
  const open = (id: string) => {
    setMobilePanel(null);
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
        variables: graphVariableScope(project, path).bindings,
        selection,
        onOpen: open,
      }),
    [graph, report, prefix, sharedGroups, selection, project, path],
  );
  const displayEdges = useMemo<Edge[]>(
    () =>
      presentEdges(graph, { report, prefix, groups: sharedGroups, selection }),
    [graph, report, prefix, sharedGroups, selection],
  );
  const selected = graph.nodes.find((n: GraphNode) => n.id === selection);
  const tokenizerInput =
    modal === "tokenizer"
      ? allNodes(project).find((item) => item.path === tokenizerTarget)?.node
      : undefined;
  const inputTokenizer = tokenizerInput?.data.tokenizer;
  const variableScope = graphVariableScope(project, path);
  const variableUses = useMemo(() => graphVariableUses(graph), [graph]);
  const hiddenVariables = tokenizerOwnsVocabulary(
    project,
    path,
    report?.tokenizers,
  )
    ? ["vocab"]
    : [];
  const updateGraphVariables = (values: Record<string, number>) =>
    change((p) => {
      if (path.length) graphAt(p, path).variables = values;
      else {
        p.config = values;
        delete p.graph.variables;
      }
    });
  const selectedTokenizer = selected
    ? (report?.tokenizers?.[prefix + selected.id] ?? selected.data.tokenizer)
    : undefined;
  const selectedVocabField = selected
    ? vocabularyField(selected.data)
    : undefined;
  const selectedAutoVocab = selected
    ? automaticVocabulary(selected.data, selectedTokenizer)
    : false;
  const fieldVariables = { ...variableScope.bindings };
  if (selectedTokenizer) {
    fieldVariables.vocab = selectedTokenizer.capacity;
    if (selectedTokenizer.mask_id != null)
      fieldVariables.mask_id = selectedTokenizer.mask_id;
  }
  const transformer = selected ? transformerSettings(selected.data) : null;
  const selectedFieldVariables = {
    ...fieldVariables,
    ...Object.fromEntries(
      Object.entries(transformer ?? {}).filter(
        ([, value]) => typeof value !== "string" || !value.startsWith("$"),
      ),
    ),
  };
  const selectedCount = graph.nodes.filter((node) => node.selected).length;
  const connection = graph.edges.find((e) => e.id === selectedEdge);
  const referenceDetails =
    splitView &&
    !selected &&
    !connection &&
    referenceInspection?.id === referenceLocation.node &&
    referenceInspection?.scope ===
      (referenceLocation.blocks.length
        ? referenceLocation.blocks.join("/") + "/"
        : "")
      ? referenceInspection
      : null;
  const connectionShape = connection
    ? edgeContract(report, prefix, connection)
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
    // Clipboard events work on LAN HTTP too, without Clipboard API permissions.
    const ownsClipboard = (event: ClipboardEvent) => {
      const target = event.target;
      return (
        !modal &&
        tab === "graph" &&
        target instanceof Element &&
        !target.closest(
          "input,textarea,select,[contenteditable],[data-reference-pane]",
        ) &&
        (canvasRef.current?.contains(target) || target === document.body) &&
        (window.getSelection()?.isCollapsed ?? true)
      );
    };
    const onCopy = (event: ClipboardEvent) => {
      if (!ownsClipboard(event) || !event.clipboardData) return;
      const text = serializeSelection(
        graphAt(ref.current, viewPath.current),
        focusedNode.current,
      );
      if (!text) return;
      event.clipboardData.setData(graphClipboardType, text);
      event.clipboardData.setData("text/plain", text);
      event.preventDefault();
      notify("Selected nodes copied");
    };
    const onPaste = (event: ClipboardEvent) => {
      if (!ownsClipboard(event) || !event.clipboardData) return;
      const text =
        event.clipboardData.getData(graphClipboardType) ||
        event.clipboardData.getData("text/plain");
      const source = parseSelection(text);
      const bounds = canvasRef.current?.getBoundingClientRect();
      if (!source || !bounds) return;
      event.preventDefault();
      const key = `${project.id}/${path.join("/")}/${text}`;
      const count =
        pasteSequence.current.key === key ? pasteSequence.current.count + 1 : 0;
      pasteSequence.current = { key, count };
      const inset = canvasFitInsets(canvasRef.current);
      const origin = reactFlow.screenToFlowPosition({
        x: bounds.x + bounds.width / 2,
        y:
          bounds.y +
          inset.top +
          Math.max(0, bounds.height - inset.top - inset.bottom) / 2,
      });
      const xs = source.nodes.map((node) => node.position.x);
      const ys = source.nodes.map((node) => node.position.y);
      // Center the copied group in the current viewport, retaining its arrangement.
      const anchor = {
        x:
          origin.x -
          (Math.max(...xs) - Math.min(...xs) + 280) / 2 +
          (count % 6) * 40,
        y:
          origin.y -
          (Math.max(...ys) - Math.min(...ys) + 240) / 2 +
          (count % 6) * 40,
      };
      changeGraph((graph) => {
        const nodes = pasteSelection(graph, source, anchor, uid);
        setSelection(nodes[0].id);
      });
      setSelectedEdge(null);
      notify(
        `${source.nodes.length} ${source.nodes.length === 1 ? "node" : "nodes"} pasted`,
      );
    };
    window.addEventListener("copy", onCopy);
    window.addEventListener("paste", onPaste);
    return () => {
      window.removeEventListener("copy", onCopy);
      window.removeEventListener("paste", onPaste);
    };
  }, [modal, tab, path, project]);
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
        if (document.activeElement?.closest("#workspace-sidebar"))
          sidebarMenuButton.current?.focus();
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
  function moeBody(): Graph {
    const body = templates
      .find((t) => t.id === "moe-transformer")
      ?.graph.nodes.find((n) => n.data.kind === "moe")?.data.body;
    if (!body)
      throw new Error("The mixture of experts definition is unavailable");
    return copy(body);
  }
  function resizeExperts(data: NodeData, value: JsonValue) {
    if (
      typeof value !== "number" ||
      !Number.isInteger(value) ||
      value < 1 ||
      value > 256
    )
      throw new Error("Expert count must be an integer between 1 and 256.");
    const body = data.body;
    if (!body) throw new Error("The MoE graph is missing.");
    const input = body.nodes.find((n) => n.data.kind === "input");
    const router = body.nodes.find((n) => n.data.kind === "moe_router");
    const combine = body.nodes.find((n) => n.data.kind === "moe_combine");
    const experts = body.nodes.filter((n) => n.data.role === "moe-expert");
    if (!input || !router || !combine || !experts.length)
      throw new Error("The MoE scaffold is incomplete.");
    for (const expert of experts.slice(value)) {
      body.nodes = body.nodes.filter((n) => n.id !== expert.id);
      body.edges = body.edges.filter(
        (edge) => edge.source !== expert.id && edge.target !== expert.id,
      );
    }
    for (let i = experts.length; i < value; i++) {
      const expert = copy(experts[0]);
      makeIndependent(expert);
      expert.id = uid();
      expert.data.label = `Expert ${i + 1}`;
      expert.position = { x: experts[0].position.x, y: i * 230 };
      body.nodes.push(expert);
      body.edges.push(
        {
          id: uid(),
          source: input.id,
          target: expert.id,
          sourceHandle: "out",
          targetHandle: "a",
          type: "smoothstep",
        },
        {
          id: uid(),
          source: expert.id,
          target: combine.id,
          sourceHandle: "out",
          targetHandle: `e${i}`,
          type: "smoothstep",
        },
      );
    }
    router.data.params.experts = value;
    combine.data.params.experts = value;
    data.params.experts = value;
    if (typeof data.params.top_k === "number" && data.params.top_k > value)
      data.params.top_k = value;
  }
  function catalogNodeData(entry: CatalogEntry): NodeData {
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
    if (kind === "moe") body = moeBody();
    if (entry.role === "moe-shared-expert") {
      const expert = moeBody().nodes.find((n) => n.data.role === "moe-expert")
        ?.data.body;
      if (!expert) throw new Error("Expert definition is unavailable");
      body = copy(expert);
    }
    if (entry.role === "transformer") body = transformerBody();
    if (entry.role === "adapter") body = adapterBody();
    const data: NodeData = {
      kind,
      label,
      params: copy(p),
      ...(body ? { body } : {}),
      ...(entry.role ? { role: entry.role } : {}),
    };
    if (entry.role === "transformer") {
      inheritTransformerDefaults(
        data,
        graphVariableScope(project, path).bindings,
      );
      exposeTransformerSettings(data);
    }
    if (
      entry.role === "transformer" ||
      ["attention", "cross_attention", "rope"].includes(kind)
    ) {
      const theta = graphVariableScope(project, path).effective.theta;
      data.params.theta = typeof theta === "number" ? "$theta" : 1_000_000;
    }
    wrapLoopTransformer(data);
    return data;
  }
  function addNode(
    entry: CatalogEntry,
    screenPosition?: { x: number; y: number },
    source?: { id: string; handle: string; targetHandle?: string },
  ) {
    addGraphNode(catalogNodeData(entry), screenPosition, source);
  }
  function addGraphNode(
    data: NodeData,
    screenPosition?: { x: number; y: number },
    source?: { id: string; handle: string; targetHandle?: string },
  ) {
    const id = uid();
    const position = reactFlow.screenToFlowPosition(
      screenPosition ?? {
        x: window.innerWidth / 2 - 70,
        y: window.innerHeight / 2,
      },
    );
    changeGraph((graph) => {
      if (source && !graph.nodes.some((node) => node.id === source.id))
        throw new Error("The source node is no longer available.");
      clearGraphSelection(graph);
      const node: GraphNode = {
        id,
        type: "architecture",
        position,
        data: copy(data),
        selected: true,
      };
      makeIndependent(node);
      const port = source?.targetHandle ?? nodePorts(node)[0];
      if (port && !nodePorts(node).includes(port))
        throw new Error("The target input port is no longer available.");
      if (source && !port)
        throw new Error("Choose a block with an input port.");
      graph.nodes.push(node);
      if (source)
        graph.edges.push({
          id: uid(),
          source: source.id,
          sourceHandle: source.handle,
          target: id,
          targetHandle: port,
          type: "smoothstep",
        });
    });
    setSelection(id);
    setSelectedEdge(null);
    setMobilePanel(window.innerWidth <= 760 ? "properties" : null);
    notify(
      data.label +
        (source ? " added and connected" : " added · connect its ports"),
    );
  }
  const connectionChoices: ConnectionBlockChoice[] = connectionPicker
    ? [
        ...catalog
          .filter((entry) => entry.kind !== "input")
          .map((entry) => ({
            id: "catalog:" + entry.label,
            label: entry.label,
            description: entry.description,
            category: entry.category,
            data: catalogNodeData(entry),
          })),
        ...customBlocks
          .filter(
            (data) =>
              nodePorts({
                id: "preset",
                type: "architecture",
                position: { x: 0, y: 0 },
                data,
              }).length > 0,
          )
          .map((data, index) => ({
            id: "saved:" + index,
            label: data.label,
            description: data.description ?? "Your reusable block",
            category: "Your blocks",
            data,
          })),
      ]
    : [];
  const closeConnectionPicker = useCallback(() => {
    setConnectionPicker(null);
    canvasRef.current?.focus({ preventScroll: true });
  }, []);
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
          edgeContract(report, key, edge)?.shape.map(String).join(" × ") || "",
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
      try {
        setSaved(savedProjectsSchema.parse(await api("/projects")));
        notify("Project saved on server");
      } catch (refreshError) {
        notify(
          "Project saved on server; project list refresh failed: " +
            describeError(refreshError),
        );
      }
      return true;
    } catch (e) {
      notify(String(e));
      return false;
    } finally {
      setSaving(false);
    }
  }
  async function loadProject(
    p: Project | SavedProject,
    options: OpenProjectOptions = {},
  ) {
    if (dirty && !options.force) {
      setPendingLoad({ project: p, options });
      setModal("unsaved");
      return false;
    }
    try {
      const loaded = options.saved
        ? projectSchema.parse(await api("/projects/" + p.id))
        : projectSchema.parse(p);
      const next = localizeLayerDefaults(loaded);
      pendingArrangement.current = new Set(
        options.arrange ? arrangeExample(next) : [],
      );
      setProject(next);
      ref.current = next;
      setPath([]);
      setSelection(null);
      setSelectedEdge(null);
      setHistory([]);
      setFuture([]);
      setDirty(!!options.newDraft);
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
  function startNewProject(starter = "blank") {
    setStarterId(starter);
    setMobilePanel(null);
    setModal("new-project");
  }
  function viewReference(example: Project) {
    setReferenceChoice("example:" + example.id);
    setReferenceLocation({ blocks: [], node: null });
    setSplitView(true);
    setTab("graph");
    setModal(null);
    setMobilePanel(null);
    notify(
      "Reference opened beside your project. Your graph is still editable on the left.",
    );
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
      const ids = new Set(
        g.nodes.filter((node) => node.selected).map((node) => node.id),
      );
      if (!ids.size) ids.add(selection);
      g.nodes = g.nodes.filter((node) => !ids.has(node.id));
      g.edges = g.edges.filter((e) => !ids.has(e.source) && !ids.has(e.target));
    });
    setSelection(null);
  }
  function duplicate() {
    const text = serializeSelection(graph, selection);
    const source = text ? parseSelection(text) : null;
    if (!source) return;
    const anchor = {
      x: Math.min(...source.nodes.map((node) => node.position.x)) + 45,
      y: Math.min(...source.nodes.map((node) => node.position.y)) + 65,
    };
    changeGraph((g) => {
      const nodes = pasteSelection(g, source, anchor, uid);
      for (const node of nodes) node.data.label += " copy";
      setSelection(nodes[0].id);
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
        input ? edgeContract(report, prefix, input) : null,
        report?.shapes[prefix + selected.id],
      );
      setSelection(next.id);
    });
    notify("Shared application added. Both instances use the same parameters.");
    setTimeout(() => fitGraph({ duration: motion(250) }), 100);
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
      setTimeout(() => fitGraph({ duration: motion(250) }), 100);
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
    setTimeout(() => fitGraph({ duration: motion(250) }), 100);
  }

  function groupSelection() {
    const ids = new Set(
      graph.nodes
        .filter((n: GraphNode) => n.selected)
        .map((n: GraphNode) => n.id),
    );
    if (!ids.size && selection) ids.add(selection);
    if (!ids.size) {
      notify(
        "Ctrl/Cmd-click nodes or drag a selection box, then create a block",
      );
      return;
    }
    const outgoing = graph.edges.filter(
      (e) => ids.has(e.source) && !ids.has(e.target),
    );
    const outputStreams = [
      ...new Map(
        outgoing.map((e) => [
          JSON.stringify([e.source, e.sourceHandle || "out"]),
          e,
        ]),
      ).values(),
    ];
    const outputIds = outputStreams.map((e) => e.source);
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
    const outputNode = selectedNodes.find((node) => node.id === output);
    const outputHandle =
      outputStreams[0]?.sourceHandle ||
      (outputNode ? outputPorts(outputNode.data)[0] : "out");
    if (
      !outputHandle ||
      (!outputStreams.length &&
        outputNode &&
        outputPorts(outputNode.data).length > 1)
    ) {
      notify(
        "Connect a single outgoing lane or include a Join lanes node before creating a block.",
      );
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
      const key = JSON.stringify([e.source, e.sourceHandle || "out"]);
      if (!external.has(key)) {
        const i = external.size;
        external.set(key, i);
        const shape = edgeContract(report, prefix, e);
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
      sourceHandle: outputHandle,
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
        ...outgoing.map((e) => ({ ...e, source: id, sourceHandle: "out" })),
      ];
    });
    setSelection(id);
    notify("Reusable block created · open to edit its internals");
  }
  function applyInputTokenizer(info: TokenizerInfo) {
    if (ref.current.id !== project.id) {
      notify("Tokenizer result ignored because the active project changed.");
      return false;
    }
    if (
      !allNodes(ref.current).some(
        (item) =>
          item.path === tokenizerTarget && item.node.data.kind === "input",
      )
    ) {
      notify("The tokenizer input is no longer available.");
      return false;
    }
    change((p) => {
      const input = allNodes(p).find(
        (item) => item.path === tokenizerTarget,
      )?.node;
      if (input) input.data.tokenizer = info;
    });
    return true;
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
        fitGraph({
          nodes: [{ id }],
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
    <div className={"app-shell" + (tab === "training" ? " training-mode" : "")}>
      <a className="skip-link" href="#workspace">
        Skip to editor
      </a>
      {mobilePanel && (
        <button
          className="drawer-backdrop"
          aria-label="Close side panel"
          onClick={() => {
            setMobilePanel(null);
            if (mobilePanel === "library") sidebarMenuButton.current?.focus();
          }}
        />
      )}
      <aside
        id="workspace-sidebar"
        className={"sidebar " + (mobilePanel === "library" ? "is-open" : "")}
        aria-label="Component library"
      >
        <button
          className="drawer-close"
          aria-label="Close library"
          onClick={() => {
            setMobilePanel(null);
            sidebarMenuButton.current?.focus();
          }}
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
          <span className="workspace-avatar">
            <FolderOpen size={18} aria-hidden="true" />
          </span>
          <span>
            Your workspace<small>Projects & examples</small>
          </span>
          <ChevronDown size={14} />
        </button>
        {tab === "training" ? (
          <div className="training-sidebar-context">
            <p>Model workspace</p>
            <strong>{project.name}</strong>
            <button
              onClick={() => {
                setTab("graph");
                setMobilePanel(null);
              }}
            >
              <Workflow size={15} aria-hidden="true" />
              Architecture
            </button>
            <button
              onClick={() => {
                setTab("code");
                setMobilePanel(null);
              }}
            >
              <Code2 size={15} aria-hidden="true" />
              PyTorch code
            </button>
            <span>
              <Activity size={15} aria-hidden="true" />
              Training
            </span>
          </div>
        ) : (
          <>
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
                Projects
              </button>
            </div>
            <div className="library-search">
              <label className="library-search-field">
                <Search size={16} aria-hidden="true" />
                <input
                  ref={librarySearchRef}
                  aria-label={
                    left === "blocks"
                      ? "Search blocks"
                      : "Search projects and examples"
                  }
                  name="library-search"
                  autoComplete="off"
                  placeholder={
                    left === "blocks"
                      ? "Find a building block…"
                      : "Find a project or example…"
                  }
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
              </label>
              {search && (
                <button
                  type="button"
                  className="library-search-clear"
                  aria-label="Clear search"
                  onClick={() => {
                    setSearch("");
                    librarySearchRef.current?.focus();
                  }}
                >
                  <X size={15} aria-hidden="true" />
                </button>
              )}
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
                          onClick={() => addGraphNode(b)}
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
                    <h3>Your projects</h3>
                    <p className="collection-description">
                      Saved on this server
                    </p>
                    {saved
                      .filter((p) =>
                        p.name
                          .toLocaleLowerCase()
                          .includes(search.toLocaleLowerCase()),
                      )
                      .map((p) => (
                        <button
                          className="model-item"
                          key={p.id}
                          onClick={() => loadProject(p, { saved: true })}
                          title="Open saved project"
                        >
                          <FolderOpen size={16} aria-hidden="true" />
                          <span>
                            {p.name}
                            <small>Open saved project</small>
                          </span>
                        </button>
                      ))}
                    {!saved.length && (
                      <p className="collection-description">
                        No saved projects yet. Create a network with New
                        project, then Save it here.
                      </p>
                    )}
                    {!!saved.length &&
                      !saved.some((p) =>
                        p.name
                          .toLocaleLowerCase()
                          .includes(search.toLocaleLowerCase()),
                      ) && (
                        <p className="collection-description" role="status">
                          No saved projects match “{search}”.
                        </p>
                      )}
                    <button
                      className="browse-projects"
                      onClick={() => {
                        setProjectTab("projects");
                        setModal("projects");
                      }}
                    >
                      <FolderOpen size={14} aria-hidden="true" />
                      Manage projects
                    </button>
                  </div>
                  <div className="library-group">
                    <h3>Reference architectures</h3>
                    {templates
                      .filter(
                        (t) =>
                          t.id !== "blank" &&
                          (t.name + " " + t.description)
                            .toLocaleLowerCase()
                            .includes(search.toLocaleLowerCase()),
                      )
                      .map((t) => (
                        <ReferenceCard
                          key={t.id}
                          project={t}
                          compact
                          onView={() => viewReference(t)}
                          onUse={() => startNewProject(t.id)}
                        />
                      ))}
                    <button
                      className="browse-projects"
                      onClick={() => {
                        setProjectTab("examples");
                        setModal("projects");
                      }}
                    >
                      Browse all examples{" "}
                      <ArrowUpRight size={14} aria-hidden="true" />
                    </button>
                  </div>
                </>
              )}
            </div>
          </>
        )}
        <div className="sidebar-bottom">
          <button
            aria-label="Open project settings"
            onClick={() => setModal("settings")}
          >
            <Settings2 size={16} aria-hidden="true" />
            Project settings
          </button>
        </div>
      </aside>
      <main className="main-workspace" id="workspace" tabIndex={-1}>
        <header className="topbar">
          <button
            ref={sidebarMenuButton}
            className="mobile-sidebar-menu"
            aria-label="Open sidebar"
            aria-controls="workspace-sidebar"
            aria-expanded={mobilePanel === "library"}
            onClick={() => {
              setMobilePanel(mobilePanel === "library" ? null : "library");
              requestAnimationFrame(() =>
                document
                  .querySelector<HTMLButtonElement>(
                    "#workspace-sidebar .drawer-close",
                  )
                  ?.focus(),
              );
            }}
          >
            <Menu size={20} aria-hidden="true" />
          </button>
          <div className="project-heading">
            <span className="project-icon">
              <GitBranch size={18} />
            </span>
            <div>
              <h1 className="project-title-heading">
                <button
                  className="project-title"
                  title="Rename and configure project"
                  aria-label={`Project settings: ${project.name}`}
                  onClick={() => setModal("settings")}
                >
                  {project.name}
                  <Pencil size={12} aria-hidden="true" />
                </button>
              </h1>
              <span className="project-meta">
                {saved.some((p) => p.id === project.id)
                  ? dirty
                    ? "Unsaved changes"
                    : "Saved on server"
                  : templates.some((p) => p.id === project.id)
                    ? dirty
                      ? "Unsaved example changes"
                      : "Example · use as starting point"
                    : "Draft · save to keep on server"}
              </span>
            </div>
          </div>
          <div className="topbar-actions">
            <ActiveTrainingRuns />
            <button
              className="subtle-button new-project-button"
              aria-label="New project"
              onClick={() => startNewProject()}
            >
              <FilePlus2 size={16} aria-hidden="true" />
              <span className="new-project-desktop">New project</span>
              <span className="new-project-mobile">New</span>
            </button>
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
              className={tab === "training" ? "active" : ""}
              aria-label="Training"
              aria-current={tab === "training" ? "page" : undefined}
              onClick={() => setTab("training")}
            >
              <Activity size={15} />
              Training
            </button>
            <button
              className={tab === "sources" ? "active" : ""}
              aria-label="Sources"
              aria-current={tab === "sources" ? "page" : undefined}
              onClick={() => setTab("sources")}
            >
              <ExternalLink size={14} />
              Sources
            </button>
          </div>
          <div className="view-actions">
            <UIUpdateNotice />
            <GPUTuning project={project} />
            <ThemePicker />
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
          </div>
        </nav>
        {tab === "training" ? (
          <TrainingWorkspace key={project.id} project={project} />
        ) : (
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
                    {path.length > 0 && (
                      <button
                        className="graph-up-button"
                        aria-label="Up one level"
                        title="Return to the enclosing graph"
                        onClick={goUp}
                      >
                        <ArrowUp size={15} />
                        <span>Up</span>
                      </button>
                    )}

                    <button
                      title="Top level model graph"
                      onClick={() => {
                        setPath([]);
                        setSelection(null);
                        setTimeout(() => fitGraph(), 80);
                      }}
                    >
                      <Box size={14} />
                      {path.length ? "Model" : "Model root"}
                    </button>
                    {crumbs.map((c, i) => (
                      <React.Fragment key={i}>
                        <ChevronRight size={12} />
                        <button
                          onClick={() => {
                            setPath(path.slice(0, i + 1));
                            setSelection(null);
                            setTimeout(() => fitGraph(), 80);
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
                      aria-label="Graph variables"
                      className="variables-button"
                      title="Graph variables: define values used by $name bindings"
                      onClick={() => setModal("variables")}
                    >
                      <Braces size={15} />
                      <span className="variables-button-label">Variables</span>
                    </button>
                    <button
                      aria-label={
                        arranging ? "Auto arrange: Arranging…" : "Auto arrange"
                      }
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
                      aria-label={
                        showAnnotations
                          ? "Hide design notes"
                          : "Show design notes"
                      }
                      title={
                        showAnnotations
                          ? "Hide design notes"
                          : "Show design notes"
                      }
                      aria-pressed={showAnnotations}
                      onClick={() => setShowAnnotations((value) => !value)}
                    >
                      <MessageSquareText size={15} aria-hidden="true" />
                    </button>
                    <button
                      aria-label="Fit graph"
                      title="Fit graph"
                      onClick={() =>
                        fitGraph({
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
                    tabIndex={0}
                    aria-label="Editable graph. Drag to select nodes; Ctrl or Command-click to extend selection. Copy and paste with Ctrl or Command C and V. Hold Space to pan."
                    onPointerDownCapture={(event) => {
                      if (
                        event.target instanceof Element &&
                        !event.target.closest(
                          "input,textarea,select,[contenteditable],button",
                        )
                      )
                        event.currentTarget.focus({ preventScroll: true });
                    }}
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
                              if (
                                changes.some(
                                  (change) => change.type === "select",
                                )
                              ) {
                                const nodes = g.nodes.filter(
                                  (node) => node.selected,
                                );
                                setSelection((current) =>
                                  nodes.some((node) => node.id === current)
                                    ? current
                                    : (nodes.at(-1)?.id ?? null),
                                );
                                if (nodes.length) setSelectedEdge(null);
                              }
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
                                    !("id" in c) ||
                                    !c.id.startsWith("weights:"),
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
                        onConnect={(connection) => {
                          const edgeId = uid();
                          setConnectionAttempt({
                            projectId: project.id,
                            path: [...path],
                            edgeId,
                          });
                          changeGraph((g) => {
                            g.edges = addEdge(
                              { ...connection, id: edgeId, type: "smoothstep" },
                              g.edges,
                            );
                          });
                        }}
                        onConnectEnd={(event, state) => {
                          if (
                            state.isValid ||
                            state.toNode ||
                            state.fromHandle?.type !== "source" ||
                            !state.fromNode ||
                            (state.fromHandle.id &&
                              state.fromHandle.id.startsWith("weights-"))
                          )
                            return;
                          const point =
                            "changedTouches" in event
                              ? event.changedTouches[0]
                              : event;
                          const bounds =
                            canvasRef.current?.getBoundingClientRect();
                          if (
                            !point ||
                            !bounds ||
                            point.clientX < bounds.left ||
                            point.clientX > bounds.right ||
                            point.clientY < bounds.top ||
                            point.clientY > bounds.bottom
                          )
                            return;
                          const target = document.elementFromPoint(
                            point.clientX,
                            point.clientY,
                          );
                          if (
                            target?.closest(
                              ".react-flow__controls,.react-flow__panel,.react-flow__minimap,button,input",
                            )
                          )
                            return;
                          setConnectionPicker({
                            projectId: project.id,
                            scope: path.join("/"),
                            source: state.fromNode.id,
                            sourceHandle: state.fromHandle.id ?? "out",
                            position: { x: point.clientX, y: point.clientY },
                            bounds: {
                              left: bounds.left,
                              top: bounds.top,
                              right: bounds.right,
                              bottom: bounds.bottom,
                            },
                          });
                        }}
                        onNodeClick={(_, n) => {
                          const nodes = graphAt(
                            ref.current,
                            viewPath.current,
                          ).nodes.filter((node) => node.selected);
                          setSelection(
                            nodes.some((node) => node.id === n.id)
                              ? n.id
                              : (nodes.at(-1)?.id ?? null),
                          );
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
                        onBeforeDelete={async ({ nodes, edges }) => {
                          const nodeIds = new Set(nodes.map((node) => node.id));
                          const edgeIds = new Set(edges.map((edge) => edge.id));
                          changeGraph((graph) => {
                            graph.nodes = graph.nodes.filter(
                              (node) => !nodeIds.has(node.id),
                            );
                            graph.edges = graph.edges.filter(
                              (edge) =>
                                !edgeIds.has(edge.id) &&
                                !nodeIds.has(edge.source) &&
                                !nodeIds.has(edge.target),
                            );
                          });
                          setSelection(null);
                          setSelectedEdge(null);
                          return false;
                        }}
                        fitView={window.innerWidth > 760 && !splitView}
                        fitViewOptions={{
                          padding: canvasFitPadding(canvasRef.current),
                        }}
                        minZoom={0.04}
                        maxZoom={1.7}
                        deleteKeyCode={["Backspace", "Delete"]}
                        selectionKeyCode="Shift"
                        multiSelectionKeyCode={["Control", "Meta", "Shift"]}
                        selectionOnDrag
                        selectionMode={SelectionMode.Partial}
                        panOnDrag={[1, 2]}
                        panActivationKeyCode="Space"
                        proOptions={{ hideAttribution: true }}
                      >
                        <Background
                          variant={BackgroundVariant.Dots}
                          gap={22}
                          size={1}
                          color="var(--graph-dots, #ced6cc)"
                        />
                        <Controls
                          showInteractive={false}
                          fitViewOptions={{
                            padding: canvasFitPadding(canvasRef.current),
                          }}
                        />
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
                          maskColor="var(--graph-minimap-mask, rgba(247,249,245,.7))"
                          pannable
                          zoomable
                        />
                        <GraphAnnotations
                          nodes={graph.nodes}
                          visible={showAnnotations}
                          onEdit={(id) => {
                            setSelection(id);
                            if (window.innerWidth <= 760)
                              setMobilePanel("properties");
                          }}
                        />
                        <Panel
                          position="top-left"
                          className="variables-canvas-panel"
                        >
                          <GraphVariableCard
                            values={variableScope.own}
                            inherited={variableScope.inherited}
                            uses={variableUses}
                            hiddenNames={hiddenVariables}
                            nested={path.length > 0}
                            onChange={updateGraphVariables}
                            onOpen={() => setModal("variables")}
                          />
                        </Panel>
                        <Panel position="bottom-center">
                          <GraphNavigator
                            graph={graph}
                            selection={selection}
                            onFocus={setSelection}
                            compact={splitView}
                            fitPadding={() =>
                              canvasFitPadding(canvasRef.current)
                            }
                          />
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
                    <pre
                      tabIndex={0}
                      role="region"
                      aria-label="Generated PyTorch code"
                    >
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
                      Export includes runtime operations, tokenizer files,
                      project JSON, dependencies, and a runnable example.
                    </div>
                  </div>
                ) : (
                  <div className="references-view">
                    <span className="eyebrow">ARCHITECTURE PROVENANCE</span>
                    <h1>Sources & implementation notes</h1>
                    <p>
                      Reference examples retain their published settings. Read
                      the notes for computational differences and unavailable
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
                                <pre
                                  tabIndex={0}
                                  role="region"
                                  aria-label={`Configuration for ${r.title}`}
                                >
                                  {JSON.stringify(r.config, null, 2)}
                                </pre>
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
                          This is a design example rather than a reproduction of
                          a published checkpoint. All weights are randomly
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
                  {!!report?.warnings.length && (
                    <button
                      className="advice-count"
                      onClick={() => setChecksOpen(!checksOpen)}
                      aria-expanded={checksOpen}
                    >
                      <Sparkles size={14} />
                      {report.warnings.length} design note
                      {report.warnings.length === 1 ? "" : "s"}
                    </button>
                  )}
                  <span>
                    {graph.nodes.length} nodes <i>·</i> {graph.edges.length}{" "}
                    connections
                  </span>
                  <span className="parameters-total">
                    {format(report?.parameters || 0)} parameters <i>·</i>{" "}
                    float32
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
                          <b>
                            {e.node || "Model"}
                            {e.port ? ` · input ${e.port}` : ""}
                          </b>
                          {e.message}
                        </span>
                        <ArrowUpRight size={14} />
                      </button>
                    ))}
                    {!!report?.warnings.length && (
                      <>
                        <h2>Design notes</h2>
                        <p>
                          Suggestions to consider. These never block export.
                        </p>
                        {report.warnings.map((issue, i) => (
                          <DesignAdvice
                            key={`${issue.node}-${issue.code}-${i}`}
                            issue={issue}
                            onLocate={focusError}
                          />
                        ))}
                      </>
                    )}
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
                    onNavigate={navigateReference}
                    onInspect={setReferenceInspection}
                    onActivate={() => {
                      if (window.innerWidth <= 760)
                        setMobilePanel("properties");
                    }}
                    onUseExample={(p) => startNewProject(p.id)}
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
              data-reference-pane={referenceDetails ? "true" : undefined}
              onKeyDownCapture={(event) => {
                if (referenceDetails && isGraphEditShortcut(event)) {
                  event.preventDefault();
                  event.stopPropagation();
                }
              }}
            >
              <div className="inspector-heading">
                <span>
                  <Settings2 size={15} />
                  {referenceDetails
                    ? "Reference properties"
                    : selected
                      ? selectedCount > 1
                        ? `${selectedCount} blocks selected`
                        : "Block properties"
                      : connection
                        ? "Connection properties"
                        : "Graph overview"}
                </span>
                <button
                  className="drawer-close"
                  aria-label="Close properties"
                  onClick={() => setMobilePanel(null)}
                >
                  <X size={20} />
                </button>
                {(selected || referenceDetails) && (
                  <button
                    aria-label="Close inspector selection"
                    title="Close inspector selection"
                    onClick={() => {
                      setSelection(null);
                      change(
                        (project) => clearGraphSelection(project.graph),
                        false,
                        false,
                      );
                      setReferenceLocation((previous) => ({
                        ...previous,
                        node: null,
                      }));
                    }}
                  >
                    <X size={14} />
                  </button>
                )}
              </div>
              <div className="inspector-scroll">
                {referenceDetails ? (
                  <ReferenceInspector inspection={referenceDetails} />
                ) : selected ? (
                  <>
                    {selectedCount > 1 && (
                      <p className="inspector-description">
                        Copy, duplicate and delete apply to all {selectedCount}{" "}
                        selected blocks. Properties below edit{" "}
                        {selected.data.label}.
                      </p>
                    )}
                    <div
                      className={
                        "selected-block-icon " + family(selected.data.kind)
                      }
                    >
                      {React.createElement(
                        icons[selected.data.kind] || Shapes,
                        {
                          size: 24,
                        },
                      )}
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
                    <label className="property-field">
                      <span>Design note</span>
                      <textarea
                        aria-label="Design note"
                        placeholder="Add a note about this block or why you chose it"
                        value={selected.data.annotation ?? ""}
                        onChange={(event) =>
                          updateNode((node) => {
                            node.data.annotation = event.target.value;
                          })
                        }
                      />
                      <small className="field-help">
                        Notes appear on the canvas, are saved with the graph,
                        and do not affect computation.
                      </small>
                    </label>
                    {selected.data.annotation?.trim() && (
                      <label className="property-field">
                        <span>Design note source URL (optional)</span>
                        <input
                          aria-label="Design note source URL"
                          type="url"
                          value={selected.data.annotationSource ?? ""}
                          onChange={(event) =>
                            updateNode((node) => {
                              node.data.annotationSource = event.target.value;
                            })
                          }
                          placeholder="https://…"
                        />
                      </label>
                    )}
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
                        Expand into editable applications{" "}
                        <GitBranch size={15} />
                      </button>
                    )}
                    {supportsWeightSharing(selected.data) && (
                      <button
                        className="inspect-open"
                        onClick={duplicateShared}
                      >
                        Add application with shared weights <Copy size={15} />
                      </button>
                    )}
                    <button
                      className="save-library"
                      onClick={() =>
                        fitGraph({
                          nodes: [{ id: selected.id }],
                          duration: motion(250),
                        })
                      }
                    >
                      <Maximize2 size={13} /> Zoom to this node
                    </button>
                    {selected.data.kind === "input" &&
                      (selected.data.tokenizer ||
                        ((selected.data.params.dtype ?? "int64") === "int64" &&
                          (selected.data.params.shape ?? ["B", "T"]).length ===
                            2)) && (
                        <section
                          className="input-tokenizer-settings"
                          aria-label="Input tokenizer"
                        >
                          <div className="section-label">Tokenizer</div>
                          <button
                            className="tokenizer-chip"
                            disabled={
                              path.length > 0 &&
                              !!selectedTokenizer &&
                              !selected.data.tokenizer
                            }
                            onClick={() => {
                              setTokenizerTarget(prefix + selected.id);
                              setModal("tokenizer");
                            }}
                          >
                            <Database size={14} />
                            <span>
                              {selectedTokenizer
                                ? tokenizerLabel(selectedTokenizer)
                                : "Select tokenizer"}
                            </span>
                            {selected.data.tokenizer && (
                              <small>
                                {format(selected.data.tokenizer.capacity)} vocab
                              </small>
                            )}
                            <ChevronRight size={14} />
                          </button>
                          {selectedTokenizer && (
                            <div className="stat-row">
                              <span>Vocabulary capacity</span>
                              <b>
                                {selectedTokenizer.capacity.toLocaleString()}
                              </b>
                            </div>
                          )}
                          {!selected.data.tokenizer && selectedTokenizer && (
                            <p className="field-help">
                              Inherited from the connected input ·{" "}
                              {format(selectedTokenizer.capacity)} token IDs.
                              Configure the originating token input to change
                              it.
                            </p>
                          )}
                          {selected.data.tokenizer && (
                            <button
                              className="small-button"
                              onClick={() =>
                                updateNode((node) => {
                                  delete node.data.tokenizer;
                                })
                              }
                            >
                              Remove tokenizer
                            </button>
                          )}
                          <p className="field-help">
                            The tokenizer defines token IDs and vocabulary size.
                            Embedding width is configured on each embedding
                            block.
                          </p>
                        </section>
                      )}
                    {selectedTokenizer && selectedVocabField && (
                      <section
                        className="property-field"
                        aria-label="Tokenizer vocabulary capacity"
                      >
                        {selectedAutoVocab && (
                          <>
                            <div className="stat-row">
                              <span>Vocabulary capacity</span>
                              <b>
                                {selectedTokenizer.capacity.toLocaleString()}
                              </b>
                            </div>
                            <small className="field-help">
                              Automatic from the connected input tokenizer.
                            </small>
                          </>
                        )}
                        <button
                          className="small-button"
                          onClick={() =>
                            updateNode((node) => {
                              node.data.params[selectedVocabField] =
                                selectedAutoVocab
                                  ? selectedTokenizer.capacity
                                  : "$vocab";
                            })
                          }
                        >
                          {selectedAutoVocab
                            ? "Set capacity manually"
                            : "Use input tokenizer capacity"}
                        </button>
                      </section>
                    )}
                    {transformer && wrappedTransformer(selected.data) && (
                      <p className="field-help">
                        These controls edit the{" "}
                        {wrappedTransformer(selected.data)?.data.label} inside
                        this loop. Opening that block shows the same settings.
                      </p>
                    )}
                    {!transformer &&
                      selected.data.body &&
                      (Object.keys(selected.data.params).some((key) =>
                        transformerSettingKeys.has(key),
                      ) ||
                        selected.data.body.nodes.filter(
                          (node) => node.data.role === "transformer",
                        ).length > 1) && (
                        <p className="field-help">
                          Configure each inner layer separately. Additional
                          container parameters affect only fields explicitly
                          linked with $name.
                        </p>
                      )}
                    {transformer && (
                      <label className="property-field">
                        <span>Attention type</span>
                        <select
                          aria-label="Attention type"
                          disabled={["heads", "kv_heads", "window"].some(
                            (key) =>
                              !!containerParameterShadowReason(
                                wrappedTransformer(selected.data)?.data ??
                                  selected.data,
                                key,
                              ),
                          )}
                          value={attentionVariant(
                            selected.data,
                            fieldVariables,
                          )}
                          onChange={(event) => {
                            if (isAttentionVariant(event.target.value)) {
                              const variant = event.target.value;
                              updateNode((node) =>
                                setAttentionVariant(
                                  node.data,
                                  variant,
                                  fieldVariables,
                                  uid,
                                ),
                              );
                            }
                          }}
                        >
                          {attentionVariants.map((variant) => (
                            <option key={variant.id} value={variant.id}>
                              {variant.label}
                            </option>
                          ))}
                        </select>
                        <small className="field-help">
                          Changes the editable attention graph inside this
                          container. Other layers are preserved.
                          {attentionVariant(selected.data, fieldVariables) ===
                            "sparse" &&
                            " The local-pattern window belongs to the Local attention window node inside; the additional attention window only adds a further restriction."}
                        </small>
                      </label>
                    )}
                    {transformer && transformerResidual(selected.data) && (
                      <label className="property-field">
                        <span>Attention residual</span>
                        <select
                          aria-label="Attention residual"
                          value={transformerResidual(selected.data) ?? "add"}
                          onChange={(event) =>
                            updateNode((node) =>
                              setTransformerResidual(
                                node.data,
                                event.target.value as "add" | "gate",
                              ),
                            )
                          }
                        >
                          <option value="add">Add</option>
                          <option value="gate">Gated</option>
                        </select>
                      </label>
                    )}
                    <div className="section-label">Configuration</div>
                    {selected.data.kind === "moe" && (
                      <label className="property-field">
                        <span>Router scoring</span>
                        <select
                          aria-label="Router scoring"
                          value={
                            typeof selected.data.params.router === "string"
                              ? selected.data.params.router
                              : "softmax"
                          }
                          onChange={(event) =>
                            updateNode((n) => {
                              n.data.params.router = event.target.value;
                            })
                          }
                        >
                          <option value="softmax">Softmax top-k</option>
                          <option value="sigmoid">Sigmoid top-k</option>
                        </select>
                        <small className="field-help">
                          Both route each token to the top-k experts. Choose how
                          the router scores them.
                        </small>
                      </label>
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
                    {Object.entries(transformer ?? selected.data.params ?? {})
                      .filter(
                        ([k]) =>
                          k !== "source" &&
                          !(selectedAutoVocab && k === selectedVocabField) &&
                          !(k === "router" && selected.data.kind === "moe") &&
                          !(
                            k === "experts" &&
                            ["moe_router", "moe_combine"].includes(
                              selected.data.kind,
                            )
                          ) &&
                          (selected.data.kind !== "attention_pattern" ||
                            visiblePatternField(selected.data.params, k)) &&
                          (!selected.data.iteration ||
                            !["iterations", "shared"].includes(k)),
                      )
                      .map(([k, v]) => {
                        const inputOwnership = inputSettingOwnership(
                          selected.data.kind,
                          path.length,
                          k,
                          report?.shapes[prefix + selected.id],
                        );
                        const transformerOwner =
                          wrappedTransformer(selected.data)?.data ??
                          selected.data;
                        const localFfnLinked =
                          typeof transformerOwner.body?.variables?.ffn_width ===
                            "number" &&
                          transformerOwner.body.nodes.some(
                            (node) =>
                              node.data.kind === "ffn" &&
                              node.data.params.hidden === "$ffn_width",
                          );
                        return (
                          <React.Fragment key={k}>
                            <Field
                              name={k}
                              variables={selectedFieldVariables}
                              displayName={
                                k === "window" &&
                                transformer &&
                                attentionVariant(
                                  selected.data,
                                  fieldVariables,
                                ) === "sparse"
                                  ? "Additional attention window"
                                  : k === "gated" &&
                                      (transformer ||
                                        [
                                          "attention",
                                          "cross_attention",
                                        ].includes(selected.data.kind))
                                    ? "Attention output gate"
                                    : undefined
                              }
                              options={
                                selected.data.kind === "lane_join" &&
                                k === "mode"
                                  ? ["stack", "mean", "sum"]
                                  : undefined
                              }
                              value={
                                inputOwnership
                                  ? inputOwnership.value
                                  : (v ?? null)
                              }
                              help={settingHelp(selected.data.kind, k)}
                              disabledReason={
                                inputOwnership?.disabledReason ??
                                (transformer && k === "ffn_width"
                                  ? undefined
                                  : containerParameterShadowReason(
                                      wrappedTransformer(selected.data)?.data ??
                                        selected.data,
                                      k,
                                    )) ??
                                generalFieldDisabledReason(
                                  selected.data.kind,
                                  k,
                                  selected.data.params,
                                ) ??
                                (transformer ||
                                [
                                  "attention",
                                  "cross_attention",
                                  "rope",
                                ].includes(selected.data.kind)
                                  ? attentionFieldDisabledReason(
                                      k,
                                      previewParameters(
                                        transformer ?? selected.data.params,
                                        fieldVariables,
                                      ),
                                      transformer
                                        ? attentionVariant(
                                            selected.data,
                                            fieldVariables,
                                          )
                                        : undefined,
                                    )
                                  : undefined)
                              }
                              onChange={(next) =>
                                updateNode((n) => {
                                  if (
                                    n.data.kind === "moe" &&
                                    k === "experts"
                                  ) {
                                    resizeExperts(n.data, next);
                                    return;
                                  }
                                  if (
                                    transformer &&
                                    transformerSettingKeys.has(k)
                                  ) {
                                    setTransformerSetting(
                                      n.data,
                                      k,
                                      next,
                                      fieldVariables,
                                    );
                                    return;
                                  }
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
                            {transformer &&
                              k === "ffn_width" &&
                              localFfnLinked && (
                                <small className="field-help">
                                  Local variable inside this Transformer; its
                                  FFN uses <code>$ffn_width</code>.
                                </small>
                              )}
                            {typeof v === "number" &&
                              !["index", "experts"].includes(k) &&
                              typeof transformerParentScope(
                                selected.data,
                                variableScope.bindings,
                              )[
                                k === "hidden" && selected.data.kind === "ffn"
                                  ? "ffn_width"
                                  : k
                              ] === "number" &&
                              !inputOwnership &&
                              ((k === "ffn_width" && transformer) ||
                                !containerParameterShadowReason(
                                  wrappedTransformer(selected.data)?.data ??
                                    selected.data,
                                  k,
                                )) && (
                                <button
                                  className="small-button"
                                  onClick={() =>
                                    updateNode((node) => {
                                      const variable =
                                        k === "hidden" &&
                                        selected.data.kind === "ffn"
                                          ? "ffn_width"
                                          : k;
                                      if (
                                        transformer &&
                                        transformerSettingKeys.has(k)
                                      )
                                        setTransformerSetting(
                                          node.data,
                                          k,
                                          "$" + variable,
                                          fieldVariables,
                                        );
                                      else node.data.params[k] = "$" + variable;
                                    })
                                  }
                                >
                                  {transformer &&
                                  k === "ffn_width" &&
                                  localFfnLinked
                                    ? "Local value"
                                    : "Independent value"}{" "}
                                  · Link to parent $
                                  {k === "hidden" &&
                                  selected.data.kind === "ffn"
                                    ? "ffn_width"
                                    : k}{" "}
                                  (
                                  {String(
                                    transformerParentScope(
                                      selected.data,
                                      variableScope.bindings,
                                    )[
                                      k === "hidden" &&
                                      selected.data.kind === "ffn"
                                        ? "ffn_width"
                                        : k
                                    ],
                                  )}
                                  )
                                </button>
                              )}
                            {k === "width" && (
                              <p className="field-help">
                                Model width: the number of features in each
                                token representation. Connected layers must
                                agree on this size.
                              </p>
                            )}
                            {k === "hidden" && selected.data.kind === "ffn" && (
                              <p className="field-help">
                                Feed-forward hidden width: the intermediate
                                number of features inside the FFN. The FFN
                                projects back to the model width afterwards.
                              </p>
                            )}
                            {typeof v === "string" && v.startsWith("$") && (
                              <p className="binding-resolution field-help">
                                {v} ={" "}
                                {String(
                                  bindingValue(v, selectedFieldVariables) ??
                                    "undefined · add it in Graph variables",
                                )}
                              </p>
                            )}
                          </React.Fragment>
                        );
                      })}
                    {selected.data.kind === "moe" && (
                      <p className="field-help">
                        Open experts to edit their individual graphs. The router
                        selects tokens and the combine sums their weighted
                        outputs. Expert count is a fixed integer; other settings
                        can use variables.
                      </p>
                    )}
                    {["moe_router", "moe_combine"].includes(
                      selected.data.kind,
                    ) && (
                      <p className="field-help">
                        Expert count is controlled by the outer Mixture of
                        experts block. Edit the graph inside each expert.
                      </p>
                    )}
                    {selected.data.kind === "attention_pattern" && (
                      <p className="field-help">
                        Connect features to a, and optionally key features to b.
                        The output is an attention pattern. Connect it to
                        attention’s optional mask port; combine patterns with
                        union or intersection. Custom matrices require fixed
                        sequence lengths. Content top-k selects from features
                        and computes dense routing scores.
                      </p>
                    )}
                    {["attention", "cross_attention", "sdpa"].includes(
                      selected.data.kind,
                    ) && (
                      <p className="field-help">
                        Dense uses PyTorch SDPA on CPU or GPU. Auto uses
                        compiled FlexAttention on CUDA; flex requires a
                        compatible GPU. A pattern defines permitted positions
                        independently of the backend.
                      </p>
                    )}
                    {selected.data.iteration && (
                      <>
                        <div className="section-label">
                          Application schedule
                        </div>
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
                    {supportsWeightSharing(selected.data) && (
                      <WeightGroupPicker
                        value={
                          selected.data.share ??
                          (selected.data.kind === "head" &&
                          selected.data.params.tie
                            ? `legacy:${selected.data.params.tie}`
                            : "")
                        }
                        matrixSharing={["embedding", "head"].includes(
                          selected.data.kind,
                        )}
                        choices={[
                          ...weightGroupChoices(
                            allNodes(project),
                            selected.data,
                          ),
                          ...(selected.data.kind === "head" &&
                          selected.data.params.tie &&
                          !selected.data.share
                            ? [
                                {
                                  id: `legacy:${selected.data.params.tie}`,
                                  label: `Legacy tie · ${graph.nodes.find((n) => n.id === selected.data.params.tie)?.data.label ?? selected.data.params.tie}`,
                                  compatible: false,
                                },
                              ]
                            : []),
                        ]}
                        onChange={(group) => {
                          if (
                            !selected.data.share &&
                            group === `legacy:${selected.data.params.tie}`
                          )
                            return;
                          updateNode((node) => {
                            delete node.data.params.tie;
                            if (group) node.data.share = group;
                            else makeIndependent(node);
                          });
                        }}
                        onCreate={() =>
                          updateNode((node) => {
                            delete node.data.params.tie;
                            makeIndependent(node);
                            node.data.share = "weights_" + uid();
                          })
                        }
                      />
                    )}
                    {selected.data.kind === "head" &&
                      selected.data.params.tie &&
                      selected.data.share && (
                        <p className="field-help">
                          This head also has a legacy embedding tie. Selecting a
                          weight group or Independent weights clears the legacy
                          tie.
                        </p>
                      )}
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
                    {report?.warnings
                      .filter((issue) => issue.node === prefix + selected.id)
                      .map((issue, i) => (
                        <DesignAdvice
                          key={`${issue.code}-${i}`}
                          issue={issue}
                          onLocate={focusError}
                        />
                      ))}
                    <div className="node-actions">
                      <button onClick={duplicate}>
                        <Copy size={14} />
                        {selectedCount > 1 ? "Duplicate selected" : "Duplicate"}
                      </button>
                      <button className="danger" onClick={removeSelected}>
                        <Trash2 size={14} />
                        {selectedCount > 1 ? "Delete selected" : "Delete"}
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
                        graph.nodes.find((n) => n.id === connection.source)
                          ?.data.label
                      }{" "}
                      →{" "}
                      {
                        graph.nodes.find((n) => n.id === connection.target)
                          ?.data.label
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
                            <option value="transformer">
                              Transformer layer
                            </option>
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
                    <h2>Select a block</h2>
                    <p className="inspector-description">
                      Click a block to edit its settings. Click a connection to
                      insert a layer.
                    </p>
                    <div className="section-label">Whole graph</div>
                    <div className="stat-row">
                      <span>Trainable parameters</span>
                      <b>{format(report?.parameters || 0)}</b>
                    </div>
                    <div className="stat-row">
                      <span>Graph validation</span>
                      <b className={report?.valid ? "valid" : "invalid"}>
                        {report?.valid ? "Compatible" : "Needs attention"}
                      </b>
                    </div>
                  </>
                )}
              </div>
            </aside>
          </div>
        )}
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
            if (tab === "training") {
              setModal("projects");
              return;
            }
            setLeft("models");
            setSearch("");
            setMobilePanel(
              mobilePanel === "library" && left === "models" ? null : "library",
            );
          }}
        >
          <FolderOpen size={18} />
          Projects
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
      {connectionPicker &&
        connectionPicker.projectId === project.id &&
        connectionPicker.scope === path.join("/") &&
        tab === "graph" &&
        !modal && (
          <ConnectionPicker
            position={connectionPicker.position}
            bounds={connectionPicker.bounds}
            choices={connectionChoices}
            project={project}
            path={path}
            sourceId={connectionPicker.source}
            sourceHandle={connectionPicker.sourceHandle}
            sourceLabel={
              graph.nodes.find((node) => node.id === connectionPicker.source)
                ?.data.label ?? "Output"
            }
            onClose={closeConnectionPicker}
            onChoose={(choice, targetHandle) => {
              const source = {
                id: connectionPicker.source,
                handle: connectionPicker.sourceHandle,
                targetHandle,
              };
              addGraphNode(choice.data, connectionPicker.position, source);
              setConnectionPicker(null);
            }}
          />
        )}
      {modal && (
        <Dialog
          key={modal}
          className={
            modal === "projects"
              ? "wide"
              : modal === "tokenizer"
                ? "tokenizer-dialog"
                : ""
          }
          label={
            modal === "tokenizer"
              ? "Tokenizer configuration"
              : modal === "variables"
                ? "Graph variables"
                : modal === "settings"
                  ? "Project settings"
                  : modal === "unsaved"
                    ? "Unsaved changes"
                    : modal === "new-project"
                      ? "New project"
                      : "Projects & examples"
          }
          onClose={() => {
            if (modal === "tokenizer" && saving) return;
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
                  ? `Tokenizer · ${tokenizerInput?.data.label ?? "Input"}`
                  : modal === "variables"
                    ? "Graph variables"
                    : modal === "projects"
                      ? "Projects & examples"
                      : modal === "new-project"
                        ? "New project"
                        : modal === "unsaved"
                          ? "Keep your changes?"
                          : "Project settings"}
              </h2>
            </div>
            <button
              disabled={modal === "tokenizer" && saving}
              onClick={() => setModal(null)}
              aria-label="Close dialog"
            >
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
                Save them before{" "}
                {pendingLoad?.options.newDraft ? "creating" : "opening"}{" "}
                {pendingLoad?.project.name}.
              </p>
              <div className="unsaved-actions">
                <button
                  className="primary"
                  disabled={saving}
                  onClick={async () => {
                    const next = pendingLoad;
                    if (next && (await saveProject()))
                      await loadProject(next.project, {
                        ...next.options,
                        force: true,
                      });
                  }}
                >
                  {pendingLoad?.options.newDraft
                    ? "Save & create"
                    : "Save & open"}
                </button>
                <button
                  onClick={() => {
                    const next = pendingLoad;
                    if (next)
                      loadProject(next.project, {
                        ...next.options,
                        force: true,
                      });
                  }}
                >
                  {pendingLoad?.options.newDraft
                    ? "Discard & create"
                    : "Discard & open"}
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
          ) : modal === "variables" ? (
            <GraphVariables
              values={variableScope.own}
              inherited={variableScope.inherited}
              uses={variableUses}
              hiddenNames={hiddenVariables}
              nested={path.length > 0}
              onChange={updateGraphVariables}
            />
          ) : modal === "tokenizer" ? (
            <TokenizerPanel
              key={project.id + ":" + tokenizerTarget}
              projectId={project.id}
              inputName={tokenizerInput?.data.label ?? "Input"}
              info={inputTokenizer}
              onApply={applyInputTokenizer}
              saving={saving}
              onClose={() => setModal(null)}
              onSave={async () => {
                if (!(await saveProject())) return false;
                setModal(null);
                notify("Tokenizer and project saved on server");
                return true;
              }}
            />
          ) : modal === "new-project" ? (
            <NewProjectForm
              templates={templates}
              starterId={starterId}
              onStarter={setStarterId}
              onCreate={(starter, name) => {
                void loadProject(
                  createProject(starter, name, "project-" + uid()),
                  { arrange: true, newDraft: true },
                );
              }}
              onCancel={() => setModal(null)}
            />
          ) : modal === "projects" ? (
            <ProjectBrowser
              templates={templates}
              saved={saved}
              activeId={project.id}
              draft={
                dirty && !saved.some((p) => p.id === project.id)
                  ? project
                  : null
              }
              tab={projectTab}
              onTab={setProjectTab}
              onNew={startNewProject}
              onOpen={(p) => {
                void loadProject(p, { saved: true });
              }}
              onView={viewReference}
              onDelete={(p) => {
                void deleteSavedProject(p);
              }}
              onResume={() => {
                setModal(null);
                setMobilePanel(null);
              }}
            />
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
      {connectionError && tab === "graph" && (
        <div className="connection-error" role="alert">
          <CircleAlert size={20} aria-hidden="true" />
          <div>
            <strong>{connectionError.title}</strong>
            {connectionError.messages.map((message) => (
              <p key={message}>{message}</p>
            ))}
            <small>
              Fix the block settings, remove the edge, or undo the connection
              (Ctrl+Z).
            </small>
          </div>
          <button
            aria-label="Dismiss connection error"
            onClick={() => setConnectionAttempt(null)}
          >
            <X size={18} />
          </button>
        </div>
      )}
      {toast && !connectionError && (
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
