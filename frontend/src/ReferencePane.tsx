import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  Panel,
  applyNodeChanges,
  useReactFlow,
  useViewport,
} from "@xyflow/react";
import type { Dispatch, SetStateAction } from "react";
import type { ReferenceLocation } from "./workspaceLocation";
import { retainedPath } from "./ui";
import type { NodeTypes } from "@xyflow/react";
import {
  Box,
  Copy,
  ChevronRight,
  LayoutGrid,
  LockKeyhole,
  Maximize2,
  RefreshCw,
  Tags,
  X,
} from "lucide-react";
import {
  describeError,
  projectSchema,
  reportSchema,
  responseData,
} from "./api";
import type { JsonValue, Project, ValidationReport } from "./types";
import { arrangeExample, computationOrder } from "./graph";
import {
  graphScope,
  parameterGroups,
  presentEdges,
  presentNodes,
} from "./graphPresentation";
import { arrangeGraph } from "./layout";
import { layoutSignature } from "./layout-core";
import { measureLabels } from "./labels";
import { edgeTypes, RoutingProvider } from "./edges";
import { GraphNavigator } from "./GraphNavigator";
import { isGraphEditShortcut } from "./ReferenceInspector";
import type { ReferenceInspection } from "./ReferenceInspector";
import { localizeLayerDefaults } from "./parameterScopes";

interface SavedReference {
  id: string;
  name: string;
}
interface ReferenceProps {
  templates: Project[];
  saved: SavedReference[];
  choice: string;
  onChoose: (value: string) => void;
  onClose: () => void;
  nodeTypes: NodeTypes;
  location: ReferenceLocation;
  onNavigate: Dispatch<SetStateAction<ReferenceLocation>>;
  onUseExample: (project: Project) => void;
  onInspect: (inspection: ReferenceInspection | null) => void;
  onActivate: () => void;
}
type ReferenceState =
  | { status: "loading"; choice: string }
  | { status: "error"; choice: string; message: string }
  | { status: "ready"; choice: string; project: Project };

/** A separate document and React Flow store: reference browsing cannot edit the workspace. */
export function ReferencePane({
  templates,
  saved,
  choice,
  onChoose,
  onClose,
  nodeTypes,
  location,
  onNavigate,
  onUseExample,
  onInspect,
  onActivate,
}: ReferenceProps) {
  const [state, setState] = useState<ReferenceState>({
    status: "loading",
    choice,
  });
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setState({ status: "loading", choice });
    async function load() {
      try {
        const template = templates.find((p) => "example:" + p.id === choice);
        const document =
          template ||
          projectSchema.parse(
            await responseData(
              await fetch(
                "/api/projects/" +
                  encodeURIComponent(choice.slice("saved:".length)),
                { signal: controller.signal },
              ),
            ),
          );
        if (!controller.signal.aborted)
          setState({
            status: "ready",
            choice,
            project: localizeLayerDefaults(document),
          });
      } catch (error) {
        if (!controller.signal.aborted)
          setState({ status: "error", choice, message: describeError(error) });
      }
    }
    void load();
    return () => controller.abort();
  }, [choice, templates, retry]);
  const ready =
    state.choice === choice && state.status === "ready" ? state.project : null;
  return (
    <section
      className="reference-pane"
      aria-label="Reference graph"
      data-reference-pane="true"
      tabIndex={0}
      onPointerDownCapture={(event) => {
        if (
          event.target instanceof Element &&
          !event.target.closest("button,input,select,a,textarea")
        ) {
          event.currentTarget.focus({ preventScroll: true });
        }
      }}
      onKeyDownCapture={(event) => {
        if (
          event.target instanceof Element &&
          event.target.closest("input,select,textarea")
        )
          return;
        if (isGraphEditShortcut(event)) {
          event.preventDefault();
          event.stopPropagation();
        }
      }}
    >
      <header className="reference-heading">
        <label>
          <span>
            <LockKeyhole size={12} aria-hidden="true" /> Reference · read-only
          </span>
          <select
            aria-label="Reference model"
            value={choice}
            onChange={(event) => onChoose(event.target.value)}
          >
            <optgroup label="Example architectures">
              {templates.map((p) => (
                <option key={p.id} value={"example:" + p.id}>
                  {p.name}
                </option>
              ))}
            </optgroup>
            {saved.length > 0 && (
              <optgroup label="Saved networks">
                {saved.map((p) => (
                  <option key={p.id} value={"saved:" + p.id}>
                    {p.name}
                  </option>
                ))}
              </optgroup>
            )}
            {choice.startsWith("saved:") &&
              !saved.some((p) => "saved:" + p.id === choice) && (
                <option value={choice}>
                  Saved reference (no longer listed)
                </option>
              )}
          </select>
        </label>
        {ready && templates.some((p) => p.id === ready.id) && (
          <button
            className="reference-use-example"
            aria-label={`Use this example: ${ready.name}`}
            title="Create project from this example"
            onClick={() => onUseExample(ready)}
          >
            <Copy size={14} aria-hidden="true" />
            <span>Use this example</span>
          </button>
        )}
        <button
          aria-label="Reload reference"
          title="Reload reference"
          onClick={() => setRetry((value) => value + 1)}
        >
          <RefreshCw size={14} aria-hidden="true" />
        </button>
        <button
          aria-label="Close reference pane"
          title="Close split view"
          onClick={onClose}
        >
          <X size={16} aria-hidden="true" />
        </button>
      </header>
      {ready ? (
        <ReactFlowProvider key={choice + ":" + retry}>
          <ReferenceCanvas
            document={ready}
            nodeTypes={nodeTypes}
            location={location}
            onNavigate={onNavigate}
            onInspect={onInspect}
            onActivate={onActivate}
          />
        </ReactFlowProvider>
      ) : (
        <div className="reference-placeholder" role="status">
          {state.choice === choice && state.status === "error" ? (
            <>
              <strong>Could not open this reference</strong>
              <p>{state.message}</p>
              <button onClick={() => setRetry((value) => value + 1)}>
                Try again
              </button>
            </>
          ) : (
            <p>Opening reference…</p>
          )}
        </div>
      )}
    </section>
  );
}
function ReferenceZoom() {
  const { zoom } = useViewport();
  const flow = useReactFlow();
  return (
    <button
      aria-label={`Reference zoom to 100 percent · current ${Math.round(zoom * 100)}%`}
      className="zoom-readout"
      title="Reference zoom to 100%"
      onClick={() => flow.zoomTo(1)}
    >
      {Math.round(zoom * 100)}%
    </button>
  );
}
function propertyValue(
  value: JsonValue | undefined,
  variables: Record<string, JsonValue | undefined>,
) {
  if (
    typeof value === "string" &&
    value.startsWith("$") &&
    value.slice(1) in variables
  )
    return `${value} → ${variables[value.slice(1)]}`;
  return typeof value === "string" ? value : JSON.stringify(value);
}
function scopeVariables(project: Project, path: string[]) {
  let graph = project.graph;
  let variables: Record<string, JsonValue | undefined> = {
    ...project.config,
    ...graph.variables,
  };
  for (const id of path) {
    const node = graph.nodes.find((node) => node.id === id);
    if (!node?.data.body) break;
    const overrides = Object.fromEntries(
      Object.entries(node.data.params).map(([key, value]) => [
        key,
        typeof value === "string" && value.startsWith("$")
          ? variables[value.slice(1)]
          : value,
      ]),
    );
    variables = { ...variables, ...overrides };
    graph = node.data.body;
    variables = { ...variables, ...graph.variables };
  }
  return variables;
}
function ReferenceCanvas({
  document,
  nodeTypes,
  location,
  onNavigate,
  onInspect,
  onActivate,
}: {
  document: Project;
  nodeTypes: NodeTypes;
  location: ReferenceLocation;
  onNavigate: Dispatch<SetStateAction<ReferenceLocation>>;
  onInspect: (inspection: ReferenceInspection | null) => void;
  onActivate: () => void;
}) {
  const [initial] = useState(() => {
    const project = structuredClone(document);
    const scopes = arrangeExample(project);
    return { project, scopes };
  });
  const [project, setProject] = useState(initial.project);
  const path = useMemo(
    () => retainedPath(project, location.blocks),
    [project, location.blocks],
  );
  const selection = location.node;
  const setPath = useCallback(
    (value: SetStateAction<string[]>) =>
      onNavigate((previous) => ({
        blocks: typeof value === "function" ? value(previous.blocks) : value,
        node: null,
      })),
    [onNavigate],
  );
  const setSelection = useCallback(
    (node: string | null) => onNavigate((previous) => ({ ...previous, node })),
    [onNavigate],
  );
  useEffect(() => {
    if (path.length !== location.blocks.length)
      onNavigate((previous) => ({ ...previous, blocks: path, node: null }));
  }, [path, location.blocks.length, onNavigate]);
  const [showLabels, setShowLabels] = useState(false);
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [error, setError] = useState("");
  const [arranging, setArranging] = useState(false);
  const pending = useRef(new Set(initial.scopes));
  const current = useRef(project),
    currentPath = useRef(path),
    currentSelection = useRef(selection);
  current.current = project;
  currentPath.current = path;
  currentSelection.current = selection;
  const mounted = useRef(false),
    request = useRef(0),
    host = useRef<HTMLDivElement>(null);
  const flow = useReactFlow();
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(document),
      signal: controller.signal,
    })
      .then(responseData)
      .then((data) => setReport(reportSchema.parse(data)))
      .catch((cause) => {
        if (!controller.signal.aborted)
          setError("Shape check unavailable: " + describeError(cause));
      });
    return () => controller.abort();
  }, [document]);
  const graph = graphScope(project, path);
  const prefix = path.length ? path.join("/") + "/" : "";
  const open = useCallback(
    (id: string) => {
      setPath((previous) => [...previous, id]);
      setSelection(null);
    },
    [setPath, setSelection],
  );
  const groups = useMemo(() => parameterGroups(project), [project]);
  const nodes = useMemo(
    () =>
      presentNodes(graph, {
        report,
        prefix,
        groups,
        selection,
        onOpen: open,
      }).map((node) => ({ ...node, selected: node.id === selection })),
    [graph, report, prefix, groups, selection, open],
  );
  const edges = useMemo(
    () => presentEdges(graph, { report, prefix, groups, selection }),
    [graph, report, prefix, groups, selection],
  );
  const focus = useCallback(() => {
    if (window.innerWidth > 760 && (host.current?.clientWidth ?? 0) >= 720) {
      void flow.fitView({ padding: 0.16 });
      return;
    }
    const graph = graphScope(current.current, currentPath.current);
    const node =
      graph.nodes.find((n) => n.id === currentSelection.current) ||
      computationOrder(graph)[0];
    if (node)
      void flow.fitView({
        nodes: [{ id: node.id }],
        padding: { top: "16px", bottom: "76px", left: "24px", right: "24px" },
        minZoom: 0.3,
        maxZoom: 0.95,
      });
  }, [flow]);
  useEffect(() => {
    const element = host.current;
    if (!element) return;
    let timer: ReturnType<typeof setTimeout>;
    const observer = new ResizeObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(focus, 100);
    });
    observer.observe(element);
    timer = setTimeout(focus, 100);
    return () => {
      observer.disconnect();
      clearTimeout(timer);
    };
  }, [path, focus]);
  const arrangeScope = useCallback(
    async (scope: string[]) => {
      const number = ++request.current;
      const snapshot = structuredClone(graphScope(current.current, scope));
      const signature = layoutSignature(snapshot);
      const prefix = scope.length ? scope.join("/") + "/" : "";
      const texts = snapshot.edges
        .map((edge) => ({
          id: edge.id,
          text:
            report?.shapes[prefix + edge.source]?.shape
              .map(String)
              .join(" × ") || "",
        }))
        .filter((label) => label.text);
      const sizes = measureLabels(texts);
      const labels = new Map(
        texts.flatMap((label) => {
          const size = sizes.get(label.id);
          return size
            ? [[label.id, { ...size, text: label.text }] as const]
            : [];
        }),
      );
      setArranging(true);
      try {
        const result = await arrangeGraph(snapshot, labels);
        if (
          !mounted.current ||
          layoutSignature(graphScope(current.current, scope)) !== signature
        )
          return;
        setProject((previous) => {
          const next = structuredClone(previous),
            target = graphScope(next, scope);
          target.nodes = result.nodes;
          target.layout = result.layout;
          return next;
        });
        if (currentPath.current.join("/") === scope.join("/"))
          requestAnimationFrame(() => {
            if (mounted.current) focus();
          });
      } catch (cause) {
        if (mounted.current)
          setError("Reference arrangement failed: " + describeError(cause));
      } finally {
        if (mounted.current && number === request.current) setArranging(false);
      }
    },
    [report, focus],
  );
  useEffect(() => {
    const scope = path.join("/");
    if (
      !report ||
      !pending.current.has(scope) ||
      !graph.nodes.length ||
      !graph.nodes.every((n) => n.measured?.width && n.measured?.height)
    )
      return;
    pending.current.delete(scope);
    void arrangeScope(path);
  }, [graph, path, report, arrangeScope]);
  const breadcrumbs = useMemo(() => {
    let graph = project.graph;
    return path.map((id) => {
      const node = graph.nodes.find((n) => n.id === id);
      if (node?.data.body) graph = node.data.body;
      return node?.data.label || id;
    });
  }, [project, path]);
  const selected = graph.nodes.find((n) => n.id === selection);
  useEffect(() => {
    const variables = scopeVariables(project, path);
    const tokenizer = selected
      ? (report?.tokenizers?.[prefix + selected.id] ?? selected.data.tokenizer)
      : undefined;
    if (tokenizer) {
      variables.vocab = tokenizer.capacity;
      if (tokenizer.mask_id != null) variables.mask_id = tokenizer.mask_id;
    }
    onInspect(
      selected
        ? {
            id: selected.id,
            scope: prefix,
            label: selected.data.label,
            kind: selected.data.kind,
            projectName: project.name,
            description: selected.data.description,
            properties: [
              ...Object.entries(selected.data.params).map(([name, value]) => ({
                name,
                value: propertyValue(value, variables) ?? "",
              })),
              ...(selected.data.kind === "input" && tokenizer
                ? [
                    {
                      name: "Tokenizer",
                      value:
                        tokenizer.repo || tokenizer.class || "Local tokenizer",
                    },
                    {
                      name: "Vocabulary capacity",
                      value: String(tokenizer.capacity),
                    },
                  ]
                : []),
            ],
            shape: report?.shapes[prefix + selected.id] ?? null,
            parameters: report?.counts[prefix + selected.id] ?? 0,
            share: selected.data.share,
          }
        : null,
    );
  }, [selected, prefix, project, path, report, onInspect]);
  useEffect(() => () => onInspect(null), [onInspect]);
  return (
    <>
      <div className="canvas-toolbar reference-toolbar">
        <div className="breadcrumbs" aria-label="Reference block path">
          <button
            onClick={() => {
              setPath([]);
              setSelection(null);
            }}
          >
            <Box size={13} aria-hidden="true" /> Model
          </button>
          {breadcrumbs.map((label, index) => (
            <span key={path.slice(0, index + 1).join("/")}>
              <ChevronRight size={12} aria-hidden="true" />
              <button
                onClick={() => {
                  setPath(path.slice(0, index + 1));
                  setSelection(null);
                }}
              >
                {label}
              </button>
            </span>
          ))}
        </div>
        <div className="graph-actions">
          <button
            aria-label="Auto arrange reference"
            title="Arrange reference view"
            disabled={arranging}
            onClick={() => void arrangeScope(path)}
          >
            <LayoutGrid size={15} aria-hidden="true" />
          </button>
          <button
            aria-label={
              showLabels ? "Hide reference labels" : "Show reference labels"
            }
            aria-pressed={showLabels}
            onClick={() => setShowLabels((value) => !value)}
          >
            <Tags size={15} aria-hidden="true" />
          </button>
          <button
            aria-label="Fit reference graph"
            title="Fit reference graph"
            onClick={() => void flow.fitView({ padding: 0.15 })}
          >
            <Maximize2 size={15} aria-hidden="true" />
          </button>
          <ReferenceZoom />
        </div>
      </div>
      {error && (
        <p className="reference-error" role="status">
          {error}
        </p>
      )}
      <div className="flow-wrap reference-flow" ref={host}>
        <RoutingProvider
          nodes={nodes}
          edges={graph.edges}
          layout={graph.layout}
          showLabels={showLabels}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            onNodesChange={(changes) => {
              const dimensions = changes.filter(
                (change) => change.type === "dimensions",
              );
              if (dimensions.length)
                setProject((previous) => {
                  const next = structuredClone(previous),
                    target = graphScope(next, path);
                  target.nodes = applyNodeChanges(dimensions, target.nodes);
                  return next;
                });
            }}
            nodesDraggable={false}
            nodesConnectable={false}
            edgesReconnectable={false}
            deleteKeyCode={null}
            selectionKeyCode={null}
            multiSelectionKeyCode={null}
            onNodeClick={(_, node) => {
              setSelection(node.id);
              onActivate();
            }}
            onNodeDoubleClick={(_, node) => {
              if (node.data.body) open(node.id);
            }}
            onPaneClick={() => setSelection(null)}
            minZoom={0.04}
            maxZoom={1.7}
            proOptions={{ hideAttribution: true }}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={22}
              size={1}
              color="var(--graph-dots, #ced6cc)"
            />
            <Controls showInteractive={false} />
            <Panel position="bottom-center">
              <GraphNavigator
                graph={graph}
                selection={selection}
                onFocus={(id) => {
                  setSelection(id);
                }}
                compact
              />
            </Panel>
          </ReactFlow>
        </RoutingProvider>
      </div>
      <div className="reference-status">
        <span>
          {graph.nodes.length} nodes · {graph.edges.length} connections
        </span>
        <span>Read-only</span>
      </div>
    </>
  );
}
