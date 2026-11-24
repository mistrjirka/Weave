import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  Panel,
  applyNodeChanges,
  useReactFlow,
  SelectionMode,
  type NodeTypes,
} from "@xyflow/react";
import { isCompactViewport } from "../compactViewport";
import { edgeContract } from "../lanePorts";
import type { ReferenceLocation } from "../workspaceLocation";
import { retainedPath } from "../ui";
import { describeError, reportSchema, responseData } from "../api";
import type { Project, ValidationReport } from "../types";
import { arrangeExample, computationOrder } from "../graph";
import {
  graphScope,
  clearGraphSelection,
  parameterGroups,
  presentEdges,
  presentNodes,
} from "../graphPresentation";
import { arrangeGraph } from "../layout";
import { layoutSignature } from "../layout-core";
import { measureLabels } from "../labels";
import { edgeTypes, RoutingProvider } from "../edges";
import { GraphNavigator } from "../GraphNavigator";
import { GraphAnnotations } from "../GraphAnnotations";
import type { ReferenceInspection } from "../ReferenceInspector";
import { graphClipboardType, serializeSelection } from "../graphClipboard";
import { graphVariableScope } from "../graphVariables";
import { ReferenceVariables } from "./ReferenceVariables";
import { ReferenceToolbar, referenceFitPadding } from "./ReferenceToolbar";
import {
  scopeVariables,
  useReferenceInspection,
} from "./useReferenceInspection";

export function ReferenceCanvas({
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
    () => retainedPath(document, location.blocks),
    [document, location.blocks],
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
  const [showAnnotations, setShowAnnotations] = useState(true);
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
        variables: graphVariableScope(project, path).bindings,
        selection,
        onOpen: open,
      }),
    [graph, report, prefix, groups, selection, open, project, path],
  );
  const edges = useMemo(
    () => presentEdges(graph, { report, prefix, groups, selection }),
    [graph, report, prefix, groups, selection],
  );
  const focus = useCallback(() => {
    if (!isCompactViewport() && (host.current?.clientWidth ?? 0) >= 720) {
      const notes =
        showAnnotations &&
        graphScope(current.current, currentPath.current).nodes.some((node) =>
          node.data.annotation?.trim(),
        );
      void flow.fitView({ padding: notes ? referenceFitPadding(true) : 0.16 });
      return;
    }
    const graph = graphScope(current.current, currentPath.current);
    const node =
      graph.nodes.find((n) => n.id === currentSelection.current) ||
      computationOrder(graph)[0];
    if (node)
      void flow.fitView({
        nodes: [{ id: node.id }],
        padding: referenceFitPadding(
          showAnnotations &&
            graph.nodes.some((item) => item.data.annotation?.trim()),
        ),
        minZoom: 0.3,
        maxZoom: 0.95,
      });
  }, [flow, showAnnotations]);
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
            edgeContract(report, prefix, edge)?.shape.map(String).join(" × ") ||
            "",
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
  // Selection is view state only; the reference computation stays read-only.
  const scopeKey = path.join("/");
  useEffect(() => {
    setProject((previous) => {
      const graph = graphScope(previous, currentPath.current);
      const chosen = graph.nodes.filter((node) => node.selected);
      if (
        selection === null
          ? !chosen.length
          : chosen.some((node) => node.id === selection)
      )
        return previous;
      const next = structuredClone(previous);
      clearGraphSelection(next.graph);
      const node = graphScope(next, currentPath.current).nodes.find(
        (node) => node.id === selection,
      );
      if (node) node.selected = true;
      return next;
    });
  }, [selection, scopeKey]);
  const [copyNotice, setCopyNotice] = useState("");
  useEffect(() => {
    if (!copyNotice) return;
    const timer = setTimeout(() => setCopyNotice(""), 3000);
    return () => clearTimeout(timer);
  }, [copyNotice]);
  const referenceClipboard = () =>
    serializeSelection(
      graphScope(current.current, currentPath.current),
      currentSelection.current,
      scopeVariables(current.current, currentPath.current),
    );
  useEffect(() => {
    const onCopy = (event: ClipboardEvent) => {
      const target = event.target;
      if (
        event.defaultPrevented ||
        !event.clipboardData ||
        !(target instanceof Element) ||
        !target.closest("[data-reference-pane]") ||
        target.closest("input,textarea,select,[contenteditable]") ||
        !(window.getSelection()?.isCollapsed ?? true)
      )
        return;
      const text = referenceClipboard();
      if (!text) return;
      event.clipboardData.setData(graphClipboardType, text);
      event.clipboardData.setData("text/plain", text);
      event.preventDefault();
      setCopyNotice("Copied · paste into your graph");
    };
    window.addEventListener("copy", onCopy);
    return () => window.removeEventListener("copy", onCopy);
  }, []);
  const selected = graph.nodes.find((n) => n.id === selection);
  useReferenceInspection({
    selected,
    prefix,
    project,
    path,
    report,
    onInspect,
  });
  useEffect(() => () => onInspect(null), [onInspect]);
  return (
    <>
      <ReferenceToolbar
        path={path}
        breadcrumbs={breadcrumbs}
        onNavigate={onNavigate}
        setPath={setPath}
        setSelection={setSelection}
        selection={selection}
        graph={graph}
        arranging={arranging}
        arrangeScope={arrangeScope}
        showLabels={showLabels}
        setShowLabels={setShowLabels}
        showAnnotations={showAnnotations}
        setShowAnnotations={setShowAnnotations}
        referenceClipboard={referenceClipboard}
        setCopyNotice={setCopyNotice}
      />
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
              const viewChanges = changes.filter(
                (change) =>
                  change.type === "dimensions" || change.type === "select",
              );
              if (!viewChanges.length) return;
              const next = structuredClone(current.current);
              const target = graphScope(next, currentPath.current);
              target.nodes = applyNodeChanges(viewChanges, target.nodes);
              current.current = next;
              setProject(next);
              if (viewChanges.some((change) => change.type === "select")) {
                const selected = target.nodes.filter((node) => node.selected);
                const previous = currentSelection.current;
                const focused = selected.some((node) => node.id === previous)
                  ? previous
                  : (selected.at(-1)?.id ?? null);
                if (focused !== previous) setSelection(focused);
              }
            }}
            nodesDraggable={false}
            nodesConnectable={false}
            edgesReconnectable={false}
            deleteKeyCode={null}
            selectionKeyCode="Shift"
            multiSelectionKeyCode={["Control", "Meta", "Shift"]}
            selectionOnDrag
            selectionMode={SelectionMode.Partial}
            panOnDrag={[1, 2]}
            panActivationKeyCode="Space"
            onNodeClick={(_, node) => {
              if (
                graphScope(current.current, currentPath.current).nodes.some(
                  (item) => item.id === node.id && item.selected,
                )
              )
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
            <Controls
              showInteractive={false}
              fitViewOptions={{
                padding: referenceFitPadding(
                  showAnnotations &&
                    graph.nodes.some((node) => node.data.annotation?.trim()),
                ),
              }}
            />
            <GraphAnnotations nodes={graph.nodes} visible={showAnnotations} />
            <ReferenceVariables project={project} path={path} report={report} />
            <Panel position="bottom-center">
              <GraphNavigator
                graph={graph}
                selection={selection}
                onFocus={(id) => {
                  setSelection(id);
                }}
                compact
                fitPadding={() =>
                  referenceFitPadding(
                    showAnnotations &&
                      graph.nodes.some((node) => node.data.annotation?.trim()),
                  )
                }
              />
            </Panel>
          </ReactFlow>
        </RoutingProvider>
      </div>
      <div className="reference-status">
        <span>
          {graph.nodes.length} nodes · {graph.edges.length} connections
        </span>
        <span role={copyNotice ? "status" : undefined}>
          {copyNotice || "Read-only · select and copy"}
        </span>
      </div>
    </>
  );
}
