import { useDocumentState } from "./useDocumentState";
import { useNotice } from "./useNotice";
import { useValidation } from "./useValidation";
import { useArrangement } from "./useArrangement";
import { useGraphClipboard } from "./useGraphClipboard";
import { useProjectActions } from "./useProjectActions";
import { useCanvasMutations } from "./useCanvasMutations";
import { useCanvasNavigation, type ConnectionPickerState } from "./useCanvasNavigation";
import { useWorkspaceNavigation } from "./useWorkspaceNavigation";
import { useSelectionActions } from "./useSelectionActions";
import { useNodeInsertion } from "./useNodeInsertion";
import { useCanvasViewport } from "./useCanvasViewport";
import { useCustomBlocks } from "./useCustomBlocks";
import { useWorkbenchShortcuts } from "./useWorkbenchShortcuts";
import { useInspectorData } from "./useInspectorData";
import { workbenchNavigationItems } from "./navigationItems";
import { graphAt } from "./document";
import { resizeExperts } from "./catalogNodeData";
import { DialogController } from "./dialogs/DialogController";
import { WorkbenchLibrary } from "./WorkbenchLibrary";
import { WorkbenchHeader } from "./WorkbenchHeader";
import { GraphCanvas } from "./GraphCanvas";
import { CanvasToolbar } from "./CanvasToolbar";
import { ChecksDrawer } from "./ChecksDrawer";
import { WorkbenchViews } from "./WorkbenchViews";
import { ValidationBar } from "./ValidationBar";
import { MobileTools } from "./MobileTools";
import { ProjectImport } from "./ProjectImport";
import { WorkbenchNotifications } from "./WorkbenchNotifications";
import { GeneratedCode } from "./GeneratedCode";
import { ProjectReferences } from "./ProjectReferences";
import { Inspector } from "./inspector/Inspector";
import { nodeTypes } from "./ArchitectureNode";
import { motion } from "./util";
import { isCompactViewport } from "../compactViewport";
import type { ConnectionAttempt } from "../connectionFeedback";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { TrainingWorkspace } from "../TrainingWorkspace";
import {
  useReactFlow,
} from "@xyflow/react";
import { Workflow } from "lucide-react";
import { ConnectionPicker } from "../ConnectionPicker";
import { graphVariableScope } from "../graphVariables";
import { retainedPath } from "../ui";
import type { InsertionKind } from "../graphOperations";
import {
  parameterGroups,
  presentNodes,
  presentEdges,
  clearGraphSelection,
} from "../graphPresentation";
import { ReferencePane } from "../ReferencePane";
import { SplitDivider } from "../SplitDivider";
import type {
  OpenProjectOptions,
  SavedProject,
} from "../projectWorkflow";
import type { ReferenceLocation } from "../workspaceLocation";
import { catalog } from "../catalog";
import {
  allNodes,
  arrangeExample,
} from "../graph";

import type { Edge } from "@xyflow/react";
import type {
  Project,
  Graph,
  ValidationIssue,
  TokenizerInfo,
} from "../types";
const copy = <T,>(v: T): T => structuredClone(v);
export function Workbench({
  initial,
  initialTemplates,
  initialSaved,
  initiallyArrange,
  initialDirty,
}: {
  initial: Project;
  initialTemplates: Project[];
  initialSaved: SavedProject[];
  initiallyArrange: boolean;
  initialDirty: boolean;
}) {
  const { toast, setToast, notify } = useNotice();
  const { project, saved, setSaved, dirty, history, setHistory, future, setFuture, saving, loadRequest, ref, change, undo, redo, dirtyAfterSave, saveProject, loadDocument } = useDocumentState(initial, initialSaved, initialDirty, notify, () => { editRevision.current++; pendingArrangement.current.delete(viewPath.current.join("/")); }, (next) => { setPath((path) => retainedPath(next, path)); setSelection(null); editRevision.current++; });
  const [templates] = useState(initialTemplates);
  const [connectionAttempt, setConnectionAttempt] =
    useState<ConnectionAttempt | null>(null);
  const { path, setPath, selection, setSelection, tab, setTab, left, setLeft, search, setSearch, modal, setModal, librarySearchRef, projectTab, setProjectTab, starterId, setStarterId, mobilePanel, setMobilePanel, visibleMobilePanel, sidebarMenuButton, splitView, setSplitView, splitRatio, setSplitRatio, referenceChoice, setReferenceChoice, referenceLocation, setReferenceLocation, referenceInspection, setReferenceInspection, splitButton, comparisonStyle, closeSplitView } = useWorkspaceNavigation(initial, initialTemplates, project.id);
  const { customBlocks, setCustomBlocks } = useCustomBlocks();
  const [checksOpen, setChecksOpen] = useState(false);
  const [showLabels, setShowLabels] = useState(false),
    [showAnnotations, setShowAnnotations] = useState(true);
  const editRevision = useRef(0);
  const [connectionPicker, setConnectionPicker] = useState<ConnectionPickerState | null>(null);
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
  const modalRef = useRef(modal);
  modalRef.current = modal;
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
  const { canvasRef, fitGraph, fitCanvas } = useCanvasViewport(reactFlow, ref, viewPath, focusedNode, tab, path, splitView);
  const fileRef = useRef<HTMLInputElement>(null);
  const { report, source, validationError, connectionError, clearValidation } = useValidation(project, connectionAttempt);
  const graph = project ? graphAt(project, path) : { nodes: [], edges: [] };
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
  const { selected, tokenizerInput, inputTokenizer, variableScope, variableUses, hiddenVariables, updateGraphVariables, selectedTokenizer, selectedVocabField, selectedAutoVocab, fieldVariables, transformer, selectedFieldVariables, selectedCount, connection, referenceDetails, connectionShape, updateNode } = useInspectorData(project, graph, path, selection, selectedEdge, modal, tokenizerTarget, prefix, report, splitView, referenceInspection, referenceLocation, change, changeGraph);
  useGraphClipboard(project, path, modal, tab, canvasRef, ref, viewPath, focusedNode, reactFlow, changeGraph, setSelection, setSelectedEdge, notify);
  const canvasMutations = useCanvasMutations(project.id, path, changeGraph, setSelection, setSelectedEdge, setConnectionAttempt);
  const canvasNavigation = useCanvasNavigation(project, path, canvasRef, ref, viewPath, setSelection, setSelectedEdge, setMobilePanel, setConnectionPicker, setHistory, setFuture, open);
  const { addNode, addGraphNode, connectionChoices, closeConnectionPicker } = useNodeInsertion(project, path, templates, customBlocks, connectionPicker, setConnectionPicker, canvasRef, reactFlow, changeGraph, setSelection, setSelectedEdge, setMobilePanel, notify);
  const { arranging, arrangeScope } = useArrangement(ref, editRevision, viewPath, report, change, fitCanvas, notify);
  function autoLayout() {
    void arrangeScope(path);
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
    return loadDocument(p, options, (next) => {
      editRevision.current++;
      pendingArrangement.current = new Set(options.arrange ? arrangeExample(next) : []);
      setPath([]);
      setSelection(null);
      setSelectedEdge(null);
    }, () => {
      setModal(null);
      setPendingLoad(null);
      setMobilePanel(null);
      clearValidation();
      setTab("graph");
      setTimeout(() => fitCanvas(), 100);
    });
  }
  function startNewProject(starter = "blank") {
    loadRequest.current++;
    setStarterId(starter);
    setMobilePanel(null);
    setModal("new-project");
  }
  function viewReference(example: Project) {
    loadRequest.current++;
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
  const { deletedProject, busy, deleteSavedProject, restoreDeletedProject, exportCode } = useProjectActions(project, setSaved, setChecksOpen, notify);
  const { removeSelected, duplicate, duplicateShared, unrollSelected, insertOnConnection, groupSelectedNodes } = useSelectionActions(project, path, templates, graph, selection, selected, connection, connectionShape, insertKind, report, prefix, changeGraph, change, setSelection, setSelectedEdge, fitGraph, notify);
  useWorkbenchShortcuts(modal, sidebarMenuButton, () => setMobilePanel(null), saveProject, undo, redo, groupSelectedNodes);
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
  const { crumbs, groups } = workbenchNavigationItems(project, path, search);
  return (
    <div className={"app-shell" + (tab === "training" ? " training-mode" : "")}>
      <a className="skip-link" href="#workspace">
        Skip to editor
      </a>
      {visibleMobilePanel && (
        <button
          className="drawer-backdrop"
          aria-label="Close side panel"
          onClick={() => {
            setMobilePanel(null);
            if (mobilePanel === "library") sidebarMenuButton.current?.focus();
          }}
        />
      )}
      <WorkbenchLibrary
        mobilePanel={mobilePanel} setMobilePanel={setMobilePanel}
        sidebarMenuButton={sidebarMenuButton} librarySearchRef={librarySearchRef}
        setModal={setModal} tab={tab} setTab={setTab} project={project}
        left={left} setLeft={setLeft} search={search} setSearch={setSearch}
        groups={groups} addNode={addNode} customBlocks={customBlocks}
        addGraphNode={addGraphNode} saved={saved} loadProject={loadProject}
        setProjectTab={setProjectTab} templates={templates}
        viewReference={viewReference} startNewProject={startNewProject}
      />
      <main className="main-workspace" id="workspace" tabIndex={-1}>
        <WorkbenchHeader
          sidebarMenuButton={sidebarMenuButton}
          sidebarOpen={mobilePanel === "library"}
          onToggleSidebar={() => {
            setMobilePanel(mobilePanel === "library" ? null : "library");
            requestAnimationFrame(() => document.querySelector<HTMLButtonElement>("#workspace-sidebar .drawer-close")?.focus());
          }}
          projectName={project.name}
          projectStatus={saved.some((p) => p.id === project.id) ? dirty ? "Unsaved changes" : "Saved on server" : templates.some((p) => p.id === project.id) ? dirty ? "Unsaved example changes" : "Example · use as starting point" : "Draft · save to keep on server"}
          onSettings={() => setModal("settings")}
          onNew={() => startNewProject()}
          onImport={() => fileRef.current?.click()}
          onSave={saveProject}
          onExport={exportCode}
          saving={saving}
          busy={busy}
        />
        <WorkbenchViews tab={tab} onTab={setTab} project={project} splitButton={splitButton} splitView={splitView} onToggleSplit={() => { setSplitView((value) => !value); if (!splitView) setTab("graph"); }} />
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
                <CanvasToolbar path={path} crumbs={crumbs} goUp={goUp} onRoot={() => { setPath([]); setSelection(null); setTimeout(() => fitGraph(), 80); }} onCrumb={(i) => { setPath(path.slice(0, i + 1)); setSelection(null); setTimeout(() => fitGraph(), 80); }} canUndo={!!history.length} canRedo={!!future.length} undo={undo} redo={redo} groupSelectedNodes={groupSelectedNodes} onVariables={() => setModal("variables")} arranging={arranging} autoLayout={autoLayout} showLabels={showLabels} setShowLabels={setShowLabels} showAnnotations={showAnnotations} setShowAnnotations={setShowAnnotations} onFit={() => fitGraph({ duration: motion(250) })} />
                {tab === "graph" ? (
                  <GraphCanvas
                    canvasRef={canvasRef}
                    graph={graph}
                    displayNodes={displayNodes}
                    displayEdges={displayEdges}
                    showLabels={showLabels}
                    showAnnotations={showAnnotations}
                    splitView={splitView}
                    onDrop={(e) => {
                      e.preventDefault();
                      const kind = e.dataTransfer.getData("application/weave");
                      const entry = catalog.find((c) => c.label === kind);
                      if (entry) addNode(entry, { x: e.clientX, y: e.clientY });
                    }}
                    onAnnotationEdit={(id) => {
                      setSelection(id);
                      if (isCompactViewport()) setMobilePanel("properties");
                    }}
                    variables={{
                      values: variableScope.own,
                      inherited: variableScope.inherited,
                      uses: variableUses,
                      hiddenNames: hiddenVariables,
                      nested: path.length > 0,
                      onChange: updateGraphVariables,
                      onOpen: () => setModal("variables"),
                    }}
                    navigator={{ selection, onFocus: setSelection }}
                    onNodesChange={canvasMutations.onNodesChange}
                    onEdgesChange={canvasMutations.onEdgesChange}
                    onConnect={canvasMutations.onConnect}
                    onConnectEnd={canvasNavigation.onConnectEnd}
                    onNodeClick={canvasNavigation.onNodeClick}
                    onNodeDoubleClick={canvasNavigation.onNodeDoubleClick}
                    onEdgeClick={canvasNavigation.onEdgeClick}
                    onPaneClick={canvasNavigation.onPaneClick}
                    onNodeDragStart={canvasNavigation.onNodeDragStart}
                    onBeforeDelete={canvasMutations.onBeforeDelete}
                  />
                ) : tab === "code" ? (
                  <GeneratedCode source={source} notify={notify} />
                ) : (
                  <ProjectReferences project={project} />
                )}
                <ValidationBar report={report} validationError={validationError} nodeCount={graph.nodes.length} edgeCount={graph.edges.length} checksOpen={checksOpen} onToggle={() => setChecksOpen(!checksOpen)} />
                {checksOpen && (
                  <ChecksDrawer report={report} validationError={validationError} onClose={() => setChecksOpen(false)} focusError={focusError} />
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
                      if (isCompactViewport()) setMobilePanel("properties");
                    }}
                    onUseExample={(p) => startNewProject(p.id)}
                    nodeTypes={nodeTypes}
                    onClose={closeSplitView}
                  />
                </>
              )}
            </div>
            <Inspector
              mobilePanel={mobilePanel} referenceDetails={referenceDetails} selected={selected} selectedCount={selectedCount} connection={connection}
              onClosePanel={() => setMobilePanel(null)}
              onClearSelection={() => { setSelection(null); change((project) => clearGraphSelection(project.graph), false, false); setReferenceLocation((previous) => ({ ...previous, node: null })); }}
              updateNode={updateNode} project={project} path={path} open={open} notify={notify} unrollSelected={unrollSelected} duplicateShared={duplicateShared} fitGraph={fitGraph}
              selectedTokenizer={selectedTokenizer} selectedVocabField={selectedVocabField} selectedAutoVocab={selectedAutoVocab}
              onChooseTokenizer={(id) => { setTokenizerTarget(prefix + id); setModal("tokenizer"); }}
              transformer={transformer} fieldVariables={fieldVariables} selectedFieldVariables={selectedFieldVariables} variableScope={variableScope} report={report} prefix={prefix} resizeExperts={resizeExperts}
              graph={graph} focusError={focusError} duplicate={duplicate} removeSelected={removeSelected}
              onSaveBlock={(data) => { setCustomBlocks((bs) => [...bs, copy(data)]); notify("Block saved to your library"); }}
              connectionShape={connectionShape} insertKind={insertKind} setInsertKind={setInsertKind} insertOnConnection={insertOnConnection} validationError={validationError}
            />
          </div>
        )}
      </main>
      <MobileTools mobilePanel={mobilePanel} libraryBlocks={left === "blocks"} onBlocks={() => { setLeft("blocks"); setSearch(""); setMobilePanel(mobilePanel === "library" && left === "blocks" ? null : "library"); }} onProjects={() => { setMobilePanel(null); setProjectTab("projects"); setModal("projects"); }} onProperties={() => setMobilePanel(mobilePanel === "properties" ? null : "properties")} />
      <ProjectImport fileRef={fileRef} loadProject={loadProject} notify={notify} />
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
      <DialogController modal={modal} saving={saving} loadRequest={loadRequest} setModal={setModal} setPendingLoad={setPendingLoad} pendingLoad={pendingLoad} project={project} saveProject={saveProject} modalRef={modalRef} dirtyAfterSave={dirtyAfterSave} loadProject={loadProject} projectRef={ref} notify={notify} variableScope={variableScope} variableUses={variableUses} hiddenVariables={hiddenVariables} path={path} updateGraphVariables={updateGraphVariables} tokenizerTarget={tokenizerTarget} tokenizerInput={tokenizerInput} inputTokenizer={inputTokenizer} applyInputTokenizer={applyInputTokenizer} templates={templates} starterId={starterId} setStarterId={setStarterId} deletedProject={deletedProject} restoreDeletedProject={restoreDeletedProject} saved={saved} dirty={dirty} projectTab={projectTab} setProjectTab={setProjectTab} startNewProject={startNewProject} viewReference={viewReference} deleteSavedProject={deleteSavedProject} setMobilePanel={setMobilePanel} change={change} />
      <WorkbenchNotifications connectionError={connectionError} graphTab={tab === "graph"} toast={toast} canUndoDelete={!!deletedProject && !modal} onDismissConnection={() => setConnectionAttempt(null)} onDismissToast={() => setToast("")} onUndoDelete={restoreDeletedProject} />
    </div>
  );
}
