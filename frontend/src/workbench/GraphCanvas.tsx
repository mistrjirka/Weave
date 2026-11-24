import type { ComponentProps, DragEventHandler, RefObject } from "react";
import {
  Background, BackgroundVariant, Controls, MiniMap, Panel, ReactFlow,
  SelectionMode, type ReactFlowProps,
} from "@xyflow/react";
import type { Edge } from "@xyflow/react";
import { canvasFitPadding } from "../canvasFitPadding";
import { GraphAnnotations } from "../GraphAnnotations";
import { GraphNavigator } from "../GraphNavigator";
import { GraphVariableCard } from "../GraphVariableCard";
import { RoutingProvider, edgeTypes } from "../edges";
import { isCompactViewport } from "../compactViewport";
import type { Graph, GraphNode } from "../types";
import { nodeTypes } from "./ArchitectureNode";
import { family } from "./presentation";

type FlowHandlers = Pick<ReactFlowProps<GraphNode, Edge>,
  "onNodesChange" | "onEdgesChange" | "onConnect" | "onConnectEnd" |
  "onNodeClick" | "onNodeDoubleClick" | "onEdgeClick" | "onPaneClick" |
  "onNodeDragStart" | "onBeforeDelete">;

type Props = FlowHandlers & {
  canvasRef: RefObject<HTMLDivElement | null>;
  onDrop: DragEventHandler<HTMLDivElement>;
  graph: Graph;
  displayNodes: GraphNode[];
  displayEdges: Edge[];
  showLabels: boolean;
  showAnnotations: boolean;
  splitView: boolean;
  onAnnotationEdit: (id: string) => void;
  variables: ComponentProps<typeof GraphVariableCard>;
  navigator: Pick<ComponentProps<typeof GraphNavigator>, "selection" | "onFocus">;
};

export function GraphCanvas({ canvasRef, onDrop, graph, displayNodes, displayEdges,
  showLabels, showAnnotations, splitView, onAnnotationEdit, variables, navigator,
  ...handlers }: Props) {
  return (
    <div
      ref={canvasRef}
      className="flow-wrap"
      tabIndex={0}
      aria-label="Editable graph. Drag to select nodes; Ctrl or Command-click to extend selection. Copy and paste with Ctrl or Command C and V. Hold Space to pan."
      onPointerDownCapture={(event) => {
        if (event.target instanceof Element && !event.target.closest("input,textarea,select,[contenteditable],button"))
          event.currentTarget.focus({ preventScroll: true });
      }}
      onDragOver={(event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
      }}
      onDrop={onDrop}
    >
      <RoutingProvider nodes={displayNodes} edges={graph.edges} layout={graph.layout} showLabels={showLabels}>
        <ReactFlow
          id="editable-flow"
          nodes={displayNodes}
          edges={displayEdges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          {...handlers}
          fitView={!isCompactViewport() && !splitView}
          fitViewOptions={{ padding: canvasFitPadding(canvasRef.current) }}
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
          <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="var(--graph-dots, #ced6cc)" />
          <Controls showInteractive={false} fitViewOptions={{ padding: canvasFitPadding(canvasRef.current) }} />
          <MiniMap<GraphNode>
            nodeColor={(n) => ({
              mint: "#8ab7a2", violet: "#afa1ce", blue: "#8cabc8", amber: "#ceb47c",
              coral: "#d49a8a", neutral: "#b2beb3", red: "#b86767",
            })[family(n.data.kind)] || "#b2beb3"}
            maskColor="var(--graph-minimap-mask, rgba(247,249,245,.7))"
            pannable zoomable
          />
          <GraphAnnotations nodes={graph.nodes} visible={showAnnotations} onEdit={onAnnotationEdit} />
          <Panel position="top-left" className="variables-canvas-panel">
            <GraphVariableCard {...variables} />
          </Panel>
          <Panel position="bottom-center">
            <GraphNavigator graph={graph} {...navigator} compact={splitView} fitPadding={() => canvasFitPadding(canvasRef.current)} />
          </Panel>
        </ReactFlow>
      </RoutingProvider>
    </div>
  );
}
