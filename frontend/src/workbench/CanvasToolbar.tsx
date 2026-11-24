import React from "react";
import { ArrowUp, Box, Braces, ChevronRight, LayoutGrid, Maximize2, MessageSquareText, Redo2, Tags, Undo2 } from "lucide-react";
import { ZoomReadout } from "./ArchitectureNode";

type Props = {
  path: string[]; crumbs: string[]; goUp: () => void; onRoot: () => void; onCrumb: (index: number) => void;
  canUndo: boolean; canRedo: boolean; undo: () => void; redo: () => void;
  groupSelectedNodes: () => void; onVariables: () => void; arranging: boolean; autoLayout: () => void;
  showLabels: boolean; setShowLabels: React.Dispatch<React.SetStateAction<boolean>>;
  showAnnotations: boolean; setShowAnnotations: React.Dispatch<React.SetStateAction<boolean>>; onFit: () => void;
};

export function CanvasToolbar({ path, crumbs, goUp, onRoot, onCrumb, canUndo, canRedo, undo, redo, groupSelectedNodes, onVariables, arranging, autoLayout, showLabels, setShowLabels, showAnnotations, setShowAnnotations, onFit }: Props) {
  return (
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
                      onClick={onRoot}
                    >
                      <Box size={14} />
                      {path.length ? "Model" : "Model root"}
                    </button>
                    {crumbs.map((c, i) => (
                      <React.Fragment key={i}>
                        <ChevronRight size={12} />
                        <button
                          onClick={() => onCrumb(i)}
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
                      disabled={!canUndo}
                      onClick={undo}
                    >
                      <Undo2 size={15} />
                    </button>
                    <button
                      aria-label="Redo"
                      title="Redo"
                      disabled={!canRedo}
                      onClick={redo}
                    >
                      <Redo2 size={15} />
                    </button>
                    <i />
                    <button
                      aria-label="Create reusable block (Ctrl+G)"
                      title="Create reusable block (Ctrl+G)"
                      onClick={groupSelectedNodes}
                    >
                      <Box size={15} />
                    </button>
                    <button
                      aria-label="Graph variables"
                      className="variables-button"
                      title="Graph variables: define values used by $name bindings"
                      onClick={onVariables}
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
                      onClick={onFit}
                    >
                      <Maximize2 size={15} />
                    </button>
                    <ZoomReadout />
                  </div>
                </div>
  );
}
