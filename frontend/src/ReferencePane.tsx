import { useEffect, useState, type Dispatch, type SetStateAction } from "react";
import { ReactFlowProvider, type NodeTypes } from "@xyflow/react";
import { Copy, LockKeyhole, RefreshCw, X } from "lucide-react";
import { describeError, projectSchema, responseData } from "./api";
import type { Project } from "./types";
import type { ReferenceLocation } from "./workspaceLocation";
import {
  isGraphEditShortcut,
  type ReferenceInspection,
} from "./ReferenceInspector";
import { localizeLayerDefaults } from "./parameterScopes";
import { ReferenceCanvas } from "./reference/ReferenceCanvas";

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
