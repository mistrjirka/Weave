import { createRoot } from "react-dom/client";
import { ReactFlowProvider } from "@xyflow/react";
import { Workflow } from "lucide-react";
import "@xyflow/react/dist/style.css";
import "./style.css";
import "./theme.css";
import { useBootstrap } from "./workbench/useBootstrap";
import { Workbench } from "./workbench/Workbench";

function Bootstrap() {
  const state = useBootstrap();
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
      initialDirty={state.initialDirty}
    />
  );
}
createRoot(document.getElementById("root")!).render(
  <ReactFlowProvider>
    <Bootstrap />
  </ReactFlowProvider>,
);
