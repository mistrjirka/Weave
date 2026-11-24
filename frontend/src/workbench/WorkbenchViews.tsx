import type { RefObject } from "react";
import { Activity, Code2, Columns2, ExternalLink, Workflow } from "lucide-react";
import { UIUpdateNotice } from "../UIUpdateNotice";
import { GPUTuning } from "../GPUTuning";
import { ThemePicker } from "../ThemePicker";
import type { Project } from "../types";
import type { WorkspaceTab } from "../workspaceLocation";

export function WorkbenchViews({ tab, onTab, project, splitButton, splitView, onToggleSplit }: { tab: WorkspaceTab; onTab: (tab: WorkspaceTab) => void; project: Project; splitButton: RefObject<HTMLButtonElement | null>; splitView: boolean; onToggleSplit: () => void }) {
  return (
        <nav className="viewbar">
          <div className="view-tabs">
            <button
              className={tab === "graph" ? "active" : ""}
              aria-label="Architecture"
              aria-current={tab === "graph" ? "page" : undefined}
              onClick={() => onTab("graph")}
            >
              <Workflow size={15} />
              Architecture
            </button>
            <button
              className={tab === "code" ? "active" : ""}
              aria-label="PyTorch code"
              aria-current={tab === "code" ? "page" : undefined}
              onClick={() => onTab("code")}
            >
              <Code2 size={15} />
              PyTorch code
            </button>
            <button
              className={tab === "training" ? "active" : ""}
              aria-label="Training"
              aria-current={tab === "training" ? "page" : undefined}
              onClick={() => onTab("training")}
            >
              <Activity size={15} />
              Training
            </button>
            <button
              className={tab === "sources" ? "active" : ""}
              aria-label="Sources"
              aria-current={tab === "sources" ? "page" : undefined}
              onClick={() => onTab("sources")}
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
              onClick={onToggleSplit}
            >
              <Columns2 size={15} aria-hidden="true" />
              <span>Split view</span>
            </button>
          </div>
        </nav>
  );
}
