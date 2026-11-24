import type { RefObject } from "react";
import { ChevronDown, Download, FilePlus2, GitBranch, Menu, Pencil, Save, Upload } from "lucide-react";
import { ActiveTrainingRuns } from "../ActiveTrainingRuns";

export function WorkbenchHeader({ sidebarMenuButton, sidebarOpen, onToggleSidebar, projectName, projectStatus, onSettings, onNew, onImport, onSave, onExport, saving, busy }: {
  sidebarMenuButton: RefObject<HTMLButtonElement | null>; sidebarOpen: boolean; onToggleSidebar: () => void; projectName: string; projectStatus: string; onSettings: () => void; onNew: () => void; onImport: () => void; onSave: () => void; onExport: () => void; saving: boolean; busy: boolean;
}) {
  return (
        <header className="topbar">
          <button
            ref={sidebarMenuButton}
            className="mobile-sidebar-menu"
            aria-label="Open sidebar"
            aria-controls="workspace-sidebar"
            aria-expanded={sidebarOpen}
            onClick={onToggleSidebar}
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
                  aria-label={`Project settings: ${projectName}`}
                  onClick={onSettings}
                >
                  <span>{projectName}</span>
                  <Pencil size={12} aria-hidden="true" />
                </button>
              </h1>
              <span className="project-meta">
                {projectStatus}
              </span>
            </div>
          </div>
          <div className="topbar-actions">
            <ActiveTrainingRuns />
            <button
              className="subtle-button new-project-button"
              aria-label="New project"
              onClick={onNew}
            >
              <FilePlus2 size={16} aria-hidden="true" />
              <span className="new-project-desktop">New project</span>
              <span className="new-project-mobile">New</span>
            </button>
            <button
              className="subtle-button"
              aria-label="Import project JSON"
              onClick={onImport}
              title="Import project JSON"
            >
              <Upload size={15} />
              <span>Import</span>
            </button>
            <button
              className="subtle-button"
              aria-label={saving ? "Saving project" : "Save"}
              onClick={onSave}
              disabled={saving}
            >
              <Save size={15} />
              <span>{saving ? "Saving…" : "Save"}</span>
            </button>
            <button
              className="export-button"
              aria-label="Export PyTorch"
              onClick={onExport}
              disabled={busy}
            >
              <Download size={15} />
              {busy ? "Working…" : "Export PyTorch"}
              <ChevronDown size={13} />
            </button>
          </div>
        </header>
  );
}
