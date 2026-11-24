import type { RefObject } from "react";
import { Activity, ArrowUpRight, Box, ChevronDown, Code2, FolderOpen, Plus, Search, Settings2, Shapes, Workflow, X } from "lucide-react";
import { ReferenceCard } from "../ProjectBrowser";
import { catalog } from "../catalog";
import type { Project, NodeData, CatalogEntry } from "../types";
import type { SavedProject, ProjectTab, OpenProjectOptions } from "../projectWorkflow";
import type { WorkspacePanel, WorkspaceTab } from "../workspaceLocation";
import { icons, family } from "./presentation";

interface WorkbenchLibraryProps {
  mobilePanel: WorkspacePanel;
  setMobilePanel: (panel: WorkspacePanel) => void;
  sidebarMenuButton: RefObject<HTMLButtonElement | null>;
  librarySearchRef: RefObject<HTMLInputElement | null>;
  setModal: (modal: string | null) => void;
  tab: WorkspaceTab;
  setTab: (tab: WorkspaceTab) => void;
  project: Project;
  left: "blocks" | "models";
  setLeft: (left: "blocks" | "models") => void;
  search: string;
  setSearch: (search: string) => void;
  groups: string[];
  addNode: (entry: CatalogEntry) => void;
  customBlocks: NodeData[];
  addGraphNode: (data: NodeData) => void;
  saved: SavedProject[];
  loadProject: (project: SavedProject, options: OpenProjectOptions) => Promise<boolean>;
  setProjectTab: (tab: ProjectTab) => void;
  templates: Project[];
  viewReference: (project: Project) => void;
  startNewProject: (starter: string) => void;
}

export function WorkbenchLibrary({
  mobilePanel, setMobilePanel, sidebarMenuButton, librarySearchRef, setModal,
  tab, setTab, project, left, setLeft, search, setSearch, groups, addNode,
  customBlocks, addGraphNode, saved, loadProject, setProjectTab, templates,
  viewReference, startNewProject,
}: WorkbenchLibraryProps) {
  return (
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
  );
}
