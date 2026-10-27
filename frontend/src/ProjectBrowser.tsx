import { useRef, useState } from "react";
import {
  ArrowUpRight,
  BookOpen,
  Copy,
  FilePlus2,
  FolderOpen,
  Search,
  Trash2,
} from "lucide-react";
import type { Project } from "./types";
import type { ProjectTab, SavedProject } from "./projectWorkflow";
import { formattedDate } from "./projectWorkflow";

interface BrowserProps {
  templates: Project[];
  saved: SavedProject[];
  activeId: string;
  draft: Project | null;
  tab: ProjectTab;
  onTab: (tab: ProjectTab) => void;
  onNew: (starter?: string) => void;
  onOpen: (project: SavedProject) => void;
  onView: (project: Project) => void;
  onDelete: (project: SavedProject) => void;
  onResume: () => void;
}

export function ReferenceCard({
  project,
  onView,
  onUse,
  compact = false,
}: {
  project: Project;
  onView: () => void;
  onUse: () => void;
  compact?: boolean;
}) {
  if (compact)
    return (
      <button
        className="reference-row"
        onClick={onView}
        aria-label={`View reference ${project.name}`}
        title={project.description}
      >
        <BookOpen size={15} aria-hidden="true" />
        <span>{project.name}</span>
        <ArrowUpRight size={13} aria-hidden="true" />
      </button>
    );
  return (
    <article className="example-project-card">
      <div className="example-card-heading">
        <BookOpen size={18} aria-hidden="true" />
        <h3>{project.name}</h3>
        <span>Example</span>
      </div>
      <p>{project.description}</p>
      <div className="example-card-actions">
        <button
          className="small-button"
          onClick={onView}
          aria-label={`View reference ${project.name}`}
        >
          View reference <ArrowUpRight size={14} aria-hidden="true" />
        </button>
        <button onClick={onUse} aria-label={`Create copy of ${project.name}`}>
          <Copy size={14} aria-hidden="true" />
          Create copy
        </button>
      </div>
    </article>
  );
}

/** Personal documents and immutable examples have visibly different actions. */
export function ProjectBrowser({
  templates,
  saved,
  activeId,
  draft,
  tab,
  onTab,
  onNew,
  onOpen,
  onView,
  onDelete,
  onResume,
}: BrowserProps) {
  const [search, setSearch] = useState("");
  const query = search.trim().toLocaleLowerCase();
  const documents = saved.filter((p) =>
    p.name.toLocaleLowerCase().includes(query),
  );
  const matchingDraft =
    draft && draft.name.toLocaleLowerCase().includes(query) ? draft : null;
  const examples = templates.filter(
    (p) =>
      p.id !== "blank" &&
      (p.name + " " + p.description).toLocaleLowerCase().includes(query),
  );
  return (
    <div className="project-browser">
      <div className="project-browser-actions">
        <p>Continue your work, start a network, or explore an example.</p>
        <button className="primary" onClick={() => onNew()}>
          <FilePlus2 size={16} aria-hidden="true" />
          New project
        </button>
      </div>
      <div
        className="project-tabs"
        role="tablist"
        aria-label="Workspace collection"
        onKeyDown={(event) => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key))
            return;
          event.preventDefault();
          const next: ProjectTab =
            event.key === "Home"
              ? "projects"
              : event.key === "End"
                ? "examples"
                : tab === "projects"
                  ? "examples"
                  : "projects";
          onTab(next);
          setSearch("");
          event.currentTarget
            .querySelector<HTMLButtonElement>(`#${next}-tab`)
            ?.focus();
        }}
      >
        <button
          role="tab"
          id="projects-tab"
          tabIndex={tab === "projects" ? 0 : -1}
          aria-selected={tab === "projects"}
          aria-controls="project-collection"
          onClick={() => {
            onTab("projects");
            setSearch("");
          }}
        >
          <FolderOpen size={16} aria-hidden="true" />
          Your projects <span>{saved.length + (draft ? 1 : 0)}</span>
        </button>
        <button
          role="tab"
          id="examples-tab"
          tabIndex={tab === "examples" ? 0 : -1}
          aria-selected={tab === "examples"}
          aria-controls="project-collection"
          onClick={() => {
            onTab("examples");
            setSearch("");
          }}
        >
          <BookOpen size={16} aria-hidden="true" />
          Reference architectures
        </button>
      </div>
      <label className="project-browser-search">
        <Search size={16} aria-hidden="true" />
        <input
          type="search"
          aria-label={
            tab === "projects"
              ? "Search saved projects"
              : "Search reference architectures"
          }
          name="project-search"
          autoComplete="off"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={
            tab === "projects"
              ? "Search your projects…"
              : "Search architectures…"
          }
        />
      </label>
      <div
        role="tabpanel"
        id="project-collection"
        aria-labelledby={tab === "projects" ? "projects-tab" : "examples-tab"}
      >
        {tab === "projects" ? (
          documents.length || matchingDraft ? (
            <div className="project-document-list">
              {matchingDraft && (
                <article className="current-project-draft">
                  <FolderOpen size={22} aria-hidden="true" />
                  <div>
                    <h3>{matchingDraft.name}</h3>
                    <p>
                      Current draft · recovered in this browser. Save it to keep
                      it on the server.
                    </p>
                  </div>
                  <button className="small-button" onClick={onResume}>
                    Continue editing
                  </button>
                </article>
              )}
              {documents.map((p) => (
                <div className="saved-project" key={p.id}>
                  <button
                    className="open-saved-project"
                    onClick={() => (p.id === activeId ? onResume() : onOpen(p))}
                    title="Open saved project"
                  >
                    <FolderOpen size={19} aria-hidden="true" />
                    <span>
                      <strong>{p.name}</strong>
                      <small>
                        {formattedDate(p.updated)}
                        {p.id === activeId ? " · Current project" : ""}
                      </small>
                    </span>
                    <ArrowUpRight size={15} aria-hidden="true" />
                  </button>
                  <button
                    title="Delete saved project"
                    aria-label={`Delete ${p.name}`}
                    onClick={() => onDelete(p)}
                  >
                    <Trash2 size={16} aria-hidden="true" />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="project-empty" role="status">
              <FolderOpen size={30} aria-hidden="true" />
              <h3>
                {query
                  ? "No matching projects"
                  : "Your next network starts here"}
              </h3>
              <p>
                {query
                  ? "Try a different name or clear the search."
                  : "Create a project, then use Save to keep it on this server. Examples are available in Reference architectures."}
              </p>
              {!query && (
                <button className="primary" onClick={() => onNew()}>
                  Create your first project
                </button>
              )}
            </div>
          )
        ) : (
          <>
            <p className="reference-explanation">
              View a reference beside your work, or use it as the starting point
              for a new project.
            </p>
            <div className="example-project-grid">
              {examples.map((p) => (
                <ReferenceCard
                  key={p.id}
                  project={p}
                  onView={() => onView(p)}
                  onUse={() => onNew(p.id)}
                />
              ))}
            </div>
            {!examples.length && (
              <p role="status">
                No architectures match “{search}”. Try transformer, diffusion,
                or Needle.
              </p>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export function NewProjectForm({
  templates,
  starterId,
  onStarter,
  onCreate,
  onCancel,
}: {
  templates: Project[];
  starterId: string;
  onStarter: (id: string) => void;
  onCreate: (starter: Project, name: string) => void;
  onCancel: () => void;
}) {
  const initial =
    templates.find((p) => p.id === starterId) ||
    templates.find((p) => p.id === "blank");
  const [name, setName] = useState(
    initial?.id === "blank"
      ? "Untitled network"
      : `${initial?.name || "Network"} copy`,
  );
  const [nameEdited, setNameEdited] = useState(false);
  const [error, setError] = useState("");
  const nameInput = useRef<HTMLInputElement>(null);
  const starter = templates.find((p) => p.id === starterId) || initial;
  return (
    <form
      className="new-project-form"
      onSubmit={(e) => {
        e.preventDefault();
        if (!name.trim()) {
          setError("Give your project a name.");
          nameInput.current?.focus();
          return;
        }
        if (starter) onCreate(starter, name.trim());
      }}
    >
      <p>
        Create your own editable network. Save it when you’re ready to keep it
        on this server.
      </p>
      <label className="property-field">
        <span>Project name</span>
        <input
          name="project-name"
          ref={nameInput}
          autoComplete="off"
          value={name}
          aria-invalid={!!error}
          aria-describedby={error ? "project-name-error" : undefined}
          onChange={(e) => {
            setName(e.target.value);
            setNameEdited(true);
            setError("");
          }}
        />
      </label>
      {error && (
        <p className="inline-error" id="project-name-error" role="alert">
          {error}
        </p>
      )}
      <label className="property-field">
        <span>Start from</span>
        <select
          name="project-starter"
          value={starter?.id || ""}
          onChange={(e) => {
            const selected = templates.find((p) => p.id === e.target.value);
            onStarter(e.target.value);
            if (!nameEdited && selected)
              setName(
                selected.id === "blank"
                  ? "Untitled network"
                  : `${selected.name} copy`,
              );
          }}
        >
          {templates
            .filter((p) => p.id === "blank")
            .map((p) => (
              <option key={p.id} value={p.id}>
                Blank canvas
              </option>
            ))}
          <optgroup label="Example architectures">
            {templates
              .filter((p) => p.id !== "blank")
              .map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
          </optgroup>
        </select>
      </label>
      <div className="starter-description">
        <BookOpen size={18} aria-hidden="true" />
        <div>
          <strong>
            {starter?.id === "blank"
              ? "A canvas of your own"
              : `A copy of ${starter?.name}`}
          </strong>
          <p>
            {starter?.id === "blank"
              ? "Starts with token input and output nodes. Add blocks and connect them to build your network."
              : starter?.description}
          </p>
        </div>
      </div>
      <div className="new-project-actions">
        <button type="button" className="small-button" onClick={onCancel}>
          Cancel
        </button>
        <button type="submit" className="primary">
          Create project <ArrowUpRight size={15} aria-hidden="true" />
        </button>
      </div>
    </form>
  );
}
