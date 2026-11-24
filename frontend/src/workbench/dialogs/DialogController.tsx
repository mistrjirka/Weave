import type { Dispatch, RefObject, SetStateAction } from "react";
import { WorkbenchDialog } from "./WorkbenchDialog";
import { download } from "../document";
import { createProject, type OpenProjectOptions, type ProjectTab, type SavedProject } from "../../projectWorkflow";
import { newId as uid } from "../../ui";
import type { GraphNode, Project, TokenizerInfo } from "../../types";
import type { WorkspacePanel } from "../../workspaceLocation";
import type { graphVariableScope, graphVariableUses } from "../../graphVariables";

type PendingLoad = { project: Project | SavedProject; options: OpenProjectOptions } | null;
export function DialogController({ modal, saving, loadRequest, setModal, setPendingLoad, pendingLoad, project, saveProject, modalRef, dirtyAfterSave, loadProject, projectRef, notify, variableScope, variableUses, hiddenVariables, path, updateGraphVariables, tokenizerTarget, tokenizerInput, inputTokenizer, applyInputTokenizer, templates, starterId, setStarterId, deletedProject, restoreDeletedProject, saved, dirty, projectTab, setProjectTab, startNewProject, viewReference, deleteSavedProject, setMobilePanel, change }: {
  modal: string | null; saving: boolean; loadRequest: RefObject<number>; setModal: Dispatch<SetStateAction<string | null>>; setPendingLoad: Dispatch<SetStateAction<PendingLoad>>; pendingLoad: PendingLoad; project: Project;
  saveProject: () => Promise<boolean>; modalRef: RefObject<string | null>; dirtyAfterSave: () => boolean; loadProject: (project: Project | SavedProject, options?: OpenProjectOptions) => Promise<boolean>; projectRef: RefObject<Project>; notify: (message: string) => void;
  variableScope: ReturnType<typeof graphVariableScope>; variableUses: ReturnType<typeof graphVariableUses>; hiddenVariables: string[]; path: string[]; updateGraphVariables: (values: Record<string, number>) => void;
  tokenizerTarget: string | null; tokenizerInput: GraphNode | undefined; inputTokenizer: TokenizerInfo | null | undefined; applyInputTokenizer: (info: TokenizerInfo) => boolean;
  templates: Project[]; starterId: string; setStarterId: Dispatch<SetStateAction<string>>; deletedProject: Project | null; restoreDeletedProject: () => Promise<void>; saved: SavedProject[]; dirty: boolean; projectTab: ProjectTab; setProjectTab: Dispatch<SetStateAction<ProjectTab>>;
  startNewProject: (starter?: string) => void; viewReference: (project: Project) => void; deleteSavedProject: (project: { id: string }) => Promise<void>; setMobilePanel: Dispatch<SetStateAction<WorkspacePanel>>; change: (fn: (project: Project) => void) => void;
}) {
  return (
    <>
      {modal && (
        <WorkbenchDialog
          key={modal}
          saving={saving}
          onClose={() => {
            if (saving) return;
            loadRequest.current++;
            setModal(null);
            setPendingLoad(null);
          }}
          onCloseButton={() => {
            loadRequest.current++;
            setPendingLoad(null);
            setModal(null);
          }}
          payload={modal === "unsaved" ? {
            kind: "unsaved",
            projectName: project.name,
            nextProjectName: pendingLoad?.project.name,
            newDraft: !!pendingLoad?.options.newDraft,
            onSave: async () => {
              const next = pendingLoad;
              if (next && (await saveProject()) && modalRef.current === "unsaved" && !dirtyAfterSave())
                await loadProject(next.project, { ...next.options, force: true });
            },
            onDiscard: () => {
              const next = pendingLoad;
              if (next) {
                try {
                  localStorage.removeItem("weave.project." + projectRef.current.id);
                  localStorage.removeItem("weave.project.status." + projectRef.current.id);
                  localStorage.removeItem("weave.project");
                  localStorage.removeItem("weave.project.status");
                } catch {
                  notify("Could not remove the local recovery draft.");
                }
                loadProject(next.project, { ...next.options, force: true });
              }
            },
            onKeepEditing: () => { setModal(null); setPendingLoad(null); },
          } : modal === "variables" ? {
            kind: "variables",
            props: { values: variableScope.own, inherited: variableScope.inherited, uses: variableUses, hiddenNames: hiddenVariables, nested: path.length > 0, onChange: updateGraphVariables },
          } : modal === "tokenizer" ? {
            kind: "tokenizer",
            panelKey: project.id + ":" + tokenizerTarget,
            props: {
              projectId: project.id,
              inputName: tokenizerInput?.data.label ?? "Input",
              info: inputTokenizer,
              onApply: applyInputTokenizer,
              saving,
              onClose: () => setModal(null),
              onSave: async () => {
                if (!(await saveProject())) return false;
                setModal(null);
                return true;
              },
            },
          } : modal === "new-project" ? {
            kind: "new-project",
            props: {
              templates, starterId, onStarter: setStarterId,
              onCreate: (starter, name) => {
                void loadProject(createProject(starter, name, "project-" + uid()), { arrange: true, newDraft: true });
              },
              onCancel: () => setModal(null),
            },
          } : modal === "projects" ? {
            kind: "projects",
            deletedProjectName: deletedProject?.name,
            onUndoDelete: restoreDeletedProject,
            props: {
              templates, saved, activeId: project.id,
              draft: dirty && !saved.some((p) => p.id === project.id) ? project : null,
              tab: projectTab, onTab: setProjectTab, onNew: startNewProject,
              onOpen: (p) => { void loadProject(p, { saved: true }); },
              onView: viewReference,
              onDelete: (p) => { void deleteSavedProject(p); },
              onResume: () => { loadRequest.current++; setModal(null); setMobilePanel(null); },
            },
          } : {
            kind: "settings",
            name: project.name,
            description: project.description,
            onNameChange: (value) => change((p) => { p.name = value; }),
            onDescriptionChange: (value) => change((p) => { p.description = value; }),
            onDownload: () => download(project.name + ".json", JSON.stringify(project, null, 2)),
          }}
        />
      )}
    </>
  );
}
