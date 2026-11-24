import type { ComponentProps } from "react";
import { Download, X } from "lucide-react";
import { Dialog } from "../../ui";
import { GraphVariables } from "../../GraphVariables";
import { TokenizerPanel } from "../../TokenizerPanel";
import { NewProjectForm, ProjectBrowser } from "../../ProjectBrowser";
import { Field } from "../Field";

export type WorkbenchDialogPayload =
  | {
      kind: "unsaved";
      projectName: string;
      nextProjectName: string | undefined;
      newDraft: boolean;
      onSave: () => void;
      onDiscard: () => void;
      onKeepEditing: () => void;
    }
  | { kind: "variables"; props: ComponentProps<typeof GraphVariables> }
  | {
      kind: "tokenizer";
      panelKey: string;
      props: ComponentProps<typeof TokenizerPanel>;
    }
  | { kind: "new-project"; props: ComponentProps<typeof NewProjectForm> }
  | {
      kind: "projects";
      props: ComponentProps<typeof ProjectBrowser>;
      deletedProjectName: string | undefined;
      onUndoDelete: () => void;
    }
  | {
      kind: "settings";
      name: string;
      description: string;
      onNameChange: (value: string) => void;
      onDescriptionChange: (value: string) => void;
      onDownload: () => void;
    };

export function WorkbenchDialog({
  payload,
  saving,
  onClose,
  onCloseButton,
}: {
  payload: WorkbenchDialogPayload;
  saving: boolean;
  onClose: () => void;
  onCloseButton: () => void;
}) {
  const kind = payload.kind;
  const label = kind === "tokenizer" ? "Tokenizer configuration"
    : kind === "variables" ? "Graph variables"
    : kind === "settings" ? "Project settings"
    : kind === "unsaved" ? "Unsaved changes"
    : kind === "new-project" ? "New project"
    : "Projects & examples";
  const title = payload.kind === "tokenizer" ? `Tokenizer · ${payload.props.inputName}`
    : kind === "variables" ? "Graph variables"
    : kind === "projects" ? "Projects & examples"
    : kind === "new-project" ? "New project"
    : kind === "unsaved" ? "Keep your changes?"
    : "Project settings";

  return (
    <Dialog
      className={kind === "projects" ? "wide" : kind === "tokenizer" ? "tokenizer-dialog" : ""}
      label={label}
      onClose={onClose}
    >
      <div className="modal-heading">
        <div>
          <span className="eyebrow">{kind === "tokenizer" ? "MODEL INPUT" : kind === "projects" ? "YOUR WORKSPACE" : "PROJECT"}</span>
          <h2>{title}</h2>
        </div>
        <button disabled={saving} onClick={onCloseButton} aria-label="Close dialog"><X size={20} /></button>
      </div>
      {payload.kind === "projects" && payload.deletedProjectName && (
        <div className="undo-banner" aria-live="polite">
          <span>Deleted {payload.deletedProjectName}</span>
          <button onClick={payload.onUndoDelete}>Undo</button>
        </div>
      )}
      {payload.kind === "unsaved" ? (
        <>
          <p className="unsaved-description">
            You have unsaved changes in <strong>{payload.projectName}</strong>.
            Save them before {payload.newDraft ? "creating" : "opening"}{" "}
            {payload.nextProjectName}.
          </p>
          <div className="unsaved-actions">
            <button className="primary" disabled={saving} onClick={payload.onSave}>
              {payload.newDraft ? "Save & create" : "Save & open"}
            </button>
            <button disabled={saving} onClick={payload.onDiscard}>
              {payload.newDraft ? "Discard & create" : "Discard & open"}
            </button>
            <button onClick={payload.onKeepEditing}>Keep editing</button>
          </div>
        </>
      ) : payload.kind === "variables" ? (
        <GraphVariables {...payload.props} />
      ) : payload.kind === "tokenizer" ? (
        <TokenizerPanel key={payload.panelKey} {...payload.props} />
      ) : payload.kind === "new-project" ? (
        <NewProjectForm {...payload.props} />
      ) : payload.kind === "projects" ? (
        <ProjectBrowser {...payload.props} />
      ) : (
        <>
          <Field name="Project name" value={payload.name} onChange={(value) => { if (typeof value === "string") payload.onNameChange(value); }} />
          <Field name="Description" value={payload.description} onChange={(value) => { if (typeof value === "string") payload.onDescriptionChange(value); }} />
          <button className="primary" onClick={payload.onDownload}>
            <Download size={15} />
            Download project JSON
          </button>
          <p>Project files include the complete graph, reusable block definitions, tokenizer metadata, and source references.</p>
        </>
      )}
    </Dialog>
  );
}
