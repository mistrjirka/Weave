import { useState, type Dispatch, type SetStateAction } from "react";
import { api, projectSchema, savedProjectsSchema } from "../api";
import type { Project } from "../types";
import type { SavedProject } from "../projectWorkflow";
import { download } from "./document";

export function useProjectActions(project: Project, setSaved: Dispatch<SetStateAction<SavedProject[]>>, setChecksOpen: Dispatch<SetStateAction<boolean>>, notify: (message: string) => void) {
  const [deletedProject, setDeletedProject] = useState<Project | null>(null);
  const [busy, setBusy] = useState(false);
  async function deleteSavedProject(p: { id: string }) {
    try {
      const doc = projectSchema.parse(await api("/projects/" + p.id));
      await api("/projects/" + p.id, undefined, "DELETE");
      setDeletedProject(doc);
      setSaved((items) => items.filter((item) => item.id !== p.id));
      try {
        setSaved(savedProjectsSchema.parse(await api("/projects")));
      } catch {
        notify("Project deleted, but the project list could not be refreshed.");
      }
      notify("Project deleted. You can undo this action.");
    } catch (e) {
      notify(String(e));
    }
  }
  async function restoreDeletedProject() {
    if (!deletedProject) return;
    try {
      await api("/projects/" + deletedProject.id, deletedProject, "PUT");
      setSaved((items) => [
        ...items.filter((item) => item.id !== deletedProject.id),
        deletedProject,
      ]);
      setDeletedProject(null);
      try {
        setSaved(savedProjectsSchema.parse(await api("/projects")));
      } catch {
        notify(
          "Project restored, but the project list could not be refreshed.",
        );
      }
      notify("Project restored");
    } catch (e) {
      notify(String(e));
    }
  }
  async function exportCode() {
    setBusy(true);
    try {
      const r = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(project),
      });
      if (!r.ok) throw new Error("Resolve graph errors before exporting.");
      download(project.name.replaceAll(" ", "_") + ".zip", await r.blob());
      notify("PyTorch package downloaded");
    } catch (e) {
      notify(String(e));
      setChecksOpen(true);
    } finally {
      setBusy(false);
    }
  }
  return { deletedProject, busy, deleteSavedProject, restoreDeletedProject, exportCode };
}
