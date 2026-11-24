import { useEffect, useState } from "react";
import { api, projectSchema, projectsSchema, savedProjectsSchema, recoveryStatusSchema } from "../api";
import { localizeLayerDefaults } from "../parameterScopes";
import { shouldRecoverProject } from "../projectWorkflow";
import { readWorkspaceLocation } from "../workspaceLocation";
import type { Project } from "../types";
import type { SavedProject } from "../projectWorkflow";

export function readLocalProject(
  id?: string,
): { project: Project; dirty: boolean; recoverable: boolean } | null {
  try {
    const raw =
      (id && localStorage.getItem("weave.project." + id)) ||
      localStorage.getItem("weave.project");
    const parsed = projectSchema.safeParse(JSON.parse(raw || "null"));
    if (!parsed.success || (id && parsed.data.id !== id)) return null;
    const project = parsed.data;
    const statusRaw =
      localStorage.getItem("weave.project.status." + project.id) ||
      localStorage.getItem("weave.project.status");
    const status = recoveryStatusSchema.safeParse(
      JSON.parse(statusRaw || "null"),
    );
    const matchingStatus =
      status.success && status.data.id === project.id ? status.data : null;
    return {
      project,
      dirty: matchingStatus?.dirty ?? false,
      recoverable: shouldRecoverProject(
        project.id,
        matchingStatus,
        !!statusRaw && (!status.success || status.data.id === project.id),
      ),
    };
  } catch {
    return null;
  }
}

export function useBootstrap() {
  const [state, setState] = useState<
    | { status: "loading" }
    | { status: "error"; message: string }
    | {
        status: "ready";
        templates: Project[];
        saved: SavedProject[];
        initial: Project;
        arrange: boolean;
        initialDirty: boolean;
      }
  >({ status: "loading" });
  useEffect(() => {
    let active = true;
    Promise.all([api("/templates"), api("/projects")])
      .then(async ([ts, ps]) => {
        const templates = projectsSchema.parse(ts).map(localizeLayerDefaults),
          saved = savedProjectsSchema.parse(ps);
        const location = readWorkspaceLocation(window.location.search);
        let initial = templates[0];
        let arrange = true;
        if (!initial) throw new Error("No initial project is available");
        const restored = readLocalProject(location.model || undefined);
        const restoredSaved =
          restored && saved.some((p) => p.id === restored.project.id);
        let initialDirty = false;
        if (restored && (!restoredSaved || restored.recoverable)) {
          initial = restored.project;
          initialDirty =
            restored.dirty ||
            (!!restoredSaved && restored.recoverable) ||
            (initial.id.startsWith("project-") && !restoredSaved);
          arrange = false;
        } else {
          const id =
            location.model ||
            (restoredSaved ? restored?.project.id : undefined);
          if (id) {
            const template = templates.find((p) => p.id === id);
            if (template) initial = template;
            else
              initial = projectSchema.parse(
                await api("/projects/" + encodeURIComponent(id)),
              );
            arrange = !!template;
          }
        }
        if (active)
          setState({
            status: "ready",
            templates,
            saved,
            initial: localizeLayerDefaults(initial),
            arrange,
            initialDirty,
          });
      })
      .catch((e) => {
        if (active) setState({ status: "error", message: String(e) });
      });
    return () => {
      active = false;
    };
  }, []);
  return state;
}
