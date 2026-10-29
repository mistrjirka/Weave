import type { Project } from "./types";

export type ProjectTab = "projects" | "examples";
export interface SavedProject {
  id: string;
  name: string;
  updated?: string;
}
export interface OpenProjectOptions {
  saved?: boolean;
  force?: boolean;
  arrange?: boolean;
  newDraft?: boolean;
}

/** Templates remain immutable; every created network owns a distinct identity. */
export function createProject(
  starter: Project,
  name: string,
  id: string,
): Project {
  const project = structuredClone(starter);
  project.id = id;
  project.name = name.trim();
  project.config.theta ??= 1_000_000;
  project.description = starter.id === "blank" ? "" : starter.description;
  delete project.updated;
  return project;
}

export function formattedDate(value?: string) {
  if (!value) return "Saved on this server";
  // SQLite CURRENT_TIMESTAMP is UTC, but does not include a timezone suffix.
  const date = new Date(
    value.includes("T") ? value : value.replace(" ", "T") + "Z",
  );
  return Number.isNaN(date.getTime())
    ? "Saved on this server"
    : new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(date);
}
