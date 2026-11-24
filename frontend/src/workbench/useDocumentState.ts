import { useCallback, useEffect, useRef, useState } from "react";
import { synchronizeDefinitions } from "../graph";
import { api, describeError, projectSaveSchema, projectSchema, savedProjectsSchema } from "../api";
import { localizeLayerDefaults } from "../parameterScopes";
import { readLocalProject } from "./useBootstrap";
import { matchesSavedRevision, withProjectIdentity } from "../projectWorkflow";
import { newId } from "../ui";
import type { Project } from "../types";
import type { OpenProjectOptions, SavedProject } from "../projectWorkflow";

export function useDocumentState(initial: Project, initialSaved: SavedProject[], initialDirty: boolean, notify: (message: string) => void, onEdit: () => void, onRestore: (project: Project) => void) {
  const [project, setProject] = useState(initial);
  const [saved, setSaved] = useState(initialSaved);
  const [dirty, setDirty] = useState(initialDirty);
  const [history, setHistory] = useState<Project[]>([]);
  const [future, setFuture] = useState<Project[]>([]);
  const [saving, setSaving] = useState(false);
  const savingRef = useRef(false);
  const loadRequest = useRef(0);
  const savedSnapshot = useRef<string | null>(initialDirty ? null : JSON.stringify(initial));
  const ref = useRef(project);
  ref.current = project;

  useEffect(() => {
    if (!dirty) return;
    const guard = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", guard);
    return () => window.removeEventListener("beforeunload", guard);
  }, [dirty]);
  useEffect(() => {
    try {
      localStorage.setItem("weave.project.status." + project.id, JSON.stringify({ id: project.id, dirty }));
      localStorage.setItem("weave.project.status", JSON.stringify({ id: project.id, dirty }));
    } catch {
      notify("Local recovery unavailable. Save your project to the server.");
    }
  }, [project.id, dirty]);
  useEffect(() => {
    try {
      localStorage.setItem("weave.project." + project.id, JSON.stringify(project));
      localStorage.setItem("weave.project", JSON.stringify(project));
    } catch {
      notify("Local recovery unavailable. Save your project to the server.");
    }
  }, [project]);

  const change = useCallback((fn: (project: Project) => void, record = true, markDirty = true) => {
    const before = ref.current;
    const next = structuredClone(before);
    try {
      fn(next);
      synchronizeDefinitions(before, next);
    } catch (error) {
      notify(describeError(error));
      return;
    }
    if (markDirty) onEdit();
    if (record) {
      setHistory((items) => [...items.slice(-39), structuredClone(before)]);
      setFuture([]);
    }
    ref.current = next;
    setProject(next);
    if (markDirty) setDirty(!matchesSavedRevision(next, savedSnapshot.current));
  }, []);

  function undo() {
    const previous = history.at(-1);
    if (!previous) return;
    setFuture((items) => [structuredClone(ref.current), ...items]);
    setHistory((items) => items.slice(0, -1));
    setProject(previous);
    ref.current = previous;
    onRestore(previous);
    setDirty(!matchesSavedRevision(previous, savedSnapshot.current));
  }
  function redo() {
    if (!future.length) return;
    const next = future[0];
    setHistory((items) => [...items, structuredClone(ref.current)]);
    setFuture((items) => items.slice(1));
    setProject(next);
    ref.current = next;
    onRestore(next);
    setDirty(!matchesSavedRevision(next, savedSnapshot.current));
  }

  function dirtyAfterSave() {
    return !matchesSavedRevision(ref.current, savedSnapshot.current);
  }
  async function saveProject() {
    if (!ref.current || savingRef.current) return false;
    savingRef.current = true;
    setSaving(true);
    try {
      const before = ref.current;
      const navigation = loadRequest.current;
      const id = ref.current.id?.startsWith("project-") ? ref.current.id : "project-" + newId();
      const next = { ...ref.current, id };
      const result = projectSaveSchema.parse(await api("/projects/" + id, next, "PUT"));
      const message = result.stage_sync?.error
        ? "Project saved, but training stages could not be updated: " + result.stage_sync.error
        : "Project saved on server" + (result.stage_sync?.updated ? ` · ${result.stage_sync.updated} training stage(s) synchronized` : "");
      const stillCurrent = navigation === loadRequest.current && ref.current.id === before.id;
      if (stillCurrent) {
        savedSnapshot.current = JSON.stringify(next);
        const current = ref.current === before ? next : { ...ref.current, id };
        setHistory((items) => items.map((item) => withProjectIdentity(item, id)));
        setFuture((items) => items.map((item) => withProjectIdentity(item, id)));
        setProject(current);
        ref.current = current;
        setDirty(!matchesSavedRevision(current, savedSnapshot.current));
      }
      try {
        setSaved(savedProjectsSchema.parse(await api("/projects")));
        notify(message);
      } catch (refreshError) {
        notify(message + "; project list refresh failed: " + describeError(refreshError));
      }
      return stillCurrent;
    } catch (error) {
      notify(String(error));
      return false;
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  }
  async function loadDocument(
    document: Project | SavedProject,
    options: OpenProjectOptions,
    prepare: (project: Project) => void,
    finish: () => void,
  ) {
    const request = ++loadRequest.current;
    const before = ref.current;
    try {
      const draft = options.saved ? readLocalProject(document.id) : null;
      const recovered = draft?.recoverable ? draft.project : null;
      const loaded = recovered || (options.saved
        ? projectSchema.parse(await api("/projects/" + encodeURIComponent(document.id)))
        : projectSchema.parse(document));
      if (request !== loadRequest.current || ref.current !== before) return false;
      const next = localizeLayerDefaults(loaded);
      savedSnapshot.current = options.newDraft || recovered ? null : JSON.stringify(next);
      prepare(next);
      setProject(next);
      ref.current = next;
      setHistory([]);
      setFuture([]);
      setDirty(!!options.newDraft || !!recovered);
      finish();
      return true;
    } catch (error) {
      notify(String(error));
      return false;
    }
  }
  return { project, setProject, saved, setSaved, dirty, setDirty, history, setHistory, future, setFuture, saving, loadRequest, savedSnapshot, ref, change, undo, redo, dirtyAfterSave, saveProject, loadDocument };
}
