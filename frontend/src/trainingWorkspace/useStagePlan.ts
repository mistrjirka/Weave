import { useEffect, useRef, useState, type RefObject } from "react";
import { z } from "zod";
import type { Project } from "../types";
import {
  capabilitySchema,
  templateSchema,
  planSchema,
  stageSchema,
  trainingRequest,
  type TrainingConfig,
  type TrainingPlan,
  type TrainingStage,
} from "../trainingApi";
import { trainingConfigsEqual } from "../trainingConfigEquality";

const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

export function useStagePlan(
  project: Project,
  selected: string,
  pendingTasksRef: RefObject<number>,
  setNotice: (value: string) => void,
  setError: (value: string) => void,
) {
  const [capabilities, setCapabilities] = useState<z.infer<
    typeof capabilitySchema
  > | null>(null);
  const [templates, setTemplates] = useState<z.infer<
    typeof templateSchema
  > | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [bootstrapRetry, setBootstrapRetry] = useState(0);
  const authenticated = capabilities?.authenticated;
  const [plan, setPlan] = useState<TrainingPlan>({
    version: 1,
    project_id: project.id,
    stages: [],
  });
  const [savedStages, setSavedStages] = useState<Record<string, TrainingStage>>(
    {},
  );
  const savedStagesRef = useRef<Record<string, TrainingStage>>({});
  const stageSavesRef = useRef(new Map<string, Promise<TrainingStage>>());
  const stage = plan.stages.find((s) => s.id === selected) ?? plan.stages[0];
  const config = stage?.config;
  const savedStage = stage ? savedStages[stage.id] : undefined;
  const stageDirty =
    !!stage &&
    (!savedStage ||
      stage.name !== savedStage.name ||
      stage.script !== savedStage.script ||
      !trainingConfigsEqual(stage.config, savedStage.config));
  const cacheKey = `weave:training:${project.id}`;
  useEffect(() => {
    void trainingRequest("/capabilities", capabilitySchema)
      .then(setCapabilities)
      .catch((e) => setError(message(e)));
  }, []);
  useEffect(() => {
    if (!capabilities?.authenticated) return;
    const controller = new AbortController();
    Promise.all([
      trainingRequest(
        "/templates",
        templateSchema,
        undefined,
        undefined,
        controller.signal,
      ),
      trainingRequest(
        `/plans/${encodeURIComponent(project.id)}`,
        planSchema,
        undefined,
        undefined,
        controller.signal,
      ),
    ])
      .then(([t, p]) => {
        if (controller.signal.aborted) return;
        setTemplates(t);
        savedStagesRef.current = Object.fromEntries(
          p.stages.map((item) => [item.id, item]),
        );
        setSavedStages(savedStagesRef.current);
        const cached = localStorage.getItem(cacheKey);
        let recovered: TrainingPlan | null = null;
        try {
          const parsed = cached
            ? planSchema.safeParse(JSON.parse(cached))
            : null;
          recovered =
            parsed?.success && parsed.data.project_id === project.id
              ? parsed.data
              : null;
        } catch {
          /* A damaged browser draft must not hide saved stages. */
        }
        // Restore draft content while attaching paths created by file migration.
        setPlan(
          recovered
            ? {
                ...recovered,
                stages: [
                  ...recovered.stages.map((draft) => {
                    const disk = savedStagesRef.current[draft.id];
                    if (!disk) return draft;
                    const migrated =
                      disk.previous_script != null &&
                      draft.script === disk.previous_script;
                    return {
                      ...draft,
                      script: migrated ? disk.script : draft.script,
                      script_path: disk.script_path,
                      config_path: disk.config_path,
                      project_path: disk.project_path,
                      model_path: disk.model_path,
                      support_path: disk.support_path,
                      previous_script: disk.previous_script,
                      script_revision: migrated
                        ? disk.script_revision
                        : (draft.script_revision ??
                          (draft.script === disk.script
                            ? disk.script_revision
                            : null)),
                      config_revision:
                        draft.config_revision ??
                        (trainingConfigsEqual(draft.config, disk.config)
                          ? disk.config_revision
                          : null),
                    };
                  }),
                  ...p.stages.filter(
                    (saved) =>
                      !recovered.stages.some((draft) => draft.id === saved.id),
                  ),
                ],
              }
            : p,
        );
        setLoaded(true);
        setError("");
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(message(e));
      });
    return () => controller.abort();
  }, [capabilities?.authenticated, project.id, cacheKey, bootstrapRetry]);
  useEffect(() => {
    if (loaded) {
      try {
        localStorage.setItem(cacheKey, JSON.stringify(plan));
      } catch {
        setError(
          "Browser draft storage is full. Use Save stages to preserve your script on the server.",
        );
      }
    }
  }, [plan, cacheKey, loaded]);
  useEffect(() => {
    if (!stage?.script_path || !authenticated) return;
    const controller = new AbortController();
    let pending = false;
    const timer = window.setInterval(() => {
      if (pending || pendingTasksRef.current > 0) return;
      pending = true;
      void trainingRequest(
        `/workspaces/${encodeURIComponent(project.id)}/stages/${encodeURIComponent(stage.id)}`,
        stageSchema,
        undefined,
        undefined,
        controller.signal,
      )
        .then((remote) => {
          if (controller.signal.aborted) return;
          const previous = savedStagesRef.current[remote.id];
          setPlan((current) => ({
            ...current,
            stages: current.stages.map((item) =>
              item.id === remote.id &&
              previous &&
              item.script === previous.script &&
              trainingConfigsEqual(item.config, previous.config)
                ? { ...remote, name: item.name }
                : item,
            ),
          }));
          savedStagesRef.current = {
            ...savedStagesRef.current,
            [remote.id]: remote,
          };
          setSavedStages(savedStagesRef.current);
        })
        .catch(() => {
          /* Explicit reload/save surfaces file errors; polling must not erase a draft. */
        })
        .finally(() => {
          pending = false;
        });
    }, 4000);
    return () => {
      window.clearInterval(timer);
      controller.abort();
    };
  }, [stage?.id, stage?.script_path, authenticated, project.id]);
  function updateStage(patch: Partial<TrainingStage>) {
    if (!stage) return;
    setNotice("");
    setPlan((current) => ({
      ...current,
      stages: current.stages.map((s) =>
        s.id === stage.id ? { ...s, ...patch } : s,
      ),
    }));
  }
  function patchSettings(patch: Partial<TrainingConfig>) {
    if (!stage) return;
    setNotice("");
    setPlan((current) => ({
      ...current,
      stages: current.stages.map((item) =>
        item.id === stage.id
          ? { ...item, config: { ...item.config, ...patch } }
          : item,
      ),
    }));
  }
  function setting<K extends keyof TrainingConfig>(
    key: K,
    value: TrainingConfig[K],
  ) {
    if (config) patchSettings({ [key]: value });
  }
  function applySavedStage(saved: TrainingStage, submitted: TrainingStage) {
    savedStagesRef.current = { ...savedStagesRef.current, [saved.id]: saved };
    setSavedStages(savedStagesRef.current);
    setPlan((current) => ({
      ...current,
      stages: current.stages.map((item) =>
        item.id !== saved.id
          ? item
          : {
              ...saved,
              name: item.name === submitted.name ? saved.name : item.name,
              script:
                item.script === submitted.script ? saved.script : item.script,
              config: trainingConfigsEqual(item.config, submitted.config)
                ? saved.config
                : item.config,
            },
      ),
    }));
  }
  async function saveStageFile(submitted = stage, announce = true) {
    if (!submitted) return;
    const key = `${project.id}/${submitted.id}`;
    const previous = stageSavesRef.current.get(key);
    const persist = async () => {
      // A queued edit is based on this browser's preceding write. SSH changes
      // still conflict against that write's revision, as they should.
      const latest = previous
        ? savedStagesRef.current[submitted.id]
        : undefined;
      const body = latest
        ? {
            ...submitted,
            script_revision: latest.script_revision,
            config_revision: latest.config_revision,
          }
        : submitted;
      const saved = await trainingRequest(
        `/workspaces/${encodeURIComponent(project.id)}/stages/${encodeURIComponent(submitted.id)}`,
        stageSchema,
        { project, stage: body },
        "PUT",
      );
      applySavedStage(saved, submitted);
      if (announce) setNotice(`Saved server file: ${saved.script_path}`);
      return saved;
    };
    const pending = previous ? previous.then(persist) : persist();
    stageSavesRef.current.set(key, pending);
    try {
      return await pending;
    } finally {
      if (stageSavesRef.current.get(key) === pending)
        stageSavesRef.current.delete(key);
    }
  }
  async function reloadStageFile() {
    if (!stage) return;
    const remote = await trainingRequest(
      `/workspaces/${encodeURIComponent(project.id)}/stages/${encodeURIComponent(stage.id)}`,
      stageSchema,
    );
    savedStagesRef.current = { ...savedStagesRef.current, [remote.id]: remote };
    setSavedStages(savedStagesRef.current);
    setPlan((current) => ({
      ...current,
      stages: current.stages.map((item) =>
        item.id === remote.id ? remote : item,
      ),
    }));
    setNotice("Reloaded the server script and launch settings.");
  }
  async function savePlan() {
    const saved = await trainingRequest(
      `/plans/${encodeURIComponent(project.id)}`,
      planSchema,
      plan,
      "PUT",
    );
    for (const item of saved.stages) {
      const submitted = plan.stages.find((value) => value.id === item.id);
      if (submitted) applySavedStage(item, submitted);
    }
    setNotice("Saved stage scripts and launch settings on the server.");
  }
  return {
    capabilities,
    setCapabilities,
    templates,
    loaded,
    setBootstrapRetry,
    plan,
    setPlan,
    stage,
    config,
    savedStage,
    stageDirty,
    savedStages,
    setSavedStages,
    savedStagesRef,
    cacheKey,
    updateStage,
    patchSettings,
    setting,
    saveStageFile,
    reloadStageFile,
    savePlan,
  };
}
