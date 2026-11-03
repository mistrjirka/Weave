import "./training.css";
import "./trainingSetup.css";
import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { z } from "zod";
import { createPortal } from "react-dom";
import {
  Activity,
  History,
  X,
  ArrowDown,
  ArrowUp,
  Copy,
  Download,
  Play,
  Pause,
  Plus,
  Save,
  ShieldCheck,
  Square,
  Trash2,
  Upload,
} from "lucide-react";
import type { Project } from "./types";
import { GPUTuningStatus } from "./GPUTuningStatus";
import { FP16Status } from "./FP16Status";
import { TrainingMessages } from "./TrainingMessages";
import { ValidationStatus } from "./ValidationStatus";
import { GeneratedExamples } from "./GeneratedExamples";
import { RunValidationControls } from "./RunValidationControls";
import { CopyTrainingScript } from "./CopyTrainingScript";
import { RunOverview } from "./RunOverview";
import { ExecutionStatus } from "./ExecutionStatus";
import {
  preferredTrainingRun,
  runningTrainingStates as running,
} from "./trainingRunSelection";
import { TrainingCounters } from "./TrainingCounters";
import { TrainingChart } from "./TrainingChart";
import { TrainingETA } from "./TrainingETA";
import {
  TrainingComputeSettings,
  useStageSettings,
  computeSummary,
} from "./TrainingComputeSettings";
import { metricsForDisplay } from "./trainingMetrics";
import { CheckpointChat } from "./chat/CheckpointChat";
import { newId, copyText, Dialog } from "./ui";
import type { PythonEditorHandle } from "./PythonEditor";
const PythonEditor = lazy(() =>
  import("./PythonEditor").then((module) => ({ default: module.PythonEditor })),
);
import {
  capabilitySchema,
  detailSchema,
  eventSchema,
  hardwareSchema,
  inspectTrainingModel,
  planSchema,
  stageSchema,
  pythonEnvironmentSchema,
  runSchema,
  templateSchema,
  trainingRequest,
  TrainingRequestError,
  trainingURL,
  type RunDetail,
  type TrainingConfig,
  type TrainingEvent,
  type TrainingGPU,
  type TrainingPlan,
  type TrainingStage,
  type TrainingRun,
} from "./trainingApi";

const reportingHooks: Record<string, string> = {
  "Report loss with automatic step tracking":
    "# Set up once: with Run(optimizer=optimizer, total_steps=total_steps) as run:\n# After optimizer.step() and scheduler.step(), if used:\nrun.log_loss(loss)\n",
  "Log scalar metrics":
    '# After an optimizer update; adjust names to your loop.\nrun.log({"train/loss": loss.item()}, step=step, tokens=tokens)\n',
  "Automatically track optimizer learning rates":
    "# Inside with Run(), once after constructing the optimizer.\n# Uses the official PyTorch optimizer hook; reports every 10 updates.\nrun.watch_optimizer(optimizer, every=10)\n",
  "Report console text": 'run.console("Training data prepared")\n',
  "Save a generated example":
    "# After generating text with your model.\nrun.sample(prompt=prompt, output=generated_text, checkpoint=checkpoint_path)\n",
  "Poll stop / snapshot requests":
    '# Poll at a safe optimizer boundary, then handle these requests.\ncommands = run.poll_commands()\nstop_requested = any(command["kind"] == "stop" for command in commands)\nsnapshot_ids = [command["id"] for command in commands if command["kind"] == "snapshot"]\n',
  "Publish a completed checkpoint":
    '# Save weights + optimizer/scheduler/scaler/RNG/data state first.\n# Include snapshot_ids from run.poll_commands() to fulfill UI snapshots.\nrun.checkpoint(checkpoint_path, metadata={"snapshot_requests": snapshot_ids})\n',
};

const previewSchema = z.object({
  records: z.array(
    z.object({
      tokens: z.array(
        z.object({ id: z.number(), text: z.string(), supervised: z.boolean() }),
      ),
      supervised_count: z.number(),
    }),
  ),
  note: z.string(),
});
const responseOK = z.object({ authenticated: z.boolean() });
const commandSchema = z.object({
  id: z.string(),
  artifact: z.string().optional(),
});
const bytes = (value: number) =>
  value >= 1024 ** 3
    ? `${(value / 1024 ** 3).toFixed(2)} GB`
    : `${(value / 1024 ** 2).toFixed(1)} MB`;
const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

export function TrainingWorkspace({ project }: { project: Project }) {
  const [capabilities, setCapabilities] = useState<z.infer<
    typeof capabilitySchema
  > | null>(null);
  const [templates, setTemplates] = useState<z.infer<
    typeof templateSchema
  > | null>(null);
  const [plan, setPlan] = useState<TrainingPlan>({
    version: 1,
    project_id: project.id,
    stages: [],
  });
  const [selected, setSelected] = useState(
    () => new URLSearchParams(location.search).get("stage") ?? "",
  );
  const [runId, setRunId] = useState(
    () => new URLSearchParams(location.search).get("run") ?? "",
  );
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [gpus, setGPUs] = useState<TrainingGPU[]>([]);
  const [initialViewReady, setInitialViewReady] = useState(false);
  const [model, setModel] = useState<Awaited<
    ReturnType<typeof inspectTrainingModel>
  > | null>(null);
  const [error, setError] = useState("");
  const [monitorError, setMonitorError] = useState("");
  const [lastRefreshed, setLastRefreshed] = useState<number | null>(null);
  const [notice, setNotice] = useState("");
  const [token, setToken] = useState("");
  const [reportBaseURL, setReportBaseURL] = useState(
    () => `http://localhost:${location.port || "8173"}`,
  );
  const [busy, setBusy] = useState(false);
  const [pauseRequested, setPauseRequested] = useState<string | null>(null);
  const pendingTasksRef = useRef(0);
  const stageSavesRef = useRef(new Map<string, Promise<TrainingStage>>());
  const launchInFlight = useRef(false);
  const [launchPhase, setLaunchPhase] = useState<
    "idle" | "saving" | "starting"
  >("idle");
  const [launchError, setLaunchError] = useState("");
  const launchErrorRef = useRef<HTMLDivElement>(null);
  const dashboardRef = useRef<HTMLElement>(null);
  const scrollToRun = useRef<string | null>(null);
  useEffect(() => {
    if (launchError) {
      launchErrorRef.current?.focus({ preventScroll: true });
      launchErrorRef.current?.scrollIntoView({ block: "nearest" });
    }
  }, [launchError]);
  useEffect(() => {
    if (detail && scrollToRun.current === detail.run.id) {
      dashboardRef.current?.scrollIntoView({ block: "start" });
      scrollToRun.current = null;
    }
  }, [detail?.run.id]);
  const [loaded, setLoaded] = useState(false);
  const initialPanel = useRef(
    new URLSearchParams(location.search).get("training_panel"),
  );
  const [editor, setEditor] = useState<"settings" | "script">(() =>
    initialPanel.current === "script" ? "script" : "settings",
  );
  const [setupOpen, setSetupOpen] = useState(
    () =>
      initialPanel.current === "setup" ||
      initialPanel.current === "script" ||
      (!runId && initialPanel.current !== "monitor"),
  );
  const [historyOpen, setHistoryOpen] = useState(false);
  const [dashboard, setDashboard] = useState<
    "metrics" | "samples" | "hardware" | "logs" | "checkpoints" | "chat"
  >("metrics");
  const [axis, setAxis] = useState<"step" | "tokens" | "time">("step");
  const [smooth, setSmooth] = useState(false);
  const [comparisonId, setComparisonId] = useState("");
  const [comparison, setComparison] = useState<RunDetail | null>(null);
  const [deletedStage, setDeletedStage] = useState<TrainingStage | null>(null);
  const [preview, setPreview] = useState<z.infer<typeof previewSchema> | null>(
    null,
  );
  const [includeData, setIncludeData] = useState(false);
  const [promptsText, setPromptsText] = useState("");
  const [savedStages, setSavedStages] = useState<Record<string, TrainingStage>>(
    {},
  );
  const savedStagesRef = useRef<Record<string, TrainingStage>>({});
  const [pythonEnvironment, setPythonEnvironment] = useState<z.infer<
    typeof pythonEnvironmentSchema
  > | null>(null);
  const [environmentError, setEnvironmentError] = useState("");
  const codeEditorRef = useRef<PythonEditorHandle>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const datasetRef = useRef<HTMLInputElement>(null);
  const stage = plan.stages.find((s) => s.id === selected) ?? plan.stages[0];
  const config = stage?.config;
  const savedStage = stage ? savedStages[stage.id] : undefined;
  const stageDirty =
    !!stage &&
    (!savedStage ||
      stage.name !== savedStage.name ||
      stage.script !== savedStage.script ||
      JSON.stringify(stage.config) !== JSON.stringify(savedStage.config));
  const codeFirst = /^\s*from\s+weave_support\.trainer\s+import\b/m.test(
    stage?.script ?? "",
  );
  const effective = useStageSettings(stage, codeFirst);
  const [parentCheckpoints, setParentCheckpoints] = useState<TrainingEvent[]>(
    [],
  );
  const [parentCheckpointError, setParentCheckpointError] = useState("");
  useEffect(() => {
    setParentCheckpoints([]);
    setParentCheckpointError("");
    if (
      !capabilities?.authenticated ||
      !config?.parent_run ||
      config.start_mode === "random"
    )
      return;
    const controller = new AbortController();
    const parent = config.parent_run;
    void (async () => {
      let after = "";
      let records: TrainingEvent[] = [];
      while (!controller.signal.aborted) {
        const page = await trainingRequest(
          `/runs/${encodeURIComponent(parent)}/event-records?kind=checkpoint&limit=100&after=${encodeURIComponent(after)}`,
          z.object({ events: z.array(eventSchema), next_after: z.string() }),
          undefined,
          undefined,
          controller.signal,
        );
        if (!page.events.length) break;
        records = [...records, ...page.events].slice(-500);
        after = page.next_after;
      }
      if (!controller.signal.aborted) setParentCheckpoints(records);
    })().catch((failure: unknown) => {
      if (!controller.signal.aborted)
        setParentCheckpointError(message(failure));
    });
    return () => controller.abort();
  }, [capabilities?.authenticated, config?.parent_run, config?.start_mode]);

  useEffect(() => {
    if (detail?.reporting_url)
      setReportBaseURL(
        detail.reporting_url.replace(/\/api\/training\/reports\/[^/]+$/, ""),
      );
  }, [detail?.run.id, detail?.reporting_url]);
  useEffect(() => {
    if (
      !initialViewReady &&
      loaded &&
      (setupOpen || !runId || detail || monitorError)
    ) {
      setInitialViewReady(true);
    }
  }, [initialViewReady, loaded, setupOpen, runId, detail, monitorError]);
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
      trainingRequest(
        `/runs?project_id=${encodeURIComponent(project.id)}`,
        z.array(runSchema),
        undefined,
        undefined,
        controller.signal,
      ),
      trainingRequest(
        "/hardware",
        hardwareSchema,
        undefined,
        undefined,
        controller.signal,
      ),
    ])
      .then(([t, p, r, h]) => {
        if (controller.signal.aborted) return;
        setTemplates(t);
        savedStagesRef.current = Object.fromEntries(
          p.stages.map((item) => [item.id, item]),
        );
        setSavedStages(savedStagesRef.current);
        setRuns(r);
        setGPUs(h.gpus);
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
                stages: recovered.stages.map((draft) => {
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
                      (JSON.stringify(draft.config) ===
                      JSON.stringify(disk.config)
                        ? disk.config_revision
                        : null),
                  };
                }),
              }
            : p,
        );
        setLoaded(true);
        if (!runId || !r.some((run) => run.id === runId)) {
          setMonitorError("");
          const replacement = preferredTrainingRun(r, selected);
          setRunId(replacement?.id ?? "");
          if (replacement) {
            setSelected(replacement.stage_id);
            if (!initialPanel.current) setSetupOpen(false);
          } else setSetupOpen(true);
          if (runId) {
            setNotice(
              replacement
                ? "The linked run is unavailable. Opened the latest available run."
                : "The linked run is unavailable. No saved runs were found for this project.",
            );
          }
        }
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(message(e));
      });
    return () => controller.abort();
    // Initial run choice must not reload unsaved stages on every selection.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [capabilities?.authenticated, project.id, cacheKey]);
  useEffect(() => {
    if (!capabilities?.authenticated || !setupOpen) return;
    const controller = new AbortController();
    setModel(null);
    void inspectTrainingModel(project)
      .then((value) => {
        if (!controller.signal.aborted) setModel(value);
      })
      .catch(() => {
        if (!controller.signal.aborted) setModel(null);
      });
    return () => controller.abort();
  }, [project, capabilities?.authenticated, setupOpen]);
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
    trainingURL({
      stage: stage?.id,
      run: runId || undefined,
      training_panel: setupOpen
        ? editor === "script"
          ? "script"
          : "setup"
        : "monitor",
    });
  }, [stage?.id, runId, setupOpen, editor]);
  useEffect(() => {
    setPreview(null);
  }, [
    stage?.id,
    config?.dataset,
    config?.sequence_length,
    config?.recipe,
    config?.chat_template,
  ]);
  useEffect(() => {
    setPromptsText(JSON.stringify(stage?.config.prompts ?? [], null, 2));
  }, [stage?.id]);
  useEffect(() => {
    if (!capabilities?.authenticated || !setupOpen) return;
    const controller = new AbortController();
    setPythonEnvironment(null);
    setEnvironmentError("");
    const executable = config?.python ?? capabilities.python;
    const timer = window.setTimeout(() => {
      void trainingRequest(
        `/environment?python=${encodeURIComponent(executable)}`,
        pythonEnvironmentSchema,
        undefined,
        undefined,
        controller.signal,
      )
        .then(setPythonEnvironment)
        .catch((e) => {
          if (!controller.signal.aborted) setEnvironmentError(message(e));
        });
    }, 600);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [
    config?.python,
    capabilities?.python,
    capabilities?.authenticated,
    setupOpen,
  ]);
  useEffect(() => {
    if (!stage?.script_path || !capabilities?.authenticated) return;
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
              JSON.stringify(item.config) === JSON.stringify(previous.config)
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
  }, [stage?.id, stage?.script_path, capabilities?.authenticated, project.id]);
  useEffect(() => {
    if (!runId || !capabilities?.authenticated) {
      setDetail(null);
      setMonitorError("");
      setLastRefreshed(null);
      return;
    }
    const controller = new AbortController();
    setDetail(null);
    setMonitorError("");
    setLastRefreshed(null);
    let pending = false;
    const runEndpoint = `/runs/${encodeURIComponent(runId)}`;
    const recoverMissingRun = async () => {
      const list = await trainingRequest(
        `/runs?project_id=${encodeURIComponent(project.id)}`,
        z.array(runSchema),
        undefined,
        undefined,
        controller.signal,
      );
      if (controller.signal.aborted) return;
      const replacement = preferredTrainingRun(
        list.filter((run) => run.id !== runId),
        selected,
      );
      setRuns(list);
      setMonitorError("");
      setRunId(replacement?.id ?? "");
      if (replacement) setSelected(replacement.stage_id);
      else setSetupOpen(true);
      setNotice(
        replacement
          ? "The selected run is unavailable. Opened the latest available run."
          : "The selected run is unavailable. No saved runs were found for this project.",
      );
    };
    const refresh = async () => {
      if (pending) return;
      pending = true;
      try {
        const value = await trainingRequest(
          runEndpoint,
          detailSchema,
          undefined,
          undefined,
          controller.signal,
        );
        if (!controller.signal.aborted) setDetail(value);
        const list = await trainingRequest(
          `/runs?project_id=${encodeURIComponent(project.id)}`,
          z.array(runSchema),
          undefined,
          undefined,
          controller.signal,
        );
        if (!controller.signal.aborted) {
          setRuns(list);
          setMonitorError("");
          setLastRefreshed(Date.now());
        }
      } catch (e) {
        if (controller.signal.aborted) return;
        if (
          e instanceof TrainingRequestError &&
          e.status === 404 &&
          e.path === runEndpoint
        ) {
          try {
            await recoverMissingRun();
          } catch (recoveryError) {
            if (!controller.signal.aborted)
              setMonitorError(message(recoveryError));
          }
        } else setMonitorError(message(e));
      } finally {
        pending = false;
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 3000);
    const stream = new EventSource(
      `/api/training/runs/${encodeURIComponent(runId)}/events`,
    );
    stream.onmessage = () => void refresh();
    return () => {
      controller.abort();
      window.clearInterval(timer);
      stream.close();
    };
  }, [runId, capabilities?.authenticated, project.id, selected]);
  useEffect(() => {
    if (!comparisonId) {
      setComparison(null);
      return;
    }
    const controller = new AbortController();
    void trainingRequest(
      `/runs/${encodeURIComponent(comparisonId)}`,
      detailSchema,
      undefined,
      undefined,
      controller.signal,
    )
      .then(setComparison)
      .catch((e) => {
        if (!controller.signal.aborted) setError(message(e));
      });
    return () => controller.abort();
  }, [comparisonId]);

  async function task(action: () => Promise<unknown>) {
    pendingTasksRef.current += 1;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await action();
    } catch (e) {
      setError(message(e));
    } finally {
      pendingTasksRef.current -= 1;
      setBusy(pendingTasksRef.current > 0);
    }
  }
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
              config:
                JSON.stringify(item.config) === JSON.stringify(submitted.config)
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
  function createStage(next: TrainingStage) {
    const clean = {
      ...next,
      script_path: null,
      config_path: null,
      script_revision: null,
      config_revision: null,
      workspace_warning: null,
      project_path: null,
      model_path: null,
      support_path: null,
      previous_script: null,
    };
    setPlan((current) => ({ ...current, stages: [...current.stages, clean] }));
    setSelected(clean.id);
    setRunId("");
    setDetail(null);
    setSetupOpen(true);
    setHistoryOpen(false);
    setEditor("settings");
    void task(() => saveStageFile(clean));
  }
  function addStage(duplicate = false) {
    if (!templates) return;
    const id = newId();
    const next: TrainingStage =
      duplicate && stage
        ? { ...structuredClone(stage), id, name: `${stage.name} copy` }
        : {
            id,
            name: `Stage ${plan.stages.length + 1}`,
            config: {
              ...structuredClone(templates.config),
              python: capabilities?.python ?? templates.config.python,
            },
            script: templates.scripts.causal,
          };
    createStage(next);
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
  function reorder(direction: number) {
    if (!stage) return;
    setPlan((current) => {
      const stages = [...current.stages],
        index = stages.findIndex((s) => s.id === stage.id),
        target = index + direction;
      if (target < 0 || target >= stages.length) return current;
      [stages[index], stages[target]] = [stages[target], stages[index]];
      return { ...current, stages };
    });
  }
  async function launch() {
    if (!stage || launchInFlight.current) return;
    launchInFlight.current = true;
    setLaunchError("");
    setLaunchPhase("saving");
    try {
      // The Python recipe owns its prompts. Only legacy preset recipes expose
      // the editable JSON field, so a hidden draft must not block code-first runs.
      const submitted = codeFirst
        ? stage
        : {
            ...stage,
            config: {
              ...stage.config,
              prompts: z
                .array(z.union([z.string(), z.record(z.string(), z.unknown())]))
                .parse(JSON.parse(promptsText)),
            },
          };
      const saved = await saveStageFile(submitted, false);
      if (!saved) throw new Error("Select a training stage before launching.");
      setLaunchPhase("starting");
      const run = await trainingRequest("/runs", runSchema, {
        project,
        stage: saved,
      });
      scrollToRun.current = run.id;
      setDetail(null);
      setDashboard("metrics");
      setRuns((current) => [
        run,
        ...current.filter((item) => item.id !== run.id),
      ]);
      setSelected(run.stage_id);
      setRunId(run.id);
      setSetupOpen(false);
      setNotice(
        run.execution === "monitor"
          ? "Stage saved. Package prepared for terminal training."
          : "Stage saved. New training run created.",
      );
    } catch (error) {
      setLaunchError(message(error));
    } finally {
      setLaunchPhase("idle");
      launchInFlight.current = false;
    }
  }
  async function command(kind: string) {
    const result = await trainingRequest(
      `/runs/${encodeURIComponent(runId)}/commands`,
      commandSchema,
      { kind, include_data: includeData },
    );
    if (kind === "stop") setPauseRequested(runId);
    setNotice(
      result.artifact
        ? "Snapshot saved. Open Checkpoints to download it."
        : kind === "snapshot"
          ? "Snapshot requested at the next safe training boundary. A custom script must support checkpoint requests."
          : kind === "stop"
            ? "Pause requested. Waiting for the safe checkpoint boundary…"
            : "Force stop requested.",
    );
    if (kind === "snapshot") setDashboard("checkpoints");
  }
  function numeric(
    key: keyof Pick<
      TrainingConfig,
      | "sequence_length"
      | "batch_size"
      | "accumulation"
      | "max_steps"
      | "learning_rate"
      | "weight_decay"
      | "gradient_clip"
      | "warmup_steps"
      | "log_every"
      | "checkpoint_every"
      | "eval_batches"
      | "generation_tokens"
      | "seed"
      | "cpu_threads"
      | "summary_every"
    >,
    label: string,
    help: string,
  ) {
    return (
      <label>
        {label}
        <input
          type="number"
          aria-label={label}
          value={config?.[key] ?? 0}
          step={key === "learning_rate" || key === "weight_decay" ? "any" : 1}
          min={
            key === "warmup_steps" || key === "weight_decay" || key === "seed"
              ? 0
              : undefined
          }
          onChange={(e) => setting(key, e.target.valueAsNumber || 0)}
        />
        <small>{help}</small>
      </label>
    );
  }
  const checkpoints =
    detail?.events.filter((e) => e.kind === "checkpoint") ?? [];
  const metrics = metricsForDisplay(
    detail?.events ?? [],
    Boolean(detail?.run.code_first),
  );
  const active = detail && running.has(detail.status.state);
  const otherActiveRun =
    detail && !active
      ? runs.find(
          (run) =>
            run.id !== detail.run.id &&
            run.stage_id === detail.run.stage_id &&
            running.has(run.status?.state ?? ""),
        )
      : undefined;
  function selectStage(id: string) {
    setSelected(id);
    const latest = preferredTrainingRun(
      runs.filter((run) => run.stage_id === id),
      id,
    );
    setRunId(latest?.id ?? "");
    setDetail(null);
    setSetupOpen(true);
    setEditor("settings");
    setNotice("");
    setLaunchError("");
  }
  function selectRun(run: TrainingRun) {
    setDetail(null);
    setRunId(run.id);
    setSelected(run.stage_id);
    setSetupOpen(false);
    setHistoryOpen(false);
    setNotice("");
  }
  function runLink(run: TrainingRun) {
    const params = new URLSearchParams(location.search);
    params.set("model", project.id);
    params.set("view", "training");
    params.set("run", run.id);
    params.set("stage", run.stage_id);
    return `?${params.toString()}`;
  }
  const latestHardware = detail?.hardware.at(-1);
  const artifactURL = (path: string) =>
    `/api/training/runs/${encodeURIComponent(runId)}/artifact?path=${encodeURIComponent(path)}`;

  const historyButton = (
    <button
      className="training-history-button"
      aria-label="Stages and run history"
      disabled={launchPhase !== "idle"}
      aria-haspopup="dialog"
      onClick={() => setHistoryOpen(true)}
    >
      <History size={14} aria-hidden="true" /> <span>Stages & history</span>
    </button>
  );

  if (!capabilities || (capabilities.authenticated && !initialViewReady))
    return (
      <section
        className="training-workspace training-loading"
        aria-label="Training workspace"
        aria-busy="true"
      >
        <p role="status">Loading training workspace…</p>
        {error || monitorError ? (
          <p role="alert">{error || monitorError}</p>
        ) : (
          <div className="training-loading-layout" aria-hidden="true">
            <div className="training-loading-bar" />
            <div className="training-loading-card">
              <div className="training-loading-bar" />
              <div className="training-loading-plots">
                <div />
                <div />
                <div />
                <div />
              </div>
            </div>
          </div>
        )}
      </section>
    );
  if (!capabilities.authenticated)
    return (
      <section className="training-workspace training-unlock">
        <ShieldCheck aria-hidden="true" size={32} />
        <h2>Train and track your model</h2>
        <p>
          Create editable training stages, watch generated examples, and keep
          checkpoints with their data and scripts.
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void task(async () => {
              await trainingRequest("/session", responseOK, { token });
              setToken("");
              setCapabilities(
                await trainingRequest("/capabilities", capabilitySchema),
              );
            });
          }}
        >
          <label>
            Operator access code
            <input
              type="password"
              autoComplete="current-password"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              required
            />
          </label>
          <button className="primary" disabled={busy}>
            Unlock training
          </button>
        </form>
        <p className="training-muted">
          This server has optional training authentication enabled. Use the
          WEAVE_TRAINING_TOKEN supplied by its operator.
        </p>
        {!capabilities.enabled && (
          <p>
            Local launch is disabled. Run{" "}
            <code>bash scripts/install.sh --training cpu</code> on the server,
            or configure an existing training Python, then restart Weave.
          </p>
        )}
        {error && <p role="alert">{error}</p>}
      </section>
    );

  return (
    <section className="training-workspace" aria-label="Training workspace">
      {stage && (
        <div className="training-workflow-bar">
          <label className="training-stage-picker">
            <span>Stage</span>
            <select
              aria-label="Training stage"
              disabled={launchPhase !== "idle"}
              value={stage.id}
              onChange={(event) => selectStage(event.target.value)}
            >
              {plan.stages.map((item, index) => (
                <option key={item.id} value={item.id}>
                  {index + 1}. {item.name}
                </option>
              ))}
            </select>
          </label>
          <nav className="training-workflow-tabs" aria-label="Stage workspace">
            <button
              disabled={launchPhase !== "idle"}
              aria-pressed={setupOpen && editor === "settings"}
              onClick={() => {
                setSetupOpen(true);
                setEditor("settings");
              }}
            >
              Setup
            </button>
            <button
              disabled={launchPhase !== "idle"}
              aria-pressed={setupOpen && editor === "script"}
              onClick={() => {
                setSetupOpen(true);
                setEditor("script");
              }}
            >
              Python script
            </button>
            <button
              aria-label="Monitor"
              aria-pressed={!setupOpen}
              disabled={!runId || launchPhase !== "idle"}
              onClick={() => setSetupOpen(false)}
            >
              Monitor
              {detail && (
                <span
                  className="training-status-dot"
                  data-state={detail.status.state}
                  aria-hidden="true"
                  title={detail.status.state}
                />
              )}
            </button>
          </nav>
          {historyButton}
        </div>
      )}
      {!setupOpen && otherActiveRun && launchPhase === "idle" && (
        <div
          className="training-notice training-current-run-notice"
          role="status"
        >
          <span>
            You’re viewing an earlier {detail?.status.state} run. Another run of
            this stage is running.
          </span>
          <a
            href={runLink(otherActiveRun)}
            onClick={(event) => {
              if (
                event.button !== 0 ||
                event.metaKey ||
                event.ctrlKey ||
                event.shiftKey ||
                event.altKey
              )
                return;
              event.preventDefault();
              selectRun(otherActiveRun);
            }}
          >
            <Activity size={14} aria-hidden="true" /> View running training
          </a>
        </div>
      )}
      {!setupOpen &&
        detail?.status.state === "stopped" &&
        !otherActiveRun &&
        launchPhase === "idle" && (
          <p
            className="training-stopped-banner"
            role="status"
            aria-label="Training stopped"
          >
            <Square size={16} aria-hidden="true" />
            <strong>Training stopped</strong>
            {detail.run.code_first && detail.restart_context?.exact_resume && (
              <span>
                Resume training restores the saved checkpoint without starting
                over.
              </span>
            )}
          </p>
        )}
      {error && (
        <p className="training-alert" role="alert">
          {error}
        </p>
      )}
      {detail?.resume_problem && (
        <div className="training-alert training-resume-problem" role="alert">
          <strong>Checkpoint resume failed.</strong> No training updates were
          applied. Use Retry resume to choose the previous script or try your
          corrected edits.
          <details>
            <summary>Compatibility details</summary>
            <pre>{detail.resume_problem.message}</pre>
            <p>{detail.resume_problem.recovery}</p>
          </details>
        </div>
      )}
      {!setupOpen && monitorError && (
        <p className="training-alert" role="alert">
          Live updates interrupted; displayed run data may be stale. Training
          can continue independently. Retrying automatically. {monitorError}
        </p>
      )}
      {notice && (
        <p className="training-notice" role="status">
          {notice}
        </p>
      )}
      {historyOpen &&
        createPortal(
          <Dialog
            label="Stages and run history"
            className="training-controls training-history-dialog"
            onClose={() => setHistoryOpen(false)}
          >
            <header className="training-history-heading">
              <h2>Stages & run history</h2>
              <button
                className="modal-close"
                aria-label="Close dialog"
                onClick={() => setHistoryOpen(false)}
              >
                <X size={18} aria-hidden="true" />
              </button>
            </header>
            <div className="training-workspace training-history-content">
              <aside className="training-stages" aria-label="Training stages">
                <input
                  hidden
                  ref={fileRef}
                  type="file"
                  accept=".zip"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    void task(async () => {
                      const form = new FormData();
                      form.append("file", file);
                      const imported = await trainingRequest(
                        "/import",
                        runSchema,
                        form,
                      );
                      if (imported.project_id !== project.id)
                        setNotice(
                          `Snapshot belongs to project ${imported.project_id}. Its recorded graph is available in the package.`,
                        );
                      setRunId(imported.id);
                      setRuns(
                        await trainingRequest(
                          `/runs?project_id=${encodeURIComponent(project.id)}`,
                          z.array(runSchema),
                        ),
                      );
                    });
                    e.target.value = "";
                  }}
                />

                <button
                  onClick={() => fileRef.current?.click()}
                  disabled={busy}
                >
                  <Upload aria-hidden="true" size={14} /> Import snapshot
                </button>
                <div className="training-section-heading">
                  <h3>Stages</h3>
                  <button
                    aria-label="Add training stage"
                    onClick={() => addStage()}
                    disabled={!templates}
                  >
                    <Plus aria-hidden="true" size={16} />
                  </button>
                </div>
                {!plan.stages.length && (
                  <p className="training-muted">
                    Start with pretraining, then add a fine-tuning stage using
                    its checkpoint.
                  </p>
                )}
                {plan.stages.map((item, index) => (
                  <button
                    className={`training-stage ${stage?.id === item.id ? "is-active" : ""}`}
                    key={item.id}
                    onClick={() => {
                      selectStage(item.id);
                      setHistoryOpen(false);
                    }}
                  >
                    <span>{index + 1}</span>
                    <div>
                      <strong>{item.name}</strong>
                      <small>
                        {item.config.recipe === "sft"
                          ? "Chat / tool-call SFT"
                          : item.config.recipe === "custom"
                            ? "Custom Python"
                            : "Text pretraining"}
                      </small>
                    </div>
                  </button>
                ))}
                {stage && (
                  <label className="training-stage-rename">
                    Stage name
                    <input
                      aria-label="Stage name"
                      value={stage.name}
                      onChange={(event) =>
                        updateStage({ name: event.target.value })
                      }
                    />
                  </label>
                )}
                <div className="training-stage-tools">
                  <button
                    aria-label="Duplicate stage"
                    disabled={!stage}
                    onClick={() => addStage(true)}
                  >
                    <Copy aria-hidden="true" size={14} />
                  </button>
                  <button
                    aria-label="Move stage up"
                    disabled={!stage || plan.stages[0]?.id === stage.id}
                    onClick={() => reorder(-1)}
                  >
                    <ArrowUp aria-hidden="true" size={14} />
                  </button>
                  <button
                    aria-label="Move stage down"
                    disabled={!stage || plan.stages.at(-1)?.id === stage.id}
                    onClick={() => reorder(1)}
                  >
                    <ArrowDown aria-hidden="true" size={14} />
                  </button>
                  <button
                    aria-label="Delete stage"
                    disabled={!stage}
                    onClick={() => {
                      setDeletedStage(stage ?? null);
                      setPlan((current) => ({
                        ...current,
                        stages: current.stages.filter(
                          (s) => s.id !== stage?.id,
                        ),
                      }));
                      selectStage(
                        plan.stages.find((item) => item.id !== stage?.id)?.id ??
                          "",
                      );
                    }}
                  >
                    <Trash2 aria-hidden="true" size={14} />
                  </button>
                </div>
                {deletedStage && (
                  <button
                    onClick={() => {
                      setPlan((current) => ({
                        ...current,
                        stages: [...current.stages, deletedStage],
                      }));
                      selectStage(deletedStage.id);
                      setDeletedStage(null);
                    }}
                  >
                    Undo stage deletion
                  </button>
                )}
                <button
                  disabled={busy || !loaded}
                  onClick={() => void task(savePlan)}
                >
                  <Save size={14} aria-hidden="true" /> Save stage order
                </button>
                <h3>Run history</h3>
                {!runs.length && (
                  <p className="training-muted">
                    Your training runs will appear here.
                  </p>
                )}
                {runs.map((run) => (
                  <button
                    key={run.id}
                    className={`training-run ${runId === run.id ? "is-active" : ""}`}
                    onClick={() => {
                      selectRun(run);
                    }}
                  >
                    <strong>{run.name}</strong>
                    <small>
                      {run.status?.state ?? "prepared"} ·{" "}
                      {new Date(run.created).toLocaleString()}
                    </small>
                    {run.parent && (
                      <small>
                        {run.revision
                          ? `Resumed · ${run.revision.script_source === "edited" ? "edited script" : "previous script"} · snapshot saved`
                          : "Continues a previous checkpoint"}
                      </small>
                    )}
                  </button>
                ))}
              </aside>
            </div>
          </Dialog>,
          document.body,
        )}
      <div className="training-layout">
        <div className="training-main">
          {stage && config ? (
            setupOpen && (
              <section
                className="training-card training-setup-card"
                aria-label="Stage setup"
              >
                {active && (
                  <p className="training-next-run-note" role="status">
                    These settings apply to the next run. Current training
                    continues independently. Change live validation in Monitor.
                  </p>
                )}
                <div className="training-stage-body">
                  {editor === "script" && (
                    <details
                      key={editor}
                      className="training-environment-details"
                    >
                      <summary>
                        Server files & Python environment
                        {pythonEnvironment
                          ? ` · Python ${pythonEnvironment.python_version}`
                          : ""}
                      </summary>
                      <div className="training-server-files">
                        <div>
                          <strong>Editable files on the Weave server</strong>
                          <small>
                            These are server paths, not files on your browser’s
                            computer. Open them in your SSH-connected VS Code.
                            Edit train.py and your own helpers. Generated
                            network code, tokenizer assets and trainer support
                            are kept separately in weave_support/ and refreshed
                            when you save.
                          </small>
                        </div>
                        <div className="training-server-path">
                          <span>Project folder</span>
                          <code>
                            {stage.project_path ??
                              savedStages[stage.id]?.project_path ??
                              "Creating project folder…"}
                          </code>
                        </div>
                        <div className="training-server-path">
                          <span>Generated model</span>
                          <code>
                            {stage.model_path ??
                              savedStages[stage.id]?.model_path ??
                              "Save the stage to generate weave_support/model.py"}
                          </code>
                        </div>
                        <div className="training-server-path">
                          <span>Python script</span>
                          <code>
                            {stage.script_path ??
                              savedStages[stage.id]?.script_path ??
                              "Creating server file…"}
                          </code>
                          <button
                            aria-label="Copy Python script server path"
                            disabled={
                              !(
                                stage.script_path ??
                                savedStages[stage.id]?.script_path
                              )
                            }
                            onClick={() =>
                              void task(async () => {
                                await copyText(
                                  stage.script_path ??
                                    savedStages[stage.id]?.script_path ??
                                    "",
                                );
                                setNotice("Server script path copied.");
                              })
                            }
                          >
                            Copy path
                          </button>
                        </div>
                        {!codeFirst && (
                          <div className="training-server-path">
                            <span>Preset values</span>
                            <code>
                              {stage.config_path ??
                                savedStages[stage.id]?.config_path ??
                                "Creating config.json…"}
                            </code>
                            <button
                              aria-label="Copy preset config server path"
                              disabled={
                                !(
                                  stage.config_path ??
                                  savedStages[stage.id]?.config_path
                                )
                              }
                              onClick={() =>
                                void task(async () => {
                                  await copyText(
                                    stage.config_path ??
                                      savedStages[stage.id]?.config_path ??
                                      "",
                                  );
                                  setNotice("Server config path copied.");
                                })
                              }
                            >
                              Copy path
                            </button>
                          </div>
                        )}
                        {savedStages[stage.id] && (
                          <small>
                            {stage.script === savedStages[stage.id].script &&
                            JSON.stringify(config) ===
                              JSON.stringify(savedStages[stage.id].config)
                              ? "Browser matches the server files."
                              : "Browser edits are not saved to the server yet."}
                          </small>
                        )}
                        {savedStages[stage.id] &&
                          ((stage.script_revision !==
                            savedStages[stage.id].script_revision &&
                            stage.script !== savedStages[stage.id].script) ||
                            (stage.config_revision !==
                              savedStages[stage.id].config_revision &&
                              JSON.stringify(config) !==
                                JSON.stringify(
                                  savedStages[stage.id].config,
                                ))) && (
                            <p className="training-alert" role="status">
                              Files changed on the server while this browser has
                              a draft. Reload the server files, or duplicate
                              this stage to preserve your browser draft.
                            </p>
                          )}
                        {stage.workspace_warning && (
                          <p className="training-alert">
                            {stage.workspace_warning}
                          </p>
                        )}
                      </div>
                      <div
                        className="training-python-environment"
                        role="status"
                      >
                        {pythonEnvironment ? (
                          <>
                            <strong>
                              Python {pythonEnvironment.python_version}
                            </strong>
                            <span>
                              PyTorch{" "}
                              {pythonEnvironment.torch_version ??
                                "not installed"}{" "}
                              ·{" "}
                              {pythonEnvironment.cuda_available
                                ? `CUDA ${pythonEnvironment.cuda_version} · ${pythonEnvironment.device_count} GPU(s)`
                                : "CPU environment"}
                            </span>
                            <code>{pythonEnvironment.executable}</code>
                          </>
                        ) : (
                          <span>
                            {environmentError ||
                              "Detecting the selected server Python environment…"}
                          </span>
                        )}
                      </div>
                    </details>
                  )}
                  {editor === "script" ? (
                    <>
                      <div className="training-script-location">
                        <code>
                          {stage.script_path ??
                            savedStages[stage.id]?.script_path ??
                            "Save the stage to create train.py"}
                        </code>
                      </div>
                      <p className="training-field-note">
                        Edit data loading, the loss, optimizer and schedule
                        here. Setup controls devices for GUI launches. Saving
                        writes this server file; clean files reload after SSH
                        edits.
                      </p>
                      <div className="training-script-toolbar">
                        <button
                          disabled={busy}
                          onClick={() => void task(() => saveStageFile())}
                        >
                          <Save size={14} aria-hidden="true" />
                          Save file
                        </button>
                        <button
                          disabled={busy || !stage.script_path}
                          onClick={() => void task(reloadStageFile)}
                        >
                          Reload server files
                        </button>
                        <CopyTrainingScript
                          projectId={project.id}
                          disabled={busy}
                          onCopy={(script, source) => {
                            updateStage({ script });
                            setNotice(
                              `Copied script from ${source} into the editor. Review it, then Save file.`,
                            );
                          }}
                        />
                        {!codeFirst && (
                          <label>
                            Insert reporting hook
                            <select
                              aria-label="Insert reporting hook"
                              value=""
                              onChange={(event) => {
                                const snippet =
                                  reportingHooks[event.target.value];
                                if (snippet)
                                  codeEditorRef.current?.insertSnippet(snippet);
                              }}
                            >
                              <option value="">Choose a hook…</option>
                              {Object.keys(reportingHooks).map((key) => (
                                <option value={key} key={key}>
                                  {key}
                                </option>
                              ))}
                            </select>
                          </label>
                        )}
                        <small>Ctrl/Cmd+S save · Ctrl/Cmd+F find</small>
                      </div>
                      <Suspense
                        fallback={<p role="status">Loading Python editor…</p>}
                      >
                        <PythonEditor
                          key={stage.id}
                          ref={codeEditorRef}
                          value={stage.script}
                          onChange={(script) => updateStage({ script })}
                          onSave={() => void task(() => saveStageFile())}
                        />
                      </Suspense>
                      <button
                        onClick={() => {
                          if (templates)
                            createStage({
                              id: newId(),
                              name: `${stage.name} · fresh template`,
                              config: structuredClone(config),
                              script: templates.scripts[config.recipe],
                            });
                        }}
                      >
                        Open fresh template as another stage
                      </button>
                    </>
                  ) : (
                    <>
                      <TrainingComputeSettings
                        config={effective.config}
                        gpus={gpus}
                        model={model}
                        codeFirst={codeFirst}
                        unresolved={effective.unresolved}
                        error={effective.error}
                        onChange={patchSettings}
                      />
                      <details
                        className="training-runtime-settings"
                        open={config.start_mode !== "random"}
                      >
                        <summary>
                          Run environment and starting checkpoint
                        </summary>
                        <div className="training-form-grid">
                          <label>
                            Run on
                            <select
                              value={config.execution}
                              onChange={(e) =>
                                setting(
                                  "execution",
                                  z
                                    .enum(["local", "monitor"])
                                    .parse(e.target.value),
                                )
                              }
                            >
                              <option value="local">
                                Weave host · launch here
                              </option>
                              <option value="monitor">
                                Terminal · prepare package and monitor
                              </option>
                            </select>
                            <small>
                              Launch yourself on this server or another
                              computer; the reporting API sends progress back to
                              Weave.
                            </small>
                          </label>
                          <label>
                            Python executable on server
                            <input
                              value={config.python}
                              disabled={config.execution === "monitor"}
                              onChange={(e) =>
                                setting("python", e.target.value)
                              }
                            />
                            <small>
                              {config.execution === "monitor"
                                ? "Terminal launches use the Python environment you run the package with."
                                : "Use a server environment with PyTorch and your training dependencies installed."}
                            </small>
                          </label>
                        </div>
                        <div className="training-form-grid">
                          <label>
                            Training recipe
                            <select
                              value={config.recipe}
                              disabled={
                                codeFirst &&
                                !(
                                  templates &&
                                  Object.values(templates.scripts).includes(
                                    stage.script,
                                  )
                                )
                              }
                              onChange={(e) => {
                                const recipe = z
                                  .enum(["causal", "sft", "custom"])
                                  .parse(e.target.value);
                                const pristine =
                                  templates &&
                                  Object.values(templates.scripts).includes(
                                    stage.script,
                                  );
                                updateStage({
                                  config: { ...config, recipe },
                                  ...(pristine && templates
                                    ? { script: templates.scripts[recipe] }
                                    : {}),
                                });
                              }}
                            >
                              <option value="causal">
                                Text pretraining · next-token prediction
                              </option>
                              <option value="sft">
                                Chat / tool-call supervised fine-tuning
                              </option>
                              <option value="custom">Custom Python</option>
                            </select>
                            <small>
                              {codeFirst &&
                              !(
                                templates &&
                                Object.values(templates.scripts).includes(
                                  stage.script,
                                )
                              )
                                ? "Your edited Python script defines the objective. Create a new stage to use another starter template."
                                : "Select a starter template. Custom objectives are defined in Python."}
                            </small>
                          </label>
                          <label>
                            Starting state
                            <select
                              value={config.start_mode}
                              onChange={(e) =>
                                setting(
                                  "start_mode",
                                  z
                                    .enum(["random", "weights", "resume"])
                                    .parse(e.target.value),
                                )
                              }
                            >
                              <option value="random">
                                Random initialization
                              </option>
                              <option value="weights">
                                New stage from checkpoint weights
                              </option>
                              <option value="resume">
                                Resume optimizer and training state
                              </option>
                            </select>
                            <small>
                              New stages reset the optimizer; resume continues
                              its state.{" "}
                              {codeFirst &&
                                "Exact resume also requires a restorable data loader."}
                            </small>
                          </label>
                        </div>
                        {config.start_mode !== "random" && (
                          <div className="training-form-grid">
                            <label>
                              Parent run
                              <select
                                value={config.parent_run ?? ""}
                                onChange={(e) => {
                                  updateStage({
                                    config: {
                                      ...config,
                                      parent_run: e.target.value,
                                      parent_checkpoint: null,
                                    },
                                  });
                                }}
                              >
                                <option value="">Select run…</option>
                                {runs.map((run) => (
                                  <option value={run.id} key={run.id}>
                                    {run.name} ·{" "}
                                    {new Date(run.created).toLocaleString()}
                                  </option>
                                ))}
                              </select>
                            </label>
                            <label>
                              Checkpoint path
                              <input
                                value={config.parent_checkpoint ?? ""}
                                onChange={(e) =>
                                  setting("parent_checkpoint", e.target.value)
                                }
                                placeholder="Choose a checkpoint or enter its path…"
                                list="training-checkpoint-paths"
                              />
                              <datalist id="training-checkpoint-paths">
                                {parentCheckpoints
                                  .filter((event) => !event.payload.remote)
                                  .map((event) => (
                                    <option
                                      key={event.id}
                                      value={String(event.payload.path)}
                                    />
                                  ))}
                              </datalist>
                              <small>
                                {parentCheckpointError
                                  ? `Could not load checkpoints: ${parentCheckpointError}`
                                  : "Checkpoints belong to the selected parent run. You can also enter a relative path."}
                              </small>
                            </label>
                          </div>
                        )}
                      </details>
                      {!codeFirst && (
                        <>
                          <h3>Data</h3>
                          <p className="training-muted">
                            The tokenizer comes from your graph’s token input.{" "}
                            <a
                              href={`?model=${encodeURIComponent(project.id)}&node=${encodeURIComponent(project.graph.nodes.find((node) => node.data.kind === "input")?.id ?? "")}`}
                            >
                              Configure the input tokenizer in Architecture
                            </a>
                            .
                          </p>
                          <div className="training-form-grid">
                            <label>
                              Training dataset path on server
                              <input
                                value={config.dataset}
                                placeholder="/path/to/train.jsonl"
                                onChange={(e) =>
                                  setting("dataset", e.target.value)
                                }
                              />
                              <small>
                                Host path: TXT, JSONL, JSON, or Parquet.
                                Pretokenized input_ids/labels also work.
                              </small>
                            </label>
                            <label>
                              Validation dataset path on server
                              <input
                                value={config.eval_dataset}
                                placeholder="Optional held-out dataset"
                                onChange={(e) =>
                                  setting("eval_dataset", e.target.value)
                                }
                              />
                              <small>
                                Use held-out data to track generalization.
                              </small>
                            </label>
                          </div>
                          <input
                            ref={datasetRef}
                            hidden
                            type="file"
                            accept=".txt,.json,.jsonl,.parquet"
                            onChange={(e) => {
                              const file = e.target.files?.[0];
                              if (!file) return;
                              void task(async () => {
                                const form = new FormData();
                                form.append("file", file);
                                const uploaded = await trainingRequest(
                                  "/datasets",
                                  z.object({
                                    path: z.string(),
                                    sha256: z.string(),
                                    bytes: z.number(),
                                  }),
                                  form,
                                );
                                setting("dataset", uploaded.path);
                                setNotice(
                                  `Dataset uploaded and fingerprinted: ${uploaded.sha256.slice(0, 12)}…`,
                                );
                              });
                              e.target.value = "";
                            }}
                          />
                          <button
                            onClick={() => datasetRef.current?.click()}
                            disabled={busy}
                          >
                            <Upload aria-hidden="true" size={14} />
                            Upload training data
                          </button>
                          <button
                            onClick={() =>
                              void task(async () => {
                                setPreview(
                                  await trainingRequest(
                                    "/preview",
                                    previewSchema,
                                    {
                                      project,
                                      stage,
                                    },
                                  ),
                                );
                              })
                            }
                            disabled={busy || !config.dataset}
                          >
                            Preview tokens & loss mask
                          </button>
                          {preview && (
                            <div className="training-data-preview">
                              <p>{preview.note}</p>
                              {preview.records.map((row, index) => (
                                <div key={index}>
                                  <small>
                                    Record {index + 1}: {row.supervised_count}{" "}
                                    supervised target tokens
                                  </small>
                                  <div>
                                    {row.tokens.map((token, i) => (
                                      <span
                                        key={i}
                                        className={
                                          token.supervised
                                            ? "is-supervised"
                                            : ""
                                        }
                                        title={`Token ${token.id}: ${token.supervised ? "contributes to loss" : "ignored as target"}`}
                                      >
                                        {token.text}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                          {config.recipe === "sft" && (
                            <p className="training-muted">
                              Use messages and optional tools schemas.
                              Assistant-only training requires a chat template
                              that supplies assistant token masks, or
                              pretokenized labels with -100 on ignored tokens.
                            </p>
                          )}
                        </>
                      )}
                      {!codeFirst && (
                        <>
                          <details>
                            <summary>
                              Optimizer, schedule, and evaluation
                            </summary>
                            <label className="training-inline-check">
                              <input
                                type="checkbox"
                                checked={config.inspect_blocks}
                                onChange={(e) =>
                                  setting("inspect_blocks", e.target.checked)
                                }
                              />
                              Sample per-block gradient and activation summaries
                              (adds overhead)
                            </label>
                            <div className="training-form-grid">
                              {numeric(
                                "sequence_length",
                                "Context length",
                                "Maximum input tokens per example.",
                              )}
                              {numeric(
                                "batch_size",
                                "Microbatch size",
                                "Examples per GPU per forward pass.",
                              )}
                              {numeric(
                                "accumulation",
                                "Gradient accumulation",
                                "Microbatches per optimizer update.",
                              )}
                              {numeric(
                                "max_steps",
                                "Total optimizer steps",
                                "Resume continues toward this total.",
                              )}
                              {numeric(
                                "learning_rate",
                                "Learning rate",
                                "AdamW learning rate after warmup.",
                              )}
                              {numeric(
                                "weight_decay",
                                "Weight decay",
                                "AdamW regularization.",
                              )}
                              {numeric(
                                "gradient_clip",
                                "Gradient clipping",
                                "Maximum total gradient norm.",
                              )}
                              {numeric(
                                "warmup_steps",
                                "Warmup steps",
                                "Linear warmup, then constant learning rate.",
                              )}
                              {numeric(
                                "checkpoint_every",
                                "Checkpoint interval",
                                "Optimizer steps between checkpoints and evaluations.",
                              )}
                              {numeric(
                                "log_every",
                                "Logging interval",
                                "Optimizer steps between scalar reports.",
                              )}
                              {numeric(
                                "eval_batches",
                                "Validation batches",
                                "Maximum held-out batches per evaluation.",
                              )}
                              {numeric(
                                "generation_tokens",
                                "Generated tokens",
                                "Maximum new tokens per evaluation prompt.",
                              )}
                              {numeric(
                                "seed",
                                "Random seed",
                                "Recorded with each run.",
                              )}
                              {numeric(
                                "cpu_threads",
                                "CPU threads",
                                "PyTorch CPU threads per process. New scripts default to all CPUs available on the machine; set a lower limit to reserve CPU capacity.",
                              )}
                              {numeric(
                                "summary_every",
                                "Block summary interval",
                                "Optimizer steps between sampled block summaries.",
                              )}
                            </div>
                            <label>
                              Evaluation prompts (JSON array)
                              <textarea
                                aria-label="Evaluation prompts"
                                rows={5}
                                value={promptsText}
                                onChange={(e) => {
                                  setPromptsText(e.target.value);
                                  try {
                                    const value = z
                                      .array(
                                        z.union([
                                          z.string(),
                                          z.record(z.string(), z.unknown()),
                                        ]),
                                      )
                                      .parse(JSON.parse(e.target.value));
                                    setting("prompts", value);
                                  } catch {
                                    /* Keep incomplete JSON editable; validate on launch. */
                                  }
                                }}
                              />
                              <small>
                                Strings for plain text, or objects with
                                messages, tools, expected_tool, and
                                expected_arguments.
                              </small>
                            </label>
                            <label>
                              Chat template override (optional)
                              <textarea
                                rows={3}
                                value={config.chat_template}
                                onChange={(e) =>
                                  setting("chat_template", e.target.value)
                                }
                                placeholder="Uses the selected input tokenizer’s template by default"
                              />
                            </label>
                            <label>
                              Custom script configuration (JSON)
                              <textarea
                                rows={3}
                                defaultValue={JSON.stringify(
                                  config.custom,
                                  null,
                                  2,
                                )}
                                key={stage.id}
                                onBlur={(e) => {
                                  try {
                                    setting(
                                      "custom",
                                      z
                                        .record(z.string(), z.unknown())
                                        .parse(JSON.parse(e.target.value)),
                                    );
                                  } catch {
                                    setError(
                                      "Custom configuration must be a JSON object.",
                                    );
                                  }
                                }}
                              />
                            </label>
                          </details>{" "}
                        </>
                      )}
                    </>
                  )}
                  <footer className="training-launch">
                    <span>
                      {effective.config
                        ? computeSummary(effective.config)
                        : "Reading training settings…"}
                      <small role="status">
                        {stageDirty
                          ? "Unsaved changes. Start saves the stage first."
                          : "Stage saved. Start creates a new run."}
                      </small>
                    </span>
                    {editor === "settings" && (
                      <button
                        onClick={() => void task(() => saveStageFile())}
                        disabled={busy || !loaded}
                      >
                        <Save aria-hidden="true" size={14} /> Save stage
                      </button>
                    )}
                    <button
                      className="primary"
                      disabled={
                        busy ||
                        (!capabilities.enabled && config.execution === "local")
                      }
                      onClick={() => void task(launch)}
                    >
                      <Play aria-hidden="true" size={15} />
                      {launchPhase === "saving"
                        ? "Saving stage…"
                        : launchPhase === "starting"
                          ? "Starting training…"
                          : config.execution === "monitor"
                            ? "Prepare training package"
                            : "Start training stage"}
                    </button>
                  </footer>
                  {launchPhase !== "idle" && (
                    <p role="status" aria-label="Training launch progress">
                      {launchPhase === "saving"
                        ? "Saving this stage and its generated model on the server…"
                        : "Stage saved. Creating the new run…"}
                    </p>
                  )}
                  {launchError && (
                    <div
                      className="training-alert"
                      role="alert"
                      aria-label="Training launch failed"
                      ref={launchErrorRef}
                      tabIndex={-1}
                    >
                      <strong>Training did not start.</strong> {launchError}
                    </div>
                  )}
                  {!capabilities.enabled && (
                    <p className="training-alert">
                      Local execution is disabled. Run{" "}
                      <code>bash scripts/install.sh --training cpu</code> on the
                      server, or select an existing environment with
                      scripts/configure.sh, then restart Weave.
                    </p>
                  )}
                </div>
              </section>
            )
          ) : (
            <section className="training-card training-empty">
              <Activity aria-hidden="true" size={32} />
              <h3>Your first training stage</h3>
              {historyButton}
              <p>
                Start with a readable text-training script. Add further stages
                when you’re ready to fine-tune.
              </p>
              <button
                className="primary"
                onClick={() => addStage()}
                disabled={!templates}
              >
                <Plus aria-hidden="true" size={16} />
                Create training stage
              </button>
            </section>
          )}

          {!setupOpen && runId && !detail && launchPhase === "idle" && (
            <p role="status">Loading selected run…</p>
          )}
          {!setupOpen && detail && launchPhase === "idle" && (
            <section
              className="training-card training-dashboard"
              aria-label="Run dashboard"
              ref={dashboardRef}
            >
              <div className="training-section-heading">
                <div>
                  {launchError && (
                    <p className="training-muted">
                      Previous run · the new launch failed
                    </p>
                  )}
                  <h3>
                    {detail.run.name}{" "}
                    <small className="training-run-state">
                      {detail.status.state}
                    </small>
                  </h3>
                  <ExecutionStatus detail={detail} />
                  {lastRefreshed !== null && (
                    <p className="training-muted" data-testid="run-freshness">
                      {monitorError
                        ? "Showing last successful update"
                        : active
                          ? "Live monitoring"
                          : "Run data refreshed"}
                      {" · "}
                      <time dateTime={new Date(lastRefreshed).toISOString()}>
                        {new Date(lastRefreshed).toLocaleTimeString()}
                      </time>
                      {!detail.events.some(
                        (event) => event.kind === "progress",
                      ) && <TrainingETA detail={detail} />}
                    </p>
                  )}
                </div>
                <div className="training-actions">
                  {runs.length > 1 && (
                    <label className="training-run-picker">
                      <span>Run</span>
                      <select
                        aria-label="Training run"
                        value={runId}
                        onChange={(event) => {
                          const run = runs.find(
                            (item) => item.id === event.target.value,
                          );
                          if (run) selectRun(run);
                        }}
                      >
                        {runs.map((run) => (
                          <option key={run.id} value={run.id}>
                            {run.status?.state ?? "prepared"} ·{" "}
                            {new Date(run.created).toLocaleString(undefined, {
                              month: "short",
                              day: "numeric",
                              hour: "2-digit",
                              minute: "2-digit",
                            })}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                  {detail.run.code_first && (
                    <RunValidationControls
                      key={detail.run.id}
                      detail={detail}
                      gpus={gpus}
                      enabled={capabilities.enabled}
                      onRestart={(run) => {
                        setRuns((current) => [
                          run,
                          ...current.filter((item) => item.id !== run.id),
                        ]);
                        setDetail(null);
                        setRunId(run.id);
                        setSetupOpen(false);
                        setNotice(
                          "Restart requested. Restoring the saved checkpoint in a new process. Your editable script was preserved.",
                        );
                      }}
                    />
                  )}
                  {active && (
                    <>
                      <button
                        disabled={busy || pauseRequested === detail.run.id}
                        title="Save a checkpoint and stop workers. Resume restores training state."
                        onClick={() => void task(() => command("stop"))}
                      >
                        <Pause aria-hidden="true" size={14} />
                        {pauseRequested === detail.run.id
                          ? "Pausing…"
                          : "Pause training"}
                      </button>
                      <button
                        disabled={busy || detail.run.execution === "monitor"}
                        onClick={() => void task(() => command("kill"))}
                      >
                        Force stop
                      </button>
                    </>
                  )}
                </div>
              </div>
              <GPUTuningStatus events={detail.events} />
              {detail.status.state === "failed" && detail.status.message && (
                <p role="status" className="training-alert">
                  {detail.status.message}
                </p>
              )}
              {detail.events
                .filter(
                  (event) =>
                    event.kind === "preparation" && event.channel === "system",
                )
                .slice(-1)
                .map((event) => {
                  const documents =
                    typeof event.payload.documents === "number"
                      ? event.payload.documents
                      : 0;
                  const total =
                    typeof event.payload.total_documents === "number"
                      ? event.payload.total_documents
                      : undefined;
                  const cacheReady =
                    event.payload.phase === "complete" ||
                    event.payload.phase === "reused";
                  const preparationInterrupted = ["failed", "stopped"].includes(
                    detail.status.state,
                  );
                  const trainingHasMetrics = detail.events.some(
                    (entry) =>
                      entry.kind === "metrics" && entry.channel === "training",
                  );
                  return (
                    !trainingHasMetrics && (
                      <section
                        key={event.id}
                        className="training-server-files"
                        aria-label="Data preparation progress"
                      >
                        <strong>
                          {preparationInterrupted
                            ? "Data preparation interrupted"
                            : event.payload.phase === "failed"
                              ? "Data preparation failed"
                              : cacheReady
                                ? "Token cache ready"
                                : "Preparing data"}
                        </strong>
                        <p role="status">
                          {preparationInterrupted
                            ? "This run ended before training started. "
                            : event.payload.phase === "waiting"
                              ? "Waiting for another process to finish this cache. "
                              : event.payload.phase === "initializing"
                                ? "Loading tokenizer and creating data loaders. "
                                : ""}
                          {documents.toLocaleString()}
                          {total ? ` / ${total.toLocaleString()}` : ""}{" "}
                          documents ·{" "}
                          {typeof event.payload.tokens === "number"
                            ? event.payload.tokens.toLocaleString()
                            : "0"}{" "}
                          tokens
                        </p>
                        {!cacheReady &&
                          !preparationInterrupted &&
                          event.payload.phase !== "failed" && (
                            <progress
                              aria-label="Cache preparation progress"
                              value={total ? documents : undefined}
                              max={total}
                            />
                          )}
                        {!!event.payload.message && (
                          <p>{String(event.payload.message)}</p>
                        )}
                        {!total && !cacheReady && (
                          <small>
                            The reader did not supply a document total; progress
                            is shown as counts rather than a percentage.
                          </small>
                        )}
                      </section>
                    )
                  );
                })}
              {detail.events
                .filter((e) => e.kind === "progress")
                .slice(-1)
                .map((event) => {
                  const useEpochs =
                    typeof event.payload.total !== "number" &&
                    typeof event.payload.max_epochs === "number";
                  const completed = useEpochs
                    ? event.payload.epochs_completed
                    : event.payload.completed;
                  const total = useEpochs
                    ? event.payload.max_epochs
                    : event.payload.total;
                  const unit = useEpochs ? "epochs" : event.payload.unit;
                  return (
                    <div key={event.id} className="training-progress">
                      <progress
                        aria-label="Training progress"
                        value={
                          typeof completed === "number" &&
                          typeof total === "number"
                            ? completed
                            : undefined
                        }
                        max={typeof total === "number" ? total : undefined}
                      />
                      <span className="training-progress-caption">
                        <span>
                          {typeof completed === "number"
                            ? completed.toLocaleString()
                            : String(completed)}{" "}
                          /{" "}
                          {typeof total === "number"
                            ? total.toLocaleString()
                            : "unknown"}{" "}
                          {String(unit)}
                        </span>
                        <TrainingETA detail={detail} />
                      </span>
                    </div>
                  );
                })}
              <RunOverview detail={detail} />
              <ValidationStatus events={detail.events} />
              <nav className="training-tabs" aria-label="Run views">
                {(
                  [
                    "metrics",
                    "samples",
                    "hardware",
                    "logs",
                    "checkpoints",
                    "chat",
                  ] as const
                ).map((tab) => (
                  <button
                    key={tab}
                    aria-pressed={dashboard === tab}
                    onClick={() => setDashboard(tab)}
                  >
                    {
                      {
                        metrics: "Metrics",
                        samples: "Generated examples",
                        hardware: "Hardware",
                        logs: "Logs",
                        checkpoints: "Checkpoints & snapshots",
                        chat: "Chat & text completion",
                      }[tab]
                    }
                  </button>
                ))}
              </nav>
              {detail.run.execution === "monitor" && (
                <details className="training-reporting-options">
                  <summary>Independent training & reporting</summary>
                  <section
                    className="training-server-files"
                    aria-label="Independent training reporting"
                  >
                    <p>
                      Download the training package and start it yourself. It
                      contains the network, training script, configuration and
                      reporting.json. Reports are sent in the background; a
                      disconnected server does not stop training.
                    </p>
                    <label>
                      Reporting server URL
                      <input
                        type="url"
                        value={reportBaseURL}
                        onChange={(event) =>
                          setReportBaseURL(event.target.value)
                        }
                      />
                      <small>
                        Localhost works when training and Weave run on the same
                        computer, or through an SSH tunnel. On another computer,
                        use this server’s reachable address (for example{" "}
                        {location.origin}).
                      </small>
                    </label>
                    <button
                      disabled={busy}
                      onClick={() =>
                        void task(async () => {
                          await trainingRequest(
                            `/runs/${encodeURIComponent(runId)}/reporting`,
                            z.object({ url: z.string(), token: z.string() }),
                            { base_url: reportBaseURL },
                          );
                          setNotice(
                            "Reporting connection saved. Download the training package again to use this address.",
                          );
                          setDetail(
                            await trainingRequest(
                              `/runs/${encodeURIComponent(runId)}`,
                              detailSchema,
                            ),
                          );
                        })
                      }
                    >
                      Save reporting address
                    </button>
                    <small>
                      Packaged endpoint: <code>{detail.reporting_url}</code>
                    </small>
                    <small>
                      Connection file on server:{" "}
                      <code>{detail.directory}/reporting.json</code>. A scoped
                      reporting key is included automatically; no training login
                      is needed.
                    </small>
                    <details>
                      <summary>
                        Launch command, reporting hooks and remote files
                      </summary>
                      <pre className="training-log">
                        {detail?.run.code_first
                          ? "python train.py"
                          : "python train.py --config config.json"}
                      </pre>
                      <p>
                        Run from the extracted package folder. Edit train.py for
                        that computer’s data paths, devices and Python
                        environment. Custom scripts report loss with run.log();
                        run.watch_optimizer(optimizer) can track learning rates
                        automatically. NVIDIA GPU readings are collected when
                        nvidia-smi is available.
                      </p>
                      <p>
                        Remote checkpoint files and datasets stay on the
                        training computer. Server snapshots preserve the
                        received history and packaged code; copy remote weights
                        separately to back them up.
                      </p>
                    </details>
                    {detail.reporting && (
                      <small>
                        Last report received:{" "}
                        {new Date(
                          detail.reporting.last_received * 1000,
                        ).toLocaleString()}
                        . This is the last report time, not a process liveness
                        check.
                      </small>
                    )}
                  </section>
                </details>
              )}
              {detail.run.execution === "monitor" && (
                <details>
                  <summary>Run this package in your terminal</summary>
                  <p>
                    On this host, keep the working directory and reporting
                    directory pointed at the prepared run:
                  </p>
                  <pre className="training-log">
                    cd {JSON.stringify(detail.directory)}
                    {"\n"}CUDA_DEVICE_ORDER=PCI_BUS_ID WEAVE_RUN_DIR=
                    {JSON.stringify(detail.directory)}{" "}
                    {JSON.stringify(detail.config.python)}{" "}
                    {!detail.run.code_first && detail.config.strategy === "ddp"
                      ? `-m torch.distributed.run --standalone --nproc_per_node=${detail.config.devices.length} `
                      : ""}
                    {detail.run.code_first
                      ? "train.py"
                      : "train.py --config config.json"}
                  </pre>
                  <p>
                    This command is displayed for copying; Weave does not
                    execute shell text.
                  </p>
                </details>
              )}
              {dashboard === "chat" && (
                <CheckpointChat
                  key={runId}
                  runId={runId}
                  gpus={gpus}
                  enabled={capabilities.enabled}
                />
              )}
              {dashboard === "metrics" && (
                <>
                  <details className="training-chart-options">
                    <summary>Chart options</summary>
                    <div className="training-chart-controls">
                      <label>
                        Horizontal axis
                        <select
                          value={axis}
                          onChange={(e) =>
                            setAxis(
                              z
                                .enum(["step", "tokens", "time"])
                                .parse(e.target.value),
                            )
                          }
                        >
                          <option value="step">Optimizer steps</option>
                          <option value="tokens">Tokens processed</option>
                          <option value="time">Elapsed seconds</option>
                        </select>
                      </label>
                      <label>
                        Compare run
                        <select
                          value={comparisonId}
                          onChange={(e) => setComparisonId(e.target.value)}
                        >
                          <option value="">No comparison</option>
                          {runs
                            .filter((r) => r.id !== runId)
                            .map((r) => (
                              <option key={r.id} value={r.id}>
                                {r.name} ·{" "}
                                {new Date(r.created).toLocaleString()}
                              </option>
                            ))}
                        </select>
                      </label>
                      <label className="training-inline-check">
                        <input
                          type="checkbox"
                          checked={smooth}
                          onChange={(e) => setSmooth(e.target.checked)}
                        />
                        Smooth curves (raw values retained)
                      </label>
                    </div>
                  </details>
                  <div className="training-charts">
                    {metrics.map((metric) => (
                      <TrainingChart
                        key={metric}
                        events={detail.events}
                        comparison={comparison?.events}
                        metric={metric}
                        axis={axis}
                        smooth={smooth}
                      />
                    ))}
                  </div>
                  {!!Object.keys(detail.module_nodes).length &&
                    config?.inspect_blocks && (
                      <details>
                        <summary>Locate blocks in the graph</summary>
                        {Object.entries(detail.module_nodes)
                          .filter(([node]) =>
                            project.graph.nodes.some((n) => n.id === node),
                          )
                          .map(([node, name]) => (
                            <p key={node}>
                              <a
                                href={`?model=${encodeURIComponent(project.id)}&node=${encodeURIComponent(node)}`}
                              >
                                {name} · current graph
                              </a>
                            </p>
                          ))}
                      </details>
                    )}
                  {detail.events.length >= 10000 && (
                    <p className="training-muted">
                      Showing the latest 10,000 events. Full event history is
                      preserved in the run and snapshots.
                    </p>
                  )}
                  {!detail.events.some((event) => event.kind === "metrics") && (
                    <p className="training-muted">
                      {active
                        ? "No training metrics yet. Check Logs for data preparation and startup progress."
                        : detail.status.state === "prepared"
                          ? "Start the prepared package to receive training metrics."
                          : "No training metrics were recorded for this run."}
                    </p>
                  )}
                  <TrainingMessages key={detail.run.id} detail={detail} />
                  <FP16Status events={detail.events} />
                </>
              )}
              {dashboard === "samples" && (
                <GeneratedExamples key={detail.run.id} events={detail.events} />
              )}
              {dashboard === "hardware" && (
                <>
                  <p className="training-muted">
                    GPU readings are device-wide and can include other jobs.
                    Host-process RAM is reported separately.
                  </p>
                  <div className="training-hardware-grid">
                    {latestHardware?.gpus.map((gpu) => (
                      <article className="training-sample" key={gpu.uuid}>
                        <h4>
                          GPU {gpu.index} · {gpu.name}
                        </h4>
                        <p>
                          {gpu.utilization == null
                            ? "Utilization unavailable"
                            : `${gpu.utilization}% utilization`}
                        </p>
                        {gpu.memory_total_mb != null &&
                        gpu.memory_used_mb != null ? (
                          <>
                            <meter
                              aria-label={`GPU ${gpu.index} VRAM used`}
                              min={0}
                              max={gpu.memory_total_mb ?? 1}
                              value={gpu.memory_used_mb ?? 0}
                            />
                            <p>
                              {((gpu.memory_used_mb ?? 0) / 1024).toFixed(1)} /{" "}
                              {((gpu.memory_total_mb ?? 0) / 1024).toFixed(1)}{" "}
                              GB VRAM
                            </p>
                          </>
                        ) : (
                          <p>VRAM readings unavailable</p>
                        )}
                        <p>
                          {gpu.temperature ?? "—"} °C · {gpu.power_watts ?? "—"}{" "}
                          W
                        </p>
                      </article>
                    ))}
                  </div>
                  {latestHardware?.process_rss_bytes !== undefined && (
                    <p>
                      Training launcher RAM:{" "}
                      {bytes(latestHardware.process_rss_bytes)} · Host CPU:{" "}
                      {latestHardware.cpu_percent?.toFixed(0) ?? "—"}%
                    </p>
                  )}
                  <div className="training-charts">
                    {latestHardware?.gpus.map((gpu) => (
                      <TrainingChart
                        key={gpu.uuid}
                        metric={`gpu/${gpu.index}/utilization`}
                        axis="time"
                        smooth={false}
                        events={detail.hardware.map((sample, index) => ({
                          version: 1,
                          id: String(index),
                          time: sample.time,
                          kind: "metrics",
                          step: index,
                          tokens: 0,
                          payload: {
                            [`gpu/${gpu.index}/utilization`]: sample.gpus.find(
                              (g) => g.uuid === gpu.uuid,
                            )?.utilization,
                          },
                        }))}
                      />
                    ))}
                  </div>
                  {!latestHardware && (
                    <p>
                      No hardware samples received. Terminal-launched packages
                      report script events; host monitoring requires the local
                      supervisor.
                    </p>
                  )}
                </>
              )}
              {dashboard === "logs" && (
                <>
                  <TrainingMessages
                    key={detail.run.id}
                    detail={detail}
                    expanded
                  />
                  <h4>Training output</h4>
                  <pre className="training-log" tabIndex={0}>
                    {detail.logs || "No console output yet."}
                  </pre>
                  {detail.evaluation_logs && (
                    <>
                      <h4>Evaluation worker</h4>
                      <pre className="training-log" tabIndex={0}>
                        {detail.evaluation_logs}
                      </pre>
                    </>
                  )}
                  <a href={artifactURL("output.log")}>
                    Download full training log
                  </a>
                  {detail.events
                    .filter(
                      (event) =>
                        event.kind === "profile" &&
                        typeof event.payload.path === "string",
                    )
                    .map((event) => (
                      <p key={event.id}>
                        <a href={artifactURL(String(event.payload.path))}>
                          Download performance trace
                        </a>{" "}
                        · Open in Perfetto or Chrome tracing
                      </p>
                    ))}
                </>
              )}
              {dashboard === "checkpoints" && (
                <>
                  <div className="training-snapshot">
                    <div>
                      <h4>Snapshot this run</h4>
                      <p>
                        Preserve weights, training state, graph, tokenizer,
                        script, data fingerprints, metrics, and saved examples.
                      </p>
                      <label className="training-inline-check">
                        <input
                          type="checkbox"
                          checked={includeData}
                          onChange={(e) => setIncludeData(e.target.checked)}
                        />
                        Include source dataset files (
                        {bytes(detail.source_data_bytes)} additional)
                      </label>
                      <small>
                        Source paths, sizes and modification times are recorded.
                        Full content hashes are included when requested in
                        train.py; source files are copied only when selected
                        here.
                      </small>
                    </div>
                    <button
                      className="primary"
                      disabled={busy}
                      onClick={() => void task(() => command("snapshot"))}
                    >
                      Create snapshot
                    </button>
                  </div>
                  {detail.snapshots.map((snapshot) => (
                    <article key={snapshot.path} className="training-sample">
                      <a href={artifactURL(snapshot.path)}>
                        <Download aria-hidden="true" size={14} />
                        Download snapshot · {bytes(snapshot.bytes)}
                      </a>
                      <p>
                        {snapshot.checkpoint ??
                          "Run record only · no resumable weights"}
                      </p>
                      <small>SHA-256: {snapshot.sha256}</small>
                      {!!snapshot.missing_data.length && (
                        <p className="training-alert">
                          Some source data was missing or changed. See the
                          archive manifest.
                        </p>
                      )}
                    </article>
                  ))}
                  <h4>Published checkpoints</h4>
                  {checkpoints.map((event) => (
                    <article className="training-checkpoint" key={event.id}>
                      <div>
                        <strong>Step {event.step}</strong>
                        <small>
                          {String(event.payload.path)} ·{" "}
                          {bytes(
                            typeof event.payload.bytes === "number"
                              ? event.payload.bytes
                              : 0,
                          )}
                        </small>
                      </div>
                      {event.payload.remote ? (
                        <small>File on training computer · not uploaded</small>
                      ) : (
                        <>
                          <a href={artifactURL(String(event.payload.path))}>
                            Weights & training state
                          </a>
                          <button
                            onClick={() => {
                              if (!stage) return;
                              updateStage({
                                config: {
                                  ...stage.config,
                                  start_mode: "weights",
                                  parent_run: runId,
                                  parent_checkpoint: String(event.payload.path),
                                },
                              });
                              setSetupOpen(true);
                              setEditor("settings");
                              setNotice(
                                "Checkpoint selected for this stage. Choose Resume if you want to restore optimizer state too.",
                              );
                            }}
                          >
                            Continue from here
                          </button>
                        </>
                      )}
                    </article>
                  ))}
                  {!checkpoints.length && (
                    <p>No checkpoint has been published yet.</p>
                  )}
                  <details>
                    <summary>Dataset provenance</summary>
                    {Object.entries(detail.datasets).map(([key, dataset]) => (
                      <div key={key}>
                        <h4>{key}</h4>
                        <p>{dataset.path}</p>
                        <code>
                          {dataset.sha256 ??
                            "Path, size and modification time recorded · content hash not requested"}
                        </code>
                      </div>
                    ))}
                    <p>
                      Source manifests list declared files, which may include
                      files not reached yet. Consumed batch traces record the
                      file, row and token span when the loader supplies source
                      metadata. Enable data_fingerprint="sha256" in train.py for
                      full file hashes; batch hashing is separately optional.
                    </p>
                  </details>
                </>
              )}
              <details>
                <summary>Run files & diagnostics</summary>
                <a
                  className="training-download"
                  href={`/api/training/runs/${encodeURIComponent(runId)}/package`}
                >
                  <Download aria-hidden="true" size={14} />
                  Training package
                </a>
                {detail.status.message && <p>{detail.status.message}</p>}
                <p className="training-muted">
                  Frozen run script on server:{" "}
                  <code>{detail.directory}/train.py</code>. Editing your stage
                  file does not change this run.
                </p>
                <p>{bytes(detail.disk_bytes)} on disk</p>
                <TrainingCounters events={detail.events} />
              </details>
              <button
                onClick={() => {
                  const id = newId();
                  const checkpoint = checkpoints
                    .filter((event) => !event.payload.remote)
                    .at(-1);
                  const next: TrainingStage = {
                    id,
                    name: `${detail.run.name} · next stage`,
                    script: detail.script,
                    config: {
                      ...structuredClone(detail.config),
                      start_mode: checkpoint ? "weights" : "random",
                      parent_run: checkpoint ? detail.run.id : null,
                      parent_checkpoint: checkpoint
                        ? String(checkpoint.payload.path)
                        : null,
                    },
                  };
                  createStage(next);
                  setNotice(
                    "Created an editable stage from the recorded script and configuration.",
                  );
                }}
              >
                Create stage from this run
              </button>
            </section>
          )}
        </div>
      </div>
    </section>
  );
}
