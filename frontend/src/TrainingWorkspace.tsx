import "./training.css";
import { useEffect, useRef, useState } from "react";
import { z } from "zod";
import {
  Activity,
  ArrowDown,
  ArrowUp,
  Copy,
  Download,
  Play,
  Plus,
  Save,
  ShieldCheck,
  Square,
  Trash2,
  Upload,
} from "lucide-react";
import type { Project } from "./types";
import { TrainingChart } from "./TrainingChart";
import { newId } from "./ui";
import {
  capabilitySchema,
  detailSchema,
  hardwareSchema,
  inspectTrainingModel,
  planSchema,
  runSchema,
  templateSchema,
  trainingRequest,
  trainingURL,
  type RunDetail,
  type TrainingConfig,
  type TrainingGPU,
  type TrainingPlan,
  type TrainingStage,
  type TrainingRun,
} from "./trainingApi";

const running = new Set(["starting", "running", "evaluating"]);
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
  const [model, setModel] = useState<Awaited<
    ReturnType<typeof inspectTrainingModel>
  > | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [editor, setEditor] = useState<"settings" | "script">("settings");
  const [dashboard, setDashboard] = useState<
    "metrics" | "samples" | "hardware" | "logs" | "checkpoints"
  >("metrics");
  const [axis, setAxis] = useState<"step" | "tokens" | "time">("step");
  const [smooth, setSmooth] = useState(false);
  const [comparisonId, setComparisonId] = useState("");
  const [comparison, setComparison] = useState<RunDetail | null>(null);
  const [sampleCheckpoint, setSampleCheckpoint] = useState("");
  const [deletedStage, setDeletedStage] = useState<TrainingStage | null>(null);
  const [preview, setPreview] = useState<z.infer<typeof previewSchema> | null>(
    null,
  );
  const [includeData, setIncludeData] = useState(false);
  const [promptsText, setPromptsText] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const datasetRef = useRef<HTMLInputElement>(null);
  const stage = plan.stages.find((s) => s.id === selected) ?? plan.stages[0];
  const config = stage?.config;
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
        setTemplates(t);
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
        setPlan(recovered ?? p);
        setLoaded(true);
        if (!runId && r[0]) setRunId(r[0].id);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(message(e));
      });
    return () => controller.abort();
    // Initial run choice must not reload unsaved stages on every selection.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [capabilities?.authenticated, project.id, cacheKey]);
  useEffect(() => {
    if (!capabilities?.authenticated) return;
    const controller = new AbortController();
    void inspectTrainingModel(project)
      .then((value) => {
        if (!controller.signal.aborted) setModel(value);
      })
      .catch(() => setModel(null));
    return () => controller.abort();
  }, [project, capabilities?.authenticated]);
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
    trainingURL({ stage: stage?.id, run: runId || undefined });
  }, [stage?.id, runId]);
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
    if (!runId || !capabilities?.authenticated) {
      setDetail(null);
      return;
    }
    const controller = new AbortController();
    setDetail(null);
    let pending = false;
    const refresh = async () => {
      if (pending) return;
      pending = true;
      try {
        const value = await trainingRequest(
          `/runs/${encodeURIComponent(runId)}`,
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
        if (!controller.signal.aborted) setRuns(list);
      } catch (e) {
        if (!controller.signal.aborted) setError(message(e));
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
  }, [runId, capabilities?.authenticated, project.id]);
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

  async function task(action: () => Promise<void>) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await action();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  function updateStage(patch: Partial<TrainingStage>) {
    if (!stage) return;
    setPlan((current) => ({
      ...current,
      stages: current.stages.map((s) =>
        s.id === stage.id ? { ...s, ...patch } : s,
      ),
    }));
  }
  function setting<K extends keyof TrainingConfig>(
    key: K,
    value: TrainingConfig[K],
  ) {
    if (config) updateStage({ config: { ...config, [key]: value } });
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
    setPlan((current) => ({ ...current, stages: [...current.stages, next] }));
    setSelected(id);
    setEditor("settings");
  }
  async function savePlan() {
    await trainingRequest(
      `/plans/${encodeURIComponent(project.id)}`,
      planSchema,
      plan,
      "PUT",
    );
    setNotice("Training stages saved.");
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
    if (!stage) return;
    const prompts: unknown = JSON.parse(promptsText);
    const checked = z
      .array(z.union([z.string(), z.record(z.string(), z.unknown())]))
      .parse(prompts);
    const updated = { ...stage, config: { ...stage.config, prompts: checked } };
    updateStage(updated);
    await trainingRequest(
      `/plans/${encodeURIComponent(project.id)}`,
      planSchema,
      {
        ...plan,
        stages: plan.stages.map((s) => (s.id === stage.id ? updated : s)),
      },
      "PUT",
    );
    const run = await trainingRequest("/runs", runSchema, {
      project,
      stage: updated,
    });
    setRuns((current) => [run, ...current]);
    setRunId(run.id);
    setNotice(
      run.execution === "monitor"
        ? "Package prepared. Download it and launch its script in your terminal."
        : "Training started from a frozen copy of this graph and script.",
    );
  }
  async function command(kind: string) {
    const result = await trainingRequest(
      `/runs/${encodeURIComponent(runId)}/commands`,
      commandSchema,
      { kind, include_data: includeData },
    );
    setNotice(
      result.artifact
        ? "Snapshot saved. Open Checkpoints to download it."
        : kind === "snapshot"
          ? "Snapshot requested at the next safe training boundary. A custom script must support checkpoint requests."
          : `${kind === "stop" ? "Safe stop" : "Force stop"} requested.`,
    );
    setDashboard("checkpoints");
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
  const sampleEvents = detail?.events.filter((e) => e.kind === "sample") ?? [];
  const metrics = [
    ...new Set(
      detail?.events
        .filter((e) => e.kind === "metrics")
        .flatMap((e) =>
          Object.entries(e.payload)
            .filter(([, v]) => typeof v === "number")
            .map(([key]) => key),
        ) ?? [],
    ),
  ];
  const active = detail && running.has(detail.status.state);
  const latestHardware = detail?.hardware.at(-1);
  const artifactURL = (path: string) =>
    `/api/training/runs/${encodeURIComponent(runId)}/artifact?path=${encodeURIComponent(path)}`;

  if (!capabilities)
    return (
      <section className="training-workspace">
        <p role="status">Loading training workspace…</p>
        {error && <p role="alert">{error}</p>}
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
          The code is stored on the Weave host at{" "}
          <code>{capabilities.token_file}</code>. Training executes your
          editable Python on that host.
        </p>
        {!capabilities.enabled && (
          <p>
            Local launch is disabled. Enable it with{" "}
            <code>WEAVE_TRAINING_ENABLED=1</code> when starting Weave.
          </p>
        )}
        {error && <p role="alert">{error}</p>}
      </section>
    );

  return (
    <section className="training-workspace" aria-label="Training workspace">
      <header className="training-heading">
        <div>
          <h2>
            <Activity aria-hidden="true" size={20} />
            Training
          </h2>
          <p>Editable stages · frozen run history · reproducible snapshots</p>
        </div>
        <div className="training-actions">
          <button onClick={() => fileRef.current?.click()} disabled={busy}>
            <Upload aria-hidden="true" size={14} />
            Import snapshot
          </button>
          <button
            onClick={() => void task(savePlan)}
            disabled={busy || !loaded}
          >
            <Save aria-hidden="true" size={14} />
            Save stages
          </button>
        </div>
      </header>
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
            const imported = await trainingRequest("/import", runSchema, form);
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
      {error && (
        <p className="training-alert" role="alert">
          {error}
        </p>
      )}
      {notice && (
        <p className="training-notice" role="status">
          {notice}
        </p>
      )}
      <div className="training-layout">
        <aside className="training-stages" aria-label="Training stages">
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
              Start with pretraining, then add a fine-tuning stage using its
              checkpoint.
            </p>
          )}
          {plan.stages.map((item, index) => (
            <button
              className={`training-stage ${stage?.id === item.id ? "is-active" : ""}`}
              key={item.id}
              onClick={() => setSelected(item.id)}
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
                  stages: current.stages.filter((s) => s.id !== stage?.id),
                }));
                setSelected("");
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
                setSelected(deletedStage.id);
                setDeletedStage(null);
              }}
            >
              Undo stage deletion
            </button>
          )}
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
              onClick={() => setRunId(run.id)}
            >
              <strong>{run.name}</strong>
              <small>
                {run.status?.state ?? "prepared"} ·{" "}
                {new Date(run.created).toLocaleString()}
              </small>
              {run.parent && (
                <small>Continues {run.parent.run_id.slice(0, 14)}…</small>
              )}
            </button>
          ))}
        </aside>
        <div className="training-main">
          {stage && config ? (
            <section className="training-card">
              <div className="training-section-heading">
                <input
                  className="training-stage-name"
                  aria-label="Stage name"
                  value={stage.name}
                  onChange={(e) => updateStage({ name: e.target.value })}
                />
                <div className="training-tabs">
                  <button
                    aria-pressed={editor === "settings"}
                    onClick={() => setEditor("settings")}
                  >
                    Settings
                  </button>
                  <button
                    aria-pressed={editor === "script"}
                    onClick={() => setEditor("script")}
                  >
                    Python script
                  </button>
                </div>
              </div>
              {editor === "script" ? (
                <>
                  <p className="training-muted">
                    Your script controls the model, data, loss, and optimizer.
                    Settings are passed through config.json. Form edits never
                    overwrite Python.
                  </p>
                  <textarea
                    className="training-script"
                    aria-label="Training Python script"
                    spellCheck={false}
                    value={stage.script}
                    onChange={(e) => updateStage({ script: e.target.value })}
                  />
                  <button
                    onClick={() => {
                      if (templates) {
                        const id = newId();
                        setPlan((current) => ({
                          ...current,
                          stages: [
                            ...current.stages,
                            {
                              id,
                              name: `${stage.name} · fresh template`,
                              config: structuredClone(config),
                              script: templates.scripts[config.recipe],
                            },
                          ],
                        }));
                        setSelected(id);
                      }
                    }}
                  >
                    Open fresh template as another stage
                  </button>
                </>
              ) : (
                <>
                  <div className="training-form-grid">
                    <label>
                      Training recipe
                      <select
                        value={config.recipe}
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
                        Other objectives remain available through your script.
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
                        <option value="random">Random initialization</option>
                        <option value="weights">
                          New stage from checkpoint weights
                        </option>
                        <option value="resume">
                          Resume optimizer and training state
                        </option>
                      </select>
                      <small>
                        New stages reset the optimizer; resume continues its
                        state.
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
                            setRunId(e.target.value);
                          }}
                        >
                          <option value="">Select run…</option>
                          {runs.map((run) => (
                            <option value={run.id} key={run.id}>
                              {run.name} · {run.id.slice(0, 14)}
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
                          placeholder="Select a checkpoint below"
                          list="training-checkpoint-paths"
                        />
                        <datalist id="training-checkpoint-paths">
                          {checkpoints.map((event) => (
                            <option
                              key={event.id}
                              value={String(event.payload.path)}
                            />
                          ))}
                        </datalist>
                        <small>
                          Use “Continue from here” in checkpoint history to
                          select it.
                        </small>
                      </label>
                    </div>
                  )}
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
                      Training dataset path
                      <input
                        value={config.dataset}
                        placeholder="/path/to/train.jsonl"
                        onChange={(e) => setting("dataset", e.target.value)}
                      />
                      <small>
                        Host path: TXT, JSONL, JSON, or Parquet. Pretokenized
                        input_ids/labels also work.
                      </small>
                    </label>
                    <label>
                      Validation dataset path
                      <input
                        value={config.eval_dataset}
                        placeholder="Optional held-out dataset"
                        onChange={(e) =>
                          setting("eval_dataset", e.target.value)
                        }
                      />
                      <small>Use held-out data to track generalization.</small>
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
                          await trainingRequest("/preview", previewSchema, {
                            project,
                            stage,
                          }),
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
                                  token.supervised ? "is-supervised" : ""
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
                      Use messages and optional tools schemas. Assistant-only
                      training requires a chat template that supplies assistant
                      token masks, or pretokenized labels with -100 on ignored
                      tokens.
                    </p>
                  )}
                  <h3>Compute</h3>
                  <div className="training-form-grid">
                    <label>
                      Run on
                      <select
                        value={config.execution}
                        onChange={(e) =>
                          setting(
                            "execution",
                            z.enum(["local", "monitor"]).parse(e.target.value),
                          )
                        }
                      >
                        <option value="local">Weave host · launch here</option>
                        <option value="monitor">
                          Terminal · prepare package and monitor
                        </option>
                      </select>
                      <small>
                        Remote worker launch is planned for a later version.
                      </small>
                    </label>
                    <label>
                      Python interpreter
                      <input
                        value={config.python}
                        onChange={(e) => setting("python", e.target.value)}
                      />
                      <small>
                        Use an environment with PyTorch and your training
                        dependencies installed.
                      </small>
                    </label>
                    <label>
                      GPU strategy
                      <select
                        value={config.strategy}
                        onChange={(e) => {
                          const strategy = z
                            .enum(["single", "ddp", "split", "train_eval"])
                            .parse(e.target.value);
                          const devices =
                            strategy === "single" || strategy === "train_eval"
                              ? config.devices.slice(0, 1)
                              : config.devices;
                          updateStage({
                            config: {
                              ...config,
                              strategy,
                              devices,
                              precision:
                                strategy === "split"
                                  ? "fp32"
                                  : config.precision,
                              evaluation_device:
                                strategy === "train_eval"
                                  ? (gpus.find(
                                      (gpu) => !devices.includes(gpu.index),
                                    )?.index ?? null)
                                  : config.evaluation_device,
                            },
                          });
                        }}
                      >
                        <option value="single">Single device</option>
                        <option value="ddp" disabled={gpus.length < 2}>
                          Data parallel · model copy on each GPU
                        </option>
                        <option
                          value="split"
                          disabled={!model?.split_supported || gpus.length < 2}
                        >
                          Split model · top-level block placement
                        </option>
                        <option value="train_eval" disabled={gpus.length < 2}>
                          Training GPU + evaluation GPU
                        </option>
                      </select>
                      <small>
                        {config.strategy === "ddp"
                          ? "Improves throughput; each GPU must fit the full model and optimizer."
                          : config.strategy === "split"
                            ? "Sequential model parallelism with gradient-preserving transfers. Keep shared weights on one GPU."
                            : "Choose CPU or an available GPU below."}
                      </small>
                    </label>
                    <label>
                      Precision
                      <select
                        value={config.precision}
                        onChange={(e) =>
                          setting(
                            "precision",
                            z
                              .enum(["fp32", "fp16", "bf16"])
                              .parse(e.target.value),
                          )
                        }
                      >
                        <option value="fp32">FP32</option>
                        <option
                          value="fp16"
                          disabled={
                            !config.devices.length ||
                            config.strategy === "split"
                          }
                        >
                          FP16 with gradient scaling
                        </option>
                        <option
                          value="bf16"
                          disabled={config.strategy === "split"}
                        >
                          BF16 · hardware checked at launch
                        </option>
                      </select>
                    </label>
                  </div>
                  <fieldset className="training-devices">
                    <legend>Training devices</legend>
                    <label>
                      <input
                        type="checkbox"
                        checked={!config.devices.length}
                        onChange={() =>
                          updateStage({
                            config: {
                              ...config,
                              devices: [],
                              strategy: "single",
                              precision: "fp32",
                            },
                          })
                        }
                      />
                      CPU
                    </label>
                    {gpus.map((gpu) => (
                      <label key={gpu.uuid}>
                        <input
                          type="checkbox"
                          checked={config.devices.includes(gpu.index)}
                          onChange={(e) =>
                            setting(
                              "devices",
                              e.target.checked
                                ? config.strategy === "single" ||
                                  config.strategy === "train_eval"
                                  ? [gpu.index]
                                  : [...config.devices, gpu.index]
                                : config.devices.filter((i) => i !== gpu.index),
                            )
                          }
                        />
                        <span>
                          GPU {gpu.index} · {gpu.name}
                          <small>
                            {gpu.memory_total_mb === null
                              ? "Memory unavailable"
                              : `${(gpu.memory_total_mb / 1024).toFixed(1)} GB total · ${((gpu.memory_used_mb ?? 0) / 1024).toFixed(1)} GB used`}
                          </small>
                        </span>
                      </label>
                    ))}
                  </fieldset>
                  {config.strategy === "train_eval" && (
                    <label>
                      Evaluation GPU
                      <select
                        value={config.evaluation_device ?? ""}
                        onChange={(e) =>
                          setting(
                            "evaluation_device",
                            e.target.value === ""
                              ? null
                              : Number(e.target.value),
                          )
                        }
                      >
                        <option value="">Select a separate GPU…</option>
                        {gpus.map((gpu) => (
                          <option
                            key={gpu.index}
                            value={gpu.index}
                            disabled={config.devices.includes(gpu.index)}
                          >
                            GPU {gpu.index} · {gpu.name}
                          </option>
                        ))}
                      </select>
                      <small>
                        Evaluates completed checkpoints; the queue keeps the
                        newest pending checkpoint.
                      </small>
                    </label>
                  )}
                  {config.strategy === "split" && (
                    <div className="training-form-grid">
                      {model?.modules.map((name) => (
                        <label key={name}>
                          {name}
                          <select
                            value={config.placement[name] ?? ""}
                            onChange={(e) =>
                              setting("placement", {
                                ...config.placement,
                                [name]: Number(e.target.value),
                              })
                            }
                          >
                            <option value="">Assign GPU…</option>
                            {config.devices.map((index) => (
                              <option key={index} value={index}>
                                GPU {index}
                              </option>
                            ))}
                          </select>
                        </label>
                      ))}
                    </div>
                  )}
                  {!model?.split_supported &&
                    model?.split_reasons.map((reason) => (
                      <p className="training-muted" key={reason}>
                        Split placement: {reason}
                      </p>
                    ))}
                  <details>
                    <summary>Optimizer, schedule, and evaluation</summary>
                    <label className="training-inline-check">
                      <input
                        type="checkbox"
                        checked={config.inspect_blocks}
                        onChange={(e) =>
                          setting("inspect_blocks", e.target.checked)
                        }
                      />
                      Sample per-block gradient and activation summaries (adds
                      overhead)
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
                        "PyTorch CPU worker threads per process.",
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
                        Strings for plain text, or objects with messages, tools,
                        expected_tool, and expected_arguments.
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
                        defaultValue={JSON.stringify(config.custom, null, 2)}
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
                  </details>
                </>
              )}
              <footer className="training-launch">
                <span>
                  {config.devices.length
                    ? `${config.devices.length} training GPU${config.devices.length > 1 ? "s" : ""}`
                    : "CPU"}{" "}
                  · effective batch{" "}
                  {config.batch_size *
                    config.accumulation *
                    (config.strategy === "ddp" ? config.devices.length : 1)}
                </span>
                <button
                  className="primary"
                  disabled={
                    busy ||
                    (!capabilities.enabled && config.execution === "local")
                  }
                  onClick={() => void task(launch)}
                >
                  <Play aria-hidden="true" size={15} />
                  {config.execution === "monitor"
                    ? "Prepare training package"
                    : "Start training stage"}
                </button>
              </footer>
              {!capabilities.enabled && (
                <p className="training-alert">
                  Local execution is disabled. Set WEAVE_TRAINING_ENABLED=1 when
                  starting Weave.
                </p>
              )}
            </section>
          ) : (
            <section className="training-card training-empty">
              <Activity aria-hidden="true" size={32} />
              <h3>Your first training stage</h3>
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

          {detail && (
            <section
              className="training-card training-dashboard"
              aria-label="Run dashboard"
            >
              <div className="training-section-heading">
                <div>
                  <h3>{detail.run.name}</h3>
                  <p className="training-muted">
                    {detail.status.state} · {bytes(detail.disk_bytes)} on disk ·{" "}
                    {detail.run.id}
                  </p>
                </div>
                <div className="training-actions">
                  {active && (
                    <>
                      <button
                        disabled={busy}
                        onClick={() => void task(() => command("stop"))}
                      >
                        <Square aria-hidden="true" size={14} />
                        Stop safely
                      </button>
                      <button
                        disabled={busy || detail.run.execution === "monitor"}
                        onClick={() => void task(() => command("kill"))}
                      >
                        Force stop
                      </button>
                    </>
                  )}
                  <a
                    className="training-download"
                    href={`/api/training/runs/${encodeURIComponent(runId)}/package`}
                  >
                    <Download aria-hidden="true" size={14} />
                    Training package
                  </a>
                </div>
              </div>
              <button
                onClick={() => {
                  const id = newId();
                  const checkpoint = checkpoints.at(-1);
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
                  setPlan((current) => ({
                    ...current,
                    stages: [...current.stages, next],
                  }));
                  setSelected(id);
                  setNotice(
                    "Created an editable stage from the recorded script and configuration.",
                  );
                }}
              >
                Create stage from this run
              </button>
              {detail.status.message && (
                <p role="status" className="training-alert">
                  {detail.status.message}
                </p>
              )}
              {detail.events
                .filter((e) => e.kind === "progress")
                .slice(-1)
                .map((event) => (
                  <div key={event.id} className="training-progress">
                    <progress
                      aria-label="Training progress"
                      value={
                        typeof event.payload.completed === "number" &&
                        typeof event.payload.total === "number"
                          ? event.payload.completed
                          : undefined
                      }
                      max={
                        typeof event.payload.total === "number"
                          ? event.payload.total
                          : undefined
                      }
                    />
                    <span>
                      {String(event.payload.completed)} /{" "}
                      {String(event.payload.total ?? "unknown")}{" "}
                      {String(event.payload.unit)}
                    </span>
                  </div>
                ))}
              <nav className="training-tabs" aria-label="Run views">
                {(
                  [
                    "metrics",
                    "samples",
                    "hardware",
                    "logs",
                    "checkpoints",
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
                      }[tab]
                    }
                  </button>
                ))}
              </nav>
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
                    {detail.config.strategy === "ddp"
                      ? `-m torch.distributed.run --standalone --nproc_per_node=${detail.config.devices.length} `
                      : ""}
                    train.py --config config.json
                  </pre>
                  <p>
                    This command is displayed for copying; Weave does not
                    execute shell text.
                  </p>
                </details>
              )}
              {dashboard === "metrics" && (
                <>
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
                              {r.name} · {r.id.slice(0, 14)}
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
                  {!metrics.length && (
                    <p className="training-muted">
                      Metrics appear when the script calls run.log().
                    </p>
                  )}
                  {detail.diagnostics.map((diagnostic) => (
                    <p className="training-muted" key={diagnostic}>
                      {diagnostic}
                    </p>
                  ))}
                  {detail.run.warnings.map((warning) => (
                    <p className="training-muted" key={warning.code}>
                      Script check: {warning.message} This is advisory.
                    </p>
                  ))}
                  {detail.events
                    .filter((e) => e.kind === "warning" || e.kind === "error")
                    .slice(-20)
                    .map((event) => (
                      <p className="training-alert" key={event.id}>
                        Step {event.step}: {String(event.payload.message)}
                      </p>
                    ))}
                </>
              )}
              {dashboard === "samples" && (
                <>
                  <label>
                    Checkpoint
                    <select
                      value={sampleCheckpoint}
                      onChange={(e) => setSampleCheckpoint(e.target.value)}
                    >
                      <option value="">All checkpoints</option>
                      {[
                        ...new Set(
                          sampleEvents.map((e) =>
                            String(e.payload.checkpoint ?? "Unspecified"),
                          ),
                        ),
                      ].map((name) => (
                        <option key={name} value={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="training-samples">
                    {sampleEvents
                      .filter(
                        (e) =>
                          !sampleCheckpoint ||
                          String(e.payload.checkpoint ?? "Unspecified") ===
                            sampleCheckpoint,
                      )
                      .map((event) => (
                        <article key={event.id} className="training-sample">
                          <header>
                            <strong>Step {event.step}</strong>
                            <small>
                              {String(
                                event.payload.checkpoint ??
                                  "Checkpoint unspecified",
                              )}
                            </small>
                          </header>
                          <h4>Prompt</h4>
                          <pre>{String(event.payload.prompt)}</pre>
                          <h4>Model output</h4>
                          <pre>{String(event.payload.output)}</pre>
                          <details>
                            <summary>Decoding and evaluation details</summary>
                            <pre>
                              {JSON.stringify(event.payload.metadata, null, 2)}
                            </pre>
                          </details>
                        </article>
                      ))}
                  </div>
                  {!sampleEvents.length && (
                    <p className="training-muted">
                      Add fixed evaluation prompts and select a tokenizer to see
                      how generations change across checkpoints. Custom scripts
                      can call run.sample().
                    </p>
                  )}
                </>
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
                        <p>{gpu.utilization ?? "—"}% utilization</p>
                        <meter
                          aria-label={`GPU ${gpu.index} VRAM used`}
                          min={0}
                          max={gpu.memory_total_mb ?? 1}
                          value={gpu.memory_used_mb ?? 0}
                        />
                        <p>
                          {((gpu.memory_used_mb ?? 0) / 1024).toFixed(1)} /{" "}
                          {((gpu.memory_total_mb ?? 0) / 1024).toFixed(1)} GB
                          VRAM
                        </p>
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
                        Without source files, hashes identify the data but do
                        not back it up.
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
                          setNotice(
                            "Checkpoint selected for this stage. Choose Resume if you want to restore optimizer state too.",
                          );
                        }}
                      >
                        Continue from here
                      </button>
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
                        <code>{dataset.sha256}</code>
                      </div>
                    ))}
                    <p>
                      Source manifests describe declared data. Prepared records
                      and consumed-record traces are recorded by the built-in
                      templates.
                    </p>
                  </details>
                </>
              )}
            </section>
          )}
        </div>
      </div>
    </section>
  );
}
