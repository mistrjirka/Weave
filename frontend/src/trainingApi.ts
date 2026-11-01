import { z } from "zod";
import type { Project } from "./types";

export const trainingConfigSchema = z.object({
  recipe: z.enum(["causal", "sft", "custom"]),
  strategy: z.enum(["single", "ddp", "split", "train_eval"]),
  execution: z.enum(["local", "monitor"]),
  python: z.string(),
  devices: z.array(z.number().int()),
  evaluation_device: z
    .union([z.number().int().nonnegative(), z.literal("cpu")])
    .nullable(),
  evaluation_mode: z.enum(["script", "inline", "async"]).default("script"),
  compute_mode: z.enum(["script", "override"]).default("script"),
  performance_mode: z.enum(["script", "override"]).default("script"),
  compile_mode: z
    .enum(["off", "default", "reduce-overhead", "max-autotune"])
    .default("off"),
  fused_optimizer: z.boolean().default(false),
  profile_steps: z.number().int().min(0).max(100).default(0),
  placement: z.record(z.string(), z.number()),
  use_gpu_tuning: z.boolean().default(true),
  dataset: z.string(),
  eval_dataset: z.string(),
  sequence_length: z.number(),
  batch_size: z.number(),
  accumulation: z.number(),
  max_steps: z.number(),
  learning_rate: z.number(),
  weight_decay: z.number(),
  gradient_clip: z.number(),
  warmup_steps: z.number(),
  log_every: z.number(),
  checkpoint_every: z.number(),
  eval_batches: z.number(),
  generation_tokens: z.number(),
  seed: z.number(),
  precision: z.enum(["fp32", "fp16", "bf16"]),
  start_mode: z.enum(["random", "weights", "resume"]),
  parent_run: z.string().nullable(),
  parent_checkpoint: z.string().nullable(),
  chat_template: z.string(),
  prompts: z.array(z.union([z.string(), z.record(z.string(), z.unknown())])),
  cpu_threads: z.number(),
  inspect_blocks: z.boolean(),
  summary_every: z.number(),
  custom: z.record(z.string(), z.unknown()),
});
export type TrainingConfig = z.infer<typeof trainingConfigSchema>;
export const stageSchema = z.object({
  id: z.string(),
  name: z.string(),
  config: trainingConfigSchema,
  script: z.string(),
  script_path: z.string().nullable().optional(),
  config_path: z.string().nullable().optional(),
  script_revision: z.string().nullable().optional(),
  config_revision: z.string().nullable().optional(),
  workspace_warning: z.string().nullable().optional(),
  project_path: z.string().nullable().optional(),
  model_path: z.string().nullable().optional(),
  support_path: z.string().nullable().optional(),
  previous_script: z.string().nullable().optional(),
});
export type TrainingStage = z.infer<typeof stageSchema>;
export const planSchema = z.object({
  version: z.literal(1),
  project_id: z.string(),
  stages: z.array(stageSchema),
});
export type TrainingPlan = z.infer<typeof planSchema>;
const modelSchema = z.object({
  modules: z.array(z.string()),
  split_supported: z.boolean(),
  split_reasons: z.array(z.string()),
});
export const runSchema = z.object({
  revision: z
    .object({ script_source: z.enum(["previous", "edited"]) })
    .nullable()
    .optional(),
  id: z.string(),
  project_id: z.string(),
  stage_id: z.string(),
  name: z.string(),
  created: z.string(),
  recipe: z.string(),
  strategy: z.string(),
  execution: z.string(),
  code_first: z.boolean().optional(),
  parent: z
    .object({ run_id: z.string(), checkpoint: z.string(), sha256: z.string() })
    .nullable(),
  warnings: z.array(z.object({ code: z.string(), message: z.string() })),
  model: modelSchema,
  status: z.object({ state: z.string() }).optional(),
});
export type TrainingRun = z.infer<typeof runSchema>;
export const eventSchema = z.object({
  version: z.literal(1),
  id: z.string(),
  time: z.number(),
  kind: z.string(),
  step: z.number(),
  tokens: z.number(),
  payload: z.record(z.string(), z.unknown()),
  channel: z.string().optional(),
});
export type TrainingEvent = z.infer<typeof eventSchema>;
const gpuSchema = z.object({
  index: z.number(),
  uuid: z.string(),
  name: z.string(),
  memory_total_mb: z.number().nullable(),
  memory_used_mb: z.number().nullable(),
  utilization: z.number().nullable(),
  temperature: z.number().nullable(),
  power_watts: z.number().nullable(),
});
export type TrainingGPU = z.infer<typeof gpuSchema>;
export const hardwareSchema = z.object({
  gpus: z.array(gpuSchema),
  scope: z.string(),
});
export const capabilitySchema = z.object({
  enabled: z.boolean(),
  authenticated: z.boolean(),
  python: z.string(),
  token_file: z.string().nullable(),
  auth_required: z.boolean().default(false),
  remote_launch: z.boolean(),
  projects_directory: z.string().optional(),
});
export const pythonEnvironmentSchema = z.object({
  executable: z.string(),
  python_version: z.string(),
  torch_version: z.string().nullable(),
  cuda_version: z.string().nullable(),
  cuda_available: z.boolean(),
  device_count: z.number(),
});
export const templateSchema = z.object({
  config: trainingConfigSchema,
  scripts: z.object({
    causal: z.string(),
    sft: z.string(),
    custom: z.string(),
  }),
});
export const validationSettingsSchema = z.object({
  mode: z.enum(["inline", "async"]),
  device: z.union([z.number().int().nonnegative(), z.literal("cpu")]),
  eval_every: z.number().int().positive(),
  eval_batches: z.number().int().positive(),
});
export type ValidationSettings = z.infer<typeof validationSettingsSchema>;

export const detailSchema = z.object({
  revision: z
    .object({
      script_source: z.enum(["previous", "edited"]),
      created: z.string(),
      code_sha256: z.string(),
      state_policy: z.string(),
    })
    .nullable()
    .optional(),
  restart_context: z
    .object({
      source_run: z.string(),
      directory: z.string(),
      checkpoint: z.string(),
      step: z.number(),
      exact_resume: z.boolean(),
      edited_script_path: z.string(),
    })
    .nullable()
    .optional(),
  resume_problem: z
    .object({
      message: z.string(),
      checkpoint: z.string(),
      recovery: z.string(),
    })
    .nullable()
    .optional(),
  validation_control: z
    .object({
      supported: z.boolean(),
      active: z.boolean(),
      current: validationSettingsSchema,
      pending: z
        .object({ id: z.string(), settings: validationSettingsSchema })
        .passthrough()
        .nullable(),
      applied_at_step: z.number().nullable(),
      applied_command_id: z.string().nullable(),
    })
    .optional(),
  run: runSchema,
  status: z.object({
    state: z.string(),
    message: z.string().nullable().optional(),
  }),
  events: z.array(eventSchema),
  hardware: z.array(
    z.object({
      time: z.number(),
      gpus: z.array(gpuSchema),
      cpu_percent: z.number().optional(),
      ram_used_bytes: z.number().optional(),
      ram_total_bytes: z.number().optional(),
      process_rss_bytes: z.number().optional(),
    }),
  ),
  reporting: z
    .object({ remote: z.boolean(), last_received: z.number() })
    .nullable()
    .optional(),
  reporting_url: z.string().nullable().optional(),
  logs: z.string(),
  evaluation_logs: z.string(),
  diagnostics: z.array(z.string()),
  snapshots: z.array(
    z.object({
      path: z.string(),
      checkpoint: z.string().nullable(),
      sha256: z.string(),
      bytes: z.number(),
      missing_data: z.array(z.unknown()),
    }),
  ),
  disk_bytes: z.number(),
  source_data_bytes: z.number(),
  config: trainingConfigSchema,
  module_nodes: z.record(z.string(), z.string()),
  directory: z.string(),
  script: z.string(),
  datasets: z.record(
    z.string(),
    z.object({
      path: z.string(),
      sha256: z.string().optional(),
      bytes: z.number(),
      fingerprint: z.string().optional(),
      mtime_ns: z.number().optional(),
    }),
  ),
});
export type RunDetail = z.infer<typeof detailSchema>;

export class TrainingRequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly path: string,
  ) {
    super(message);
    this.name = "TrainingRequestError";
  }
}

export async function trainingRequest<T>(
  path: string,
  schema: z.ZodType<T>,
  body?: unknown,
  method?: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch("/api/training" + path, {
    cache: "no-store",
    method: method ?? (body === undefined ? "GET" : "POST"),
    signal,
    headers:
      body instanceof FormData
        ? undefined
        : { "Content-Type": "application/json" },
    body:
      body === undefined
        ? undefined
        : body instanceof FormData
          ? body
          : JSON.stringify(body),
  });
  let result: unknown;
  try {
    result = JSON.parse(await response.text());
  } catch {
    throw new TrainingRequestError(
      `Training server returned a non-JSON response (HTTP ${response.status}) for ${method ?? (body === undefined ? "GET" : "POST")} /api/training${path}.`,
      response.status,
      path,
    );
  }
  if (!response.ok) {
    const parsed = z.object({ detail: z.unknown() }).safeParse(result);
    throw new TrainingRequestError(
      parsed.success
        ? typeof parsed.data.detail === "string"
          ? parsed.data.detail
          : JSON.stringify(parsed.data.detail)
        : `Request failed (${response.status})`,
      response.status,
      path,
    );
  }
  return schema.parse(result);
}
export const inspectTrainingModel = (project: Project) =>
  trainingRequest("/model", modelSchema, project);
export function trainingURL(values: { stage?: string; run?: string }) {
  const url = new URL(window.location.href);
  for (const key of ["stage", "run"] as const) {
    if (values[key]) url.searchParams.set(key, values[key]);
    else url.searchParams.delete(key);
  }
  window.history.replaceState(null, "", url);
}
