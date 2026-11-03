import { useEffect, useState } from "react";
import { z } from "zod";
import { Cpu } from "lucide-react";
import {
  trainingConfigSchema,
  trainingRequest,
  type TrainingConfig,
  type TrainingGPU,
  type TrainingStage,
} from "./trainingApi";

const previewSchema = z.object({
  config: trainingConfigSchema,
  unresolved: z.array(z.string()),
  error: z.string().nullable(),
  supported: z.array(z.string()).nullable().optional(),
});
const computeKeys = ["strategy", "devices", "precision", "placement"] as const;
const performanceKeys = [
  "compile_mode",
  "fused_optimizer",
  "profile_steps",
] as const;
function pick<K extends keyof TrainingConfig>(
  config: TrainingConfig,
  keys: readonly K[],
) {
  return Object.fromEntries(keys.map((key) => [key, config[key]])) as Pick<
    TrainingConfig,
    K
  >;
}

/** Preview Python literals on the server; never run a script just to populate a form. */
export function useStageSettings(
  stage: TrainingStage | undefined,
  codeFirst: boolean,
) {
  const [preview, setPreview] = useState<{
    stageId: string;
    script: string;
    value: z.infer<typeof previewSchema>;
  } | null>(null);
  const [error, setError] = useState("");
  const configJSON = JSON.stringify(stage?.config);
  useEffect(() => {
    if (!stage) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setError("");
      void trainingRequest(
        "/settings/preview",
        previewSchema,
        stage,
        "POST",
        controller.signal,
      )
        .then((value) => {
          if (!controller.signal.aborted)
            setPreview({ stageId: stage.id, script: stage.script, value });
        })
        .catch((failure: unknown) => {
          if (!controller.signal.aborted)
            setError(
              failure instanceof Error ? failure.message : String(failure),
            );
        });
    }, 180);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [stage?.id, stage?.script, configJSON]);
  const current =
    preview && preview.stageId === stage?.id && preview.script === stage?.script
      ? preview.value
      : null;
  if (!stage || !current)
    return { config: null, unresolved: [], error, supported: null };
  // Reflect changes immediately, while the debounced preview request catches up.
  const config = {
    ...stage.config,
    ...pick(current.config, computeKeys),
    ...pick(current.config, performanceKeys),
    evaluation_mode: current.config.evaluation_mode,
    evaluation_device: current.config.evaluation_device,
    ...(stage.config.compute_mode === "override"
      ? pick(stage.config, computeKeys)
      : {}),
    ...(stage.config.performance_mode === "override"
      ? pick(stage.config, performanceKeys)
      : {}),
    ...(codeFirst && stage.config.evaluation_mode !== "script"
      ? pick(stage.config, ["evaluation_mode", "evaluation_device"])
      : {}),
  };
  return {
    config,
    unresolved: current.unresolved,
    supported: current.supported ?? null,
    error: error || current.error || "",
  };
}

export function computeSummary(config: TrainingConfig) {
  const devices = config.devices.length
    ? config.devices.map((index) => `GPU ${index}`).join(" + ")
    : "CPU";
  const strategy =
    config.strategy === "ddp"
      ? "Data parallel"
      : config.strategy === "split"
        ? "Split model"
        : "Single device";
  return `${devices} · ${strategy} · ${config.precision.toUpperCase()}`;
}

type ModelPlacement = {
  modules: string[];
  split_supported: boolean;
  split_reasons: string[];
};
export function TrainingComputeSettings({
  config,
  gpus,
  model,
  codeFirst,
  unresolved,
  error,
  onChange,
  supported = null,
}: {
  config: TrainingConfig | null;
  gpus: TrainingGPU[];
  model: ModelPlacement | null;
  codeFirst: boolean;
  supported?: string[] | null;
  unresolved: string[];
  error: string;
  onChange: (patch: Partial<TrainingConfig>) => void;
}) {
  if (!config)
    return <p role="status">{error || "Reading training settings…"}</p>;
  const accepts = (field: string) =>
    supported === null || supported.includes(field);
  if (supported?.length === 0)
    return (
      <section aria-label="Custom runtime settings">
        <h3>Script-managed training</h3>
        <p className="training-field-note">
          This custom script owns its devices, precision and evaluation. Weave
          reports the applied settings when the script starts.
        </p>
        <p className="training-field-note">
          To enable supported GUI controls, read launch_settings() in Python and
          declare WEAVE_SUPPORTED_SETTINGS. See the{" "}
          <a
            href="https://github.com/mistrjirka/Weave/blob/main/docs/custom-training.md"
            target="_blank"
            rel="noreferrer"
          >
            custom training guide
          </a>
          .
        </p>
      </section>
    );
  function compute(patch: Partial<TrainingConfig>) {
    if (!config) return;
    onChange({
      ...pick(config, computeKeys),
      compute_mode: "override",
      ...patch,
    });
  }
  function performance(patch: Partial<TrainingConfig>) {
    if (!config) return;
    onChange({
      ...pick(config, performanceKeys),
      performance_mode: "override",
      ...patch,
    });
  }
  function selectDevices(devices: number[]) {
    if (!config) return;
    compute({
      devices,
      ...(devices.length === 0
        ? { strategy: "single", precision: "fp32" }
        : {}),
    });
    if (!devices.length && config.fused_optimizer)
      performance({ fused_optimizer: false });
  }
  const unknown = new Set(unresolved);
  const missingDevices = config.devices.filter(
    (index) => !gpus.some((gpu) => gpu.index === index),
  );
  return (
    <div className="training-compute-settings">
      {error && (
        <p className="training-alert" role="alert">
          {error}
        </p>
      )}
      {unresolved.length > 0 && (
        <p className="training-alert" role="status">
          Python calculates {unresolved.join(", ")} at runtime. Their values
          cannot be previewed. Selecting values here sets them for this stage.
        </p>
      )}
      <div className="training-compute-columns">
        <section
          className="training-setup-section"
          aria-label="Compute configuration"
        >
          <header>
            <h3>
              <Cpu size={16} aria-hidden="true" /> Training devices
            </h3>
            <p>Choose where the model trains.</p>
          </header>
          <fieldset
            className="training-device-options"
            disabled={!accepts("devices")}
          >
            <legend>Choose CPU or GPUs</legend>
            <label className="training-device-option">
              <input
                type="checkbox"
                checked={!unknown.has("devices") && !config.devices.length}
                onChange={() => selectDevices([])}
              />
              <span>
                <strong>CPU</strong>
                <small>Use the host processor</small>
              </span>
            </label>
            {gpus.map((gpu) => (
              <label key={gpu.uuid} className="training-device-option">
                <input
                  type="checkbox"
                  checked={
                    !unknown.has("devices") &&
                    config.devices.includes(gpu.index)
                  }
                  onChange={(event) => {
                    const devices = event.target.checked
                      ? config.strategy === "single" ||
                        config.strategy === "train_eval"
                        ? [gpu.index]
                        : [...config.devices, gpu.index]
                      : config.devices.filter((index) => index !== gpu.index);
                    selectDevices(devices);
                  }}
                />
                <span>
                  <strong>
                    GPU {gpu.index} · {gpu.name}
                  </strong>
                  <small>
                    {gpu.memory_total_mb === null
                      ? "Memory unavailable"
                      : `${(gpu.memory_total_mb / 1024).toFixed(1)} GB total · ${((gpu.memory_used_mb ?? 0) / 1024).toFixed(1)} GB used`}
                  </small>
                </span>
              </label>
            ))}
            {missingDevices.map((index) => (
              <label
                key={`missing-${index}`}
                className="training-device-option"
              >
                <input
                  type="checkbox"
                  checked
                  onChange={() =>
                    selectDevices(
                      config.devices.filter((device) => device !== index),
                    )
                  }
                />
                <span>
                  <strong>GPU {index}</strong>
                  <small>Not visible on this host</small>
                </span>
              </label>
            ))}
          </fieldset>
          {missingDevices.length > 0 && (
            <p className="training-alert">
              Selected GPU {missingDevices.join(", ")} is not visible on this
              host. Check the environment before launching.
            </p>
          )}
          <div className="training-form-grid">
            <label>
              GPU strategy
              <select
                aria-label="GPU strategy"
                disabled={!accepts("strategy")}
                value={unknown.has("strategy") ? "" : config.strategy}
                onChange={(event) => {
                  const strategy = z
                    .enum(["single", "ddp", "split", "train_eval"])
                    .parse(event.target.value);
                  const devices =
                    strategy === "train_eval" && !config.devices.length
                      ? gpus.slice(0, 1).map((gpu) => gpu.index)
                      : strategy === "single" || strategy === "train_eval"
                        ? config.devices.slice(0, 1)
                        : config.devices.length >= 2
                          ? config.devices
                          : gpus.slice(0, 2).map((gpu) => gpu.index);
                  compute({
                    strategy,
                    devices,
                    precision: strategy === "split" ? "fp32" : config.precision,
                    ...(strategy === "train_eval"
                      ? {
                          evaluation_device:
                            gpus.find((gpu) => !devices.includes(gpu.index))
                              ?.index ?? null,
                        }
                      : {}),
                  });
                  if (strategy === "split" && config.compile_mode !== "off")
                    performance({ compile_mode: "off" });
                }}
              >
                {unknown.has("strategy") && (
                  <option value="">Calculated at runtime</option>
                )}
                <option value="single">Single device</option>
                <option value="ddp" disabled={gpus.length < 2}>
                  Data parallel (DDP)
                </option>
                <option
                  value="split"
                  disabled={gpus.length < 2 || !model?.split_supported}
                >
                  Split model
                </option>
                {((!codeFirst && supported === null) ||
                  config.strategy === "train_eval") && (
                  <option value="train_eval">
                    Training GPU + evaluation GPU
                  </option>
                )}
              </select>
            </label>
            <label>
              Precision
              <select
                aria-label="Precision"
                disabled={!accepts("precision")}
                value={unknown.has("precision") ? "" : config.precision}
                onChange={(event) =>
                  compute({
                    precision: z
                      .enum(["fp32", "fp16", "bf16"])
                      .parse(event.target.value),
                  })
                }
              >
                {unknown.has("precision") && (
                  <option value="">Calculated at runtime</option>
                )}
                <option value="fp32">FP32</option>
                <option
                  value="fp16"
                  disabled={
                    !config.devices.length || config.strategy === "split"
                  }
                >
                  FP16
                </option>
                <option value="bf16" disabled={config.strategy === "split"}>
                  BF16
                </option>
              </select>
            </label>
          </div>
          {config.strategy === "ddp" && (
            <p className="training-field-note">
              Each GPU holds a full model. The data loader must distribute
              batches across ranks.
            </p>
          )}
          {(config.strategy === "ddp" || config.strategy === "split") &&
            config.devices.length < 2 && (
              <p className="training-alert">
                Select at least two GPUs for{" "}
                {config.strategy === "ddp" ? "data parallel" : "split model"}{" "}
                training.
              </p>
            )}
          {config.strategy === "split" && (
            <details open>
              <summary>Block placement</summary>
              <p className="training-field-note">
                Requires FP32 and eager execution. Leave blocks unassigned to
                use a matching tuning profile, or assign every block.
              </p>
              <div className="training-form-grid">
                {model?.modules.map((name) => (
                  <label key={name}>
                    {name}
                    <select
                      value={config.placement[name] ?? ""}
                      onChange={(event) => {
                        const placement = { ...config.placement };
                        if (event.target.value === "") delete placement[name];
                        else placement[name] = Number(event.target.value);
                        compute({ placement });
                      }}
                    >
                      <option value="">Use tuning profile</option>
                      {config.devices.map((index) => (
                        <option key={index} value={index}>
                          GPU {index}
                        </option>
                      ))}
                    </select>
                  </label>
                ))}
              </div>
            </details>
          )}
        </section>
        <section
          className="training-setup-section"
          aria-label="Validation configuration"
        >
          <header>
            <h3>Validation</h3>
            <p>Check quality while training progresses.</p>
          </header>
          <div className="training-form-grid">
            <label>
              Validation execution
              <select
                aria-label="Validation execution"
                disabled={
                  !accepts("evaluation_mode") ||
                  (!codeFirst && supported === null)
                }
                value={
                  unknown.has("async_evaluation") ? "" : config.evaluation_mode
                }
                onChange={(event) =>
                  onChange({
                    evaluation_mode: z
                      .enum(["inline", "async"])
                      .parse(event.target.value),
                    evaluation_device: config.evaluation_device ?? "cpu",
                  })
                }
              >
                {unknown.has("async_evaluation") && (
                  <option value="">Calculated at runtime</option>
                )}
                <option value="inline">Inline · pause training</option>
                <option value="async">Background · keep training</option>
              </select>
            </label>
            <label>
              Validation device
              <select
                aria-label="Validation device"
                value={
                  config.evaluation_mode === "async" ||
                  config.strategy === "train_eval"
                    ? unknown.has("evaluation_device")
                      ? ""
                      : (config.evaluation_device ?? "cpu")
                    : "training"
                }
                disabled={
                  !accepts("evaluation_device") ||
                  (config.evaluation_mode !== "async" &&
                    config.strategy !== "train_eval")
                }
                onChange={(event) =>
                  onChange({
                    evaluation_mode: "async",
                    evaluation_device:
                      event.target.value === "cpu"
                        ? "cpu"
                        : Number(event.target.value),
                  })
                }
              >
                {config.evaluation_mode !== "async" &&
                  config.strategy !== "train_eval" && (
                    <option value="training">Training device</option>
                  )}
                {unknown.has("evaluation_device") && (
                  <option value="">Calculated at runtime</option>
                )}
                <option value="cpu" disabled={!codeFirst && supported === null}>
                  CPU
                </option>
                {gpus.map((gpu) => (
                  <option
                    key={gpu.uuid}
                    value={gpu.index}
                    disabled={
                      !codeFirst &&
                      supported === null &&
                      config.devices.includes(gpu.index)
                    }
                  >
                    GPU {gpu.index} · {gpu.name}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <p className="training-field-note">
            {supported !== null
              ? "Your Python script implements evaluation. The optional AsyncEvaluator runs one immutable snapshot at a time."
              : config.evaluation_mode === "async"
                ? "One background worker evaluates checkpoints. Newer pending checkpoints replace older ones."
                : "Training waits for each validation pass, using the same device."}
          </p>
          {config.evaluation_mode === "async" &&
            typeof config.evaluation_device === "number" &&
            config.devices.includes(config.evaluation_device) && (
              <p className="training-field-note">
                Validation shares a training GPU and will compete for memory and
                compute.
              </p>
            )}
          <p className="training-field-note">
            {supported !== null
              ? "Only declared settings are passed to your evaluation code. Custom live controls must be implemented by the script."
              : codeFirst
                ? "Validation frequency and batches are defined in the Python script. Live changes are available in Monitor."
                : "This legacy recipe validates at checkpoint intervals. Select Training GPU + evaluation GPU for background validation."}
          </p>
        </section>
      </div>
      {codeFirst && (
        <details className="training-performance-settings">
          <summary>Performance and profiling</summary>
          <div className="training-form-grid">
            <label>
              Compilation
              <select
                aria-label="Compilation"
                value={unknown.has("compile_mode") ? "" : config.compile_mode}
                disabled={
                  !accepts("compile_mode") || config.strategy === "split"
                }
                onChange={(event) =>
                  performance({
                    compile_mode: z
                      .enum([
                        "off",
                        "default",
                        "reduce-overhead",
                        "max-autotune",
                      ])
                      .parse(event.target.value),
                  })
                }
              >
                {unknown.has("compile_mode") && (
                  <option value="">Calculated at runtime</option>
                )}
                <option value="off">Off · eager PyTorch</option>
                <option value="default">Default · kernel fusion</option>
                <option value="reduce-overhead">
                  Reduce overhead · CUDA graphs
                </option>
                <option value="max-autotune">Maximum autotuning</option>
              </select>
              <small>
                The first use of a batch shape may compile slowly and use extra
                memory.
              </small>
            </label>
            <label>
              Profile updates (0 disables capture)
              <input
                aria-label="Profile updates (0 disables capture)"
                disabled={!accepts("profile_steps")}
                type="number"
                min={0}
                max={100}
                value={config.profile_steps}
                onChange={(event) => {
                  const value = event.target.valueAsNumber;
                  if (Number.isInteger(value) && value >= 0 && value <= 100)
                    performance({ profile_steps: value });
                }}
              />
              <small>
                Capture after warm-up. Profiling temporarily slows training.
              </small>
            </label>
          </div>
          <label className="training-inline-check">
            <input
              type="checkbox"
              checked={config.fused_optimizer}
              disabled={!accepts("fused_optimizer") || !config.devices.length}
              onChange={(event) =>
                performance({ fused_optimizer: event.target.checked })
              }
            />
            Fused AdamW (CUDA only)
          </label>
        </details>
      )}
      <details>
        <summary>GPU tuning</summary>
        <label className="training-inline-check">
          <input
            type="checkbox"
            disabled={supported !== null}
            checked={config.use_gpu_tuning}
            onChange={(event) =>
              onChange({ use_gpu_tuning: event.target.checked })
            }
          />
          Use a matching saved GPU tuning profile
        </label>
        <p className="training-field-note">
          Manage benchmarks in Weave settings. Explicit block placement takes
          precedence.
        </p>
      </details>
    </div>
  );
}
