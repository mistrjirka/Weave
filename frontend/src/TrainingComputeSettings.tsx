import { TrainingValidationSettings } from "./training/TrainingValidationSettings";
import { computeValues, performanceValues } from "./training/stageSettings";
export { useStageSettings } from "./training/stageSettings";
import { z } from "zod";
import { Cpu } from "lucide-react";
import { type TrainingConfig, type TrainingGPU } from "./trainingApi";

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
      ...computeValues(config),
      compute_mode: "override",
      ...patch,
    });
  }
  function performance(patch: Partial<TrainingConfig>) {
    if (!config) return;
    onChange({
      ...performanceValues(config),
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
        <TrainingValidationSettings
          config={config}
          gpus={gpus}
          codeFirst={codeFirst}
          supported={supported}
          unknown={unknown}
          onChange={onChange}
        />
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
                  const value =
                    event.target.value === "" ? 0 : event.target.valueAsNumber;
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
