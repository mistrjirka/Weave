import { z } from "zod";
import type { TrainingConfig, TrainingGPU } from "../trainingApi";

export function TrainingValidationSettings({
  config,
  gpus,
  codeFirst,
  supported,
  unknown,
  onChange,
}: {
  config: TrainingConfig;
  gpus: TrainingGPU[];
  codeFirst: boolean;
  supported: string[] | null;
  unknown: Set<string>;
  onChange: (patch: Partial<TrainingConfig>) => void;
}) {
  const accepts = (field: string) =>
    supported === null || supported.includes(field);
  return (
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
              !accepts("evaluation_mode") || (!codeFirst && supported === null)
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
  );
}
