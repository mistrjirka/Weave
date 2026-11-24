import { z } from "zod";
import type { TrainingConfig, TrainingStage } from "../trainingApi";

type Props = {
  config: TrainingConfig;
  stage: TrainingStage;
  codeFirst: boolean;
  promptsText: string;
  setPromptsText: (value: string) => void;
  setError: (value: string) => void;
  setting: <K extends keyof TrainingConfig>(
    key: K,
    value: TrainingConfig[K],
  ) => void;
};
export function StageOptimizer({
  config,
  stage,
  codeFirst,
  promptsText,
  setPromptsText,
  setError,
  setting,
}: Props) {
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
    const min =
      key === "sequence_length"
        ? 2
        : key === "learning_rate" ||
            key === "gradient_clip" ||
            key === "warmup_steps" ||
            key === "weight_decay" ||
            key === "seed"
          ? 0
          : 1;
    return (
      <label>
        {label}
        <input
          type="number"
          aria-label={label}
          value={config?.[key] ?? min}
          step={
            key === "learning_rate" ||
            key === "weight_decay" ||
            key === "gradient_clip"
              ? "any"
              : 1
          }
          min={min}
          onChange={({ currentTarget: input }) => {
            const value = input.valueAsNumber;
            if (
              Number.isFinite(value) &&
              input.validity.valid &&
              (value > 0 ||
                (key !== "learning_rate" && key !== "gradient_clip"))
            )
              setting(key, value);
          }}
        />
        <small>{help}</small>
      </label>
    );
  }
  return (
    <>
      {!codeFirst && config.recipe !== "custom" && (
        <>
          <details>
            <summary>Optimizer, schedule, and evaluation</summary>
            <label className="training-inline-check">
              <input
                type="checkbox"
                checked={config.inspect_blocks}
                onChange={(e) => setting("inspect_blocks", e.target.checked)}
              />
              Sample per-block gradient and activation summaries (adds overhead)
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
              {numeric("weight_decay", "Weight decay", "AdamW regularization.")}
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
              {numeric("seed", "Random seed", "Recorded with each run.")}
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
                Strings for plain text, or objects with messages, tools,
                expected_tool, and expected_arguments.
              </small>
            </label>
            <label>
              Chat template override (optional)
              <textarea
                rows={3}
                value={config.chat_template}
                onChange={(e) => setting("chat_template", e.target.value)}
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
                    setError("Custom configuration must be a JSON object.");
                  }
                }}
              />
            </label>
          </details>{" "}
        </>
      )}
    </>
  );
}
