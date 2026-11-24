import { z } from "zod";
import {
  templateSchema,
  type TrainingConfig,
  type TrainingEvent,
  type TrainingRun,
  type TrainingStage,
} from "../trainingApi";

type Props = {
  stage: TrainingStage;
  codeFirst: boolean;
  templates: z.infer<typeof templateSchema> | null;
  runs: TrainingRun[];
  parentCheckpoints: TrainingEvent[];
  parentCheckpointError: string;
  updateStage: (patch: Partial<TrainingStage>) => void;
  setting: <K extends keyof TrainingConfig>(
    key: K,
    value: TrainingConfig[K],
  ) => void;
};
export function StageRuntime({
  stage,
  codeFirst,
  templates,
  runs,
  parentCheckpoints,
  parentCheckpointError,
  updateStage,
  setting,
}: Props) {
  const config = stage.config;
  return (
    <details
      className="training-runtime-settings"
      open={config.start_mode !== "random"}
    >
      <summary>Run environment and starting checkpoint</summary>
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
            Launch yourself on this server or another computer; the reporting
            API sends progress back to Weave.
          </small>
        </label>
        <label>
          Python executable on server
          <input
            value={config.python}
            disabled={config.execution === "monitor"}
            onChange={(e) => setting("python", e.target.value)}
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
                Object.values(templates.scripts).includes(stage.script)
              )
            }
            onChange={(e) => {
              const recipe = z
                .enum(["causal", "sft", "custom"])
                .parse(e.target.value);
              const pristine =
                templates &&
                Object.values(templates.scripts).includes(stage.script);
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
            <option value="sft">Chat / tool-call supervised fine-tuning</option>
            <option value="custom">Custom Python</option>
          </select>
          <small>
            {codeFirst &&
            !(
              templates &&
              Object.values(templates.scripts).includes(stage.script)
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
                z.enum(["random", "weights", "resume"]).parse(e.target.value),
              )
            }
          >
            <option value="random">Random initialization</option>
            <option value="weights">New stage from checkpoint weights</option>
            <option value="resume">Resume optimizer and training state</option>
          </select>
          <small>
            New stages reset the optimizer; resume continues its state.{" "}
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
                  {run.name} · {new Date(run.created).toLocaleString()}
                </option>
              ))}
            </select>
          </label>
          <label>
            Checkpoint path
            <input
              value={config.parent_checkpoint ?? ""}
              onChange={(e) => setting("parent_checkpoint", e.target.value)}
              placeholder="Choose a checkpoint or enter its path…"
              list="training-checkpoint-paths"
            />
            <datalist id="training-checkpoint-paths">
              {parentCheckpoints
                .filter((event) => !event.payload.remote)
                .map((event) => (
                  <option key={event.id} value={String(event.payload.path)} />
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
  );
}
