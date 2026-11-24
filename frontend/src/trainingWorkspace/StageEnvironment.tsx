import { z } from "zod";
import { pythonEnvironmentSchema, type TrainingStage } from "../trainingApi";
import { trainingConfigsEqual } from "../trainingConfigEquality";
import { copyText } from "../ui";

type Props = {
  stage: TrainingStage;
  savedStage: TrainingStage | undefined;
  codeFirst: boolean;
  pythonEnvironment: z.infer<typeof pythonEnvironmentSchema> | null;
  environmentError: string;
  task: (action: () => Promise<unknown>) => Promise<void>;
  setNotice: (notice: string) => void;
};
export function StageEnvironment({
  stage,
  savedStage,
  codeFirst,
  pythonEnvironment,
  environmentError,
  task,
  setNotice,
}: Props) {
  const config = stage.config;
  return (
    <details className="training-environment-details">
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
            These are server paths, not files on your browser’s computer. Open
            them in your SSH-connected VS Code. Edit train.py and your own
            helpers. Generated network code, tokenizer assets and trainer
            support are kept separately in weave_support/ and refreshed when you
            save.
          </small>
        </div>
        <div className="training-server-path">
          <span>Project folder</span>
          <code>
            {stage.project_path ??
              savedStage?.project_path ??
              "Creating project folder…"}
          </code>
        </div>
        <div className="training-server-path">
          <span>Generated model</span>
          <code>
            {stage.model_path ??
              savedStage?.model_path ??
              "Save the stage to generate weave_support/model.py"}
          </code>
        </div>
        <div className="training-server-path">
          <span>Python script</span>
          <code>
            {stage.script_path ??
              savedStage?.script_path ??
              "Creating server file…"}
          </code>
          <button
            aria-label="Copy Python script server path"
            disabled={!(stage.script_path ?? savedStage?.script_path)}
            onClick={() =>
              void task(async () => {
                await copyText(
                  stage.script_path ?? savedStage?.script_path ?? "",
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
                savedStage?.config_path ??
                "Creating config.json…"}
            </code>
            <button
              aria-label="Copy preset config server path"
              disabled={!(stage.config_path ?? savedStage?.config_path)}
              onClick={() =>
                void task(async () => {
                  await copyText(
                    stage.config_path ?? savedStage?.config_path ?? "",
                  );
                  setNotice("Server config path copied.");
                })
              }
            >
              Copy path
            </button>
          </div>
        )}
        {savedStage && (
          <small>
            {stage.script === savedStage.script &&
            trainingConfigsEqual(config, savedStage.config)
              ? "Browser matches the server files."
              : "Browser edits are not saved to the server yet."}
          </small>
        )}
        {savedStage &&
          ((stage.script_revision !== savedStage.script_revision &&
            stage.script !== savedStage.script) ||
            (stage.config_revision !== savedStage.config_revision &&
              !trainingConfigsEqual(config, savedStage.config))) && (
            <p className="training-alert" role="status">
              Files changed on the server while this browser has a draft. Reload
              the server files, or duplicate this stage to preserve your browser
              draft.
            </p>
          )}
        {stage.workspace_warning && (
          <p className="training-alert">{stage.workspace_warning}</p>
        )}
      </div>
      <div className="training-python-environment" role="status">
        {pythonEnvironment ? (
          <>
            <strong>Python {pythonEnvironment.python_version}</strong>
            <span>
              PyTorch {pythonEnvironment.torch_version ?? "not installed"} ·{" "}
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
  );
}
