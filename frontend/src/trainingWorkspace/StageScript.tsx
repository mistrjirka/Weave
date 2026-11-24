import { lazy, Suspense, useRef } from "react";
import { Save } from "lucide-react";
import { z } from "zod";
import { CopyTrainingScript } from "../CopyTrainingScript";
import type { PythonEditorHandle } from "../PythonEditor";
import { newId } from "../ui";
import { templateSchema, type TrainingStage } from "../trainingApi";
const PythonEditor = lazy(() =>
  import("../PythonEditor").then((module) => ({
    default: module.PythonEditor,
  })),
);
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

type Props = {
  stage: TrainingStage;
  savedStage: TrainingStage | undefined;
  projectId: string;
  templates: z.infer<typeof templateSchema> | null;
  busy: boolean;
  codeFirst: boolean;
  updateStage: (patch: Partial<TrainingStage>) => void;
  setNotice: (notice: string) => void;
  createStage: (stage: TrainingStage) => void;
  task: (action: () => Promise<unknown>) => Promise<void>;
  saveStageFile: () => Promise<TrainingStage | undefined>;
  reloadStageFile: () => Promise<void>;
};
export function StageScript({
  stage,
  savedStage,
  projectId,
  templates,
  busy,
  codeFirst,
  updateStage,
  setNotice,
  createStage,
  task,
  saveStageFile,
  reloadStageFile,
}: Props) {
  const codeEditorRef = useRef<PythonEditorHandle>(null);
  const config = stage.config;
  return (
    <>
      <div className="training-script-location">
        <code>
          {stage.script_path ??
            savedStage?.script_path ??
            "Save the stage to create train.py"}
        </code>
      </div>
      <p className="training-field-note">
        Edit data loading, the loss, optimizer and schedule here. Setup controls
        devices for GUI launches. Saving writes this server file; clean files
        reload after SSH edits.
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
          projectId={projectId}
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
                const snippet = reportingHooks[event.target.value];
                if (snippet) codeEditorRef.current?.insertSnippet(snippet);
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
      <Suspense fallback={<p role="status">Loading Python editor…</p>}>
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
  );
}
