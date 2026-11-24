import type { RefObject } from "react";
import { Play, Save } from "lucide-react";
import {
  computeSummary,
  type useStageSettings,
} from "../TrainingComputeSettings";
import type { TrainingConfig } from "../trainingApi";
import type { useStageLaunch } from "./useStageLaunch";
export function StageActiveNotice({ active }: { active: boolean }) {
  return active ? (
    <p className="training-next-run-note" role="status">
      These settings apply to the next run. Current training continues
      independently. Change live validation in Monitor.
    </p>
  ) : null;
}

type Props = {
  config: TrainingConfig;
  computeConfig: ReturnType<typeof useStageSettings>["config"];
  codeFirst: boolean;
  stageDirty: boolean;
  editor: "settings" | "script";
  busy: boolean;
  loaded: boolean;
  launchReady: boolean;
  enabled: boolean;
  launchPhase: ReturnType<typeof useStageLaunch>["launchPhase"];
  launchError: string;
  launchErrorRef: RefObject<HTMLDivElement | null>;
  onSave: () => void;
  onLaunch: () => void;
};
export function StageLaunchControls({
  config,
  computeConfig,
  codeFirst,
  stageDirty,
  editor,
  busy,
  loaded,
  launchReady,
  enabled,
  launchPhase,
  launchError,
  launchErrorRef,
  onSave,
  onLaunch,
}: Props) {
  return (
    <>
      <footer className="training-launch">
        <span>
          {config.recipe === "custom" && !codeFirst
            ? "Custom Python · compute applied by script"
            : computeConfig
              ? computeSummary(computeConfig)
              : "Reading training settings…"}
          <small role="status">
            {stageDirty
              ? "Unsaved changes. Start saves the stage first."
              : "Stage saved. Start creates a new run."}
          </small>
        </span>
        {editor === "settings" && (
          <button onClick={onSave} disabled={busy || !loaded}>
            <Save aria-hidden="true" size={14} /> Save stage
          </button>
        )}
        <button
          className="primary"
          disabled={
            busy || !launchReady || (!enabled && config.execution === "local")
          }
          onClick={onLaunch}
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
      {!enabled && (
        <p className="training-alert">
          Local execution is disabled. Run{" "}
          <code>bash scripts/install.sh --training cpu</code> on the server, or
          select an existing environment with scripts/configure.sh, then restart
          Weave.
        </p>
      )}
    </>
  );
}
