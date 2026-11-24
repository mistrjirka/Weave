import { useEffect } from "react";
import { trainingURL, type TrainingStage } from "../trainingApi";
export function useWorkspaceURL(
  stage: TrainingStage | undefined,
  runId: string,
  setupOpen: boolean,
  editor: "settings" | "script",
  loaded: boolean,
) {
  useEffect(() => {
    if (!loaded) return;
    trainingURL({
      stage: stage?.id,
      run: runId || undefined,
      training_panel: setupOpen
        ? editor === "script"
          ? "script"
          : "setup"
        : "monitor",
    });
  }, [stage?.id, runId, setupOpen, editor, loaded]);
}
