import { useRef, useState } from "react";
import { z } from "zod";
import type { Project } from "../types";
import {
  runSchema,
  trainingRequest,
  type TrainingRun,
  type TrainingStage,
} from "../trainingApi";
const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);
export function useStageLaunch(
  project: Project,
  stage: TrainingStage | undefined,
  launchReady: boolean,
  codeFirst: boolean,
  promptsText: string,
  saveStageFile: (
    stage: TrainingStage,
    announce: boolean,
  ) => Promise<TrainingStage | undefined>,
  onStarted: (run: TrainingRun) => void,
) {
  const launchInFlight = useRef(false);
  const [launchPhase, setLaunchPhase] = useState<
    "idle" | "saving" | "starting"
  >("idle");
  const [launchError, setLaunchError] = useState("");
  async function launch() {
    if (!stage || !launchReady || launchInFlight.current) return;
    launchInFlight.current = true;
    setLaunchError("");
    setLaunchPhase("saving");
    try {
      // The Python recipe owns its prompts. Only legacy preset recipes expose
      // the editable JSON field, so a hidden draft must not block code-first runs.
      const submitted = codeFirst
        ? stage
        : {
            ...stage,
            config: {
              ...stage.config,
              prompts: z
                .array(z.union([z.string(), z.record(z.string(), z.unknown())]))
                .parse(JSON.parse(promptsText)),
            },
          };
      const saved = await saveStageFile(submitted, false);
      if (!saved) throw new Error("Select a training stage before launching.");
      setLaunchPhase("starting");
      const run = await trainingRequest("/runs", runSchema, {
        project,
        stage: saved,
      });
      onStarted(run);
    } catch (error) {
      setLaunchError(message(error));
    } finally {
      setLaunchPhase("idle");
      launchInFlight.current = false;
    }
  }
  return { launch, launchPhase, launchError, setLaunchError };
}
