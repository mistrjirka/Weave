import { useEffect, useState } from "react";
import { z } from "zod";
import { stableJSON } from "../trainingConfigEquality";
import {
  trainingConfigSchema,
  trainingRequest,
  type TrainingConfig,
  type TrainingStage,
} from "../trainingApi";

const previewSchema = z.object({
  config: trainingConfigSchema,
  code_first: z.boolean().nullable(),
  unresolved: z.array(z.string()),
  error: z.string().nullable(),
  supported: z.array(z.string()).nullable().optional(),
});
export function computeValues(config: TrainingConfig) {
  return {
    strategy: config.strategy,
    devices: config.devices,
    precision: config.precision,
    placement: config.placement,
  };
}
export function performanceValues(config: TrainingConfig) {
  return {
    compile_mode: config.compile_mode,
    fused_optimizer: config.fused_optimizer,
    profile_steps: config.profile_steps,
  };
}

/** Preview Python literals on the server; never run a script just to populate a form. */
export function useStageSettings(stage: TrainingStage | undefined) {
  const [preview, setPreview] = useState<{
    stageId: string;
    script: string;
    configJSON: string;
    value: z.infer<typeof previewSchema>;
  } | null>(null);
  const [error, setError] = useState("");
  const configJSON = stableJSON(stage?.config);
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
            setPreview({
              stageId: stage.id,
              script: stage.script,
              configJSON,
              value,
            });
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
    preview &&
    preview.stageId === stage?.id &&
    preview.script === stage?.script &&
    preview.configJSON === configJSON
      ? preview.value
      : null;
  if (!stage || !current)
    return {
      config: null,
      codeFirst: null,
      unresolved: [],
      error,
      supported: null,
    };
  // Reflect changes immediately, while the debounced preview request catches up.
  const config = {
    ...stage.config,
    ...computeValues(current.config),
    ...performanceValues(current.config),
    evaluation_mode: current.config.evaluation_mode,
    evaluation_device: current.config.evaluation_device,
    ...(stage.config.compute_mode === "override"
      ? computeValues(stage.config)
      : {}),
    ...(stage.config.performance_mode === "override"
      ? performanceValues(stage.config)
      : {}),
    ...(current.code_first && stage.config.evaluation_mode !== "script"
      ? {
          evaluation_mode: stage.config.evaluation_mode,
          evaluation_device: stage.config.evaluation_device,
        }
      : {}),
  };
  return {
    config,
    codeFirst: current.code_first,
    unresolved: current.unresolved,
    supported: current.supported ?? null,
    error: error || current.error || "",
  };
}
