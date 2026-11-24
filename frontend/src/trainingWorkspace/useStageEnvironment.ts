import { useEffect, useState } from "react";
import { z } from "zod";
import type { Project } from "../types";
import {
  capabilitySchema,
  inspectTrainingModel,
  monitorEventsSchema,
  pythonEnvironmentSchema,
  trainingRequest,
  type TrainingEvent,
  type TrainingStage,
} from "../trainingApi";
const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);
export function useStageEnvironment(
  project: Project,
  stage: TrainingStage | undefined,
  capabilities: z.infer<typeof capabilitySchema> | null,
  setupOpen: boolean,
) {
  const config = stage?.config;
  const [model, setModel] = useState<Awaited<
    ReturnType<typeof inspectTrainingModel>
  > | null>(null);
  const [pythonEnvironment, setPythonEnvironment] = useState<z.infer<
    typeof pythonEnvironmentSchema
  > | null>(null);
  const [environmentError, setEnvironmentError] = useState("");
  const [parentCheckpoints, setParentCheckpoints] = useState<TrainingEvent[]>(
    [],
  );
  const [parentCheckpointError, setParentCheckpointError] = useState("");
  useEffect(() => {
    setParentCheckpoints([]);
    setParentCheckpointError("");
    if (
      !capabilities?.authenticated ||
      !config?.parent_run ||
      config.start_mode === "random"
    )
      return;
    const controller = new AbortController();
    const parent = config.parent_run;
    void trainingRequest(
      `/runs/${encodeURIComponent(parent)}/event-records?kind=checkpoint&latest=true&limit=500`,
      monitorEventsSchema,
      undefined,
      undefined,
      controller.signal,
    )
      .then(({ events }) => {
        if (!controller.signal.aborted) setParentCheckpoints(events);
      })
      .catch((failure: unknown) => {
        if (!controller.signal.aborted)
          setParentCheckpointError(message(failure));
      });
    return () => controller.abort();
  }, [capabilities?.authenticated, config?.parent_run, config?.start_mode]);

  useEffect(() => {
    if (!capabilities?.authenticated || !setupOpen) return;
    const controller = new AbortController();
    setModel(null);
    void inspectTrainingModel(project)
      .then((value) => {
        if (!controller.signal.aborted) setModel(value);
      })
      .catch(() => {
        if (!controller.signal.aborted) setModel(null);
      });
    return () => controller.abort();
  }, [project, capabilities?.authenticated, setupOpen]);
  useEffect(() => {
    if (!capabilities?.authenticated || !setupOpen) return;
    const controller = new AbortController();
    setPythonEnvironment(null);
    setEnvironmentError("");
    const executable = config?.python ?? capabilities.python;
    const timer = window.setTimeout(() => {
      void trainingRequest(
        `/environment?python=${encodeURIComponent(executable)}`,
        pythonEnvironmentSchema,
        undefined,
        undefined,
        controller.signal,
      )
        .then(setPythonEnvironment)
        .catch((e) => {
          if (!controller.signal.aborted) setEnvironmentError(message(e));
        });
    }, 600);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [
    config?.python,
    capabilities?.python,
    capabilities?.authenticated,
    setupOpen,
  ]);
  return {
    model,
    pythonEnvironment,
    environmentError,
    parentCheckpoints,
    parentCheckpointError,
  };
}
