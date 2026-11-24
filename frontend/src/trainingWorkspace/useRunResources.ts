import { useEffect, useRef, useState } from "react";
import { z } from "zod";
import {
  hardwareSchema,
  runSchema,
  trainingRequest,
  type TrainingGPU,
  type TrainingRun,
} from "../trainingApi";
const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);
export function useRunResources(
  authenticated: boolean | undefined,
  projectId: string,
  onHistory: (history: TrainingRun[]) => void,
) {
  const [gpus, setGPUs] = useState<TrainingGPU[]>([]);
  const [auxiliaryRetry, setAuxiliaryRetry] = useState(0);
  const [auxiliaryError, setAuxiliaryError] = useState("");
  const onHistoryRef = useRef(onHistory);
  onHistoryRef.current = onHistory;
  useEffect(() => {
    if (!authenticated) return;
    const onHistory = onHistoryRef.current;
    const controller = new AbortController();
    setAuxiliaryError("");
    void trainingRequest(
      `/runs?project_id=${encodeURIComponent(projectId)}`,
      z.array(runSchema),
      undefined,
      undefined,
      controller.signal,
    )
      .then((history) => {
        if (controller.signal.aborted) return;
        onHistory(history);
      })
      .catch((failure: unknown) => {
        if (!controller.signal.aborted)
          setAuxiliaryError(
            (current) => `${current} Run history: ${message(failure)}`,
          );
      });
    void trainingRequest(
      "/hardware",
      hardwareSchema,
      undefined,
      undefined,
      controller.signal,
    )
      .then((hardware) => {
        if (!controller.signal.aborted) setGPUs(hardware.gpus);
      })
      .catch((failure: unknown) => {
        if (!controller.signal.aborted)
          setAuxiliaryError(
            (current) => `${current} Hardware: ${message(failure)}`,
          );
      });
    return () => controller.abort();
  }, [authenticated, projectId, auxiliaryRetry]);
  return { gpus, auxiliaryError, setAuxiliaryRetry };
}
