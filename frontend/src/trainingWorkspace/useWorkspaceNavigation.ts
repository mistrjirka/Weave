import { useRef, useState } from "react";
import type { TrainingRun } from "../trainingApi";
export function trainingRunLink(projectId: string, run: TrainingRun) {
  const params = new URLSearchParams(location.search);
  params.set("model", projectId);
  params.set("view", "training");
  params.set("run", run.id);
  params.set("stage", run.stage_id);
  return `?${params.toString()}`;
}

export function trainingArtifactURL(runId: string, path: string) {
  return `/api/training/runs/${encodeURIComponent(runId)}/artifact?path=${encodeURIComponent(path)}`;
}

export function useWorkspaceNavigation() {
  const [selected, setSelected] = useState(
    () => new URLSearchParams(location.search).get("stage") ?? "",
  );
  const [runId, setRunId] = useState(
    () => new URLSearchParams(location.search).get("run") ?? "",
  );
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const selectionTouched = useRef(false);
  const initialPanel = useRef(
    new URLSearchParams(location.search).get("training_panel"),
  );
  const [editor, setEditor] = useState<"settings" | "script">(() =>
    initialPanel.current === "script" ? "script" : "settings",
  );
  const [setupOpen, setSetupOpen] = useState(
    () =>
      initialPanel.current === "setup" ||
      initialPanel.current === "script" ||
      (!runId && initialPanel.current !== "monitor"),
  );
  const [historyOpen, setHistoryOpen] = useState(false);
  return {
    selected,
    setSelected,
    runId,
    setRunId,
    runs,
    setRuns,
    selectionTouched,
    initialPanel,
    editor,
    setEditor,
    setupOpen,
    setSetupOpen,
    historyOpen,
    setHistoryOpen,
  };
}
