import type { ReactNode } from "react";
import type { RunDetail, TrainingStage } from "../trainingApi";
type Props = {
  stage: TrainingStage | undefined;
  stages: TrainingStage[];
  launching: boolean;
  setupOpen: boolean;
  editor: "settings" | "script";
  runId: string;
  detail: RunDetail | null;
  historyButton: ReactNode;
  selectStage: (id: string) => void;
  setSetupOpen: (open: boolean) => void;
  setEditor: (editor: "settings" | "script") => void;
};
export function WorkflowNavigation({
  stage,
  stages,
  launching,
  setupOpen,
  editor,
  runId,
  detail,
  historyButton,
  selectStage,
  setSetupOpen,
  setEditor,
}: Props) {
  return (
    <>
      {stage && (
        <div className="training-workflow-bar">
          <label className="training-stage-picker">
            <span>Stage</span>
            <select
              aria-label="Training stage"
              disabled={launching}
              value={stage.id}
              onChange={(event) => selectStage(event.target.value)}
            >
              {stages.map((item, index) => (
                <option key={item.id} value={item.id}>
                  {index + 1}. {item.name}
                </option>
              ))}
            </select>
          </label>
          <nav className="training-workflow-tabs" aria-label="Stage workspace">
            <button
              disabled={launching}
              aria-pressed={setupOpen && editor === "settings"}
              onClick={() => {
                setSetupOpen(true);
                setEditor("settings");
              }}
            >
              Setup
            </button>
            <button
              disabled={launching}
              aria-pressed={setupOpen && editor === "script"}
              onClick={() => {
                setSetupOpen(true);
                setEditor("script");
              }}
            >
              Python script
            </button>
            <button
              aria-label="Monitor"
              aria-pressed={!setupOpen}
              disabled={!runId || launching}
              onClick={() => setSetupOpen(false)}
            >
              Monitor
              {detail && (
                <span
                  className="training-status-dot"
                  data-state={detail.status.state}
                  aria-hidden="true"
                  title={detail.status.state}
                />
              )}
            </button>
          </nav>
          {historyButton}
        </div>
      )}
    </>
  );
}
