import { GPUTuningStatus } from "../GPUTuningStatus";
import { RunOverview } from "../RunOverview";
import { TrainingCounters } from "../TrainingCounters";
import { ValidationStatus } from "../ValidationStatus";
import { RunProgress } from "./RunProgress";
import type { RunDetail } from "../trainingApi";
export type DashboardTab =
  | "metrics"
  | "samples"
  | "hardware"
  | "logs"
  | "checkpoints"
  | "chat"
  | "artifacts";
export function RunSummary({
  detail,
  active,
  customRun,
  dashboard,
  setDashboard,
}: {
  detail: RunDetail;
  active: boolean;
  customRun: boolean;
  dashboard: DashboardTab;
  setDashboard: (tab: DashboardTab) => void;
}) {
  return (
    <>
      <GPUTuningStatus events={detail.events} />
      {(detail.status.state === "failed" ||
        detail.status.state === "unknown") &&
        detail.status.message && (
          <p role="status" className="training-alert">
            {detail.status.message}
          </p>
        )}
      <RunProgress detail={detail} active={!!active} />
      <RunOverview detail={detail} />
      {customRun && <TrainingCounters events={detail.events} compact />}
      <ValidationStatus events={detail.events} />
      <nav className="training-tabs" aria-label="Run views">
        {(
          [
            "metrics",
            "samples",
            "hardware",
            "logs",
            "checkpoints",
            "chat",
            "artifacts",
          ] as const
        ).map((tab) => (
          <button
            key={tab}
            aria-pressed={dashboard === tab}
            onClick={() => setDashboard(tab)}
          >
            {
              {
                metrics: "Metrics",
                samples: "Generated examples",
                hardware: "Hardware",
                logs: "Logs",
                checkpoints: "Checkpoints & snapshots",
                chat: "Chat & text completion",
                artifacts: "Tables & artifacts",
              }[tab]
            }
          </button>
        ))}
      </nav>
    </>
  );
}
