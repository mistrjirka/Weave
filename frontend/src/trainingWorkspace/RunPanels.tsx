import { TrainingArtifacts } from "../TrainingArtifacts";
import { GeneratedExamples } from "../GeneratedExamples";
import { CheckpointChat } from "../chat/CheckpointChat";
import type { Project } from "../types";
import type { RunDetail, TrainingGPU, TrainingRun } from "../trainingApi";
import type { useRunDashboard } from "./useRunDashboard";
import { RunReporting } from "./RunReporting";
import { RunMetrics } from "./RunMetrics";
import { RunDiagnostics } from "./RunDiagnostics";
import { RunCheckpoints } from "./RunCheckpoints";
type Props = {
  state: ReturnType<typeof useRunDashboard>;
  runId: string;
  detail: RunDetail;
  project: Project;
  runs: TrainingRun[];
  gpus: TrainingGPU[];
  enabled: boolean;
  inspectBlocks: boolean | undefined;
  busy: boolean;
  artifactURL: (path: string) => string;
  task: (action: () => Promise<unknown>) => Promise<void>;
  setNotice: (notice: string) => void;
  setDetail: (detail: RunDetail) => void;
  onContinue: (path: string) => void;
};
export function RunPanels({
  state,
  runId,
  detail,
  project,
  runs,
  gpus,
  enabled,
  inspectBlocks,
  busy,
  artifactURL,
  task,
  setNotice,
  setDetail,
  onContinue,
}: Props) {
  const {
    dashboard,
    comparison,
    axis,
    setAxis,
    smooth,
    setSmooth,
    comparisonId,
    setComparisonId,
    reportBaseURL,
    setReportBaseURL,
    checkpoints,
    active,
    customRun,
    scriptCapabilities,
    includeData,
    setIncludeData,
    command,
  } = state;
  return (
    <>
      <RunReporting
        detail={detail}
        runId={runId}
        busy={busy}
        reportBaseURL={reportBaseURL}
        setReportBaseURL={setReportBaseURL}
        task={task}
        setNotice={setNotice}
        setDetail={setDetail}
      />
      {dashboard === "chat" && (
        <CheckpointChat
          key={runId}
          runId={runId}
          gpus={gpus}
          enabled={enabled}
        />
      )}
      {dashboard === "metrics" && (
        <RunMetrics
          detail={detail}
          comparison={comparison}
          runs={runs}
          project={project}
          inspectBlocks={inspectBlocks}
          active={!!active}
          axis={axis}
          setAxis={setAxis}
          smooth={smooth}
          setSmooth={setSmooth}
          comparisonId={comparisonId}
          setComparisonId={setComparisonId}
        />
      )}
      {dashboard === "artifacts" && (
        <TrainingArtifacts events={detail.events} artifactURL={artifactURL} />
      )}
      {dashboard === "samples" && (
        <GeneratedExamples key={detail.run.id} events={detail.events} />
      )}
      {(dashboard === "hardware" || dashboard === "logs") && (
        <RunDiagnostics
          detail={detail}
          view={dashboard}
          artifactURL={artifactURL}
        />
      )}
      {dashboard === "checkpoints" && (
        <RunCheckpoints
          detail={detail}
          checkpoints={checkpoints}
          active={!!active}
          customRun={!!customRun}
          scriptCapabilities={scriptCapabilities}
          includeData={includeData}
          setIncludeData={setIncludeData}
          busy={busy}
          artifactURL={artifactURL}
          onSnapshot={() => void task(() => command("snapshot"))}
          onContinue={onContinue}
        />
      )}
    </>
  );
}
