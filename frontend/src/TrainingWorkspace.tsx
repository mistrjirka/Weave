import "./training.css";
import "./trainingSetup.css";
import { useState } from "react";
import type { Project } from "./types";
import { preferredTrainingRun } from "./trainingRunSelection";
import {
  TrainingComputeSettings,
  useStageSettings,
} from "./TrainingComputeSettings";
import { useRunMonitor } from "./trainingWorkspace/useRunMonitor";
import { StageOptimizer } from "./trainingWorkspace/StageOptimizer";
import { StageEnvironment } from "./trainingWorkspace/StageEnvironment";
import { StageRuntime } from "./trainingWorkspace/StageRuntime";
import { StageData } from "./trainingWorkspace/StageData";
import { StageScript } from "./trainingWorkspace/StageScript";
import { useStagePlan } from "./trainingWorkspace/useStagePlan";
import { StageList } from "./trainingWorkspace/StageList";
import { RunHeader } from "./trainingWorkspace/RunHeader";
import { useStageLaunch } from "./trainingWorkspace/useStageLaunch";
import { StageLaunchControls } from "./trainingWorkspace/StageLaunchControls";
import { StagePane } from "./trainingWorkspace/StagePane";
import { useStageEnvironment } from "./trainingWorkspace/useStageEnvironment";
import { WorkflowNavigation } from "./trainingWorkspace/WorkflowNavigation";
import { WorkspaceAlerts } from "./trainingWorkspace/WorkspaceAlerts";
import { HistoryDialog } from "./trainingWorkspace/HistoryDialog";
import { RunSummary } from "./trainingWorkspace/RunSummary";
import { RunFiles } from "./trainingWorkspace/RunFiles";
import { TrainingAccess } from "./trainingWorkspace/TrainingAccess";
import { useRunDashboard } from "./trainingWorkspace/useRunDashboard";
import { stageActions } from "./trainingWorkspace/stageActions";
import { useRunResources } from "./trainingWorkspace/useRunResources";
import { RunPanels } from "./trainingWorkspace/RunPanels";
import { useStageDraft } from "./trainingWorkspace/useStageDraft";
import { useWorkspaceTask } from "./trainingWorkspace/useWorkspaceTask";
import { EmptyStage } from "./trainingWorkspace/EmptyStage";
import { useWorkspaceURL } from "./trainingWorkspace/useWorkspaceURL";
import { useLaunchFocus } from "./trainingWorkspace/useLaunchFocus";
import {
  useWorkspaceNavigation,
  trainingRunLink,
  trainingArtifactURL,
} from "./trainingWorkspace/useWorkspaceNavigation";
import {
  AuxiliaryError,
  HistoryButton,
} from "./trainingWorkspace/WorkspaceNavigationControls";
import { type TrainingStage, type TrainingRun } from "./trainingApi";

export function TrainingWorkspace({ project }: { project: Project }) {
  const {
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
  } = useWorkspaceNavigation();
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const { task, busy, pendingTasksRef } = useWorkspaceTask(setError, setNotice);
  const stagePlan = useStagePlan(
    project,
    selected,
    pendingTasksRef,
    setNotice,
    setError,
  );
  const {
    capabilities,
    setCapabilities,
    templates,
    loaded,
    setBootstrapRetry,
    plan,
    setPlan,
    stage,
    config,
    savedStage,
    stageDirty,
    updateStage,
    patchSettings,
    setting,
    saveStageFile,
    reloadStageFile,
    savePlan,
  } = stagePlan;
  const { detail, setDetail, monitorError, lastRefreshed } = useRunMonitor(
    runId,
    capabilities?.authenticated,
    project.id,
    selected,
    setRuns,
    setRunId,
    setSelected,
    setSetupOpen,
    setNotice,
  );
  const { gpus, auxiliaryError, setAuxiliaryRetry } = useRunResources(
    capabilities?.authenticated,
    project.id,
    (history) => {
      setRuns(history);
      if (!selectionTouched.current && !runId) {
        const replacement = preferredTrainingRun(history, selected);
        if (replacement) {
          setRunId(replacement.id);
          setSelected(replacement.stage_id);
          if (!initialPanel.current) setSetupOpen(false);
        }
      }
    },
  );
  const dashboardState = useRunDashboard(
    runId,
    detail,
    runs,
    setError,
    setNotice,
  );
  const {
    dashboard,
    setDashboard,
    pauseRequested,
    command,
    checkpoints,
    scriptCapabilities,
    customRun,
    active,
    otherActiveRun,
  } = dashboardState;
  const [deletedStage, setDeletedStage] = useState<{
    stage: TrainingStage;
    index: number;
  } | null>(null);
  const { preview, setPreview, promptsText, setPromptsText } =
    useStageDraft(stage);
  const {
    model,
    pythonEnvironment,
    environmentError,
    parentCheckpoints,
    parentCheckpointError,
  } = useStageEnvironment(project, stage, capabilities, setupOpen);
  const effective = useStageSettings(stage);
  const codeFirst = effective.codeFirst === true;
  const launchReady = effective.codeFirst !== null && !effective.error;
  const { launch, launchPhase, launchError, setLaunchError } = useStageLaunch(
    project,
    stage,
    launchReady,
    codeFirst,
    promptsText,
    saveStageFile,
    (run) => {
      scrollToRun.current = run.id;
      setDetail(null);
      setDashboard("metrics");
      setRuns((current) => [
        run,
        ...current.filter((item) => item.id !== run.id),
      ]);
      setSelected(run.stage_id);
      setRunId(run.id);
      setSetupOpen(false);
      setNotice(
        run.execution === "monitor"
          ? "Stage saved. Package prepared for terminal training."
          : "Stage saved. New training run created.",
      );
    },
  );
  const { launchErrorRef, dashboardRef, scrollToRun } = useLaunchFocus(
    launchError,
    detail,
  );
  useWorkspaceURL(stage, runId, setupOpen, editor, loaded);
  const { createStage, addStage, reorder } = stageActions(
    stagePlan,
    task,
    (clean) => {
      setSelected(clean.id);
      setRunId("");
      setDetail(null);
      setSetupOpen(true);
      setHistoryOpen(false);
      setEditor("settings");
    },
  );
  function selectStage(id: string) {
    selectionTouched.current = true;
    setSelected(id);
    const latest = preferredTrainingRun(
      runs.filter((run) => run.stage_id === id),
      id,
    );
    setRunId(latest?.id ?? "");
    setDetail(null);
    setSetupOpen(true);
    setEditor("settings");
    setNotice("");
    setLaunchError("");
  }
  function selectRun(run: TrainingRun) {
    selectionTouched.current = true;
    setDetail(null);
    setRunId(run.id);
    setSelected(run.stage_id);
    setSetupOpen(false);
    setHistoryOpen(false);
    setNotice("");
  }
  const runLink = (run: TrainingRun) => trainingRunLink(project.id, run);
  const artifactURL = (path: string) => trainingArtifactURL(runId, path);

  const historyButton = (
    <HistoryButton
      disabled={launchPhase !== "idle"}
      onOpen={() => setHistoryOpen(true)}
    />
  );

  if (!capabilities || !capabilities.authenticated || !loaded)
    return (
      <TrainingAccess
        capabilities={capabilities}
        loaded={loaded}
        error={error}
        monitorError={monitorError}
        busy={busy}
        task={task}
        setError={setError}
        setCapabilities={setCapabilities}
        setBootstrapRetry={setBootstrapRetry}
      />
    );

  return (
    <section className="training-workspace" aria-label="Training workspace">
      <AuxiliaryError
        error={auxiliaryError}
        onRetry={() => setAuxiliaryRetry((value) => value + 1)}
      />
      <WorkflowNavigation
        stage={stage}
        stages={plan.stages}
        launching={launchPhase !== "idle"}
        setupOpen={setupOpen}
        editor={editor}
        runId={runId}
        detail={detail}
        historyButton={historyButton}
        selectStage={selectStage}
        setSetupOpen={setSetupOpen}
        setEditor={setEditor}
      />
      <WorkspaceAlerts
        setupOpen={setupOpen}
        launching={launchPhase !== "idle"}
        detail={detail}
        otherActiveRun={otherActiveRun}
        error={error}
        monitorError={monitorError}
        notice={notice}
        runLink={runLink}
        selectRun={selectRun}
      />
      <HistoryDialog
        open={historyOpen}
        busy={busy}
        projectId={project.id}
        runId={runId}
        runs={runs}
        onClose={() => setHistoryOpen(false)}
        selectRun={selectRun}
        task={task}
        setNotice={setNotice}
        setRunId={setRunId}
        setRuns={setRuns}
      >
        <StageList
          plan={plan}
          stage={stage}
          setPlan={setPlan}
          deletedStage={deletedStage}
          setDeletedStage={setDeletedStage}
          selectStage={selectStage}
          updateStage={updateStage}
          addStage={addStage}
          reorder={reorder}
          onClose={() => setHistoryOpen(false)}
          onSave={() => void task(savePlan)}
          canAdd={!!templates}
          busy={busy}
          loaded={loaded}
        />
      </HistoryDialog>
      <div className="training-layout">
        <div className="training-main">
          {stage && config ? (
            setupOpen && (
              <StagePane active={!!active}>
                {editor === "script" && (
                  <StageEnvironment
                    stage={stage}
                    savedStage={savedStage}
                    codeFirst={codeFirst}
                    pythonEnvironment={pythonEnvironment}
                    environmentError={environmentError}
                    task={task}
                    setNotice={setNotice}
                  />
                )}
                {editor === "script" ? (
                  <StageScript
                    stage={stage}
                    savedStage={savedStage}
                    projectId={project.id}
                    templates={templates}
                    busy={busy}
                    codeFirst={codeFirst}
                    updateStage={updateStage}
                    setNotice={setNotice}
                    createStage={createStage}
                    task={task}
                    saveStageFile={saveStageFile}
                    reloadStageFile={reloadStageFile}
                  />
                ) : (
                  <>
                    <TrainingComputeSettings
                      config={effective.config}
                      gpus={gpus}
                      model={model}
                      codeFirst={codeFirst}
                      unresolved={effective.unresolved}
                      supported={effective.supported}
                      error={effective.error}
                      onChange={patchSettings}
                    />
                    <StageRuntime
                      stage={stage}
                      codeFirst={codeFirst}
                      templates={templates}
                      runs={runs}
                      parentCheckpoints={parentCheckpoints}
                      parentCheckpointError={parentCheckpointError}
                      updateStage={updateStage}
                      setting={setting}
                    />
                    <StageData
                      project={project}
                      stage={stage}
                      codeFirst={codeFirst}
                      busy={busy}
                      preview={preview}
                      setPreview={setPreview}
                      task={task}
                      setNotice={setNotice}
                      setting={setting}
                    />
                    <StageOptimizer
                      config={config}
                      stage={stage}
                      codeFirst={codeFirst}
                      promptsText={promptsText}
                      setPromptsText={setPromptsText}
                      setError={setError}
                      setting={setting}
                    />
                  </>
                )}
                <StageLaunchControls
                  config={config}
                  computeConfig={effective.config}
                  codeFirst={codeFirst}
                  stageDirty={stageDirty}
                  editor={editor}
                  busy={busy}
                  loaded={loaded}
                  launchReady={launchReady}
                  enabled={capabilities.enabled}
                  launchPhase={launchPhase}
                  launchError={launchError}
                  launchErrorRef={launchErrorRef}
                  onSave={() => void task(() => saveStageFile())}
                  onLaunch={() => void task(launch)}
                />
              </StagePane>
            )
          ) : (
            <EmptyStage
              historyButton={historyButton}
              onCreate={() => addStage()}
              canCreate={!!templates}
            />
          )}

          {!setupOpen && runId && !detail && launchPhase === "idle" && (
            <p role="status">Loading selected run…</p>
          )}
          {!setupOpen && detail && launchPhase === "idle" && (
            <section
              className="training-card training-dashboard"
              aria-label="Run dashboard"
              ref={dashboardRef}
            >
              <RunHeader
                detail={detail}
                runs={runs}
                runId={runId}
                gpus={gpus}
                enabled={capabilities.enabled}
                active={!!active}
                busy={busy}
                customRun={!!customRun}
                scriptCapabilities={scriptCapabilities}
                pauseRequested={pauseRequested}
                launchError={launchError}
                lastRefreshed={lastRefreshed}
                monitorError={monitorError}
                selectRun={selectRun}
                onCommand={(kind) => void task(() => command(kind))}
                onRestart={(run, announce) => {
                  setRuns((current) => [
                    run,
                    ...current.filter((item) => item.id !== run.id),
                  ]);
                  setDetail(null);
                  setRunId(run.id);
                  setSetupOpen(false);
                  if (announce)
                    setNotice(
                      "Restart requested. Restoring the saved checkpoint in a new process. Your editable script was preserved.",
                    );
                }}
              />
              <RunSummary
                detail={detail}
                active={!!active}
                customRun={!!customRun}
                dashboard={dashboard}
                setDashboard={setDashboard}
              />
              <RunPanels
                state={dashboardState}
                runId={runId}
                detail={detail}
                project={project}
                runs={runs}
                gpus={gpus}
                enabled={capabilities.enabled}
                inspectBlocks={config?.inspect_blocks}
                busy={busy}
                artifactURL={artifactURL}
                task={task}
                setNotice={setNotice}
                setDetail={setDetail}
                onContinue={(path) => {
                  if (!stage) return;
                  updateStage({
                    config: {
                      ...stage.config,
                      start_mode: "weights",
                      parent_run: runId,
                      parent_checkpoint: path,
                    },
                  });
                  setSetupOpen(true);
                  setEditor("settings");
                  setNotice(
                    "Checkpoint selected for this stage. Choose Resume if you want to restore optimizer state too.",
                  );
                }}
              />
              <RunFiles
                detail={detail}
                checkpoints={checkpoints}
                runId={runId}
                customRun={!!customRun}
                createStage={createStage}
                setNotice={setNotice}
              />
            </section>
          )}
        </div>
      </div>
    </section>
  );
}
