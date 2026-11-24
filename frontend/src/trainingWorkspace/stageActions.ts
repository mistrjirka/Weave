import { newId } from "../ui";
import type { TrainingStage } from "../trainingApi";
import type { useStagePlan } from "./useStagePlan";
export function stageActions(
  planState: ReturnType<typeof useStagePlan>,
  task: (action: () => Promise<unknown>) => Promise<void>,
  onCreated: (stage: TrainingStage) => void,
) {
  const { plan, setPlan, stage, templates, capabilities, saveStageFile } =
    planState;
  function createStage(next: TrainingStage) {
    const clean = {
      ...next,
      script_path: null,
      config_path: null,
      script_revision: null,
      config_revision: null,
      workspace_warning: null,
      project_path: null,
      model_path: null,
      support_path: null,
      previous_script: null,
    };
    setPlan((current) => ({ ...current, stages: [...current.stages, clean] }));
    onCreated(clean);
    void task(() => saveStageFile(clean));
  }
  function addStage(duplicate = false) {
    if (!templates) return;
    const id = newId();
    const next: TrainingStage =
      duplicate && stage
        ? { ...structuredClone(stage), id, name: `${stage.name} copy` }
        : {
            id,
            name: `Stage ${plan.stages.length + 1}`,
            config: {
              ...structuredClone(templates.config),
              python: capabilities?.python ?? templates.config.python,
            },
            script: templates.scripts.causal,
          };
    createStage(next);
  }
  function reorder(direction: number) {
    if (!stage) return;
    setPlan((current) => {
      const stages = [...current.stages],
        index = stages.findIndex((s) => s.id === stage.id),
        target = index + direction;
      if (target < 0 || target >= stages.length) return current;
      [stages[index], stages[target]] = [stages[target], stages[index]];
      return { ...current, stages };
    });
  }
  return { createStage, addStage, reorder };
}
