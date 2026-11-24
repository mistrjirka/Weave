import { ArrowDown, ArrowUp, Copy, Plus, Save, Trash2 } from "lucide-react";
import type { Dispatch, SetStateAction } from "react";
import type { TrainingPlan, TrainingStage } from "../trainingApi";
type DeletedStage = { stage: TrainingStage; index: number } | null;
type Props = {
  plan: TrainingPlan;
  stage: TrainingStage | undefined;
  setPlan: Dispatch<SetStateAction<TrainingPlan>>;
  deletedStage: DeletedStage;
  setDeletedStage: Dispatch<SetStateAction<DeletedStage>>;
  selectStage: (id: string) => void;
  updateStage: (patch: Partial<TrainingStage>) => void;
  addStage: (duplicate?: boolean) => void;
  reorder: (direction: number) => void;
  onClose: () => void;
  onSave: () => void;
  canAdd: boolean;
  busy: boolean;
  loaded: boolean;
};
export function StageList({
  plan,
  stage,
  setPlan,
  deletedStage,
  setDeletedStage,
  selectStage,
  updateStage,
  addStage,
  reorder,
  onClose,
  onSave,
  canAdd,
  busy,
  loaded,
}: Props) {
  return (
    <>
      <div className="training-section-heading">
        <h3>Stages</h3>
        <button
          aria-label="Add training stage"
          onClick={() => addStage()}
          disabled={!canAdd}
        >
          <Plus aria-hidden="true" size={16} />
        </button>
      </div>
      {!plan.stages.length && (
        <p className="training-muted">
          Start with pretraining, then add a fine-tuning stage using its
          checkpoint.
        </p>
      )}
      {plan.stages.map((item, index) => (
        <button
          className={`training-stage ${stage?.id === item.id ? "is-active" : ""}`}
          key={item.id}
          onClick={() => {
            selectStage(item.id);
            onClose();
          }}
        >
          <span>{index + 1}</span>
          <div>
            <strong>{item.name}</strong>
            <small>
              {item.config.recipe === "sft"
                ? "Chat / tool-call SFT"
                : item.config.recipe === "custom"
                  ? "Custom Python"
                  : "Text pretraining"}
            </small>
          </div>
        </button>
      ))}
      {stage && (
        <label className="training-stage-rename">
          Stage name
          <input
            aria-label="Stage name"
            value={stage.name}
            onChange={(event) => updateStage({ name: event.target.value })}
          />
        </label>
      )}
      <div className="training-stage-tools">
        <button
          aria-label="Duplicate stage"
          disabled={!stage}
          onClick={() => addStage(true)}
        >
          <Copy aria-hidden="true" size={14} />
        </button>
        <button
          aria-label="Move stage up"
          disabled={!stage || plan.stages[0]?.id === stage.id}
          onClick={() => reorder(-1)}
        >
          <ArrowUp aria-hidden="true" size={14} />
        </button>
        <button
          aria-label="Move stage down"
          disabled={!stage || plan.stages.at(-1)?.id === stage.id}
          onClick={() => reorder(1)}
        >
          <ArrowDown aria-hidden="true" size={14} />
        </button>
        <button
          aria-label="Delete stage"
          disabled={!stage}
          onClick={() => {
            if (!stage) return;
            setDeletedStage({
              stage,
              index: plan.stages.findIndex((item) => item.id === stage.id),
            });
            setPlan((current) => ({
              ...current,
              stages: current.stages.filter((s) => s.id !== stage?.id),
            }));
            selectStage(
              plan.stages.find((item) => item.id !== stage?.id)?.id ?? "",
            );
          }}
        >
          <Trash2 aria-hidden="true" size={14} />
        </button>
      </div>
      {deletedStage && (
        <button
          onClick={() => {
            setPlan((current) => {
              if (
                current.stages.some((item) => item.id === deletedStage.stage.id)
              )
                return current;
              const stages = [...current.stages];
              stages.splice(deletedStage.index, 0, deletedStage.stage);
              return { ...current, stages };
            });
            selectStage(deletedStage.stage.id);
            setDeletedStage(null);
          }}
        >
          Undo stage deletion
        </button>
      )}
      <button disabled={busy || !loaded} onClick={onSave}>
        <Save size={14} aria-hidden="true" /> Save stage order
      </button>
    </>
  );
}
