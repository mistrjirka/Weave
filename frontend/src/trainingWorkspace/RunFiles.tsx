import { Download } from "lucide-react";
import { TrainingCounters } from "../TrainingCounters";
import { newId } from "../ui";
import type { RunDetail, TrainingEvent, TrainingStage } from "../trainingApi";
const bytes = (value: number) =>
  value >= 1024 ** 3
    ? `${(value / 1024 ** 3).toFixed(2)} GB`
    : `${(value / 1024 ** 2).toFixed(1)} MB`;
export function RunFiles({
  detail,
  checkpoints,
  runId,
  customRun,
  createStage,
  setNotice,
}: {
  detail: RunDetail;
  checkpoints: TrainingEvent[];
  runId: string;
  customRun: boolean;
  createStage: (stage: TrainingStage) => void;
  setNotice: (notice: string) => void;
}) {
  return (
    <>
      <details>
        <summary>Run files & diagnostics</summary>
        <a
          className="training-download"
          href={`/api/training/runs/${encodeURIComponent(runId)}/package`}
        >
          <Download aria-hidden="true" size={14} />
          Training package
        </a>
        {detail.status.message && <p>{detail.status.message}</p>}
        <p className="training-muted">
          Frozen run script on server: <code>{detail.directory}/train.py</code>.
          Editing your stage file does not change this run.
        </p>
        <p>{bytes(detail.disk_bytes)} on disk</p>
        {!customRun && <TrainingCounters events={detail.events} />}
      </details>
      <button
        onClick={() => {
          const id = newId();
          const checkpoint = checkpoints
            .filter((event) => !event.payload.remote)
            .at(-1);
          const next: TrainingStage = {
            id,
            name: `${detail.run.name} · next stage`,
            script: detail.script,
            config: {
              ...structuredClone(detail.config),
              start_mode: checkpoint ? "weights" : "random",
              parent_run: checkpoint ? detail.run.id : null,
              parent_checkpoint: checkpoint
                ? String(checkpoint.payload.path)
                : null,
            },
          };
          createStage(next);
          setNotice(
            "Created an editable stage from the recorded script and configuration.",
          );
        }}
      >
        Create stage from this run
      </button>
    </>
  );
}
