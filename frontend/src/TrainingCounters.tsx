import { namedCounters, counterLabel } from "./trainingContract";
import type { TrainingEvent } from "./trainingApi";
import { formatMetric } from "./trainingMetrics";

export function TrainingCounters({
  events,
  compact = false,
}: {
  events: TrainingEvent[];
  compact?: boolean;
}) {
  const named = namedCounters(events);
  if (Object.keys(named).length)
    return (
      <dl
        className={compact ? "training-named-counters" : "training-counters"}
        aria-label="Training work counters"
      >
        {Object.entries(named)
          .slice(0, compact ? 6 : 32)
          .map(([name, value]) => (
            <div key={name}>
              <dt>{counterLabel(name)}</dt>
              <dd>{formatMetric(value)}</dd>
            </div>
          ))}
      </dl>
    );
  const progress = events
    .filter((event) => event.kind === "progress")
    .at(-1)?.payload;
  if (
    !progress ||
    ![
      progress.batches_processed,
      progress.optimizer_updates,
      progress.tokens_processed,
    ].some((value) => typeof value === "number")
  )
    return null;
  const maxEpochs = progress.max_epochs;
  return (
    <dl className="training-counters" aria-label="Training work counters">
      <div>
        <dt>
          {typeof maxEpochs === "number"
            ? "Epochs completed"
            : "Epoch tracking"}
        </dt>
        <dd>
          {typeof maxEpochs === "number"
            ? `${progress.epochs_completed} / ${maxEpochs}`
            : "Update-based budget"}
        </dd>
      </div>
      <div>
        <dt>
          {typeof progress.world_size === "number" && progress.world_size > 1
            ? "Microbatches per GPU"
            : "Microbatches processed"}
        </dt>
        <dd>{formatMetric(Number(progress.batches_processed))}</dd>
      </div>
      <div>
        <dt>Optimizer updates</dt>
        <dd>{formatMetric(Number(progress.optimizer_updates))}</dd>
      </div>
      <div>
        <dt>Input tokens processed</dt>
        <dd>{formatMetric(Number(progress.tokens_processed))}</dd>
      </div>
    </dl>
  );
}
