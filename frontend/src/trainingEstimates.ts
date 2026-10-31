import type { TrainingEvent } from "./trainingApi";

export function estimateTrainingFinish(events: TrainingEvent[]) {
  const progress = events.filter(
    (event) => event.kind === "progress" && event.channel !== "evaluation",
  );
  const latest = progress.at(-1);
  if (!latest) return null;
  const epochs = typeof latest.payload.total !== "number";
  const total = epochs ? latest.payload.max_epochs : latest.payload.total;
  if (typeof total !== "number" || !Number.isFinite(total) || total <= 0)
    return null;
  const points = progress
    .flatMap((event) => {
      const payload = event.payload;
      const completed = epochs ? payload.epochs_completed : payload.completed;
      if (typeof completed !== "number" || !Number.isFinite(completed))
        return [];
      const fraction =
        epochs &&
        typeof payload.batches_per_epoch === "number" &&
        payload.batches_per_epoch > 0 &&
        typeof payload.batches_in_epoch === "number"
          ? payload.batches_in_epoch / payload.batches_per_epoch
          : 0;
      return [{ time: event.time, completed: completed + fraction }];
    })
    .filter((point) => point.time >= latest.time - 300)
    .slice(-60);
  const last = points.at(-1);
  if (!last || last.completed >= total) return null;
  const first = points.find(
    (point) => point.completed < last.completed && last.time - point.time >= 10,
  );
  if (!first) return null;
  const seconds =
    ((total - last.completed) * (last.time - first.time)) /
    (last.completed - first.completed);
  return Number.isFinite(seconds) && seconds > 0
    ? { seconds, finishTime: last.time + seconds }
    : null;
}

export function formatTimeRemaining(seconds: number) {
  if (seconds < 60) return "<1m";
  const minutes = Math.ceil(seconds / 60);
  if (minutes >= 1440)
    return `${Math.floor(minutes / 1440)}d ${Math.floor((minutes % 1440) / 60)}h`;
  if (minutes >= 60) return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
  return `${minutes}m`;
}
