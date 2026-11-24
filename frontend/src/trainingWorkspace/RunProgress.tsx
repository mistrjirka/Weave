import type { RunDetail } from "../trainingApi";
import { TrainingETA } from "../TrainingETA";

export function RunProgress({
  detail,
  active,
}: {
  detail: RunDetail;
  active: boolean;
}) {
  return (
    <>
      {detail.events
        .filter(
          (event) => event.kind === "preparation" && event.channel === "system",
        )
        .slice(-1)
        .map((event) => {
          const documents =
            typeof event.payload.documents === "number"
              ? event.payload.documents
              : 0;
          const total =
            typeof event.payload.total_documents === "number"
              ? event.payload.total_documents
              : undefined;
          const cacheReady =
            event.payload.phase === "complete" ||
            event.payload.phase === "reused";
          const preparationInterrupted = ["failed", "stopped"].includes(
            detail.status.state,
          );
          const trainingHasMetrics = detail.events.some(
            (entry) => entry.kind === "metrics" && entry.channel === "training",
          );
          return (
            !trainingHasMetrics && (
              <section
                key={event.id}
                className="training-server-files"
                aria-label="Data preparation progress"
              >
                <strong>
                  {preparationInterrupted
                    ? "Data preparation interrupted"
                    : event.payload.phase === "failed"
                      ? "Data preparation failed"
                      : cacheReady
                        ? "Token cache ready"
                        : "Preparing data"}
                </strong>
                <p role="status">
                  {preparationInterrupted
                    ? "This run ended before training started. "
                    : event.payload.phase === "waiting"
                      ? "Waiting for another process to finish this cache. "
                      : event.payload.phase === "initializing"
                        ? "Loading tokenizer and creating data loaders. "
                        : ""}
                  {documents.toLocaleString()}
                  {total ? ` / ${total.toLocaleString()}` : ""} documents ·{" "}
                  {typeof event.payload.tokens === "number"
                    ? event.payload.tokens.toLocaleString()
                    : "0"}{" "}
                  tokens
                </p>
                {!cacheReady &&
                  !preparationInterrupted &&
                  event.payload.phase !== "failed" && (
                    <progress
                      aria-label="Cache preparation progress"
                      value={total ? documents : undefined}
                      max={total}
                    />
                  )}
                {!!event.payload.message && (
                  <p>{String(event.payload.message)}</p>
                )}
                {!total && !cacheReady && (
                  <small>
                    The reader did not supply a document total; progress is
                    shown as counts rather than a percentage.
                  </small>
                )}
              </section>
            )
          );
        })}
      {detail.events
        .filter(
          (e) =>
            e.kind === "progress" && (e.channel ?? "training") === "training",
        )
        .slice(-1)
        .map((event) => {
          const useEpochs =
            typeof event.payload.total !== "number" &&
            typeof event.payload.max_epochs === "number";
          const completed = useEpochs
            ? event.payload.epochs_completed
            : event.payload.completed;
          const total = useEpochs
            ? event.payload.max_epochs
            : event.payload.total;
          const unit = useEpochs ? "epochs" : event.payload.unit;
          return (
            <div key={event.id} className="training-progress">
              {active && (
                <progress
                  aria-label="Training progress"
                  value={
                    typeof completed === "number" && typeof total === "number"
                      ? completed
                      : undefined
                  }
                  max={typeof total === "number" ? total : undefined}
                />
              )}
              <span className="training-progress-caption">
                <span>
                  {!active && "Last reported progress · "}
                  {typeof completed === "number"
                    ? completed.toLocaleString()
                    : String(completed)}{" "}
                  /{" "}
                  {typeof total === "number"
                    ? total.toLocaleString()
                    : "unknown"}{" "}
                  {String(unit)}
                </span>
                <TrainingETA detail={detail} />
              </span>
            </div>
          );
        })}
    </>
  );
}
