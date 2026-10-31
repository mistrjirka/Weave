import { useState } from "react";
import type { TrainingEvent } from "./trainingApi";
import { groupTrainingSamples } from "./trainingSamples";

export function GeneratedExamples({ events }: { events: TrainingEvent[] }) {
  const [selected, setSelected] = useState("latest");
  const groups = groupTrainingSamples(events);
  const latest = groups[0];
  const visible =
    selected === "all"
      ? groups
      : groups.filter(
          (group) =>
            group.key === (selected === "latest" ? latest?.key : selected),
        );
  const trainingStep = Math.max(
    0,
    ...events
      .filter(
        (event) =>
          event.channel !== "evaluation" &&
          (event.kind === "progress" || event.kind === "metrics"),
      )
      .map((event) => event.step),
  );
  return (
    <>
      <div className="training-examples-toolbar">
        <label>
          Examples from
          <select
            aria-label="Generated example checkpoint"
            value={selected}
            onChange={(event) => setSelected(event.target.value)}
          >
            <option value="latest">
              Follow latest
              {latest ? ` · update ${latest.step.toLocaleString()}` : ""}
            </option>
            <option value="all">
              All evaluated checkpoints · newest first
            </option>
            {groups.map((group) => (
              <option key={group.key} value={group.key}>
                Update {group.step.toLocaleString()} · {group.events.length}{" "}
                examples
              </option>
            ))}
          </select>
        </label>
        {latest && (
          <p className="training-muted" role="status">
            Latest examples: update {latest.step.toLocaleString()}
            {trainingStep > latest.step
              ? ` · training is at ${trainingStep.toLocaleString()}. New examples arrive after validation.`
              : ""}
          </p>
        )}
      </div>
      <div className="training-samples">
        {visible.flatMap((group) =>
          group.events.map((event) => {
            const metadata = event.payload.metadata;
            const decoding =
              typeof metadata === "object" &&
              metadata !== null &&
              "decoding" in metadata
                ? String(metadata.decoding)
                : null;
            return (
              <article key={event.id} className="training-sample">
                <header>
                  <strong>
                    Step {event.step.toLocaleString()}
                    {group === latest ? " · latest" : ""}
                  </strong>
                  <small>{group.checkpoint}</small>
                </header>
                {decoding && (
                  <small className="training-muted">
                    {decoding === "greedy"
                      ? "Greedy decoding · most likely token at each step"
                      : `Decoding: ${decoding}`}
                  </small>
                )}
                <h4>Prompt</h4>
                <pre>{String(event.payload.prompt)}</pre>
                <h4>Model output</h4>
                <pre>{String(event.payload.output)}</pre>
                <details>
                  <summary>Decoding and evaluation details</summary>
                  <pre>{JSON.stringify(metadata, null, 2)}</pre>
                </details>
              </article>
            );
          }),
        )}
      </div>
      {!groups.length && (
        <p className="training-muted">
          No generated examples received yet. They appear when evaluation
          reaches sample generation. Set sample_prompts in train.py, or use
          run.sample() in a custom loop.
        </p>
      )}
    </>
  );
}
