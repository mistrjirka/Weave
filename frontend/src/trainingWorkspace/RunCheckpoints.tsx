import { Download } from "lucide-react";
import type { RunDetail, TrainingEvent } from "../trainingApi";

const bytes = (value: number) =>
  value >= 1024 ** 3
    ? `${(value / 1024 ** 3).toFixed(2)} GB`
    : `${(value / 1024 ** 2).toFixed(1)} MB`;
type Props = {
  detail: RunDetail;
  checkpoints: TrainingEvent[];
  active: boolean;
  customRun: boolean;
  scriptCapabilities: Record<string, unknown>;
  includeData: boolean;
  setIncludeData: (value: boolean) => void;
  busy: boolean;
  artifactURL: (path: string) => string;
  onSnapshot: () => void;
  onContinue: (path: string) => void;
};
export function RunCheckpoints({
  detail,
  checkpoints,
  active,
  customRun,
  scriptCapabilities,
  includeData,
  setIncludeData,
  busy,
  artifactURL,
  onSnapshot,
  onContinue,
}: Props) {
  return (
    <>
      <div className="training-snapshot">
        <div>
          <h4>Snapshot this run</h4>
          <p>
            Preserve weights, training state, graph, tokenizer, script, data
            fingerprints, metrics, and saved examples.
          </p>
          <label className="training-inline-check">
            <input
              type="checkbox"
              checked={includeData}
              onChange={(e) => setIncludeData(e.target.checked)}
            />
            Include source dataset files ({bytes(detail.source_data_bytes)}{" "}
            additional)
          </label>
          <small>
            Source paths, sizes and modification times are recorded. Full
            content hashes are included when requested in train.py; source files
            are copied only when selected here.
          </small>
        </div>
        <button
          className="primary"
          disabled={
            busy ||
            Boolean(
              active &&
              customRun &&
              (scriptCapabilities.commands !== true ||
                scriptCapabilities.checkpoint !== true),
            )
          }
          title={
            active && customRun && scriptCapabilities.checkpoint !== true
              ? "The custom script has not declared checkpoint support."
              : undefined
          }
          onClick={onSnapshot}
        >
          Create snapshot
        </button>
      </div>
      {detail.snapshots.map((snapshot) => (
        <article key={snapshot.path} className="training-sample">
          <a href={artifactURL(snapshot.path)}>
            <Download aria-hidden="true" size={14} />
            Download snapshot · {bytes(snapshot.bytes)}
          </a>
          <p>
            {snapshot.checkpoint ?? "Run record only · no resumable weights"}
          </p>
          <small>SHA-256: {snapshot.sha256}</small>
          {!!snapshot.missing_data.length && (
            <p className="training-alert">
              Some source data was missing or changed. See the archive manifest.
            </p>
          )}
        </article>
      ))}
      <h4>Published checkpoints</h4>
      {checkpoints.map((event) => (
        <article className="training-checkpoint" key={event.id}>
          <div>
            <strong>Step {event.step}</strong>
            <small>
              {String(event.payload.path)} ·{" "}
              {bytes(
                typeof event.payload.bytes === "number"
                  ? event.payload.bytes
                  : 0,
              )}
            </small>
          </div>
          {event.payload.remote ? (
            <small>File on training computer · not uploaded</small>
          ) : (
            <>
              <a href={artifactURL(String(event.payload.path))}>
                Weights & training state
              </a>
              <button onClick={() => onContinue(String(event.payload.path))}>
                Continue from here
              </button>
            </>
          )}
        </article>
      ))}
      {!checkpoints.length && <p>No checkpoint has been published yet.</p>}
      <details>
        <summary>Dataset provenance</summary>
        {Object.entries(detail.datasets).map(([key, dataset]) => (
          <div key={key}>
            <h4>{key}</h4>
            <p>{dataset.path}</p>
            <code>
              {dataset.sha256 ??
                "Path, size and modification time recorded · content hash not requested"}
            </code>
          </div>
        ))}
        <p>
          Source manifests list declared files, which may include files not
          reached yet. Consumed batch traces record the file, row and token span
          when the loader supplies source metadata. Enable
          data_fingerprint="sha256" in train.py for full file hashes; batch
          hashing is separately optional.
        </p>
      </details>
    </>
  );
}
