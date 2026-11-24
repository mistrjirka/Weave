import type { RunDetail } from "../trainingApi";

export function RestartSnapshot({ detail }: { detail: RunDetail }) {
  return (
    <>
      {detail.revision && (
        <details>
          <summary>
            Restart snapshot ·{" "}
            {detail.revision.script_source === "edited"
              ? "edited script"
              : "previous script"}
          </summary>
          <p className="training-muted">
            Script, settings and file hashes are recorded in{" "}
            <code>{detail.directory}/revision.json</code>.{" "}
            {detail.revision.state_policy}
          </p>
        </details>
      )}
    </>
  );
}
