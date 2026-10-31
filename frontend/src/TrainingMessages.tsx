import { useState } from "react";
import type { RunDetail } from "./trainingApi";

export function TrainingMessages({
  detail,
  expanded = false,
}: {
  detail: RunDetail;
  expanded?: boolean;
}) {
  const [open, setOpen] = useState(expanded);
  const messages = detail.events.filter(
    (event) => event.kind === "warning" || event.kind === "error",
  );
  return (
    <details
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        Warnings and diagnostics · {messages.length} recorded events
      </summary>
      <div
        className="training-message-log"
        role="region"
        aria-label="Training warning log"
        tabIndex={0}
      >
        {detail.diagnostics.map((message) => (
          <p key={message}>Info · {message}</p>
        ))}
        {detail.run.warnings.map((warning) => (
          <p key={warning.code}>Script check · {warning.message}</p>
        ))}
        {messages
          .slice(-100)
          .reverse()
          .map((event) => (
            <p
              key={event.id}
              className={
                event.kind === "error" ? "training-message-error" : undefined
              }
            >
              <time dateTime={new Date(event.time * 1000).toISOString()}>
                {new Date(event.time * 1000).toLocaleTimeString()}
              </time>
              {" · "}
              {event.kind === "error" ? "Error" : "Warning"} · Update{" "}
              {event.step}: {String(event.payload.message)}
            </p>
          ))}
        {messages.length > 100 && (
          <p>
            Showing the latest 100 warning/error entries. Full history is
            preserved in the run.
          </p>
        )}
        {!messages.length &&
          !detail.diagnostics.length &&
          !detail.run.warnings.length && <p>No warnings or errors recorded.</p>}
      </div>
    </details>
  );
}
