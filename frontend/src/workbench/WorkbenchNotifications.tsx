import { CheckCircle2, CircleAlert, X } from "lucide-react";
import type { connectionFeedback } from "../connectionFeedback";

export function WorkbenchNotifications({ connectionError, graphTab, toast, canUndoDelete, onDismissConnection, onDismissToast, onUndoDelete }: { connectionError: ReturnType<typeof connectionFeedback> | null; graphTab: boolean; toast: string; canUndoDelete: boolean; onDismissConnection: () => void; onDismissToast: () => void; onUndoDelete: () => void }) {
  return (
    <>
      {connectionError && graphTab && (
        <div className="connection-error" role="alert">
          <CircleAlert size={20} aria-hidden="true" />
          <div>
            <strong>{connectionError.title}</strong>
            {connectionError.messages.map((message) => (
              <p key={message}>{message}</p>
            ))}
            <small>
              Fix the block settings, remove the edge, or undo the connection
              (Ctrl+Z).
            </small>
          </div>
          <button
            aria-label="Dismiss connection error"
            onClick={onDismissConnection}
          >
            <X size={18} />
          </button>
        </div>
      )}
      {toast && !connectionError && (
        <div className="toast" role="status">
          <CheckCircle2 size={16} />
          {toast}
          <button
            aria-label="Dismiss notification"
            onClick={onDismissToast}
          >
            <X size={14} />
          </button>
          {canUndoDelete && toast.startsWith("Project deleted") && (
            <button className="toast-undo" onClick={onUndoDelete}>
              Undo
            </button>
          )}
        </div>
      )}
    </>
  );
}
