import { History } from "lucide-react";

export function HistoryButton({
  disabled,
  onOpen,
}: {
  disabled: boolean;
  onOpen: () => void;
}) {
  return (
    <button
      className="training-history-button"
      aria-label="Stages and run history"
      disabled={disabled}
      aria-haspopup="dialog"
      onClick={onOpen}
    >
      <History size={14} aria-hidden="true" /> <span>Stages & history</span>
    </button>
  );
}

export function AuxiliaryError({
  error,
  onRetry,
}: {
  error: string;
  onRetry: () => void;
}) {
  return error ? (
    <div className="training-alert" role="alert">
      {error} <button onClick={onRetry}>Retry</button>
    </div>
  ) : null;
}
