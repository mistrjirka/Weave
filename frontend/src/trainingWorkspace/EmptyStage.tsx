import { Activity, Plus } from "lucide-react";
import type { ReactNode } from "react";
export function EmptyStage({
  historyButton,
  onCreate,
  canCreate,
}: {
  historyButton: ReactNode;
  onCreate: () => void;
  canCreate: boolean;
}) {
  return (
    <section className="training-card training-empty">
      <Activity aria-hidden="true" size={32} />
      <h3>Your first training stage</h3>
      {historyButton}
      <p>
        Start with a readable text-training script. Add further stages when
        you’re ready to fine-tune.
      </p>
      <button className="primary" onClick={onCreate} disabled={!canCreate}>
        <Plus aria-hidden="true" size={16} />
        Create training stage
      </button>
    </section>
  );
}
