import { ArrowUpRight, CheckCircle2, CircleAlert, X } from "lucide-react";
import { DesignAdvice } from "../DesignAdvice";
import type { ValidationIssue, ValidationReport } from "../types";

type Props = {
  report: ValidationReport | null;
  validationError: string | null;
  onClose: () => void;
  focusError: (issue: ValidationIssue) => void;
};

export function ChecksDrawer({ report, validationError, onClose, focusError }: Props) {
  return (
    <div className="checks-drawer">
      <div>
        <strong>Graph validation</strong>
        <button aria-label="Close validation issues" onClick={onClose}>
          <X size={15} />
        </button>
      </div>
      {validationError && <p role="alert">Validation unavailable: {validationError}</p>}
      {report?.valid && (
        <p className="valid"><CheckCircle2 size={15} /> All connected tensor shapes are compatible.</p>
      )}
      {report?.errors.map((e: ValidationIssue, i: number) => (
        <button className="check-error" key={i} onClick={() => focusError(e)}>
          <CircleAlert size={15} />
          <span><b>{e.node || "Model"}{e.port ? ` · input ${e.port}` : ""}</b>{e.message}</span>
          <ArrowUpRight size={14} />
        </button>
      ))}
      {!!report?.warnings.length && (
        <>
          <h2>Design notes</h2>
          <p>Suggestions to consider. These never block export.</p>
          {report.warnings.map((issue, i) => (
            <DesignAdvice key={`${issue.node}-${issue.code}-${i}`} issue={issue} onLocate={focusError} />
          ))}
        </>
      )}
    </div>
  );
}
