import { CheckCircle2, ChevronDown, CircleAlert, Sparkles } from "lucide-react";
import type { ValidationReport } from "../types";
import { format } from "./util";

export function ValidationBar({ report, validationError, nodeCount, edgeCount, checksOpen, onToggle }: { report: ValidationReport | null; validationError: string; nodeCount: number; edgeCount: number; checksOpen: boolean; onToggle: () => void }) {
  return (
                <div
                  className="validation-bar"
                  aria-live="polite"
                  aria-atomic="true"
                >
                  <button
                    onClick={onToggle}
                    aria-expanded={checksOpen}
                    className={report?.valid ? "valid" : "invalid"}
                  >
                    {report?.valid ? (
                      <CheckCircle2 size={14} />
                    ) : (
                      <CircleAlert size={14} />
                    )}{" "}
                    {report
                      ? report.valid
                        ? "Shapes compatible"
                        : report.errors.length + " graph issues"
                      : validationError
                        ? "Validation unavailable"
                        : "Checking graph…"}
                    <ChevronDown size={12} />
                  </button>
                  {!!report?.warnings.length && (
                    <button
                      className="advice-count"
                      onClick={onToggle}
                      aria-expanded={checksOpen}
                    >
                      <Sparkles size={14} />
                      {report.warnings.length} design note
                      {report.warnings.length === 1 ? "" : "s"}
                    </button>
                  )}
                  <span>
                    {nodeCount} nodes <i>·</i> {edgeCount}{" "}
                    connections
                  </span>
                  <span className="parameters-total">
                    {format(report?.parameters || 0)} parameters <i>·</i>{" "}
                    float32
                  </span>
                </div>
  );
}
