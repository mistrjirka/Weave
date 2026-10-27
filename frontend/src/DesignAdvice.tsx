import { ArrowUpRight, Lightbulb } from "lucide-react";
import type { ValidationIssue } from "./types";

export function DesignAdvice({
  issue,
  onLocate,
}: {
  issue: ValidationIssue;
  onLocate: (issue: ValidationIssue) => void;
}) {
  return (
    <article className="design-advice" data-rule={issue.code}>
      <div className="advice-heading">
        <Lightbulb size={16} aria-hidden="true" />
        <strong>{issue.title || "Contract note"}</strong>
      </div>
      {issue.node && (
        <button
          className="advice-locate"
          aria-label={`Locate block ${issue.node}`}
          onClick={() => onLocate(issue)}
        >
          {issue.node}
          <ArrowUpRight size={14} aria-hidden="true" />
        </button>
      )}
      <p>{issue.message}</p>
      {issue.suggestion && (
        <p className="advice-suggestion">{issue.suggestion}</p>
      )}
      {issue.source && (
        <a href={issue.source} target="_blank" rel="noreferrer">
          Read the reference <ArrowUpRight size={12} aria-hidden="true" />
        </a>
      )}
    </article>
  );
}
