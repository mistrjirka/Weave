import { useState } from "react";
import { Panel } from "@xyflow/react";
import { graphVariableScope } from "../graphVariables";
import { tokenizerOwnsVocabulary } from "../tokenizerVocabulary";
import type { Project, ValidationReport } from "../types";

export function ReferenceVariables({
  project,
  path,
  report,
}: {
  project: Project;
  path: string[];
  report: ValidationReport | null;
}) {
  const [variablesOpen, setVariablesOpen] = useState(true);
  return (
    <Panel position="top-left" className="variables-canvas-panel">
      <aside
        className="graph-variable-card reference-variables nodrag nopan nowheel"
        aria-label="Reference variables"
      >
        <header className="variable-card-heading">
          <strong>Variables · read-only</strong>
          <button
            aria-label={
              variablesOpen
                ? "Minimize reference variables"
                : "Expand reference variables"
            }
            onClick={() => setVariablesOpen((open) => !open)}
          >
            {variablesOpen ? "−" : "+"}
          </button>
        </header>
        {variablesOpen && (
          <div className="variable-card-body">
            <dl>
              {Object.entries(graphVariableScope(project, path).effective)
                .filter(
                  ([name]) =>
                    name !== "vocab" ||
                    !tokenizerOwnsVocabulary(project, path, report?.tokenizers),
                )
                .map(([name, value]) => (
                  <div key={name}>
                    <dt>${name}</dt>
                    <dd>{value}</dd>
                  </div>
                ))}
            </dl>
            <small>
              Values include inherited variables and enclosing block settings.
            </small>
          </div>
        )}
      </aside>
    </Panel>
  );
}
