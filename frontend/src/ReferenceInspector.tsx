import { LockKeyhole } from "lucide-react";
import type { TensorContract } from "./types";

export interface ReferenceInspection {
  id: string;
  scope: string;
  label: string;
  kind: string;
  projectName: string;
  description?: string;
  properties: { name: string; value: string }[];
  shape: TensorContract | null;
  parameters: number;
  share?: string;
}

export function isGraphEditShortcut(
  event: Pick<KeyboardEvent, "key" | "ctrlKey" | "metaKey">,
) {
  return (
    ["Delete", "Backspace"].includes(event.key) ||
    ((event.ctrlKey || event.metaKey) &&
      ["z", "g", "s"].includes(event.key.toLowerCase()))
  );
}

/** The shared inspector renders reference values without editable controls. */
export function ReferenceInspector({
  inspection,
}: {
  inspection: ReferenceInspection;
}) {
  return (
    <section
      className="reference-details"
      aria-label="Reference properties"
      data-reference-pane="true"
      tabIndex={0}
      onKeyDownCapture={(event) => {
        if (isGraphEditShortcut(event)) {
          event.preventDefault();
          event.stopPropagation();
        }
      }}
    >
      <div className="reference-readonly">
        <LockKeyhole size={13} aria-hidden="true" /> Reference · read-only
      </div>
      <p className="reference-model-name">{inspection.projectName}</p>
      <span className="eyebrow">{inspection.kind.replaceAll("_", " ")}</span>
      <h2>{inspection.label}</h2>
      {inspection.description && (
        <p className="inspector-description">{inspection.description}</p>
      )}
      <div className="section-label">Block settings</div>
      {inspection.properties.length ? (
        <dl>
          {inspection.properties.map((property) => (
            <div key={property.name}>
              <dt>{property.name.replaceAll("_", " ")}</dt>
              <dd>{property.value}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="field-help">No configurable parameters.</p>
      )}
      {inspection.share && (
        <p className="field-help">
          Weight group: <code>{inspection.share}</code>
        </p>
      )}
      <div className="section-label">Output contract</div>
      <div className="shape-card">
        <span>Tensor shape</span>
        <code>
          {inspection.shape
            ? `[${inspection.shape.shape.join(", ")}]`
            : "Unresolved"}
        </code>
        {inspection.shape && <small>{inspection.shape.dtype}</small>}
      </div>
      <div className="stat-row">
        <span>Parameters</span>
        <b>
          {new Intl.NumberFormat(undefined, {
            notation: "compact",
            maximumFractionDigits: 1,
          }).format(inspection.parameters)}
        </b>
      </div>
    </section>
  );
}
