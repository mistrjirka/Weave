import { runContract } from "./trainingContract";
import { z } from "zod";
import type { TrainingEvent } from "./trainingApi";
import { formatMetric } from "./trainingMetrics";

const fileSchema = z.object({
  path: z.string(),
  name: z.string(),
  bytes: z.number(),
  sha256: z.string(),
  remote: z.boolean().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});
const tableSchema = fileSchema.extend({
  rows: z.number(),
  columns: z.array(z.string()),
  preview: z.array(z.record(z.string(), z.unknown())),
});
function cellText(value: unknown) {
  if (value === null || value === undefined) return "—";
  return typeof value === "string" ? value : JSON.stringify(value);
}
export function TrainingArtifacts({
  events,
  artifactURL,
}: {
  events: TrainingEvent[];
  artifactURL: (path: string) => string;
}) {
  const contract = runContract(events);
  const artifacts = events.filter(
    (event) => event.kind === "table" || event.kind === "artifact",
  );
  return (
    <section className="training-artifacts" aria-label="Tables and artifacts">
      {!artifacts.length && (
        <p className="training-muted">
          No tables or artifacts reported. Custom scripts can publish
          trajectories with run.table() and completed files with run.artifact().
        </p>
      )}
      {artifacts
        .slice(-50)
        .reverse()
        .map((event) => {
          const file = fileSchema.safeParse(event.payload);
          if (!file.success) return null;
          const table = tableSchema.safeParse(event.payload);
          return (
            <article key={event.id} className="training-sample">
              <header>
                <h4>{file.data.name}</h4>
                <small>
                  {event.step} {contract.step_unit} ·{" "}
                  {formatMetric(file.data.bytes)} bytes
                  {table.success && ` · ${formatMetric(table.data.rows)} rows`}
                </small>
              </header>
              {table.success && (
                <div
                  className="training-table-scroll"
                  role="region"
                  aria-label={`${file.data.name} preview`}
                  tabIndex={0}
                >
                  <table>
                    <thead>
                      <tr>
                        {table.data.columns.map((column) => (
                          <th key={column} scope="col">
                            {column}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {table.data.preview.map((row, index) => (
                        <tr key={index}>
                          {table.data.columns.map((column) => (
                            <td key={column}>{cellText(row[column])}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {file.data.remote ? (
                <p className="training-muted">
                  File stays on the training computer. The preview is available
                  here.
                </p>
              ) : (
                <a href={artifactURL(file.data.path)}>
                  Download {table.success ? "full JSONL table" : file.data.name}
                </a>
              )}
              {table.success && (
                <small>
                  Preview: {table.data.preview.length} of {table.data.rows}{" "}
                  rows.
                </small>
              )}
              {!!file.data.metadata &&
                Object.keys(file.data.metadata).length > 0 && (
                  <details>
                    <summary>Provenance</summary>
                    <pre className="training-log">
                      {JSON.stringify(file.data.metadata, null, 2)}
                    </pre>
                  </details>
                )}
            </article>
          );
        })}
    </section>
  );
}
