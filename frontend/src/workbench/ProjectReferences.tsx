import { ArrowUpRight, ExternalLink, GitBranch } from "lucide-react";
import type { Project } from "../types";

export function ProjectReferences({ project }: { project: Project }) {
  return (
                  <div className="references-view">
                    <span className="eyebrow">ARCHITECTURE PROVENANCE</span>
                    <h1>Sources & implementation notes</h1>
                    <p>
                      Reference examples retain their published settings. Read
                      the notes for computational differences and unavailable
                      operations.
                    </p>
                    {project.references?.length ? (
                      project.references.map((r, i: number) => (
                        <article key={i}>
                          <ExternalLink size={20} />
                          <div>
                            <h3>{r.title}</h3>
                            <a href={r.url} target="_blank" rel="noreferrer">
                              View upstream source <ArrowUpRight size={13} />
                            </a>
                            <code>{r.revision}</code>
                            <p>{r.notes}</p>
                            {r.config && (
                              <details>
                                <summary>Published configuration</summary>
                                <pre
                                  tabIndex={0}
                                  role="region"
                                  aria-label={`Configuration for ${r.title}`}
                                >
                                  {JSON.stringify(r.config, null, 2)}
                                </pre>
                              </details>
                            )}
                          </div>
                        </article>
                      ))
                    ) : (
                      <div className="empty-reference">
                        <GitBranch size={30} />
                        <h3>Built from composable primitives</h3>
                        <p>
                          This is a design example rather than a reproduction of
                          a published checkpoint. All weights are randomly
                          initialized on export.
                        </p>
                      </div>
                    )}
                  </div>
  );
}
