import { ArrowUpRight, Copy, GitBranch, Maximize2 } from "lucide-react";
import { api, graphSchema } from "../../api";
import { supportsWeightSharing } from "../../weightGroups";
import { motion } from "../util";
import type { FitViewOptions } from "@xyflow/react";
import type { GraphNode, Project } from "../../types";

interface BlockOperationsProps {
  selected: GraphNode;
  project: Project;
  path: string[];
  updateNode: (fn: (node: GraphNode) => void) => void;
  open: (id: string) => void;
  notify: (message: string) => void;
  unrollSelected: () => void;
  duplicateShared: () => void;
  fitGraph: (options: FitViewOptions) => unknown;
}
export function BlockOperations({ selected, project, path, updateNode, open, notify, unrollSelected, duplicateShared, fitGraph }: BlockOperationsProps) {
  return <>
                    {["attention", "cross_attention"].includes(
                      selected.data.kind,
                    ) && (
                      <button
                        className="inspect-open"
                        onClick={async () => {
                          try {
                            const body = graphSchema.parse(
                              await api("/attention/expand", {
                                project,
                                path,
                                node_id: selected.id,
                              }),
                            );
                            updateNode((n) => {
                              n.data.kind = "block";
                              n.data.body = body;
                              n.data.params = {};
                              n.data.description =
                                "Editable Q/K/V projections, head transforms, positions and attention computation.";
                            });
                            open(selected.id);
                          } catch (e) {
                            notify(String(e));
                          }
                        }}
                      >
                        Edit attention internals
                        <ArrowUpRight size={15} />
                      </button>
                    )}
                    {selected.data.body && (
                      <button
                        className="inspect-open"
                        onClick={() => open(selected.id)}
                      >
                        Open internal graph
                        <ArrowUpRight size={15} />
                      </button>
                    )}
                    {selected.data.kind === "loop" && (
                      <button className="inspect-open" onClick={unrollSelected}>
                        Expand into editable applications{" "}
                        <GitBranch size={15} />
                      </button>
                    )}
                    {supportsWeightSharing(selected.data) && (
                      <button
                        className="inspect-open"
                        onClick={duplicateShared}
                      >
                        Add application with shared weights <Copy size={15} />
                      </button>
                    )}
                    <button
                      className="save-library"
                      onClick={() =>
                        fitGraph({
                          nodes: [{ id: selected.id }],
                          duration: motion(250),
                        })
                      }
                    >
                      <Maximize2 size={13} /> Zoom to this node
                    </button>
  </>;
}
