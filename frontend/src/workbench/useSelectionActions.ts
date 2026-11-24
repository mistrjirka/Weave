import type { Dispatch, SetStateAction } from "react";
import type { Edge, FitViewOptions } from "@xyflow/react";
import { api, unrollSchema } from "../api";
import { edgeContract } from "../lanePorts";
import {
  serializeSelection,
  parseSelection,
  pasteSelection,
} from "../graphClipboard";
import {
  addSharedApplication,
  buildLayer,
  insertOnEdge,
  type InsertionKind,
} from "../graphOperations";
import { layoutGraph } from "../graph";
import { newId as uid } from "../ui";
import type { Project, Graph, GraphNode, ValidationReport } from "../types";
import { groupSelection } from "./groupSelection";
import { transformerBody } from "./catalogNodeData";
import { replaceGraph } from "./document";
import { motion } from "./util";

export function useSelectionActions(
  project: Project,
  path: string[],
  templates: Project[],
  graph: Graph,
  selection: string | null,
  selected: GraphNode | undefined,
  connection: Edge | undefined,
  connectionShape: ReturnType<typeof edgeContract>,
  insertKind: InsertionKind,
  report: ValidationReport | null,
  prefix: string,
  changeGraph: (fn: (graph: Graph) => void) => void,
  change: (fn: (project: Project) => void) => void,
  setSelection: Dispatch<SetStateAction<string | null>>,
  setSelectedEdge: Dispatch<SetStateAction<string | null>>,
  fitGraph: (options?: Omit<FitViewOptions, "padding">) => unknown,
  notify: (message: string) => void,
) {
  function removeSelected() {
    if (!selection) return;
    changeGraph((g) => {
      const ids = new Set(
        g.nodes.filter((node) => node.selected).map((node) => node.id),
      );
      if (!ids.size) ids.add(selection);
      g.nodes = g.nodes.filter((node) => !ids.has(node.id));
      g.edges = g.edges.filter((e) => !ids.has(e.source) && !ids.has(e.target));
    });
    setSelection(null);
  }
  function duplicate() {
    const text = serializeSelection(graph, selection);
    const source = text ? parseSelection(text) : null;
    if (!source) return;
    const anchor = {
      x: Math.min(...source.nodes.map((node) => node.position.x)) + 45,
      y: Math.min(...source.nodes.map((node) => node.position.y)) + 65,
    };
    changeGraph((g) => {
      const nodes = pasteSelection(g, source, anchor, uid);
      for (const node of nodes) node.data.label += " copy";
      setSelection(nodes[0].id);
    });
  }
  function duplicateShared() {
    if (!selected) return;
    const input = graph.edges.find(
      (e) => e.target === selected.id && (e.targetHandle || "a") === "a",
    );
    changeGraph((g) => {
      const next = addSharedApplication(
        g,
        selected.id,
        input ? edgeContract(report, prefix, input) : null,
        report?.shapes[prefix + selected.id],
      );
      setSelection(next.id);
    });
    notify("Shared application added. Both instances use the same parameters.");
    setTimeout(() => fitGraph({ duration: motion(250) }), 100);
  }
  async function unrollSelected() {
    if (!selected) return;
    try {
      const result = unrollSchema.parse(
        await api("/graph/unroll", { project, path, node_id: selected.id }),
      );
      change((p) => {
        layoutGraph(result.graph);
        replaceGraph(p, path, result.graph);
      });
      setSelection(result.first_id);
      setSelectedEdge(null);
      notify(
        "Loop expanded into editable applications. Dashed links identify shared weights.",
      );
      setTimeout(() => fitGraph({ duration: motion(250) }), 100);
    } catch (e) {
      notify(String(e));
    }
  }
  function insertOnConnection() {
    if (!connection || connectionShape?.dtype !== "float32") return;
    const width = connectionShape.shape.at(-1);
    if (typeof width !== "number") return;
    const data = buildLayer(insertKind, {
      width,
      heads: project.config.heads,
      transformerBody: () => transformerBody(templates),
    });
    changeGraph((g) => {
      const next = insertOnEdge(g, connection.id, data);
      setSelection(next.id);
    });
    setSelectedEdge(null);
    notify(
      "Independent learned layer inserted. Shared transformer weights remain tied.",
    );
    setTimeout(() => fitGraph({ duration: motion(250) }), 100);
  }

  const groupSelectedNodes = () =>
    groupSelection(
      graph,
      selection,
      report,
      prefix,
      uid,
      changeGraph,
      setSelection,
      notify,
    );
  return {
    removeSelected,
    duplicate,
    duplicateShared,
    unrollSelected,
    insertOnConnection,
    groupSelectedNodes,
  };
}
