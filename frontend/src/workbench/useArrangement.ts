import { useRef, useState, type RefObject } from "react";
import { edgeContract } from "../lanePorts";
import { arrangeGraph } from "../layout";
import { layoutSignature } from "../layout-core";
import { measureLabels } from "../labels";
import { describeError } from "../api";
import type { Project, ValidationReport } from "../types";
import { graphAt } from "./document";

export function useArrangement(ref: RefObject<Project>, editRevision: RefObject<number>, viewPath: RefObject<string[]>, report: ValidationReport | null, change: (fn: (project: Project) => void, record?: boolean, markDirty?: boolean) => void, fitCanvas: () => void, notify: (message: string) => void) {
  const arrangeRequest = useRef(0);
  const [arranging, setArranging] = useState(false);
  const copy = <T,>(value: T): T => structuredClone(value);
async function arrangeScope(scope: string[], record = true) {
  const request = ++arrangeRequest.current,
    revision = editRevision.current,
    id = ref.current.id,
    snapshot = copy(graphAt(ref.current, scope)),
    signature = layoutSignature(snapshot);
  const key = scope.length ? scope.join("/") + "/" : "";
  const texts = snapshot.edges
    .map((edge) => ({
      id: edge.id,
      text:
        edgeContract(report, key, edge)?.shape.map(String).join(" × ") || "",
    }))
    .filter((label) => label.text);
  const sizes = measureLabels(texts);
  const labels = new Map<string, { width: number; height: number; text: string }>();
  for (const label of texts) {
    const size = sizes.get(label.id);
    if (size) labels.set(label.id, { ...size, text: label.text });
  }
  setArranging(true);
  try {
    const result = await arrangeGraph(snapshot, labels);
    if (ref.current.id !== id) return;
    if (
      editRevision.current !== revision ||
      layoutSignature(graphAt(ref.current, scope)) !== signature
    ) {
      if (record)
        notify("The graph changed while arranging. Try Auto arrange again.");
      return;
    }
    change(
      (project) => {
        const graph = graphAt(project, scope),
          positions = new Map(result.nodes.map((n) => [n.id, n.position]));
        for (const node of graph.nodes) {
          const position = positions.get(node.id);
          if (position) node.position = position;
        }
        graph.layout = result.layout;
      },
      record,
      record,
    );
    const fitRevision = editRevision.current;
    if (viewPath.current.join("/") === scope.join("/"))
      setTimeout(() => {
        if (
          editRevision.current === fitRevision &&
          viewPath.current.join("/") === scope.join("/")
        )
          fitCanvas();
      }, 80);
    if (record)
      notify(
        "Graph arranged with ELK: port-aware routes, crossing reduction, and space for labels.",
      );
  } catch (error) {
    notify("Auto arrange failed: " + describeError(error));
  } finally {
    if (request === arrangeRequest.current) setArranging(false);
  }
}

  return { arranging, arrangeScope };
}
