import { useMemo } from "react";
import { allNodes } from "../graph";
import { graphVariableScope, graphVariableUses } from "../graphVariables";
import { automaticVocabulary, tokenizerOwnsVocabulary, vocabularyField } from "../tokenizerVocabulary";
import { transformerSettings } from "../transformerControls";
import { edgeContract } from "../lanePorts";
import type { ReferenceInspection } from "../ReferenceInspector";
import type { ReferenceLocation } from "../workspaceLocation";
import type { Graph, GraphNode, Project, ValidationReport } from "../types";
import { graphAt } from "./document";

export function useInspectorData(project: Project, graph: Graph, path: string[], selection: string | null, selectedEdge: string | null, modal: string | null, tokenizerTarget: string | null, prefix: string, report: ValidationReport | null, splitView: boolean, referenceInspection: ReferenceInspection | null, referenceLocation: ReferenceLocation, change: (fn: (project: Project) => void) => void, changeGraph: (fn: (graph: Graph) => void) => void) {
  const selected = graph.nodes.find((n: GraphNode) => n.id === selection);
  const tokenizerInput =
    modal === "tokenizer"
      ? allNodes(project).find((item) => item.path === tokenizerTarget)?.node
      : undefined;
  const inputTokenizer = tokenizerInput?.data.tokenizer;
  const variableScope = graphVariableScope(project, path);
  const variableUses = useMemo(() => graphVariableUses(graph), [graph]);
  const hiddenVariables = tokenizerOwnsVocabulary(
    project,
    path,
    report?.tokenizers,
  )
    ? ["vocab"]
    : [];
  const updateGraphVariables = (values: Record<string, number>) =>
    change((p) => {
      if (path.length) graphAt(p, path).variables = values;
      else {
        p.config = values;
        delete p.graph.variables;
      }
    });
  const selectedTokenizer = selected
    ? (report?.tokenizers?.[prefix + selected.id] ?? selected.data.tokenizer)
    : undefined;
  const selectedVocabField = selected
    ? vocabularyField(selected.data)
    : undefined;
  const selectedAutoVocab = selected
    ? automaticVocabulary(selected.data, selectedTokenizer)
    : false;
  const fieldVariables = { ...variableScope.bindings };
  if (selectedTokenizer) {
    fieldVariables.vocab = selectedTokenizer.capacity;
    if (selectedTokenizer.mask_id != null)
      fieldVariables.mask_id = selectedTokenizer.mask_id;
  }
  const transformer = selected ? transformerSettings(selected.data) : null;
  const selectedFieldVariables = {
    ...fieldVariables,
    ...Object.fromEntries(
      Object.entries(transformer ?? {}).filter(
        ([, value]) => typeof value !== "string" || !value.startsWith("$"),
      ),
    ),
  };
  const selectedCount = graph.nodes.filter((node) => node.selected).length;
  const connection = graph.edges.find((e) => e.id === selectedEdge);
  const referenceDetails =
    splitView &&
    !selected &&
    !connection &&
    referenceInspection?.id === referenceLocation.node &&
    referenceInspection?.scope ===
      (referenceLocation.blocks.length
        ? referenceLocation.blocks.join("/") + "/"
        : "")
      ? referenceInspection
      : null;
  const connectionShape = connection
    ? edgeContract(report, prefix, connection)
    : null;
  const updateNode = (fn: (n: GraphNode) => void) =>
    changeGraph((g) => {
      const n = g.nodes.find((n: GraphNode) => n.id === selection);
      if (n) fn(n);
    });
  return { selected, tokenizerInput, inputTokenizer, variableScope, variableUses, hiddenVariables, updateGraphVariables, selectedTokenizer, selectedVocabField, selectedAutoVocab, fieldVariables, transformer, selectedFieldVariables, selectedCount, connection, referenceDetails, connectionShape, updateNode };
}
