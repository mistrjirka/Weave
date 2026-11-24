import { useEffect } from "react";
import type { GraphNode, JsonValue, Project, ValidationReport } from "../types";
import type { ReferenceInspection } from "../ReferenceInspector";

function propertyValue(
  value: JsonValue | undefined,
  variables: Record<string, JsonValue | undefined>,
) {
  if (
    typeof value === "string" &&
    value.startsWith("$") &&
    value.slice(1) in variables
  )
    return `${value} → ${variables[value.slice(1)]}`;
  return typeof value === "string" ? value : JSON.stringify(value);
}
export function scopeVariables(project: Project, path: string[]) {
  let graph = project.graph;
  let variables: Record<string, JsonValue | undefined> = {
    ...project.config,
    ...graph.variables,
  };
  for (const id of path) {
    const node = graph.nodes.find((node) => node.id === id);
    if (!node?.data.body) break;
    const overrides = Object.fromEntries(
      Object.entries(node.data.params).map(([key, value]) => [
        key,
        typeof value === "string" && value.startsWith("$")
          ? variables[value.slice(1)]
          : value,
      ]),
    );
    variables = { ...variables, ...overrides };
    graph = node.data.body;
    variables = { ...variables, ...graph.variables };
  }
  return variables;
}
export function useReferenceInspection({
  selected,
  prefix,
  project,
  path,
  report,
  onInspect,
}: {
  selected: GraphNode | undefined;
  prefix: string;
  project: Project;
  path: string[];
  report: ValidationReport | null;
  onInspect: (inspection: ReferenceInspection | null) => void;
}) {
  useEffect(() => {
    const variables = scopeVariables(project, path);
    const tokenizer = selected
      ? (report?.tokenizers?.[prefix + selected.id] ?? selected.data.tokenizer)
      : undefined;
    if (tokenizer) {
      variables.vocab = tokenizer.capacity;
      if (tokenizer.mask_id != null) variables.mask_id = tokenizer.mask_id;
    }
    onInspect(
      selected
        ? {
            id: selected.id,
            scope: prefix,
            label: selected.data.label,
            kind: selected.data.kind,
            projectName: project.name,
            description: selected.data.description,
            annotation: selected.data.annotation,
            properties: [
              ...Object.entries(selected.data.params).map(([name, value]) => ({
                name,
                value: propertyValue(value, variables) ?? "",
              })),
              ...(selected.data.kind === "input" && tokenizer
                ? [
                    {
                      name: "Tokenizer",
                      value:
                        tokenizer.display_name ||
                        tokenizer.repo ||
                        tokenizer.class ||
                        "Local tokenizer",
                    },
                    {
                      name: "Vocabulary capacity",
                      value: String(tokenizer.capacity),
                    },
                  ]
                : []),
            ],
            shape: report?.shapes[prefix + selected.id] ?? null,
            parameters: report?.counts[prefix + selected.id] ?? 0,
            share: selected.data.share,
          }
        : null,
    );
  }, [selected, prefix, project, path, report, onInspect]);
}
