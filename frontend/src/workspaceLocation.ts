/** Shareable view state. Graph documents stay in saved projects/local recovery. */
export type WorkspaceTab = "graph" | "code" | "sources" | "training";
export interface ReferenceLocation {
  blocks: string[];
  node: string | null;
}
export interface WorkspaceLocation {
  model: string | null;
  blocks: string[];
  node: string | null;
  tab: WorkspaceTab;
  library: "blocks" | "models";
  panel: "library" | "properties" | null;
  split: boolean;
  reference: string | null;
  referenceLocation: ReferenceLocation;
  ratio: number;
  workspace: "projects" | "examples" | "new" | null;
  starter: string | null;
}
const keys = [
  "model",
  "block",
  "node",
  "view",
  "library",
  "panel",
  "split",
  "reference",
  "refBlock",
  "refNode",
  "ratio",
  "workspace",
  "starter",
];
export function readWorkspaceLocation(search: string): WorkspaceLocation {
  const params = new URLSearchParams(search);
  const view = params.get("view"),
    panel = params.get("panel"),
    reference = params.get("reference");
  const workspace = params.get("workspace");
  const ratio = Number(params.get("ratio") || 50);
  return {
    model: params.get("model") || null,
    blocks: params.getAll("block"),
    node: params.get("node") || null,
    tab:
      view === "code" || view === "sources" || view === "training"
        ? view
        : "graph",
    library: params.get("library") === "models" ? "models" : "blocks",
    panel: panel === "library" || panel === "properties" ? panel : null,
    split: params.get("split") === "1",
    reference:
      reference && /^(example|saved):.+$/.test(reference) ? reference : null,
    referenceLocation: {
      blocks: params.getAll("refBlock"),
      node: params.get("refNode") || null,
    },
    ratio: Number.isFinite(ratio) ? Math.max(30, Math.min(70, ratio)) : 50,
    workspace:
      workspace === "projects" ||
      workspace === "examples" ||
      workspace === "new"
        ? workspace
        : null,
    starter: workspace === "new" ? params.get("starter") : null,
  };
}
export function workspaceURL(href: string, state: WorkspaceLocation): string {
  const url = new URL(href);
  if (url.searchParams.get("model") !== state.model) {
    url.searchParams.delete("stage");
    url.searchParams.delete("run");
  }
  for (const key of keys) url.searchParams.delete(key);
  const set = (key: string, value: string | null) => {
    if (value) url.searchParams.set(key, value);
  };
  set("model", state.model);
  for (const block of state.blocks) url.searchParams.append("block", block);
  set("node", state.node);
  if (state.tab !== "graph") set("view", state.tab);
  if (state.library !== "blocks") set("library", state.library);
  set("panel", state.panel);
  set("workspace", state.workspace);
  if (state.workspace === "new") set("starter", state.starter);
  if (state.split) {
    set("split", "1");
    set("reference", state.reference);
    for (const block of state.referenceLocation.blocks)
      url.searchParams.append("refBlock", block);
    set("refNode", state.referenceLocation.node);
    if (Math.abs(state.ratio - 50) > 0.05)
      set("ratio", String(Number(state.ratio.toFixed(1))));
  }
  return url.toString();
}
