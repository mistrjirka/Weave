import { useEffect, useRef, useState, type CSSProperties } from "react";
import { retainedPath } from "../ui";
import { compatiblePanel, readWorkspaceLocation, workspaceURL, type ReferenceLocation, type WorkspacePanel } from "../workspaceLocation";
import type { ReferenceInspection } from "../ReferenceInspector";
import type { ProjectTab } from "../projectWorkflow";
import type { Project } from "../types";

export function useWorkspaceNavigation(initial: Project, initialTemplates: Project[], projectId: string) {
  const [initialLocation] = useState(() =>
    readWorkspaceLocation(window.location.search),
  );
  const [path, setPath] = useState<string[]>(() =>
      retainedPath(initial, initialLocation.blocks),
    ),
    [selection, setSelection] = useState<string | null>(
      initialLocation.split && initialLocation.referenceLocation.node
        ? null
        : initialLocation.node,
    ),
    [tab, setTab] = useState(initialLocation.tab),
    [left, setLeft] = useState(initialLocation.library),
    [search, setSearch] = useState(""),
    [modal, setModal] = useState<string | null>(
      initialLocation.workspace === "new"
        ? "new-project"
        : initialLocation.workspace
          ? "projects"
          : null,
    );
  const librarySearchRef = useRef<HTMLInputElement>(null);
  const [projectTab, setProjectTab] = useState<ProjectTab>(
    initialLocation.workspace === "examples" ? "examples" : "projects",
  );
  const [starterId, setStarterId] = useState(
    initialTemplates.some((p) => p.id === initialLocation.starter)
      ? initialLocation.starter || "blank"
      : "blank",
  );
  const [mobilePanel, setMobilePanel] = useState<WorkspacePanel>(
    initialLocation.panel,
  );
  const visibleMobilePanel = compatiblePanel(tab, mobilePanel);
  useEffect(() => {
    setMobilePanel((current) => compatiblePanel(tab, current));
  }, [tab]);
  const sidebarMenuButton = useRef<HTMLButtonElement>(null);
  const [splitView, setSplitView] = useState(initialLocation.split);
  const [splitRatio, setSplitRatio] = useState(initialLocation.ratio);
  const [referenceChoice, setReferenceChoice] = useState(
    initialLocation.reference ||
      "example:" +
        (initialTemplates.find((p) => p.id !== initial.id)?.id || initial.id),
  );
  const [referenceLocation, setReferenceLocation] = useState<ReferenceLocation>(
    initialLocation.referenceLocation,
  );
  const [referenceInspection, setReferenceInspection] =
    useState<ReferenceInspection | null>(null);
  const splitButton = useRef<HTMLButtonElement>(null);
  const comparisonStyle: CSSProperties & {
    "--editable-pane-size": string;
  } = { "--editable-pane-size": `${splitRatio}%` };
  function closeSplitView() {
    setSplitView(false);
    requestAnimationFrame(() => splitButton.current?.focus());
  }
  useEffect(() => {
    const panel =
      mobilePanel === "library" || mobilePanel === "properties"
        ? mobilePanel
        : null;
    const url = workspaceURL(window.location.href, {
      model: projectId,
      blocks: path,
      node: selection,
      tab,
      library: left,
      panel,
      split: splitView,
      reference: referenceChoice,
      referenceLocation,
      ratio: splitRatio,
      workspace:
        modal === "new-project"
          ? "new"
          : modal === "projects"
            ? projectTab
            : null,
      starter: modal === "new-project" ? starterId : null,
    });
    if (url !== window.location.href)
      window.history.replaceState(window.history.state, "", url);
  }, [
    projectId,
    path,
    selection,
    tab,
    left,
    mobilePanel,
    splitView,
    referenceChoice,
    referenceLocation,
    splitRatio,
    modal,
    projectTab,
    starterId,
  ]);
  return { path, setPath, selection, setSelection, tab, setTab, left, setLeft, search, setSearch, modal, setModal, librarySearchRef, projectTab, setProjectTab, starterId, setStarterId, mobilePanel, setMobilePanel, visibleMobilePanel, sidebarMenuButton, splitView, setSplitView, splitRatio, setSplitRatio, referenceChoice, setReferenceChoice, referenceLocation, setReferenceLocation, referenceInspection, setReferenceInspection, splitButton, comparisonStyle, closeSplitView };
}
