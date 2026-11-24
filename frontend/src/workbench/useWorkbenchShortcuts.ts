import { useEffect, type RefObject } from "react";

export function useWorkbenchShortcuts(modal: string | null, sidebarMenuButton: RefObject<HTMLButtonElement | null>, closePanel: () => void, save: () => Promise<boolean>, undo: () => void, redo: () => void, group: () => void) {
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        if (!modal) void save();
        return;
      }
      const target = event.target;
      if (modal || (target instanceof Element && target.closest("[data-reference-pane]"))) return;
      if (event.key === "Escape") {
        closePanel();
        if (document.activeElement?.closest("#workspace-sidebar")) sidebarMenuButton.current?.focus();
        return;
      }
      if (target instanceof Element && target.closest("input,textarea,select,[contenteditable]")) return;
      if ((event.ctrlKey || event.metaKey) && event.key === "z") {
        event.preventDefault();
        event.shiftKey ? redo() : undo();
      }
      if ((event.ctrlKey || event.metaKey) && event.key === "g") {
        event.preventDefault();
        group();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [modal, sidebarMenuButton, closePanel, save, undo, redo, group]);
}
