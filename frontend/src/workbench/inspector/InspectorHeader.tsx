import { Settings2, X } from "lucide-react";

interface InspectorHeaderProps {
  hasReference: boolean;
  hasSelection: boolean;
  selectedCount: number;
  hasConnection: boolean;
  onClosePanel: () => void;
  onClearSelection: () => void;
}

export function InspectorHeader({ hasReference, hasSelection, selectedCount, hasConnection, onClosePanel, onClearSelection }: InspectorHeaderProps) {
  return (
    <div className="inspector-heading">
      <span>
        <Settings2 size={15} />
        {hasReference ? "Reference properties" : hasSelection ? selectedCount > 1 ? `${selectedCount} blocks selected` : "Block properties" : hasConnection ? "Connection properties" : "Graph overview"}
      </span>
      <button className="drawer-close" aria-label="Close properties" onClick={onClosePanel}><X size={20} /></button>
      {(hasSelection || hasReference) && <button aria-label="Close inspector selection" title="Close inspector selection" onClick={onClearSelection}><X size={14} /></button>}
    </div>
  );
}
