import { FolderOpen, Plus, Settings2 } from "lucide-react";
import type { WorkspacePanel } from "../workspaceLocation";

export function MobileTools({ mobilePanel, libraryBlocks, onBlocks, onProjects, onProperties }: { mobilePanel: WorkspacePanel; libraryBlocks: boolean; onBlocks: () => void; onProjects: () => void; onProperties: () => void }) {
  return (
      <nav className="mobile-tools" aria-label="Editor panels">
        <button
          aria-expanded={mobilePanel === "library" && libraryBlocks}
          onClick={onBlocks}
        >
          <Plus size={18} />
          Add blocks
        </button>
        <button
          aria-haspopup="dialog"
          onClick={onProjects}
        >
          <FolderOpen size={18} />
          Projects
        </button>
        <button
          aria-expanded={mobilePanel === "properties"}
          onClick={onProperties}
        >
          <Settings2 size={18} />
          Properties
        </button>
      </nav>
  );
}
