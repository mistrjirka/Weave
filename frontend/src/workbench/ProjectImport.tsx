import type { RefObject } from "react";
import { projectSchema } from "../api";
import type { Project } from "../types";

export function ProjectImport({ fileRef, loadProject, notify }: { fileRef: RefObject<HTMLInputElement | null>; loadProject: (project: Project) => Promise<boolean>; notify: (message: string) => void }) {
  return (
      <input
        hidden
        ref={fileRef}
        type="file"
        accept=".json"
        onChange={async (e) => {
          try {
            const f = e.target.files?.[0];
            if (!f) return;
            const p = projectSchema.parse(JSON.parse(await f.text()));
            await loadProject(p);
            notify("Project imported");
          } catch (e) {
            notify(String(e));
          } finally {
            if (fileRef.current) fileRef.current.value = "";
          }
        }}
      />
  );
}
