import { useRef, useState } from "react";
const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);
export function useWorkspaceTask(
  setError: (error: string) => void,
  setNotice: (notice: string) => void,
) {
  const pendingTasksRef = useRef(0);
  const [busy, setBusy] = useState(false);
  async function task(action: () => Promise<unknown>) {
    pendingTasksRef.current += 1;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await action();
    } catch (e) {
      setError(message(e));
    } finally {
      pendingTasksRef.current -= 1;
      setBusy(pendingTasksRef.current > 0);
    }
  }
  return { task, busy, pendingTasksRef };
}
