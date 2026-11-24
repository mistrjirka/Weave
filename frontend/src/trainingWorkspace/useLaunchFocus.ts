import { useEffect, useRef } from "react";
import type { RunDetail } from "../trainingApi";
export function useLaunchFocus(launchError: string, detail: RunDetail | null) {
  const launchErrorRef = useRef<HTMLDivElement>(null);
  const dashboardRef = useRef<HTMLElement>(null);
  const scrollToRun = useRef<string | null>(null);
  useEffect(() => {
    if (launchError) {
      launchErrorRef.current?.focus({ preventScroll: true });
      launchErrorRef.current?.scrollIntoView({ block: "nearest" });
    }
  }, [launchError]);
  useEffect(() => {
    if (detail && scrollToRun.current === detail.run.id) {
      dashboardRef.current?.scrollIntoView({ block: "start" });
      scrollToRun.current = null;
    }
  }, [detail?.run.id]);

  return { launchErrorRef, dashboardRef, scrollToRun };
}
