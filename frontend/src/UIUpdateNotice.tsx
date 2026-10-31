import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";

/** Refresh only on request: training survives reloads, but a draft may be open. */
export function UIUpdateNotice() {
  const [available, setAvailable] = useState(false);
  useEffect(() => {
    const current = document.querySelector<HTMLScriptElement>(
      'script[type="module"][src]',
    )?.src;
    if (!current) return;
    let disposed = false;
    let pending: AbortController | null = null;
    const check = async () => {
      if (pending || document.visibilityState === "hidden") return;
      const controller = new AbortController();
      pending = controller;
      const timeout = window.setTimeout(() => controller.abort(), 10000);
      try {
        const response = await fetch("/", {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok) return;
        const page = new DOMParser().parseFromString(
          await response.text(),
          "text/html",
        );
        const source = page
          .querySelector('script[type="module"][src]')
          ?.getAttribute("src");
        if (!disposed && source)
          setAvailable(new URL(source, location.href).href !== current);
      } catch {
        /* Keep the current UI usable during server/network outages. */
      } finally {
        window.clearTimeout(timeout);
        pending = null;
      }
    };
    const timer = window.setInterval(() => void check(), 60000);
    const focus = () => void check();
    window.addEventListener("focus", focus);
    return () => {
      disposed = true;
      pending?.abort();
      window.clearInterval(timer);
      window.removeEventListener("focus", focus);
    };
  }, []);
  return available ? (
    <button
      title="A new interface is available. Reloading this page does not stop training."
      onClick={() => location.reload()}
    >
      <RefreshCw size={14} aria-hidden="true" /> Reload UI
    </button>
  ) : null;
}
