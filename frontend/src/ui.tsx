import React, { useEffect, useRef } from "react";
import type { Project } from "./types";

// getRandomValues works on LAN HTTP; randomUUID and Clipboard API require HTTPS.
export function newId() {
  const bytes = new Uint8Array(12);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function copyText(text: string) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {}
  }
  const previous = document.activeElement as HTMLElement | null;
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.cssText = "position:fixed;left:-9999px;top:0";
  document.body.append(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  previous?.focus({ preventScroll: true });
  if (!copied)
    throw new Error("Copy is unavailable. Download the code instead.");
}

export function retainedPath(project: Project, path: string[]) {
  let graph = project.graph;
  const valid: string[] = [];
  for (const id of path) {
    const node = graph.nodes.find((n) => n.id === id);
    if (!node?.data.body) break;
    valid.push(id);
    graph = node.data.body;
  }
  return valid;
}

export function Dialog({
  children,
  label,
  className = "",
  onClose,
}: {
  children: React.ReactNode;
  label: string;
  className?: string;
  onClose: () => void;
}) {
  const root = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const background = Array.from(
      document.querySelectorAll<HTMLElement>(
        ".app-shell > main, .app-shell > aside, .app-shell > nav, .app-shell > .drawer-backdrop",
      ),
    );
    background.forEach((el) => {
      el.inert = true;
    });
    const frame = requestAnimationFrame(() =>
      root.current
        ?.querySelector<HTMLElement>('[aria-label="Close dialog"], button')
        ?.focus(),
    );
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        close.current();
        return;
      }
      if (e.key !== "Tab") return;
      const fields = Array.from(
        root.current?.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled):not([type=hidden]), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex="0"]',
        ) || [],
      ).filter((el) => el.getClientRects().length);
      const first = fields[0],
        last = fields.at(-1);
      if (!first) {
        e.preventDefault();
        return;
      }
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last?.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", onKey, true);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("keydown", onKey, true);
      background.forEach((el) => {
        el.inert = false;
      });
      if (previous?.isConnected) previous.focus({ preventScroll: true });
    };
  }, []);
  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={root}
        className={"modal " + className}
        role="dialog"
        aria-modal="true"
        aria-label={label}
      >
        {children}
      </div>
    </div>
  );
}
