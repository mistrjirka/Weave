import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Search, X } from "lucide-react";
import { z } from "zod";
import { responseData } from "./api";
import type { NodeData, Project } from "./types";

export type ConnectionBlockChoice = {
  id: string;
  label: string;
  description: string;
  category: string;
  data: NodeData;
};
const compatibilitySchema = z.object({
  source: z
    .object({
      shape: z.array(z.union([z.string(), z.number()])),
      dtype: z.string(),
    })
    .nullable(),
  matches: z.array(
    z.object({ id: z.string(), ports: z.array(z.string()).min(1) }),
  ),
  message: z.string().nullable(),
});

export function ConnectionPicker({
  position,
  bounds,
  choices,
  project,
  path,
  sourceId,
  sourceHandle,
  sourceLabel,
  onChoose,
  onClose,
}: {
  position: { x: number; y: number };
  bounds: { left: number; top: number; right: number; bottom: number };
  choices: ConnectionBlockChoice[];
  project: Project;
  path: string[];
  sourceId: string;
  sourceHandle: string;
  sourceLabel: string;
  onChoose: (choice: ConnectionBlockChoice, port: string) => void;
  onClose: () => void;
}) {
  const root = useRef<HTMLDivElement>(null),
    input = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const id = useId();
  const [attempt, setAttempt] = useState(0);
  const [resolution, setResolution] = useState<{
    request: string;
    data: z.infer<typeof compatibilitySchema>;
  } | null>(null);
  const [failure, setFailure] = useState<{
    request: string;
    message: string;
  } | null>(null);
  const requestBody = useMemo(
    () =>
      JSON.stringify({
        project,
        path,
        source: sourceId,
        handle: sourceHandle,
        candidates: choices.map((choice) => ({
          id: choice.id,
          data: choice.data,
        })),
      }),
    [project, path, sourceId, sourceHandle, choices],
  );
  useEffect(() => {
    const controller = new AbortController();
    let disposed = false;
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    void fetch("/api/connections/compatible", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: requestBody,
      signal: controller.signal,
    })
      .then(responseData)
      .then((value) => {
        if (!disposed)
          setResolution({
            request: requestBody,
            data: compatibilitySchema.parse(value),
          });
      })
      .catch((error) => {
        if (!disposed)
          setFailure({
            request: requestBody,
            message: controller.signal.aborted
              ? "Checking compatible inputs timed out."
              : error instanceof Error
                ? error.message
                : String(error),
          });
      })
      .finally(() => window.clearTimeout(timeout));
    return () => {
      disposed = true;
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, [requestBody, attempt]);
  const result = resolution?.request === requestBody ? resolution.data : null;
  const error = failure?.request === requestBody ? failure.message : null;
  const loading = !result && !error;
  const available = new Map(
    result?.matches.map((match) => [match.id, match.ports]) ?? [],
  );
  const words = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  const matches = choices.filter(
    (choice) =>
      available.has(choice.id) &&
      words.every((word) =>
        `${choice.label} ${choice.description} ${choice.category}`
          .toLocaleLowerCase()
          .includes(word),
      ),
  );
  function choose(choice: ConnectionBlockChoice) {
    const port = available.get(choice.id)?.[0];
    if (port) onChoose(choice, port);
  }
  const index = Math.min(active, Math.max(0, matches.length - 1));
  useEffect(() => {
    input.current?.focus();
  }, []);
  useEffect(() => {
    const closeOutside = (event: PointerEvent) => {
      if (event.target instanceof Node && !root.current?.contains(event.target))
        onClose();
    };
    document.addEventListener("pointerdown", closeOutside, true);
    window.addEventListener("resize", onClose);
    return () => {
      document.removeEventListener("pointerdown", closeOutside, true);
      window.removeEventListener("resize", onClose);
    };
  }, [onClose]);
  useEffect(() => {
    root.current
      ?.querySelector('[aria-selected="true"]')
      ?.scrollIntoView({ block: "nearest" });
  }, [index, query]);
  const width = Math.min(330, bounds.right - bounds.left - 16);
  const height = Math.min(390, bounds.bottom - bounds.top - 16);
  const left = Math.max(
    bounds.left + 8,
    Math.min(position.x, bounds.right - width - 8),
  );
  const top = Math.max(
    bounds.top + 8,
    Math.min(position.y, bounds.bottom - height - 8),
  );
  return (
    <div
      ref={root}
      role="dialog"
      aria-label="Add connected block"
      className="connection-picker"
      style={{ left, top, width, maxHeight: height }}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          onClose();
          return;
        }
        if (event.target !== input.current) return;
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          setActive(
            matches.length
              ? (index +
                  (event.key === "ArrowDown" ? 1 : -1) +
                  matches.length) %
                  matches.length
              : 0,
          );
        }
        if (event.key === "Enter" && matches[index]) {
          event.preventDefault();
          choose(matches[index]);
        }
      }}
    >
      <div className="connection-picker-heading">
        <div>
          <strong>Add connected block</strong>
          <small>
            From {sourceLabel}
            {result?.source
              ? ` · ${result.source.dtype} [${result.source.shape.join(", ")}]`
              : ""}
          </small>
        </div>
        <button aria-label="Cancel adding block" onClick={onClose}>
          <X size={16} />
        </button>
      </div>
      <label className="connection-picker-search">
        <Search size={16} />
        <input
          ref={input}
          role="combobox"
          aria-label="Search connected blocks"
          aria-expanded="true"
          aria-controls={id}
          aria-autocomplete="list"
          aria-activedescendant={matches[index] ? `${id}-${index}` : undefined}
          value={query}
          placeholder="Search blocks…"
          onChange={(event) => {
            setQuery(event.target.value);
            setActive(0);
          }}
        />
      </label>
      <div
        id={id}
        role="listbox"
        aria-label="Available blocks"
        className="connection-picker-results"
      >
        {matches.map((choice, i) => (
          <button
            key={choice.id}
            id={`${id}-${i}`}
            role="option"
            aria-selected={i === index}
            tabIndex={-1}
            onPointerMove={() => setActive(i)}
            onClick={() => choose(choice)}
          >
            <span>
              <strong>{choice.label}</strong>
              <small>{choice.description}</small>
            </span>
            <small>
              {choice.category}
              {available.get(choice.id)?.[0] !== "a" && (
                <span> · Input {available.get(choice.id)?.[0]}</span>
              )}
            </small>
          </button>
        ))}
        {loading && <p role="status">Checking compatible inputs…</p>}
        {error && (
          <div role="alert">
            <p>{error}</p>
            <button
              onClick={() => {
                setFailure(null);
                setResolution(null);
                setAttempt((value) => value + 1);
              }}
            >
              Retry compatibility check
            </button>
          </div>
        )}
        {result?.message && <p role="status">{result.message}</p>}
        {result && !result.message && !matches.length && (
          <p>
            {query
              ? "No matching blocks with compatible inputs. Try another name."
              : "No compatible blocks for this tensor."}
          </p>
        )}
      </div>
      <p className="connection-picker-hint">
        Compatible inputs only · ↑ ↓ to choose · Enter to add · Esc to cancel
      </p>
    </div>
  );
}
