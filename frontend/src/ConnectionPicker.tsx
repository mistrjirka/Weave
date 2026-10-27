import { useEffect, useId, useRef, useState } from "react";
import { Search, X } from "lucide-react";
import type { CatalogEntry, NodeData } from "./types";

export type ConnectionBlockChoice = {
  id: string;
  label: string;
  description: string;
  category: string;
} & (
  | { source: "catalog"; preset: CatalogEntry }
  | { source: "saved"; preset: NodeData }
);

export function ConnectionPicker({
  position,
  bounds,
  choices,
  sourceLabel,
  onChoose,
  onClose,
}: {
  position: { x: number; y: number };
  bounds: { left: number; top: number; right: number; bottom: number };
  choices: ConnectionBlockChoice[];
  sourceLabel: string;
  onChoose: (choice: ConnectionBlockChoice) => void;
  onClose: () => void;
}) {
  const root = useRef<HTMLDivElement>(null),
    input = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const id = useId();
  const words = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  const matches = choices.filter((choice) =>
    words.every((word) =>
      `${choice.label} ${choice.description} ${choice.category}`
        .toLocaleLowerCase()
        .includes(word),
    ),
  );
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
          onChoose(matches[index]);
        }
      }}
    >
      <div className="connection-picker-heading">
        <div>
          <strong>Add connected block</strong>
          <small>From {sourceLabel}</small>
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
            onClick={() => onChoose(choice)}
          >
            <span>
              <strong>{choice.label}</strong>
              <small>{choice.description}</small>
            </span>
            <small>{choice.category}</small>
          </button>
        ))}
        {!matches.length && <p>No matching blocks. Try another name.</p>}
      </div>
      <p className="connection-picker-hint">
        ↑ ↓ to choose · Enter to add · Esc to cancel
      </p>
    </div>
  );
}
