import React, { useEffect, useState } from "react";
import { z } from "zod";
import { bindingValue } from "../graphVariables";
import type { JsonValue, Parameters } from "../types";

const jsonValue: z.ZodType<JsonValue> = z.lazy(() => z.union([
  z.string(), z.number().finite(), z.boolean(), z.null(),
  z.array(jsonValue), z.record(z.string(), jsonValue),
]));

export function Field({
  name,
  value,
  onChange,
  disabledReason,
  help,
  options,
  variables,
  displayName,
  parseValue,
}: {
  name: string;
  options?: string[];
  variables?: Parameters;
  displayName?: string;
  value: JsonValue;
  onChange: (value: JsonValue) => void;
  parseValue?: (value: JsonValue) => JsonValue | null;
  disabledReason?: string;
  help?: string;
}) {
  const [helpOpen, setHelpOpen] = useState(false);
  const [error, setError] = useState("");
  const errorId = React.useId();
  const [draft, setDraft] = useState(
    typeof value === "object" ? JSON.stringify(value) : String(value ?? ""),
  );
  useEffect(() => {
    setError("");
    setDraft(
      typeof value === "object" ? JSON.stringify(value) : String(value ?? ""),
    );
  }, [value]);
  const label = displayName ?? name.replaceAll("_", " ");
  const helpId = `${errorId}-explanation`;
  const heading = (
    <span className="setting-heading">
      <span>{label}</span>
      {help && (
        <button
          type="button"
          className="setting-help-button"
          aria-label={`Explain ${label}`}
          aria-expanded={helpOpen}
          aria-controls={helpId}
          onClick={() => setHelpOpen((open) => !open)}
        >
          ?
        </button>
      )}
    </span>
  );
  const explanation =
    help && helpOpen ? (
      <p id={helpId} className="setting-explanation">
        {help}
      </p>
    ) : null;
  const inheritedValue =
    typeof value === "string" && value.startsWith("$")
      ? bindingValue(value, variables ?? {})
      : undefined;
  if (inheritedValue !== undefined && typeof inheritedValue !== "number")
    return (
      <div className="property-field">
        {heading}
        <div className="stat-row">
          <code>{String(value)}</code>
          <b>
            {typeof inheritedValue === "object"
              ? JSON.stringify(inheritedValue)
              : String(inheritedValue)}
          </b>
        </div>
        <small className="field-help">
          Linked to the enclosing parameter. Change it there, or set an
          independent value here.
        </small>
        <button
          className="small-button"
          disabled={!!disabledReason}
          onClick={() => onChange(inheritedValue)}
        >
          Set independently
        </button>
        {disabledReason && (
          <small className="field-help">{disabledReason}</small>
        )}
        {explanation}
      </div>
    );
  if (typeof value === "boolean")
    return (
      <div className="setting-boolean">
        <div className="property-row">
          {heading}
          <button
            className={"toggle " + (value ? "on" : "")}
            role="switch"
            aria-checked={value}
            aria-label={label}
            aria-describedby={
              disabledReason ? `${errorId}-help` : helpOpen ? helpId : undefined
            }
            disabled={!!disabledReason}
            onClick={() => onChange(!value)}
          >
            <i />
          </button>
        </div>
        {disabledReason && (
          <small id={`${errorId}-help`} className="field-help">
            {disabledReason}
          </small>
        )}
        {explanation}
      </div>
    );
  const selects: Record<string, string[]> = {
    variant: ["two_stage", "three_stage"],
    rope_scaling: ["none", "yarn"],
    approximate: ["none", "tanh"],
    mode: ["max", "average"],
    dtype: ["int64", "float32", "bool", "attention_pattern"],
    output_dtype: ["float32", "int64", "bool", "attention_pattern"],
    strategy: ["confidence", "random"],
    pattern: [
      "local",
      "global",
      "block",
      "random",
      "causal",
      "full",
      "content_topk",
      "custom",
    ],
    backend: ["dense", "auto", "flex"],
  };
  const choices = options ?? selects[name];
  if (choices)
    return (
      <div className="property-field">
        {heading}
        <select
          aria-label={label}
          name={name}
          disabled={!!disabledReason}
          aria-describedby={
            disabledReason ? `${errorId}-help` : helpOpen ? helpId : undefined
          }
          value={String(value)}
          onChange={(e) => onChange(e.target.value)}
        >
          {choices.map((v: string) => (
            <option key={v}>{v}</option>
          ))}
        </select>
        {disabledReason && (
          <small id={`${errorId}-help`} className="field-help">
            {disabledReason}
          </small>
        )}
        {explanation}
      </div>
    );
  const commit = () => {
    if (
      draft ===
      (typeof value === "object" ? JSON.stringify(value) : String(value))
    ) {
      setError("");
      return;
    }
    let next: JsonValue = draft;
    if (
      typeof value === "number" ||
      (typeof value === "string" && value.startsWith("$"))
    ) {
      if (/^\$[a-zA-Z_][a-zA-Z0-9_]*$/.test(draft.trim())) next = draft.trim();
      else if (draft.trim() && Number.isFinite(Number(draft)))
        next = Number(draft);
      else {
        setError("Not applied. Enter a number or a binding such as $width.");
        return;
      }
    } else if (typeof value === "object") {
      try {
        const parsed: unknown = JSON.parse(draft);
        next = jsonValue.parse(parsed);
        if (
          Array.isArray(value)
            ? !Array.isArray(next)
            : !next || typeof next !== "object" || Array.isArray(next)
        )
          throw new Error();
      } catch {
        setError(
          Array.isArray(value)
            ? "Not applied. Enter a valid JSON array, such as [2, 3]."
            : "Not applied. Enter a valid JSON object.",
        );
        return;
      }
    }
    if (parseValue) {
      const parsed = parseValue(next);
      if (parsed === null) {
        setError("Not applied. Enter a finite number.");
        return;
      }
      setError("");
      onChange(parsed);
      return;
    }
    setError("");
    onChange(next);
  };
  return (
    <div className="property-field">
      {heading}
      {name === "source" ? (
        <textarea
          className="source-editor"
          aria-label="Python source"
          disabled={!!disabledReason}
          name={name}
          aria-invalid={!!error}
          aria-describedby={
            error
              ? errorId
              : disabledReason
                ? `${errorId}-help`
                : helpOpen
                  ? helpId
                  : undefined
          }
          value={draft}
          spellCheck={false}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
        />
      ) : (
        <input
          aria-label={label}
          disabled={!!disabledReason}
          name={name}
          autoComplete="off"
          spellCheck={false}
          aria-invalid={!!error}
          aria-describedby={
            error
              ? errorId
              : disabledReason
                ? `${errorId}-help`
                : helpOpen
                  ? helpId
                  : undefined
          }
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
          }}
        />
      )}
      {disabledReason && (
        <small id={`${errorId}-help`} className="field-help">
          {disabledReason}
        </small>
      )}
      {explanation}
      {error && (
        <span id={errorId} className="field-error" role="alert">
          {error}
        </span>
      )}
    </div>
  );
}
