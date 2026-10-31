import { useEffect, useState } from "react";
import { z } from "zod";
import { responseData, tokenizerSchema } from "./api";
import type { TokenizerInfo } from "./types";

const choicesSchema = z.array(
  z.object({
    project_id: z.string(),
    project_name: z.string(),
    input_path: z.string(),
    tokenizer: tokenizerSchema,
  }),
);

export function ProjectTokenizerPicker({
  projectId,
  disabled,
  onSelect,
}: {
  projectId: string;
  disabled: boolean;
  onSelect: (info: TokenizerInfo) => void;
}) {
  const [choices, setChoices] = useState<z.infer<typeof choicesSchema>>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    let disposed = false;
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    setLoading(true);
    setError("");
    void fetch("/api/tokenizers/projects", {
      cache: "no-store",
      signal: controller.signal,
    })
      .then(responseData)
      .then((data) => {
        if (!disposed)
          setChoices(
            choicesSchema
              .parse(data)
              .filter((choice) => choice.project_id !== projectId),
          );
      })
      .catch((failure: unknown) => {
        if (!disposed)
          setError(
            controller.signal.aborted
              ? "Loading saved tokenizers timed out."
              : failure instanceof Error
                ? failure.message
                : String(failure),
          );
      })
      .finally(() => {
        window.clearTimeout(timeout);
        if (!disposed) setLoading(false);
      });
    return () => {
      disposed = true;
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, [projectId, attempt]);

  return (
    <>
      <label className="property-field">
        <span>Tokenizer from another project</span>
        <select
          value=""
          disabled={disabled || loading || !choices.length || !!error}
          onChange={(event) => {
            const choice = choices.find(
              (item) =>
                JSON.stringify([item.project_id, item.tokenizer.id]) ===
                event.target.value,
            );
            if (choice) onSelect(structuredClone(choice.tokenizer));
          }}
        >
          <option value="">
            {loading
              ? "Loading saved tokenizers…"
              : choices.length
                ? "Choose a saved tokenizer…"
                : "No tokenizers in other saved projects"}
          </option>
          {choices.map((choice) => {
            const key = JSON.stringify([
              choice.project_id,
              choice.tokenizer.id,
            ]);
            return (
              <option key={key} value={key}>
                {choice.project_name} ·{" "}
                {choice.tokenizer.repo || choice.tokenizer.class || "Tokenizer"}{" "}
                · {choice.tokenizer.capacity.toLocaleString()} tokens
              </option>
            );
          })}
        </select>
      </label>
      {error && (
        <div className="inline-error" role="alert">
          {error}{" "}
          <button onClick={() => setAttempt((value) => value + 1)}>
            Retry
          </button>
        </div>
      )}
    </>
  );
}
