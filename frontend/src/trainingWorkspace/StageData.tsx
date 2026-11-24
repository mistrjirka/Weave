import { useRef } from "react";
import { z } from "zod";
import { Upload } from "lucide-react";
import type { Project } from "../types";
import {
  trainingRequest,
  type TrainingConfig,
  type TrainingStage,
} from "../trainingApi";

export const previewSchema = z.object({
  records: z.array(
    z.object({
      tokens: z.array(
        z.object({ id: z.number(), text: z.string(), supervised: z.boolean() }),
      ),
      supervised_count: z.number(),
    }),
  ),
  note: z.string(),
});

type Props = {
  project: Project;
  stage: TrainingStage;
  codeFirst: boolean;
  busy: boolean;
  preview: z.infer<typeof previewSchema> | null;
  setPreview: (preview: z.infer<typeof previewSchema> | null) => void;
  task: (action: () => Promise<unknown>) => Promise<void>;
  setNotice: (notice: string) => void;
  setting: <K extends keyof TrainingConfig>(
    key: K,
    value: TrainingConfig[K],
  ) => void;
};
export function StageData({
  project,
  stage,
  codeFirst,
  busy,
  preview,
  setPreview,
  task,
  setNotice,
  setting,
}: Props) {
  const config = stage.config;
  const datasetRef = useRef<HTMLInputElement>(null);
  return (
    <>
      {!codeFirst && config.recipe !== "custom" && (
        <>
          <h3>Data</h3>
          <p className="training-muted">
            The tokenizer comes from your graph’s token input.{" "}
            <a
              href={`?model=${encodeURIComponent(project.id)}&node=${encodeURIComponent(project.graph.nodes.find((node) => node.data.kind === "input")?.id ?? "")}`}
            >
              Configure the input tokenizer in Architecture
            </a>
            .
          </p>
          <div className="training-form-grid">
            <label>
              Training dataset path on server
              <input
                value={config.dataset}
                placeholder="/path/to/train.jsonl"
                onChange={(e) => setting("dataset", e.target.value)}
              />
              <small>
                Host path: TXT, JSONL, JSON, or Parquet. Pretokenized
                input_ids/labels also work.
              </small>
            </label>
            <label>
              Validation dataset path on server
              <input
                value={config.eval_dataset}
                placeholder="Optional held-out dataset"
                onChange={(e) => setting("eval_dataset", e.target.value)}
              />
              <small>Use held-out data to track generalization.</small>
            </label>
          </div>
          <input
            ref={datasetRef}
            hidden
            type="file"
            accept=".txt,.json,.jsonl,.parquet"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              void task(async () => {
                const form = new FormData();
                form.append("file", file);
                const uploaded = await trainingRequest(
                  "/datasets",
                  z.object({
                    path: z.string(),
                    sha256: z.string(),
                    bytes: z.number(),
                  }),
                  form,
                );
                setting("dataset", uploaded.path);
                setNotice(
                  `Dataset uploaded and fingerprinted: ${uploaded.sha256.slice(0, 12)}…`,
                );
              });
              e.target.value = "";
            }}
          />
          <button onClick={() => datasetRef.current?.click()} disabled={busy}>
            <Upload aria-hidden="true" size={14} />
            Upload training data
          </button>
          <button
            onClick={() =>
              void task(async () => {
                setPreview(
                  await trainingRequest("/preview", previewSchema, {
                    project,
                    stage,
                  }),
                );
              })
            }
            disabled={busy || !config.dataset}
          >
            Preview tokens & loss mask
          </button>
          {preview && (
            <div className="training-data-preview">
              <p>{preview.note}</p>
              {preview.records.map((row, index) => (
                <div key={index}>
                  <small>
                    Record {index + 1}: {row.supervised_count} supervised target
                    tokens
                  </small>
                  <div>
                    {row.tokens.map((token, i) => (
                      <span
                        key={i}
                        className={token.supervised ? "is-supervised" : ""}
                        title={`Token ${token.id}: ${token.supervised ? "contributes to loss" : "ignored as target"}`}
                      >
                        {token.text}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
          {config.recipe === "sft" && (
            <p className="training-muted">
              Use messages and optional tools schemas. Assistant-only training
              requires a chat template that supplies assistant token masks, or
              pretokenized labels with -100 on ignored tokens.
            </p>
          )}
        </>
      )}
    </>
  );
}
