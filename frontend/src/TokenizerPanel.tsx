import { useEffect, useRef, useState } from "react";
import { CheckCircle2, FolderOpen, Upload } from "lucide-react";
import { previewSchema, responseData, tokenizerSchema } from "./api";
import { ProjectTokenizerPicker } from "./ProjectTokenizerPicker";
import type { TokenizerInfo, TokenPreview } from "./types";

type Method = "upload" | "source" | "saved";
const methods: { id: Method; label: string }[] = [
  { id: "upload", label: "Upload" },
  { id: "source", label: "Repository / path" },
  { id: "saved", label: "Other project" },
];

export function tokenizerLabel(info: TokenizerInfo) {
  return info.display_name || info.repo || info.class || "Tokenizer";
}

export function TokenizerPanel({
  projectId,
  inputName,
  info,
  onApply,
  onSave,
  saving,
  onClose,
}: {
  projectId: string;
  inputName: string;
  info?: TokenizerInfo | null;
  onApply: (info: TokenizerInfo) => boolean;
  onSave: () => Promise<boolean>;
  saving: boolean;
  onClose: () => void;
}) {
  const [method, setMethod] = useState<Method>("upload");
  const [source, setSource] = useState("");
  const [revision, setRevision] = useState("");
  const [pending, setPending] = useState("");
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [saveError, setSaveError] = useState("");
  const [text, setText] = useState("The cat sat on the mat.");
  const [preview, setPreview] = useState<TokenPreview | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const files = useRef<HTMLInputElement>(null);
  const folder = useRef<HTMLInputElement>(null);
  const loadRequest = useRef<AbortController | null>(null);
  const previewRequest = useRef<AbortController | null>(null);
  useEffect(
    () => () => {
      loadRequest.current?.abort();
      previewRequest.current?.abort();
    },
    [],
  );
  useEffect(() => {
    previewRequest.current?.abort();
    setPreview(null);
    setPreviewBusy(false);
  }, [info?.id, text]);

  function apply(next: TokenizerInfo) {
    if (!onApply(next)) return;
    previewRequest.current?.abort();
    setPreview(null);
    setPreviewBusy(false);
    setError("");
    const vocabulary = info
      ? `${info.capacity.toLocaleString()} → ${next.capacity.toLocaleString()}`
      : next.capacity.toLocaleString();
    setStatus(
      `Applied to ${inputName} · capacity ${vocabulary}. Not saved yet.`,
    );
  }

  async function load(url: string, options: RequestInit, message: string) {
    if (loadRequest.current) return;
    const controller = new AbortController();
    loadRequest.current = controller;
    previewRequest.current?.abort();
    setPreviewBusy(false);
    setPending(message);
    setError("");
    setStatus("");
    try {
      const result = await fetch(url, {
        ...options,
        signal: controller.signal,
      }).then(responseData);
      if (!controller.signal.aborted) apply(tokenizerSchema.parse(result));
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(
          `${failure instanceof Error ? failure.message : String(failure)} Your current tokenizer has not changed.`,
        );
    } finally {
      if (!controller.signal.aborted) setPending("");
      if (loadRequest.current === controller) loadRequest.current = null;
    }
  }

  function upload(selected: FileList | null) {
    if (!selected?.length) return;
    const bundle = Array.from(selected).filter((file) =>
      /\.(json|model|txt|vocab|merges|tiktoken)$/i.test(file.name),
    );
    if (!bundle.length) {
      setError(
        "Choose tokenizer JSON, SentencePiece .model, vocabulary or merges files. XML files are not supported.",
      );
      return;
    }
    if (new Set(bundle.map((file) => file.name)).size !== bundle.length) {
      setError(
        "This folder contains multiple files with the same name. Choose the folder for one tokenizer, or select its files together.",
      );
      return;
    }
    if (
      bundle.reduce((total, file) => total + file.size, 0) >
      64 * 1024 * 1024
    ) {
      setError(
        "Tokenizer files must total 64 MB or less. Select only the tokenizer files, without model weights.",
      );
      return;
    }
    const form = new FormData();
    bundle.forEach((file) => form.append("files", file));
    void load(
      "/api/tokenizers/upload",
      { method: "POST", body: form },
      `Uploading and inspecting ${bundle.length} file${bundle.length === 1 ? "" : "s"}…`,
    );
  }

  async function tokenize() {
    if (!info || (previewRequest.current && previewBusy)) return;
    const controller = new AbortController();
    previewRequest.current = controller;
    setPreviewBusy(true);
    setError("");
    try {
      const result = await fetch("/api/tokenizers/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: info.id, text }),
        signal: controller.signal,
      }).then(responseData);
      if (!controller.signal.aborted) setPreview(previewSchema.parse(result));
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      if (!controller.signal.aborted) setPreviewBusy(false);
    }
  }

  return (
    <div className="tokenizer-panel">
      <div className="tokenizer-body">
        <p>
          Vocabulary dimensions follow this input automatically. Manual
          dimension overrides stay unchanged.
        </p>
        <section className="tokenizer-result" aria-label="Current tokenizer">
          <div className="result-heading">
            {info && <CheckCircle2 size={18} aria-hidden="true" />}
            <b>{info ? tokenizerLabel(info) : "No tokenizer selected"}</b>
            {info && <span>Current</span>}
          </div>
          {info && (
            <>
              <div className="token-stats">
                <div>
                  <span>Vocabulary</span>
                  <b>{info.size.toLocaleString()}</b>
                </div>
                <div>
                  <span>Embedding capacity</span>
                  <b>{info.capacity.toLocaleString()}</b>
                </div>
              </div>
              <details className="tokenizer-details">
                <summary>Details & tokenization preview</summary>
                <p>
                  Base vocabulary: {info.base_size.toLocaleString()} ·{" "}
                  {info.class}
                </p>
                <div className="special-tokens">
                  {Object.entries(info.special_tokens).map(([key, value]) => (
                    <span key={key}>
                      {key.replace("_token", "")}{" "}
                      <code>
                        {value.token} · {value.id}
                      </code>
                    </span>
                  ))}
                </div>
                {!Object.keys(info.special_tokens).length && (
                  <p>
                    No special-token roles detected. Include
                    tokenizer_config.json and special_tokens_map.json if your
                    tokenizer provides them.
                  </p>
                )}
                <label className="property-field">
                  <span>Tokenization preview</span>
                  <textarea
                    value={text}
                    onChange={(event) => setText(event.target.value)}
                  />
                </label>
                <button
                  className="small-button"
                  disabled={!!pending || saving || previewBusy}
                  onClick={() => void tokenize()}
                >
                  {previewBusy ? "Tokenizing…" : "Tokenize sample"}
                </button>
                {preview && (
                  <>
                    <div className="token-pills">
                      {preview.tokens.map((token, index) => (
                        <span key={index}>
                          {token}
                          <small>{preview.ids[index]}</small>
                        </span>
                      ))}
                    </div>
                    <p className="preview-shape">
                      Token IDs: [{preview.shape.join(", ")}] · int64
                    </p>
                  </>
                )}
              </details>
            </>
          )}
        </section>
        {status && (
          <p className="tokenizer-status" role="status">
            {status}
          </p>
        )}
        <section
          className="tokenizer-replace"
          aria-label={info ? "Replace tokenizer" : "Choose tokenizer"}
        >
          <h3>{info ? "Replace tokenizer" : "Choose tokenizer"}</h3>
          <div
            className="tokenizer-methods"
            role="group"
            aria-label="Tokenizer source"
          >
            {methods.map((item) => (
              <button
                key={item.id}
                aria-pressed={method === item.id}
                disabled={!!pending || saving}
                onClick={() => {
                  setMethod(item.id);
                  setError("");
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
          {method === "upload" && (
            <div className="tokenizer-upload">
              <p>
                Select all files for one tokenizer together, or choose its
                folder. Include <code>tokenizer.json</code>,{" "}
                <code>tokenizer_config.json</code>,{" "}
                <code>special_tokens_map.json</code> and any vocabulary / merges
                files provided. No model weights needed.
              </p>
              <div className="modal-actions">
                <button
                  className="primary"
                  disabled={!!pending || saving}
                  onClick={() => files.current?.click()}
                >
                  <Upload size={16} aria-hidden="true" />
                  Upload files
                </button>
                <button
                  disabled={!!pending || saving}
                  onClick={() => folder.current?.click()}
                >
                  <FolderOpen size={16} aria-hidden="true" />
                  Upload folder
                </button>
              </div>
              <small>
                JSON, SentencePiece .model, vocabulary and merges files · 64 MB
                total
              </small>
            </div>
          )}
          {method === "source" && (
            <form
              onSubmit={(event) => {
                event.preventDefault();
                if (source.trim())
                  void load(
                    "/api/tokenizers/inspect",
                    {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({
                        source: source.trim(),
                        revision: revision.trim() || null,
                      }),
                    },
                    "Inspecting…",
                  );
              }}
            >
              <label className="property-field">
                <span>Hugging Face repository or server-local path</span>
                <input
                  placeholder="e.g. gpt2 or /path/to/tokenizer"
                  value={source}
                  autoComplete="off"
                  spellCheck={false}
                  disabled={!!pending || saving}
                  onChange={(event) => setSource(event.target.value)}
                />
              </label>
              <p className="tokenizer-hint">
                A server path refers to the computer running Weave. Use Upload
                for files on this device.
              </p>
              <details className="tokenizer-details">
                <summary>Revision (optional)</summary>
                <label className="property-field">
                  <span>Commit or tag</span>
                  <input
                    placeholder="main"
                    value={revision}
                    disabled={!!pending || saving}
                    onChange={(event) => setRevision(event.target.value)}
                  />
                </label>
              </details>
              <div className="modal-actions">
                <button
                  type="submit"
                  className="primary"
                  disabled={!!pending || saving || !source.trim()}
                >
                  {pending ? "Inspecting…" : "Load & inspect"}
                </button>
              </div>
            </form>
          )}
          {method === "saved" && (
            <ProjectTokenizerPicker
              projectId={projectId}
              disabled={!!pending || saving}
              onSelect={apply}
            />
          )}
          {pending && (
            <p className="tokenizer-status" role="status">
              {pending} Your current tokenizer stays active until this finishes.
            </p>
          )}
          {error && (
            <div className="inline-error" role="alert">
              {error}
            </div>
          )}
        </section>
        <input
          ref={folder}
          hidden
          multiple
          type="file"
          aria-label="Tokenizer folder"
          {...{ webkitdirectory: "" }}
          onChange={(event) => {
            upload(event.target.files);
            event.target.value = "";
          }}
        />
        <input
          ref={files}
          hidden
          multiple
          type="file"
          aria-label="Tokenizer files"
          accept=".json,.model,.txt,.vocab,.merges,.tiktoken"
          onChange={(event) => {
            upload(event.target.files);
            event.target.value = "";
          }}
        />
      </div>
      {saveError && (
        <div className="inline-error" role="alert">
          {saveError}
        </div>
      )}
      <footer className="tokenizer-footer">
        <button disabled={saving} onClick={onClose}>
          Close
        </button>
        <button
          className="primary"
          disabled={!!pending || saving || !info}
          onClick={async () => {
            setSaveError("");
            if (!(await onSave()))
              setSaveError(
                "The project could not be saved. Your tokenizer change is still in this draft. Try saving again.",
              );
          }}
        >
          {saving ? "Saving…" : "Save & close"}
        </button>
      </footer>
    </div>
  );
}
