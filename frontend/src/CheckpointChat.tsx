import { useEffect, useMemo, useRef, useState } from "react";
import { z } from "zod";
import { trainingRequest, type TrainingGPU } from "./trainingApi";

const toolsSchema = z
  .array(
    z
      .object({
        type: z.literal("function"),
        function: z
          .object({
            name: z.string().regex(/^[A-Za-z0-9_-]{1,64}$/),
            parameters: z.record(z.string(), z.json()).optional(),
          })
          .passthrough(),
      })
      .passthrough(),
  )
  .max(128)
  .refine(
    (tools) =>
      new Set(tools.map((tool) => tool.function.name)).size === tools.length,
    "Tool names must be unique.",
  );

const sessionSchema = z.object({
  state: z.enum(["unloaded", "loading", "ready", "generating", "error"]),
  checkpoint: z.string().nullable(),
  device: z.string(),
  messages: z.array(
    z.object({ role: z.enum(["user", "assistant"]), content: z.string() }),
  ),
  error: z.string().nullable(),
  has_chat_template: z.boolean(),
  finish_reason: z.string().nullable(),
  loading_stage: z.string().nullable().default(null),
  loading_started: z.number().nullable().default(null),
  tools: toolsSchema.nullable().default(null),
  decoder: z.string().nullable().default(null),
  decoder_label: z.string().nullable().default(null),
});
const catalogSchema = z.object({
  checkpoints: z.array(
    z.object({
      path: z.string(),
      step: z.number(),
      validation_loss: z.number().nullable(),
      available: z.boolean(),
    }),
  ),
  recommended_checkpoint: z.string().nullable(),
  recommendation: z.string(),
  problem: z.string().nullable(),
  session: sessionSchema,
  decoders: z
    .array(
      z.object({
        id: z.string(),
        label: z.string(),
        available: z.boolean(),
      }),
    )
    .default([]),
});

export function CheckpointChat({
  runId,
  gpus,
  enabled,
}: {
  runId: string;
  gpus: TrainingGPU[];
  enabled: boolean;
}) {
  const [catalog, setCatalog] = useState<z.infer<typeof catalogSchema> | null>(
    null,
  );
  const [session, setSession] = useState<z.infer<typeof sessionSchema> | null>(
    null,
  );
  const [checkpoint, setCheckpoint] = useState("");
  const [device, setDevice] = useState("cpu");
  const [mode, setMode] = useState<"chat" | "completion">("chat");
  const [prompt, setPrompt] = useState("");
  const [system, setSystem] = useState("");
  const [chatTemplate, setChatTemplate] = useState("");
  const [toolsText, setToolsText] = useState("");
  const [decoder, setDecoder] = useState("");
  const [toolsNotice, setToolsNotice] = useState("");
  const [toolsFileError, setToolsFileError] = useState("");
  const [maxTokens, setMaxTokens] = useState(128);
  const [contextLength, setContextLength] = useState(2048);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1);
  const [seed, setSeed] = useState(42);
  const [error, setError] = useState("");
  const [catalogError, setCatalogError] = useState("");
  const [pollError, setPollError] = useState("");
  const loadTimedOut = useRef(false);
  const [busy, setBusy] = useState(false);
  const [catalogVersion, setCatalogVersion] = useState(0);
  const requestPending = useRef(false);
  const operation = useRef(0);
  const initialized = useRef(false);
  const transcript = useRef<HTMLDivElement>(null);
  const endpoint = `/runs/${encodeURIComponent(runId)}/chat`;

  // Catalog reads and lightweight model status have independent lifetimes.
  // Refreshing a slow catalog must not leave a completed load looking stuck.
  useEffect(() => {
    const controller = new AbortController();
    setCatalogError("");
    let disposed = false;
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    void trainingRequest(
      endpoint,
      catalogSchema,
      undefined,
      undefined,
      controller.signal,
    )
      .then((result) => {
        if (disposed) return;
        setCatalog(result);
        setCheckpoint(
          (current) =>
            current ||
            result.session.checkpoint ||
            result.recommended_checkpoint ||
            "",
        );
        if (!initialized.current && result.session.checkpoint) {
          setDevice(result.session.device);
          setToolsText(
            result.session.tools === null
              ? ""
              : JSON.stringify(result.session.tools, null, 2),
          );
          setDecoder(result.session.decoder ?? "");
        }
        initialized.current = true;
      })
      .catch((failure) => {
        if (!disposed)
          setCatalogError(
            controller.signal.aborted
              ? "Checkpoint list took too long to load. Use Refresh checkpoints to retry."
              : failure instanceof Error
                ? failure.message
                : String(failure),
          );
      })
      .finally(() => window.clearTimeout(timeout));
    return () => {
      disposed = true;
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [endpoint, catalogVersion]);

  useEffect(() => {
    let disposed = false;
    let pending = false;
    let current: AbortController | null = null;
    const refresh = async () => {
      if (pending || requestPending.current) return;
      pending = true;
      const revision = operation.current;
      const controller = new AbortController();
      current = controller;
      const timeout = window.setTimeout(() => controller.abort(), 15000);
      try {
        const result = await trainingRequest(
          endpoint + "/session",
          sessionSchema,
          undefined,
          undefined,
          controller.signal,
        );
        if (!disposed && revision === operation.current) {
          setSession(result);
          setPollError("");
          if (loadTimedOut.current && result.state === "ready") {
            loadTimedOut.current = false;
            setError("");
          }
        }
      } catch (failure) {
        if (!disposed)
          setPollError(
            controller.signal.aborted
              ? "Model status request timed out. Retrying automatically."
              : `Cannot refresh model status: ${failure instanceof Error ? failure.message : String(failure)}. Retrying automatically.`,
          );
      } finally {
        window.clearTimeout(timeout);
        current = null;
        pending = false;
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 1000);
    return () => {
      disposed = true;
      current?.abort();
      window.clearInterval(timer);
    };
  }, [endpoint]);

  useEffect(() => {
    const element = transcript.current;
    if (
      element &&
      element.scrollHeight - element.scrollTop - element.clientHeight < 180
    )
      element.scrollTop = element.scrollHeight;
  }, [session?.messages]);

  async function action(kind: string, body: unknown = {}, method = "POST") {
    operation.current += 1;
    requestPending.current = true;
    setBusy(true);
    setError("");
    loadTimedOut.current = false;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    try {
      const result = await trainingRequest(
        endpoint + kind,
        sessionSchema,
        body,
        method,
        controller.signal,
      );
      setSession(result);
      if (kind === "/messages") setPrompt("");
      if (kind === "/settings") setToolsNotice("Tool settings applied.");
    } catch (failure) {
      loadTimedOut.current = kind === "/load" && controller.signal.aborted;
      setError(
        controller.signal.aborted
          ? "The request took too long. Model status checks will continue; you can unload and retry."
          : failure instanceof Error
            ? failure.message
            : String(failure),
      );
    } finally {
      window.clearTimeout(timeout);
      requestPending.current = false;
      setBusy(false);
    }
  }
  const generating = session?.state === "generating";
  const loaded = session?.state === "ready" || generating;
  const selectedLoaded =
    loaded && checkpoint === session.checkpoint && device === session.device;
  const parsedTools = useMemo(() => {
    if (!toolsText.trim()) return { tools: null, error: "" };
    try {
      if (new TextEncoder().encode(toolsText).length > 262144)
        return { tools: null, error: "Tool catalog must be at most 256 KiB." };
      const parsed = toolsSchema.safeParse(JSON.parse(toolsText));
      return parsed.success
        ? { tools: parsed.data, error: "" }
        : {
            tools: null,
            error:
              "Use a JSON list of function tools with unique names and JSON Schema parameters.",
          };
    } catch {
      return {
        tools: null,
        error:
          "Invalid JSON. Paste a JSON list of OpenAI-style function tools.",
      };
    }
  }, [toolsText]);
  const toolOptions = { tools: parsedTools.tools, decoder: decoder || null };
  const toolsDirty =
    loaded &&
    (JSON.stringify(parsedTools.tools) !== JSON.stringify(session.tools) ||
      (decoder || null) !== session.decoder);
  const toolsInvalid = !!parsedTools.error || !!toolsFileError;
  const missingTemplate =
    mode === "chat" &&
    loaded &&
    !session.has_chat_template &&
    !chatTemplate.trim();
  const canSend =
    selectedLoaded &&
    !generating &&
    !busy &&
    !toolsDirty &&
    !toolsInvalid &&
    !missingTemplate &&
    !!prompt.trim();
  const send = () =>
    action("/messages", {
      text: prompt,
      mode,
      system,
      chat_template: chatTemplate,
      max_new_tokens: maxTokens,
      context_length: contextLength,
      temperature,
      top_p: topP,
      repetition_penalty: repetitionPenalty,
      seed,
      ...toolOptions,
    });
  return (
    <section className="checkpoint-chat" aria-label="Checkpoint chat">
      <h3>Try a saved checkpoint</h3>
      <p className="training-muted">
        Loads the frozen network and tokenizer without running train.py. A
        pretrained model can complete text; useful conversation needs chat
        training. Tool calls are displayed as text.
      </p>
      {!catalog && <p role="status">Finding saved checkpoints…</p>}
      {catalog?.problem && (
        <p role="status" className="training-alert">
          {catalog.problem}
        </p>
      )}
      <div className="training-form-grid">
        <label>
          Checkpoint
          <select
            value={checkpoint}
            onChange={(event) => setCheckpoint(event.target.value)}
            disabled={busy || generating || session?.state === "loading"}
          >
            <option value="">Choose a saved checkpoint…</option>
            {catalog?.checkpoints.map((item) => (
              <option
                key={item.path}
                value={item.path}
                disabled={!item.available}
              >
                Update {item.step}
                {item.path === catalog.recommended_checkpoint
                  ? " · recommended"
                  : ""}
                {item.validation_loss !== null
                  ? ` · validation loss ${item.validation_loss.toFixed(4)}`
                  : " · no validation score"}
                {!item.available ? " · weights unavailable" : ""}
              </option>
            ))}
          </select>
          <small>
            {catalog?.recommendation} The choice stays fixed until you load
            another checkpoint.
          </small>
        </label>
        <label>
          Inference device
          <select
            value={device}
            onChange={(event) => setDevice(event.target.value)}
            disabled={busy || generating || session?.state === "loading"}
          >
            <option value="cpu">CPU</option>
            {gpus.map((gpu) => (
              <option key={gpu.uuid} value={`cuda:${gpu.index}`}>
                GPU {gpu.index} · {gpu.name}
              </option>
            ))}
          </select>
          <small>
            CPU is the default. A GPU needs memory for another model copy and
            may compete with training.
          </small>
        </label>
      </div>
      <div className="checkpoint-chat-actions">
        <button
          type="button"
          onClick={() => setCatalogVersion((current) => current + 1)}
          disabled={busy}
        >
          Refresh checkpoints
        </button>
        <button
          type="button"
          className="primary"
          onClick={() =>
            void action("/load", { checkpoint, device, ...toolOptions })
          }
          disabled={
            busy ||
            toolsInvalid ||
            !enabled ||
            !checkpoint ||
            !!catalog?.problem ||
            generating ||
            session?.state === "loading" ||
            selectedLoaded
          }
        >
          {session?.state === "loading"
            ? "Loading checkpoint…"
            : selectedLoaded
              ? "Checkpoint loaded"
              : "Load checkpoint"}
        </button>
        <button
          type="button"
          onClick={() => void action("", undefined, "DELETE")}
          disabled={busy || session?.state === "unloaded" || !session}
        >
          Unload model
        </button>
        <button
          type="button"
          onClick={() => void action("/clear")}
          disabled={
            busy || session?.state !== "ready" || !session.messages.length
          }
        >
          New conversation
        </button>
      </div>
      {!enabled && (
        <p className="training-muted">
          Enable local execution on the server to load a checkpoint.
        </p>
      )}
      <p
        role="status"
        aria-label="Chat model status"
        className="training-muted"
      >
        {session?.state ?? "unloaded"}
        {session?.checkpoint
          ? ` · ${session.device === "cpu" ? "CPU" : session.device.replace("cuda:", "GPU ")}`
          : ""}
        {loaded && (
          <span
            role="group"
            className="checkpoint-chat-tool-status"
            aria-label="Active tool settings"
          >
            <span>Decoder: {session.decoder_label ?? "None"}</span>
            <span>{session.tools?.length ?? 0} tools loaded</span>
          </span>
        )}
      </p>
      {loaded && !selectedLoaded && (
        <p role="status">
          Load the selected checkpoint and device before sending a message.
        </p>
      )}
      {session?.state === "loading" && (
        <p role="status" aria-label="Checkpoint loading progress">
          {session.loading_stage ?? "Starting inference worker"}
          {session.loading_started !== null &&
            ` · ${Math.max(0, Math.floor(Date.now() / 1000 - session.loading_started))}s elapsed`}
        </p>
      )}
      {(error || catalogError || pollError || session?.error) && (
        <p
          role="alert"
          className="training-alert"
          aria-label="Checkpoint error"
        >
          {error || catalogError || pollError || session?.error}
        </p>
      )}
      <div className="training-form-grid">
        <label>
          Prompt format
          <select
            value={mode}
            disabled={generating}
            onChange={(event) => {
              setMode(z.enum(["chat", "completion"]).parse(event.target.value));
            }}
          >
            <option value="chat">Conversation · saved chat template</option>
            <option value="completion">Text completion · raw prompt</option>
          </select>
          <small>
            {mode === "chat"
              ? "Formats the whole conversation with the tokenizer's chat template."
              : "Continues only the current prompt. Previous turns and the system message are not included."}
          </small>
        </label>
        <label>
          System message{" "}
          <textarea
            rows={2}
            value={system}
            onChange={(event) => setSystem(event.target.value)}
            disabled={generating || mode !== "chat"}
            placeholder="Optional; only if the training template supports it"
          />
        </label>
      </div>
      <details className="checkpoint-chat-tool-settings">
        <summary>Tools &amp; constrained decoding</summary>
        <div className="training-form-grid">
          <label>
            Decoder
            <select
              value={decoder}
              disabled={busy || generating || session?.state === "loading"}
              onChange={(event) => {
                setDecoder(event.target.value);
                setToolsNotice("");
              }}
            >
              <option value="">None · current behavior</option>
              {catalog?.decoders.map((item) => (
                <option
                  key={item.id}
                  value={item.id}
                  disabled={!item.available}
                >
                  {item.label}
                  {item.available ? "" : " · not installed"}
                </option>
              ))}
            </select>
            <small>
              Constrains the next token using a decoder installed by the server
              owner. XML tool grammar must be registered in .weave-config.json
              before selection. Tool calls are displayed and never executed.
            </small>
          </label>
          <label>
            Upload tool catalog
            <input
              type="file"
              accept=".json,application/json"
              disabled={busy || generating || session?.state === "loading"}
              onChange={async (event) => {
                const file = event.currentTarget.files?.[0];
                event.currentTarget.value = "";
                if (!file) return;
                setToolsNotice("");
                if (file.size > 262144) {
                  setToolsFileError("Tool catalog must be at most 256 KiB.");
                  return;
                }
                try {
                  setToolsText(await file.text());
                  setToolsFileError("");
                } catch {
                  setToolsFileError(
                    "Could not read the tool catalog. Try the file again or paste its JSON below.",
                  );
                }
              }}
            />
            <small>
              OpenAI-style function tools as a JSON list. Uploading fills the
              editor; apply to activate.
            </small>
          </label>
          <label className="checkpoint-chat-tools-editor">
            Tool catalog JSON
            <textarea
              rows={6}
              value={toolsText}
              spellCheck={false}
              disabled={busy || generating || session?.state === "loading"}
              aria-invalid={toolsInvalid}
              onChange={(event) => {
                setToolsText(event.target.value);
                setToolsNotice("");
                setToolsFileError("");
              }}
              placeholder={
                '[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
              }
            />
            <small>
              {parsedTools.tools?.length ?? 0} tools in draft. Passed to the
              saved chat template in conversation mode. Leave empty to remove
              the catalog.
            </small>
          </label>
        </div>
        {toolsInvalid && (
          <p role="alert" className="training-alert">
            {toolsFileError || parsedTools.error}
          </p>
        )}
        <div className="checkpoint-chat-actions">
          <button
            type="button"
            disabled={
              busy || session?.state !== "ready" || toolsInvalid || !toolsDirty
            }
            onClick={() => void action("/settings", toolOptions, "PUT")}
          >
            Apply tool settings
          </button>
          <small role="status">
            {toolsDirty
              ? "Unapplied changes. Apply before sending a message."
              : toolsNotice ||
                (loaded
                  ? "Tool settings match the loaded session."
                  : "These settings will apply when you load a checkpoint.")}
          </small>
        </div>
      </details>
      <details>
        <summary>Generation settings</summary>
        <div className="training-form-grid">
          <label>
            Maximum new tokens
            <input
              type="number"
              min={1}
              max={2048}
              value={maxTokens}
              onChange={(event) => setMaxTokens(event.target.valueAsNumber)}
            />
          </label>
          <label>
            Context token budget
            <input
              type="number"
              min={2}
              max={32768}
              value={contextLength}
              onChange={(event) => setContextLength(event.target.valueAsNumber)}
            />
            <small>
              Must fit the prompt and reply and stay within the trained context
              length. Long prompts produce an error instead of silently dropping
              history.
            </small>
          </label>
          <label>
            Temperature
            <input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              onChange={(event) => setTemperature(event.target.valueAsNumber)}
            />
            <small>
              0 selects the most likely token; higher values add randomness.
            </small>
          </label>
          <label>
            Top-p
            <input
              type="number"
              min={0.01}
              max={1}
              step={0.05}
              value={topP}
              disabled={temperature === 0}
              onChange={(event) => setTopP(event.target.valueAsNumber)}
            />
            <small>
              Sample from tokens covering this share of the probability mass.
              Ignored for greedy decoding at temperature 0.
            </small>
          </label>
          <label>
            Repetition penalty
            <input
              aria-label="Repetition penalty"
              type="number"
              min={1}
              max={2}
              step={0.05}
              value={repetitionPenalty}
              onChange={(event) =>
                setRepetitionPenalty(event.target.valueAsNumber)
              }
            />
            <small>
              1 disables the penalty. Values above 1 discourage tokens already
              in the prompt or generated text. This changes decoding, not the
              model’s learned weights.
            </small>
          </label>
          <label>
            Sampling seed
            <input
              type="number"
              min={0}
              max={4294967295}
              value={seed}
              disabled={temperature === 0}
              onChange={(event) => setSeed(event.target.valueAsNumber)}
            />
          </label>
          <label>
            Chat template override
            <textarea
              rows={4}
              value={chatTemplate}
              onChange={(event) => setChatTemplate(event.target.value)}
              disabled={generating || mode !== "chat"}
              placeholder="Optional Jinja template matching your training format"
            />
            <small>
              Use this if train.py supplied its own template. It affects this
              conversation only and changes no files in your project.
            </small>
          </label>
        </div>
      </details>
      {missingTemplate && (
        <p role="status">
          The saved tokenizer has no chat template. Choose text completion or
          supply the template used during training under Generation settings.
        </p>
      )}
      <div
        ref={transcript}
        className="checkpoint-chat-transcript"
        role="log"
        aria-label="Chat conversation"
        aria-live="polite"
        aria-relevant="additions text"
      >
        {session?.messages.map((message, index) => (
          <article
            key={index}
            className={`checkpoint-chat-message is-${message.role}`}
          >
            <strong>{message.role === "user" ? "You" : "Model"}</strong>
            <p>
              {message.content ||
                (generating ? "Generating…" : "No text generated.")}
            </p>
          </article>
        ))}
        {!session?.messages.length && (
          <p className="training-muted">
            Load a checkpoint, then send a message or a text prompt.
          </p>
        )}
      </div>
      {session?.finish_reason && (
        <p className="training-muted">
          Response ended:{" "}
          {session.finish_reason === "length"
            ? session.decoder
              ? "new-token limit reached; constrained output may be incomplete"
              : "new-token limit reached"
            : session.finish_reason === "eos"
              ? "end-of-sequence token"
              : "stopped"}
          .
        </p>
      )}
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (canSend) void send();
        }}
      >
        <label>
          Message or prompt
          <textarea
            aria-label="Message or prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            rows={3}
            maxLength={32000}
            disabled={!selectedLoaded || generating}
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                (event.ctrlKey || event.metaKey) &&
                canSend
              ) {
                event.preventDefault();
                void send();
              }
            }}
            placeholder={
              mode === "chat"
                ? "Write a message…"
                : "Write text for the model to continue…"
            }
          />
          <small>
            Generates at most {maxTokens} new tokens, stopping earlier at an
            end-of-sequence token. Change the limit in Generation settings.
          </small>
        </label>
        <div className="checkpoint-chat-actions">
          <button type="submit" className="primary" disabled={!canSend}>
            Send
          </button>
          {generating && (
            <button
              type="button"
              onClick={() => void action("/stop")}
              disabled={busy}
            >
              Stop generation
            </button>
          )}
          <small>
            Ctrl/Cmd+Enter to send. Conversations are recorded with the run,
            separately from training.
          </small>
        </div>
      </form>
    </section>
  );
}
