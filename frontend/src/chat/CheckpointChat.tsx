import { useEffect, useMemo, useRef, useState } from "react";
import { z } from "zod";
import { trainingRequest, type TrainingGPU } from "../trainingApi";
import {
  catalogSchema,
  sessionSchema,
  toolsSchema,
  diagnosticsSchema,
  liveSystem,
  downloadJsonl,
} from "./chatSchema";
import { ChatTranscript } from "./ChatTranscript";
import { ChatTestRunner } from "./ChatTestRunner";
import { copyText } from "../ui";

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
  const deviceChosen = useRef(false);
  const [settingsOpen, setSettingsOpen] = useState(
    () => window.matchMedia("(min-width: 1000px)").matches,
  );
  const [diagnostics, setDiagnostics] = useState<
    z.infer<typeof diagnosticsSchema>
  >({ errors: [], warnings: [] });
  const [diagnosticsPending, setDiagnosticsPending] = useState(false);
  const [previewKey, setPreviewKey] = useState("");
  const [mock, setMock] = useState<{ id: string; name: string } | null>(null);
  const [mockContent, setMockContent] = useState('{"ok":true}');
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
        if (!initialized.current) {
          const saved = result.session.checkpoint
            ? result.session
            : (result.preferences ?? result.session);
          if (result.session.checkpoint) {
            setDevice(result.session.device);
            deviceChosen.current = true;
          }
          setToolsText(
            saved.tools === null ? "" : JSON.stringify(saved.tools, null, 2),
          );
          setDecoder(saved.decoder ?? "");
          setSystem(saved.system);
          setChatTemplate(saved.chat_template);
          setMode(saved.mode);
          if (saved.decoder) setTemperature(0);
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
    if (deviceChosen.current || session?.checkpoint || !catalog?.supports_cuda)
      return;
    const idle = gpus
      .filter(
        (gpu) =>
          (gpu.memory_total_mb ?? 0) - (gpu.memory_used_mb ?? 0) > 1024 &&
          (gpu.utilization ?? 0) < 20,
      )
      .sort(
        (a, b) =>
          (b.memory_total_mb ?? 0) -
          (b.memory_used_mb ?? 0) -
          ((a.memory_total_mb ?? 0) - (a.memory_used_mb ?? 0)),
      );
    if (idle[0]) {
      setDevice(`cuda:${idle[0].index}`);
      deviceChosen.current = true;
    }
  }, [gpus, session?.checkpoint, catalog?.supports_cuda]);

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
      if (kind === "/settings")
        setToolsNotice("Settings saved for this project.");
      return result;
    } catch (failure) {
      loadTimedOut.current = kind === "/load" && controller.signal.aborted;
      setError(
        controller.signal.aborted
          ? "The request took too long. Model status checks will continue; you can unload and retry."
          : failure instanceof Error
            ? failure.message
            : String(failure),
      );
      return null;
    } finally {
      window.clearTimeout(timeout);
      requestPending.current = false;
      setBusy(false);
    }
  }
  const generating =
    session?.state === "generating" || session?.state === "testing";
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
  const toolOptions = useMemo(
    () => ({
      tools: parsedTools.tools,
      decoder: decoder || null,
      system,
      chat_template: chatTemplate,
      mode,
    }),
    [parsedTools.tools, decoder, system, chatTemplate, mode],
  );
  const optionsKey = JSON.stringify(toolOptions);
  const toolsDirty =
    loaded &&
    (["tools", "decoder", "system", "chat_template", "mode"] as const).some(
      (key) =>
        JSON.stringify(toolOptions[key]) !== JSON.stringify(session[key]),
    );
  const pinned =
    !!session?.conversation_settings &&
    session.messages.some((message) => message.included_in_context);
  const toolsInvalid =
    !!parsedTools.error || !!toolsFileError || diagnostics.errors.length > 0;

  useEffect(() => {
    if (parsedTools.error || !catalog) return;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => {
      setDiagnosticsPending(true);
      void trainingRequest(
        endpoint + "/validate-options",
        diagnosticsSchema,
        toolOptions,
        "POST",
        controller.signal,
      )
        .then((result) => {
          if (!controller.signal.aborted) setDiagnostics(result);
        })
        .catch((failure) => {
          if (!controller.signal.aborted)
            setDiagnostics({
              errors: [
                failure instanceof Error ? failure.message : String(failure),
              ],
              warnings: [],
            });
        })
        .finally(() => {
          if (!controller.signal.aborted) setDiagnosticsPending(false);
        });
    }, 300);
    return () => {
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [endpoint, toolOptions, parsedTools.error, catalog]);

  const historyKey = JSON.stringify(
    session?.messages.map((message) => [
      message.role,
      message.content,
      message.included_in_context,
    ]) ?? [],
  );
  const contextKey = JSON.stringify([
    optionsKey,
    prompt,
    maxTokens,
    contextLength,
    historyKey,
  ]);
  useEffect(() => {
    if (session?.state !== "ready" || busy || toolsInvalid || !catalog) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      void trainingRequest(
        endpoint + "/preview",
        sessionSchema,
        {
          ...toolOptions,
          text: prompt,
          max_new_tokens: maxTokens,
          context_length: contextLength,
        },
        "POST",
        controller.signal,
      )
        .then((result) => {
          if (!controller.signal.aborted) {
            setSession(result);
            setPreviewKey(contextKey);
          }
        })
        .catch((failure) => {
          if (!controller.signal.aborted)
            setError(
              `Context preview: ${failure instanceof Error ? failure.message : String(failure)}`,
            );
        });
    }, 500);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [endpoint, contextKey, session?.state, busy, toolsInvalid, catalog]);
  const budget =
    previewKey === contextKey && session?.context_preview?.state === "ready"
      ? session.context_preview
      : null;
  const missingTemplate =
    mode === "chat" &&
    loaded &&
    !session.has_chat_template &&
    !chatTemplate.trim();
  const canSend =
    selectedLoaded &&
    !generating &&
    !busy &&
    !toolsInvalid &&
    !diagnosticsPending &&
    !missingTemplate &&
    !!prompt.trim() &&
    !(toolsDirty && pinned) &&
    budget !== null &&
    (budget.remaining_tokens ?? -1) >= 0;
  async function applySettings(clear = false) {
    if (clear && !(await action("/clear"))) return null;
    return action("/settings", toolOptions, "PUT");
  }
  async function send() {
    if (!canSend) return;
    if (toolsDirty && !(await applySettings())) return;
    await action("/messages", {
      text: prompt,
      ...toolOptions,
      max_new_tokens: maxTokens,
      context_length: contextLength,
      temperature,
      top_p: topP,
      repetition_penalty: repetitionPenalty,
      seed,
    });
  }
  const controlsDisabled = busy || generating || session?.state === "loading";
  const activeTools = session?.tools?.length ?? 0;
  const draftCount = parsedTools.tools?.length ?? 0;
  const selectedPreset =
    catalog?.presets.find(
      (preset) =>
        JSON.stringify(preset.tools) === JSON.stringify(parsedTools.tools),
    )?.id ?? (draftCount ? "custom" : "");
  const expiry =
    session?.idle_expires == null
      ? null
      : Math.max(0, Math.ceil((session.idle_expires - Date.now() / 1000) / 60));
  function deterministic() {
    setTemperature(0);
    setTopP(1);
    setSeed(42);
    setRepetitionPenalty(1);
  }
  function usePreset(id: string) {
    if (!id) {
      setToolsText("");
      setToolsNotice("");
      return;
    }
    const preset = catalog?.presets.find((item) => item.id === id);
    if (!preset) return;
    setToolsText(JSON.stringify(preset.tools, null, 2));
    setSystem(liveSystem(preset.system_template));
    if (
      catalog?.decoders.some(
        (item) => item.id === "xml-tool-grammar" && item.available,
      )
    )
      setDecoder("xml-tool-grammar");
    deterministic();
    setToolsNotice("");
    setToolsFileError("");
  }
  return (
    <section className="checkpoint-chat" aria-label="Checkpoint chat">
      <div className="chat-model-toolbar">
        <label>
          Checkpoint
          <select
            value={checkpoint}
            onChange={(event) => setCheckpoint(event.target.value)}
            disabled={controlsDisabled}
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
                {Object.entries(item.task_metrics).length
                  ? Object.entries(item.task_metrics)
                      .map(
                        ([key, value]) =>
                          ` · ${key}: ${(value * 100).toFixed(1)}%`,
                      )
                      .join("")
                  : item.validation_loss !== null
                    ? ` · loss ${item.validation_loss.toFixed(4)}`
                    : " · no task score"}
              </option>
            ))}
          </select>
        </label>
        <label>
          Inference device
          <select
            value={device}
            disabled={controlsDisabled}
            onChange={(event) => {
              setDevice(event.target.value);
              deviceChosen.current = true;
            }}
          >
            <option value="cpu">CPU · slow smoke tests</option>
            {gpus.map((gpu) => (
              <option key={gpu.uuid} value={`cuda:${gpu.index}`}>
                GPU {gpu.index} · {gpu.name}
              </option>
            ))}
          </select>
        </label>
        <div className="checkpoint-chat-actions">
          <button
            type="button"
            className="primary"
            disabled={
              controlsDisabled ||
              selectedLoaded ||
              !enabled ||
              !checkpoint ||
              !!catalog?.problem ||
              toolsInvalid
            }
            onClick={() =>
              void action("/load", { checkpoint, device, ...toolOptions })
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
            disabled={busy || !session || session.state === "unloaded"}
            onClick={() => void action("", undefined, "DELETE")}
          >
            Unload model
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => setCatalogVersion((version) => version + 1)}
          >
            Refresh checkpoints
          </button>
        </div>
      </div>
      {!catalog && <p role="status">Finding saved checkpoints…</p>}
      <details className="chat-model-notes">
        <summary>Checkpoint recommendation &amp; device notes</summary>
        <p>{catalog?.recommendation}</p>
        <p>
          Loading another checkpoint starts a new conversation. Catalog and
          system settings are saved per project. Other runs keep their own
          loaded sessions.
        </p>
        {device === "cpu" && (
          <p>
            CPU generation can be slow. Choose an available GPU for interactive
            testing; a GPU needs memory for another model copy.
          </p>
        )}
      </details>
      <div className="chat-status-strip">
        <span role="status" aria-label="Chat model status">
          {session?.state ?? "unloaded"}
          {session?.checkpoint
            ? ` · ${session.device === "cpu" ? "CPU" : session.device.replace("cuda:", "GPU ")}`
            : ""}
        </span>
        <span
          role="group"
          aria-label="Active tool settings"
          className="checkpoint-chat-tool-status"
        >
          <span>Decoder: {session?.decoder_label ?? "None"}</span>
          <span>{activeTools} tools loaded</span>
        </span>
        {loaded && expiry !== null && (
          <small title="Configured by chat_idle_timeout_seconds in the server settings">
            Idle unload in ~{expiry} min; messages/settings kept
          </small>
        )}
      </div>
      {session?.state === "loading" && (
        <p role="status" aria-label="Checkpoint loading progress">
          {session.loading_stage ?? "Starting inference worker"}
          {session.loading_started !== null &&
            ` · ${Math.max(0, Math.floor(Date.now() / 1000 - session.loading_started))}s elapsed`}
        </p>
      )}
      {session?.notice && <p role="status">{session.notice}</p>}
      {(error ||
        catalogError ||
        pollError ||
        session?.error ||
        catalog?.problem) && (
        <p
          role="alert"
          className="training-alert"
          aria-label="Checkpoint error"
        >
          {error ||
            catalogError ||
            pollError ||
            session?.error ||
            catalog?.problem}
        </p>
      )}
      {!enabled && (
        <p className="training-alert">
          Enable local execution on the server to load a checkpoint.
        </p>
      )}
      <div className="chat-workbench">
        <details
          className="chat-settings-panel"
          open={settingsOpen}
          onToggle={(event) => setSettingsOpen(event.currentTarget.open)}
        >
          <summary>
            Test settings
            {toolsInvalid
              ? " · needs attention"
              : toolsDirty
                ? " · unapplied"
                : ""}
          </summary>
          <div className="chat-settings-body">
            <label>
              Prompt format
              <select
                value={mode}
                disabled={controlsDisabled}
                onChange={(event) =>
                  setMode(
                    z.enum(["chat", "completion"]).parse(event.target.value),
                  )
                }
              >
                <option value="chat">Conversation · saved chat template</option>
                <option value="completion">Text completion · raw prompt</option>
              </select>
            </label>
            <details className="checkpoint-chat-tool-settings" open>
              <summary>Tools &amp; constrained decoding</summary>
              <label>
                Catalog preset
                <select
                  aria-label="Catalog preset"
                  value={selectedPreset}
                  disabled={controlsDisabled}
                  onChange={(event) => usePreset(event.target.value)}
                >
                  <option value="">No catalog</option>
                  <option value="custom" disabled>
                    Custom / saved catalog
                  </option>
                  {catalog?.presets.map((preset) => (
                    <option key={preset.id} value={preset.id}>
                      {preset.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Decoder
                <select
                  value={decoder}
                  disabled={controlsDisabled}
                  onChange={(event) => {
                    setDecoder(event.target.value);
                    if (event.target.value) deterministic();
                    setToolsNotice("");
                  }}
                >
                  <option value="">None · unconstrained</option>
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
              </label>
              <label>
                Upload tool catalog
                <input
                  type="file"
                  accept=".json,application/json"
                  disabled={controlsDisabled}
                  onChange={async (event) => {
                    const file = event.currentTarget.files?.[0];
                    event.currentTarget.value = "";
                    if (!file) return;
                    if (file.size > 262144) {
                      setToolsFileError("Tool catalog must fit in 256 KiB.");
                      return;
                    }
                    try {
                      setToolsText(await file.text());
                      setToolsFileError("");
                      setToolsNotice("");
                    } catch {
                      setToolsFileError(
                        "Could not read the catalog. Paste its JSON instead.",
                      );
                    }
                  }}
                />
              </label>
              <label className="checkpoint-chat-tools-editor">
                Tool catalog JSON
                <textarea
                  rows={5}
                  value={toolsText}
                  spellCheck={false}
                  disabled={controlsDisabled}
                  aria-invalid={toolsInvalid}
                  onChange={(event) => {
                    setToolsText(event.target.value);
                    setToolsNotice("");
                    setToolsFileError("");
                  }}
                  placeholder="No catalog selected. Choose a preset, upload a JSON list, or paste your tools."
                />
                <small>
                  {draftCount} tools in draft. Tools enter the model prompt in
                  conversation mode.
                </small>
              </label>
              {!draftCount && (
                <p className="chat-warning">
                  No tools loaded —{" "}
                  {decoder
                    ? "XML tool decoding can only return <no_call/>."
                    : "select a catalog to test declared tool calls."}
                </p>
              )}
              {(parsedTools.error || toolsFileError) && (
                <p role="alert" className="training-alert">
                  {toolsFileError || parsedTools.error}
                </p>
              )}
              {diagnostics.errors.map((message, i) => (
                <p role="alert" className="training-alert" key={i}>
                  {message}
                </p>
              ))}
              {diagnostics.warnings.map((message, i) => (
                <p className="chat-warning" key={i}>
                  {message}
                </p>
              ))}
            </details>
            <label>
              System message
              <textarea
                rows={4}
                value={system}
                disabled={controlsDisabled || mode !== "chat"}
                onChange={(event) => setSystem(event.target.value)}
                placeholder="No explicit system prompt. The saved template supplies any fallback."
              />
            </label>
            {!system.trim() && mode === "chat" && (
              <small className="chat-warning">
                System prompt is empty. Date-relative requests may be ambiguous.
              </small>
            )}
            <div className="checkpoint-chat-actions">
              <button
                type="button"
                disabled={controlsDisabled || mode !== "chat"}
                onClick={() => {
                  const preset = catalog?.presets.find(
                    (item) => item.id === "mobile-actions",
                  );
                  if (preset) setSystem(liveSystem(preset.system_template));
                }}
              >
                Mobile Actions system prompt
              </button>
              <button
                type="button"
                disabled={!system}
                onClick={() => {
                  void copyText(system).then(
                    () => setToolsNotice("System prompt copied."),
                    () => setError("Could not copy system prompt."),
                  );
                }}
              >
                Copy system prompt
              </button>
            </div>
            <details>
              <summary>Generation settings</summary>
              <div className="chat-generation-fields">
                <label>
                  Maximum new tokens
                  <input
                    type="number"
                    min={1}
                    max={2048}
                    value={maxTokens}
                    disabled={controlsDisabled}
                    onChange={(event) =>
                      setMaxTokens(event.target.valueAsNumber)
                    }
                  />
                </label>
                <label>
                  Context token budget
                  <input
                    type="number"
                    min={2}
                    max={32768}
                    value={contextLength}
                    disabled={controlsDisabled}
                    onChange={(event) =>
                      setContextLength(event.target.valueAsNumber)
                    }
                  />
                </label>
                <label>
                  Temperature
                  <input
                    type="number"
                    min={0}
                    max={2}
                    step={0.1}
                    value={temperature}
                    disabled={controlsDisabled}
                    onChange={(event) =>
                      setTemperature(event.target.valueAsNumber)
                    }
                  />
                </label>
                <label>
                  Top-p
                  <input
                    type="number"
                    min={0.01}
                    max={1}
                    step={0.05}
                    value={topP}
                    disabled={controlsDisabled || temperature === 0}
                    onChange={(event) => setTopP(event.target.valueAsNumber)}
                  />
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
                    disabled={controlsDisabled}
                    onChange={(event) =>
                      setRepetitionPenalty(event.target.valueAsNumber)
                    }
                  />
                  <small>
                    1 disables the penalty. Long free-text arguments can repeat;
                    increasing it may change tool/value choices.
                  </small>
                </label>
                <label>
                  Sampling seed
                  <input
                    type="number"
                    min={0}
                    max={4294967295}
                    value={seed}
                    disabled={controlsDisabled || temperature === 0}
                    onChange={(event) => setSeed(event.target.valueAsNumber)}
                  />
                </label>
                <label className="chat-field-wide">
                  Chat template override
                  <textarea
                    rows={3}
                    value={chatTemplate}
                    disabled={controlsDisabled || mode !== "chat"}
                    onChange={(event) => setChatTemplate(event.target.value)}
                    placeholder="Optional template matching training"
                  />
                </label>
              </div>
              <div className="checkpoint-chat-actions">
                <button
                  type="button"
                  disabled={controlsDisabled}
                  onClick={deterministic}
                >
                  Deterministic tool test
                </button>
                <button
                  type="button"
                  disabled={controlsDisabled}
                  onClick={() => {
                    setMaxTokens(128);
                    setContextLength(2048);
                    setTemperature(decoder ? 0 : 0.7);
                    setTopP(0.9);
                    setRepetitionPenalty(1);
                    setSeed(42);
                  }}
                >
                  Reset generation settings
                </button>
              </div>
            </details>
            <div className="checkpoint-chat-actions">
              <button
                type="button"
                disabled={
                  session?.state !== "ready" ||
                  busy ||
                  toolsInvalid ||
                  !toolsDirty ||
                  pinned
                }
                onClick={() => void applySettings()}
              >
                Apply settings
              </button>
              <small role="status">
                {toolsNotice ||
                  (toolsDirty
                    ? pinned
                      ? "Clear messages to change pinned settings."
                      : "Pending changes will apply when you send."
                    : "Settings saved with the project when loaded/applied.")}
              </small>
            </div>
          </div>
        </details>
        <div className="chat-conversation">
          <div className="chat-conversation-heading">
            <h3>Conversation</h3>
            <div className="checkpoint-chat-actions">
              <button
                type="button"
                title="Clear messages; keep the loaded model, tools and decoder"
                disabled={controlsDisabled || !session?.messages.length}
                onClick={() => void action("/clear")}
              >
                Clear messages
              </button>
              <button
                type="button"
                disabled={!session?.messages.length}
                onClick={() => {
                  if (session)
                    downloadJsonl("chat-transcript.jsonl", [
                      { version: 1, run_id: runId, ...session },
                    ]);
                }}
              >
                Export transcript
              </button>
            </div>
          </div>
          <p className="chat-execution-note">
            Calls are parsed/displayed only; nothing is executed. “Add mock
            result” inserts a simulated response.
          </p>
          {!activeTools && (
            <p className="chat-warning" role="status">
              No tools loaded. Choose a catalog in Test settings before testing
              tool requests.
            </p>
          )}
          {toolsDirty && (
            <div className="chat-blocking-banner" role="status">
              <strong>
                {pinned
                  ? "Conversation settings are pinned."
                  : "Settings have unapplied changes."}
              </strong>
              <p>
                {pinned
                  ? "This history uses the previous tools/system/decoder. Clear messages and apply the new settings to start a clean test."
                  : "The draft catalog/system/decoder will be saved and applied before your next message."}
              </p>
              {pinned && (
                <button
                  type="button"
                  disabled={controlsDisabled || toolsInvalid}
                  onClick={() => void applySettings(true)}
                >
                  Clear messages &amp; apply settings
                </button>
              )}
            </div>
          )}
          {missingTemplate && (
            <p className="training-alert">
              No saved chat template. Choose raw text completion or provide the
              training template.
            </p>
          )}
          {loaded && !selectedLoaded && (
            <p className="training-alert">
              Load the selected checkpoint and device before sending.
            </p>
          )}
          <ChatTranscript
            session={session}
            generating={session?.state === "generating"}
            onMock={(id, name) => {
              setMock({ id, name });
              setMockContent('{"ok":true}');
            }}
          />
          {mock && (
            <div className="chat-mock-result">
              <strong>Simulated result for {mock.name}</strong>
              <label>
                Canned result
                <textarea
                  rows={3}
                  value={mockContent}
                  onChange={(event) => setMockContent(event.target.value)}
                />
              </label>
              <div className="checkpoint-chat-actions">
                <button
                  type="button"
                  disabled={controlsDisabled}
                  onClick={async () => {
                    if (
                      await action("/tool-result", {
                        tool_call_id: mock.id,
                        content: mockContent,
                        simulated: true,
                      })
                    )
                      setMock(null);
                  }}
                >
                  Insert simulated result
                </button>
                <button type="button" onClick={() => setMock(null)}>
                  Cancel
                </button>
              </div>
              <small>
                This inserts a tool/result role into the next prompt. It runs no
                API and may be outside this model's training distribution.
              </small>
            </div>
          )}
          <form
            className="chat-composer"
            onSubmit={(event) => {
              event.preventDefault();
              void send();
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
                    : "Write text to continue…"
                }
              />
            </label>
            <div
              className="chat-budget"
              role="status"
              aria-label="Context token budget"
            >
              {budget ? (
                <>
                  {budget.prompt_tokens?.toLocaleString()} prompt tokens (
                  {budget.history_tokens?.toLocaleString()} history) ·{" "}
                  {maxTokens.toLocaleString()} reserved reply ·{" "}
                  <strong>
                    {budget.remaining_tokens?.toLocaleString()} remaining
                  </strong>{" "}
                  / {contextLength.toLocaleString()}
                </>
              ) : loaded ? (
                "Counting prompt and history tokens…"
              ) : (
                "Load a checkpoint to preview token usage."
              )}
            </div>
            {budget && (budget.remaining_tokens ?? 0) < 0 && (
              <p className="training-alert">
                Prompt plus reply exceeds the context budget. Clear messages,
                shorten the prompt/reply, or increase the budget within the
                trained limit.
              </p>
            )}
            {session?.context_preview?.state === "error" && (
              <p className="training-alert">{session.context_preview.error}</p>
            )}
            <div className="checkpoint-chat-actions">
              <button type="submit" className="primary" disabled={!canSend}>
                {toolsDirty && !pinned ? "Apply & send" : "Send"}
              </button>
              {generating && (
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void action("/stop")}
                >
                  Stop generation
                </button>
              )}
              <small>Ctrl/Cmd+Enter · at most {maxTokens} new tokens</small>
            </div>
          </form>
          {session?.finish_reason && (
            <small className="chat-finish-reason">
              Response ended:{" "}
              {session.finish_reason === "length"
                ? "new-token limit reached; XML may be incomplete"
                : session.finish_reason === "turn_end"
                  ? "assistant turn ended"
                  : session.finish_reason === "eos"
                    ? "end-of-sequence token"
                    : "stopped"}
              .
            </small>
          )}
          <ChatTestRunner
            session={session}
            disabled={
              session?.state !== "ready" || busy || toolsInvalid || !!toolsDirty
            }
            onRun={(cases) =>
              action("/tests", {
                cases: cases.map((item) => ({
                  ...toolOptions,
                  max_new_tokens: maxTokens,
                  context_length: contextLength,
                  ...item,
                })),
              })
            }
            onStop={() => void action("/stop")}
          />
        </div>
      </div>
    </section>
  );
}
