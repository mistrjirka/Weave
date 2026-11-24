import { useEffect, useMemo, useRef, useState } from "react";
import { z } from "zod";
import { trainingRequest, type TrainingGPU } from "../trainingApi";
import { useChatSession } from "./useChatSession";
import {
  catalogSchema,
  sessionSchema,
  toolsSchema,
  diagnosticsSchema,
  liveSystem,
} from "./chatSchema";

export function useCheckpointChat(runId: string, gpus: TrainingGPU[]) {
  const [catalog, setCatalog] = useState<z.infer<typeof catalogSchema> | null>(
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
  const [catalogError, setCatalogError] = useState("");
  const [catalogVersion, setCatalogVersion] = useState(0);
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
  const { session, setSession, error, setError, pollError, busy, action } =
    useChatSession(endpoint, setPrompt, setToolsNotice);

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
  return {
    catalog,
    session,
    checkpoint,
    setCheckpoint,
    device,
    setDevice,
    mode,
    setMode,
    prompt,
    setPrompt,
    system,
    setSystem,
    chatTemplate,
    setChatTemplate,
    toolsText,
    setToolsText,
    decoder,
    setDecoder,
    toolsNotice,
    setToolsNotice,
    toolsFileError,
    setToolsFileError,
    maxTokens,
    setMaxTokens,
    contextLength,
    setContextLength,
    temperature,
    setTemperature,
    topP,
    setTopP,
    repetitionPenalty,
    setRepetitionPenalty,
    seed,
    setSeed,
    error,
    setError,
    catalogError,
    pollError,
    busy,
    setCatalogVersion,
    deviceChosen,
    settingsOpen,
    setSettingsOpen,
    diagnostics,
    mock,
    setMock,
    mockContent,
    setMockContent,
    action,
    generating,
    loaded,
    selectedLoaded,
    parsedTools,
    toolOptions,
    toolsDirty,
    pinned,
    toolsInvalid,
    budget,
    missingTemplate,
    canSend,
    applySettings,
    send,
    controlsDisabled,
    activeTools,
    draftCount,
    selectedPreset,
    expiry,
    deterministic,
    usePreset,
  };
}

export type CheckpointChatState = ReturnType<typeof useCheckpointChat>;
