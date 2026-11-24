import { z } from "zod";
import { copyText } from "../ui";
import { liveSystem } from "./chatSchema";
import type { CheckpointChatState } from "./useCheckpointChat";

export function ChatSettings({ chat }: { chat: CheckpointChatState }) {
  const {
    catalog,
    session,
    mode,
    setMode,
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
    setError,
    busy,
    settingsOpen,
    setSettingsOpen,
    diagnostics,
    parsedTools,
    toolsDirty,
    pinned,
    toolsInvalid,
    applySettings,
    controlsDisabled,
    draftCount,
    selectedPreset,
    deterministic,
    usePreset,
  } = chat;
  return (
    <details
      className="chat-settings-panel"
      open={settingsOpen}
      onToggle={(event) => setSettingsOpen(event.currentTarget.open)}
    >
      <summary>
        Test settings
        {toolsInvalid ? " · needs attention" : toolsDirty ? " · unapplied" : ""}
      </summary>
      <div className="chat-settings-body">
        <label>
          Prompt format
          <select
            value={mode}
            disabled={controlsDisabled}
            onChange={(event) =>
              setMode(z.enum(["chat", "completion"]).parse(event.target.value))
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
                onChange={(event) => setTemperature(event.target.valueAsNumber)}
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
  );
}
