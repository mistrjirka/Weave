import type { TrainingGPU } from "../trainingApi";
import { downloadJsonl } from "./chatSchema";
import { ChatTranscript } from "./ChatTranscript";
import { ChatTestRunner } from "./ChatTestRunner";
import { ChatSettings } from "./ChatSettings";
import { useCheckpointChat } from "./useCheckpointChat";

export function CheckpointChat({
  runId,
  gpus,
  enabled,
}: {
  runId: string;
  gpus: TrainingGPU[];
  enabled: boolean;
}) {
  const chat = useCheckpointChat(runId, gpus);
  const {
    catalog,
    session,
    checkpoint,
    setCheckpoint,
    device,
    setDevice,
    mode,
    prompt,
    setPrompt,
    maxTokens,
    contextLength,
    error,
    catalogError,
    pollError,
    busy,
    setCatalogVersion,
    deviceChosen,
    mock,
    setMock,
    mockContent,
    setMockContent,
    action,
    generating,
    loaded,
    selectedLoaded,
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
    expiry,
  } = chat;
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
        <ChatSettings chat={chat} />
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
