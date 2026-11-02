import { useRef, useState, useEffect } from "react";
import type { ChatSession, ChatMessage } from "./chatSchema";
import { downloadJsonl } from "./chatSchema";
import { copyText } from "./ui";

export function ChatTranscript({
  session,
  generating,
  onMock,
}: {
  session: ChatSession | null;
  generating: boolean;
  onMock: (callId: string, name: string) => void;
}) {
  const transcript = useRef<HTMLDivElement>(null);
  const [notice, setNotice] = useState("");
  useEffect(() => {
    const element = transcript.current;
    if (
      element &&
      element.scrollHeight - element.scrollTop - element.clientHeight < 180
    )
      element.scrollTop = element.scrollHeight;
  }, [session?.messages]);
  async function copy(message: ChatMessage) {
    try {
      await copyText(message.content);
      setNotice("Output copied.");
    } catch {
      setNotice("Could not copy. Select the raw output to copy it manually.");
    }
  }
  return (
    <>
      <div
        ref={transcript}
        className="checkpoint-chat-transcript"
        role="log"
        aria-label="Chat conversation"
        aria-live="polite"
        aria-relevant="additions text"
      >
        {!session?.messages.length && (
          <div className="chat-empty">
            <strong>Test a checkpoint</strong>
            <p>
              Load a model, choose a tool catalog if needed, then send a prompt.
              Generated calls are displayed here.
            </p>
          </div>
        )}
        {session?.messages.map((message, index) => (
          <article
            key={message.turn_id ? `${message.turn_id}-${message.role}` : index}
            className={`checkpoint-chat-message is-${message.role}`}
          >
            <div className="chat-turn-heading">
              <strong>
                {message.role === "user"
                  ? "You"
                  : message.role === "tool"
                    ? "Simulated tool result"
                    : "Model"}
              </strong>
              {message.included_in_context !== undefined &&
                !(generating && index >= session.messages.length - 2) && (
                  <span
                    title={
                      message.context_reason ??
                      "Sent in the next conversation prompt"
                    }
                    className={
                      message.included_in_context
                        ? "chat-context-included"
                        : "chat-context-excluded"
                    }
                  >
                    {message.included_in_context
                      ? "In context"
                      : "Excluded from context"}
                  </span>
                )}
            </div>
            <p>
              {message.content ||
                (generating && message.role === "assistant"
                  ? "Generating…"
                  : "No text generated.")}
            </p>
            {message.context_reason && <small>{message.context_reason}</small>}
            {message.role === "assistant" && message.content && (
              <>
                {message.parsed && message.parsed.kind !== "text" && (
                  <div className="chat-call-inspection">
                    <strong>
                      {message.parsed.errors.length
                        ? "Invalid / incomplete call"
                        : message.parsed.kind === "no_call"
                          ? "No tool selected"
                          : "Valid tool call"}
                    </strong>
                    {message.parsed.errors.map((error, i) => (
                      <p key={i} className="training-alert">
                        {error}
                      </p>
                    ))}
                    {message.parsed.calls.length > 0 && (
                      <div className="chat-table-scroll">
                        <table>
                          <thead>
                            <tr>
                              <th>Tool</th>
                              <th>Arguments</th>
                              <th>Test action</th>
                            </tr>
                          </thead>
                          <tbody>
                            {message.parsed.calls.map((call, i) => (
                              <tr key={i}>
                                <td>{call.name}</td>
                                <td>
                                  <pre>
                                    {JSON.stringify(call.arguments, null, 2)}
                                  </pre>
                                </td>
                                <td>
                                  <button
                                    type="button"
                                    onClick={() => {
                                      void copyText(
                                        JSON.stringify({
                                          name: call.name,
                                          arguments: call.arguments,
                                        }),
                                      ).then(
                                        () => setNotice("Call copied."),
                                        () => setNotice("Could not copy call."),
                                      );
                                    }}
                                  >
                                    Copy call
                                  </button>
                                  {message.tool_calls[i] && (
                                    <button
                                      type="button"
                                      disabled={
                                        generating ||
                                        !message.included_in_context ||
                                        session.messages.some(
                                          (m) =>
                                            m.tool_call_id ===
                                            message.tool_calls[i].id,
                                        )
                                      }
                                      onClick={() =>
                                        onMock(
                                          message.tool_calls[i].id,
                                          call.name,
                                        )
                                      }
                                    >
                                      Add mock result
                                    </button>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )}
                <div className="chat-output-footer">
                  <small>
                    Calls are parsed/displayed only; nothing is executed.
                  </small>
                  <div>
                    <button type="button" onClick={() => void copy(message)}>
                      Copy output
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        const user = session.messages[index - 1];
                        if (user?.role !== "user") return;
                        const expected =
                          message.parsed?.kind === "no_call"
                            ? []
                            : message.parsed?.calls.map(
                                ({ name, arguments: args }) => ({
                                  name,
                                  arguments: args,
                                }),
                              );
                        downloadJsonl("chat-regression-case.jsonl", [
                          {
                            ...message.generation,
                            text: user.content,
                            expected,
                          },
                        ]);
                      }}
                    >
                      Save as test case
                    </button>
                  </div>
                </div>
              </>
            )}
            {message.generation && (
              <small className="chat-generation-meta">
                Temperature {message.generation.temperature ?? "—"} · seed{" "}
                {message.generation.seed ?? "—"} · decoder{" "}
                {message.generation.decoder ?? "None"}
              </small>
            )}
          </article>
        ))}
      </div>
      {notice && <small role="status">{notice}</small>}
    </>
  );
}
