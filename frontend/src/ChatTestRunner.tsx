import { useState } from "react";
import type { ChatSession } from "./chatSchema";
import { downloadJsonl } from "./chatSchema";
import { z } from "zod";
const casesSchema = z
  .array(z.object({ text: z.string().min(1) }).passthrough())
  .min(1)
  .max(50);
export function ChatTestRunner({
  session,
  disabled,
  onRun,
  onStop,
}: {
  session: ChatSession | null;
  disabled: boolean;
  onRun: (cases: z.infer<typeof casesSchema>) => Promise<unknown>;
  onStop: () => void;
}) {
  const [source, setSource] = useState("");
  const [error, setError] = useState("");
  const suite = session?.test_suite;
  async function run() {
    try {
      const cases = casesSchema.parse(
        source
          .split(/\r?\n/)
          .filter((line) => line.trim())
          .map((line) => JSON.parse(line)),
      );
      setError("");
      await onRun(cases);
    } catch {
      setError(
        "Use one JSON object per line, each with text and optional expected calls; maximum 50 cases.",
      );
    }
  }
  return (
    <details className="chat-tests">
      <summary>
        Regression tests
        {suite ? ` · ${suite.results.length}/${suite.total}` : ""}
      </summary>
      <p>
        Each case is an independent prompt. Tools/system can be overridden per
        case; defaults come from this session. Expected calls compare parsed
        tool names and argument values, not XML whitespace.
      </p>
      <label>
        Import test cases (.jsonl)
        <input
          type="file"
          accept=".jsonl,.ndjson"
          disabled={disabled}
          onChange={async (event) => {
            const file = event.currentTarget.files?.[0];
            event.currentTarget.value = "";
            if (!file) return;
            if (file.size > 1048576) {
              setError("Test cases must fit in 1 MiB.");
              return;
            }
            try {
              setSource(await file.text());
              setError("");
            } catch {
              setError("Could not read test cases.");
            }
          }}
        />
      </label>
      <label>
        Test cases JSONL
        <textarea
          rows={4}
          value={source}
          disabled={disabled}
          onChange={(event) => setSource(event.target.value)}
          placeholder={
            '{"text":"Turn on the flashlight","expected":[{"name":"turn_on_flashlight","arguments":{}}]}\n{"text":"Hello","expected":[]}'
          }
        />
      </label>
      {error && (
        <p role="alert" className="training-alert">
          {error}
        </p>
      )}
      <div className="checkpoint-chat-actions">
        <button
          type="button"
          disabled={disabled || !source.trim()}
          onClick={() => void run()}
        >
          Run test cases
        </button>
        {session?.state === "testing" && (
          <button type="button" onClick={onStop}>
            Stop tests
          </button>
        )}
        {suite && (
          <button
            type="button"
            onClick={() =>
              downloadJsonl("chat-test-results.jsonl", suite.results)
            }
          >
            Export test results
          </button>
        )}
      </div>
      {suite && (
        <>
          <p role="status">
            {suite.state} · {suite.results.length}/{suite.total} tested ·{" "}
            {suite.results.filter((r) => r.exact_match === true).length} exact
            matches · {suite.results.filter((r) => r.parse_valid).length} valid
            parses
          </p>
          <div className="chat-table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Parse</th>
                  <th>Tool</th>
                  <th>Arguments / exact</th>
                  <th>Output</th>
                </tr>
              </thead>
              <tbody>
                {suite.results.map((r) => (
                  <tr key={r.index}>
                    <td>{r.name}</td>
                    <td>{r.parse_valid ? "Pass" : "Fail"}</td>
                    <td>
                      {r.tool_match == null
                        ? "Unscored"
                        : r.tool_match
                          ? "Pass"
                          : "Fail"}
                    </td>
                    <td>
                      {r.exact_match == null
                        ? "Unscored"
                        : r.exact_match
                          ? "Pass"
                          : "Fail"}
                    </td>
                    <td>
                      <pre>{r.error || r.output}</pre>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </details>
  );
}
