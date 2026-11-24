import { Copy } from "lucide-react";
import { copyText } from "../ui";

export function GeneratedCode({ source, notify }: { source: string; notify: (message: string) => void }) {
  return (
                  <div className="code-view">
                    <div className="code-heading">
                      <span>
                        <span className="file-dot" />
                        model.py <small>Generated from this graph</small>
                      </span>
                      <button
                        className="subtle-button"
                        onClick={() => {
                          copyText(source)
                            .then(() => notify("Code copied"))
                            .catch(() =>
                              notify("Use Export to download the code"),
                            );
                        }}
                      >
                        <Copy size={14} />
                        Copy code
                      </button>
                    </div>
                    <pre
                      tabIndex={0}
                      role="region"
                      aria-label="Generated PyTorch code"
                    >
                      {source.split("\n").map((line, i) => (
                        <div key={i}>
                          <span className="line-number">{i + 1}</span>
                          <code
                            className={
                              line.trim().startsWith("#") ? "comment" : ""
                            }
                          >
                            {line || " "}
                          </code>
                        </div>
                      ))}
                    </pre>
                    <div className="code-note">
                      Export includes runtime operations, tokenizer files,
                      project JSON, dependencies, and a runnable example.
                    </div>
                  </div>
  );
}
