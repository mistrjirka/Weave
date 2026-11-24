import { ChevronRight, Database } from "lucide-react";
import { tokenizerLabel } from "../../TokenizerPanel";
import { format } from "../util";
import type { GraphNode, TokenizerInfo } from "../../types";

interface InputSettingsProps {
  selected: GraphNode;
  selectedTokenizer: TokenizerInfo | null | undefined;
  selectedVocabField: string | undefined;
  selectedAutoVocab: boolean;
  path: string[];
  updateNode: (fn: (node: GraphNode) => void) => void;
  onChooseTokenizer: () => void;
}

export function InputSettings({ selected, selectedTokenizer, selectedVocabField, selectedAutoVocab, path, updateNode, onChooseTokenizer }: InputSettingsProps) {
  return <>
                    {selected.data.kind === "input" &&
                      (selected.data.tokenizer ||
                        ((selected.data.params.dtype ?? "int64") === "int64" &&
                          (selected.data.params.shape ?? ["B", "T"]).length ===
                            2)) && (
                        <section
                          className="input-tokenizer-settings"
                          aria-label="Input tokenizer"
                        >
                          <div className="section-label">Tokenizer</div>
                          <button
                            className="tokenizer-chip"
                            disabled={
                              path.length > 0 &&
                              !!selectedTokenizer &&
                              !selected.data.tokenizer
                            }
                            onClick={() => {
                              onChooseTokenizer();
                            }}
                          >
                            <Database size={14} />
                            <span>
                              {selectedTokenizer
                                ? tokenizerLabel(selectedTokenizer)
                                : "Select tokenizer"}
                            </span>
                            {selected.data.tokenizer && (
                              <small>
                                {format(selected.data.tokenizer.capacity)} vocab
                              </small>
                            )}
                            <ChevronRight size={14} />
                          </button>
                          {selectedTokenizer && (
                            <div className="stat-row">
                              <span>Vocabulary capacity</span>
                              <b>
                                {selectedTokenizer.capacity.toLocaleString()}
                              </b>
                            </div>
                          )}
                          {!selected.data.tokenizer && selectedTokenizer && (
                            <p className="field-help">
                              Inherited from the connected input ·{" "}
                              {format(selectedTokenizer.capacity)} token IDs.
                              Configure the originating token input to change
                              it.
                            </p>
                          )}
                          {selected.data.tokenizer && (
                            <button
                              className="small-button"
                              onClick={() =>
                                updateNode((node) => {
                                  delete node.data.tokenizer;
                                })
                              }
                            >
                              Remove tokenizer
                            </button>
                          )}
                          <p className="field-help">
                            The tokenizer defines token IDs and vocabulary size.
                            Embedding width is configured on each embedding
                            block.
                          </p>
                        </section>
                      )}
                    {selectedTokenizer && selectedVocabField && (
                      <section
                        className="property-field"
                        aria-label="Tokenizer vocabulary capacity"
                      >
                        {selectedAutoVocab && (
                          <>
                            <div className="stat-row">
                              <span>Vocabulary capacity</span>
                              <b>
                                {selectedTokenizer.capacity.toLocaleString()}
                              </b>
                            </div>
                            <small className="field-help">
                              Automatic from the connected input tokenizer.
                            </small>
                          </>
                        )}
                        <button
                          className="small-button"
                          onClick={() =>
                            updateNode((node) => {
                              node.data.params[selectedVocabField] =
                                selectedAutoVocab
                                  ? selectedTokenizer.capacity
                                  : "$vocab";
                            })
                          }
                        >
                          {selectedAutoVocab
                            ? "Set capacity manually"
                            : "Use input tokenizer capacity"}
                        </button>
                      </section>
                    )}
  </>;
}
