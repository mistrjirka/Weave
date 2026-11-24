import React from "react";
import { CircleAlert, Copy, Plus, Shapes, Trash2 } from "lucide-react";
import { InspectorHeader } from "./InspectorHeader";
import { InputSettings } from "./InputSettings";
import { BlockOperations } from "./BlockOperations";
import { BlockSettings } from "./BlockSettings";
import { Field } from "../Field";
import { icons, family } from "../presentation";
import { format } from "../util";
import { ReferenceInspector, isGraphEditShortcut } from "../../ReferenceInspector";
import { DesignAdvice } from "../../DesignAdvice";
import { WeightGroupPicker } from "../../WeightGroupPicker";
import { supportsWeightSharing, weightGroupChoices } from "../../weightGroups";
import { isInsertionKind } from "../../graphOperations";
import type { InsertionKind } from "../../graphOperations";
import { parseScheduleValue } from "../../projectWorkflow";
import type { WorkspacePanel } from "../../workspaceLocation";
import { catalog } from "../../catalog";
import { allNodes, makeIndependent } from "../../graph";
import { newId as uid } from "../../ui";
import type { Graph, GraphNode, NodeData, ValidationIssue } from "../../types";
import type { edgeContract } from "../../lanePorts";

type Settings = React.ComponentProps<typeof BlockSettings>;
type Operations = React.ComponentProps<typeof BlockOperations>;
type Inputs = React.ComponentProps<typeof InputSettings>;
interface InspectorProps extends Omit<Settings, "selected">, Omit<Operations, "selected" | "path" | "updateNode">, Pick<Inputs, "selectedTokenizer"> {
  selected: GraphNode | undefined;
  selectedCount: number;
  graph: Graph;
  referenceDetails: React.ComponentProps<typeof ReferenceInspector>["inspection"] | null;
  mobilePanel: WorkspacePanel;
  connection: Graph["edges"][number] | undefined;
  connectionShape: ReturnType<typeof edgeContract> | null;
  insertKind: InsertionKind;
  setInsertKind: React.Dispatch<React.SetStateAction<InsertionKind>>;
  insertOnConnection: () => void;
  validationError: unknown;
  focusError: (issue: ValidationIssue) => void;
  duplicate: () => void;
  removeSelected: () => void;
  onClosePanel: () => void;
  onClearSelection: () => void;
  onChooseTokenizer: (id: string) => void;
  onSaveBlock: (data: NodeData) => void;
}

export function Inspector({ mobilePanel, referenceDetails, selected, selectedCount, connection, onClosePanel, onClearSelection, updateNode, project, path, open, notify, unrollSelected, duplicateShared, fitGraph, selectedTokenizer, selectedVocabField, selectedAutoVocab, onChooseTokenizer, transformer, fieldVariables, selectedFieldVariables, variableScope, report, prefix, resizeExperts, graph, focusError, duplicate, removeSelected, onSaveBlock, connectionShape, insertKind, setInsertKind, insertOnConnection, validationError }: InspectorProps) {
  return (
            <aside
              className={
                "inspector " + (mobilePanel === "properties" ? "is-open" : "")
              }
              aria-label="Property inspector"
              data-reference-pane={referenceDetails ? "true" : undefined}
              onKeyDownCapture={(event) => {
                if (referenceDetails && isGraphEditShortcut(event)) {
                  event.preventDefault();
                  event.stopPropagation();
                }
              }}
            >
              <InspectorHeader
                hasReference={!!referenceDetails}
                hasSelection={!!selected}
                selectedCount={selectedCount}
                hasConnection={!!connection}
                onClosePanel={onClosePanel}
                onClearSelection={onClearSelection}
              />
              <div className="inspector-scroll">
                {referenceDetails ? (
                  <ReferenceInspector inspection={referenceDetails} />
                ) : selected ? (
                  <>
                    {selectedCount > 1 && (
                      <p className="inspector-description">
                        Copy, duplicate and delete apply to all {selectedCount}{" "}
                        selected blocks. Properties below edit{" "}
                        {selected.data.label}.
                      </p>
                    )}
                    <div
                      className={
                        "selected-block-icon " + family(selected.data.kind)
                      }
                    >
                      {React.createElement(
                        icons[selected.data.kind] || Shapes,
                        {
                          size: 24,
                        },
                      )}
                    </div>
                    <span className="eyebrow">
                      {selected.data.kind.replaceAll("_", " ")}
                    </span>
                    <input
                      className="inspector-name"
                      aria-label="Block name"
                      value={selected.data.label}
                      onChange={(e) =>
                        updateNode((n) => {
                          n.data.label = e.target.value;
                        })
                      }
                    />
                    <p className="inspector-description">
                      {selected.data.description ||
                        catalog.find((c) => c.kind === selected.data.kind)
                          ?.description ||
                        "Editable composite computation."}
                    </p>
                    <label className="property-field">
                      <span>Design note</span>
                      <textarea
                        aria-label="Design note"
                        placeholder="Add a note about this block or why you chose it"
                        value={selected.data.annotation ?? ""}
                        onChange={(event) =>
                          updateNode((node) => {
                            node.data.annotation = event.target.value;
                          })
                        }
                      />
                      <small className="field-help">
                        Notes appear on the canvas, are saved with the graph,
                        and do not affect computation.
                      </small>
                    </label>
                    {selected.data.annotation?.trim() && (
                      <label className="property-field">
                        <span>Design note source URL (optional)</span>
                        <input
                          aria-label="Design note source URL"
                          type="url"
                          value={selected.data.annotationSource ?? ""}
                          onChange={(event) =>
                            updateNode((node) => {
                              node.data.annotationSource = event.target.value;
                            })
                          }
                          placeholder="https://…"
                        />
                      </label>
                    )}
                    <BlockOperations selected={selected} project={project} path={path} updateNode={updateNode} open={open} notify={notify} unrollSelected={unrollSelected} duplicateShared={duplicateShared} fitGraph={fitGraph} />
                    <InputSettings selected={selected} selectedTokenizer={selectedTokenizer} selectedVocabField={selectedVocabField} selectedAutoVocab={selectedAutoVocab} path={path} updateNode={updateNode} onChooseTokenizer={() => onChooseTokenizer(selected.id)} />
                    <BlockSettings selected={selected} transformer={transformer} selectedAutoVocab={selectedAutoVocab} selectedVocabField={selectedVocabField} fieldVariables={fieldVariables} selectedFieldVariables={selectedFieldVariables} variableScope={variableScope} report={report} path={path} prefix={prefix} updateNode={updateNode} resizeExperts={resizeExperts} />
                    {selected.data.kind === "moe" && (
                      <p className="field-help">
                        Open experts to edit their individual graphs. The router
                        selects tokens and the combine sums their weighted
                        outputs. Expert count is a fixed integer; other settings
                        can use variables.
                      </p>
                    )}
                    {["moe_router", "moe_combine"].includes(
                      selected.data.kind,
                    ) && (
                      <p className="field-help">
                        Expert count is controlled by the outer Mixture of
                        experts block. Edit the graph inside each expert.
                      </p>
                    )}
                    {selected.data.kind === "attention_pattern" && (
                      <p className="field-help">
                        Connect features to a, and optionally key features to b.
                        The output is an attention pattern. Connect it to
                        attention’s optional mask port; combine patterns with
                        union or intersection. Custom matrices require fixed
                        sequence lengths. Content top-k selects from features
                        and computes dense routing scores.
                      </p>
                    )}
                    {["attention", "cross_attention", "sdpa"].includes(
                      selected.data.kind,
                    ) && (
                      <p className="field-help">
                        Dense uses PyTorch SDPA on CPU or GPU. Auto uses
                        compiled FlexAttention on CUDA; flex requires a
                        compatible GPU. A pattern defines permitted positions
                        independently of the backend.
                      </p>
                    )}
                    {selected.data.iteration && (
                      <>
                        <div className="section-label">
                          Application schedule
                        </div>
                        <p className="field-help">
                          Zero-based step and total are passed to sampling
                          operations. Shared applications may use different
                          schedules.
                        </p>
                        {(["index", "total"] as const).map((key) => (
                          <Field
                            key={key}
                            name={"schedule " + key}
                            parseValue={parseScheduleValue}
                            value={selected.data.iteration![key]}
                            onChange={(value) =>
                              updateNode((n) => {
                                if (n.data.iteration && typeof value === "number")
                                  n.data.iteration[key] = value;
                              })
                            }
                          />
                        ))}
                      </>
                    )}
                    {supportsWeightSharing(selected.data) && (
                      <WeightGroupPicker
                        value={
                          selected.data.share ??
                          (selected.data.kind === "head" &&
                          selected.data.params.tie
                            ? `legacy:${selected.data.params.tie}`
                            : "")
                        }
                        matrixSharing={["embedding", "head"].includes(
                          selected.data.kind,
                        )}
                        choices={[
                          ...weightGroupChoices(
                            allNodes(project),
                            selected.data,
                          ),
                          ...(selected.data.kind === "head" &&
                          selected.data.params.tie &&
                          !selected.data.share
                            ? [
                                {
                                  id: `legacy:${selected.data.params.tie}`,
                                  label: `Legacy tie · ${graph.nodes.find((n) => n.id === selected.data.params.tie)?.data.label ?? selected.data.params.tie}`,
                                  compatible: false,
                                },
                              ]
                            : []),
                        ]}
                        onChange={(group) => {
                          if (
                            !selected.data.share &&
                            group === `legacy:${selected.data.params.tie}`
                          )
                            return;
                          updateNode((node) => {
                            delete node.data.params.tie;
                            if (group) node.data.share = group;
                            else makeIndependent(node);
                          });
                        }}
                        onCreate={() =>
                          updateNode((node) => {
                            delete node.data.params.tie;
                            makeIndependent(node);
                            node.data.share = "weights_" + uid();
                          })
                        }
                      />
                    )}
                    {selected.data.kind === "head" &&
                      selected.data.params.tie &&
                      selected.data.share && (
                        <p className="field-help">
                          This head also has a legacy embedding tie. Selecting a
                          weight group or Independent weights clears the legacy
                          tie.
                        </p>
                      )}
                    {selected.data.kind === "custom" && (
                      <Field
                        name="source"
                        value={selected.data.params.source || ""}
                        onChange={(v) =>
                          updateNode((n) => {
                            if (typeof v === "string") n.data.params.source = v;
                          })
                        }
                      />
                    )}
                    <div className="section-label">Output contract</div>
                    <div className="shape-card">
                      <span>Tensor shape</span>
                      <code>
                        {report?.shapes[prefix + selected.id]
                          ? `[${report.shapes[prefix + selected.id]?.shape.join(", ")}]`
                          : "Unresolved"}
                      </code>
                      <small>
                        {report?.shapes[prefix + selected.id]?.dtype ||
                          "Connect valid inputs to infer shape"}
                      </small>
                    </div>
                    <div className="stat-row">
                      <span>Parameters</span>
                      <b>{format(report?.counts[prefix + selected.id] || 0)}</b>
                    </div>
                    {report?.errors
                      .filter((e) => e.node === prefix + selected.id)
                      .map((e: ValidationIssue, i: number) => (
                        <div className="inline-error" key={i}>
                          <CircleAlert size={15} />
                          {e.message}
                        </div>
                      ))}
                    {report?.warnings
                      .filter((issue) => issue.node === prefix + selected.id)
                      .map((issue, i) => (
                        <DesignAdvice
                          key={`${issue.code}-${i}`}
                          issue={issue}
                          onLocate={focusError}
                        />
                      ))}
                    <div className="node-actions">
                      <button onClick={duplicate}>
                        <Copy size={14} />
                        {selectedCount > 1 ? "Duplicate selected" : "Duplicate"}
                      </button>
                      <button className="danger" onClick={removeSelected}>
                        <Trash2 size={14} />
                        {selectedCount > 1 ? "Delete selected" : "Delete"}
                      </button>
                    </div>
                    {selected.data.body && (
                      <button
                        className="save-library"
                        onClick={() => onSaveBlock(selected.data)}
                      >
                        <Plus size={14} />
                        Save to block library
                      </button>
                    )}
                  </>
                ) : connection ? (
                  <div className="connection-editor">
                    <span className="eyebrow">EDIT THE COMPUTATION PATH</span>
                    <h2>Insert between nodes</h2>
                    <p>
                      {
                        graph.nodes.find((n) => n.id === connection.source)
                          ?.data.label
                      }{" "}
                      →{" "}
                      {
                        graph.nodes.find((n) => n.id === connection.target)
                          ?.data.label
                      }
                    </p>
                    <div className="shape-card">
                      <span>Connection contract</span>
                      <code>
                        {connectionShape
                          ? `[${connectionShape.shape.join(", ")}] · ${connectionShape.dtype}`
                          : "Unresolved"}
                      </code>
                    </div>
                    <p className="inspector-description">
                      An inserted layer has independent parameters. Existing
                      shared applications keep their weight group.
                    </p>
                    {connectionShape?.dtype === "float32" &&
                    typeof connectionShape.shape.at(-1) === "number" ? (
                      <>
                        <label className="property-field">
                          <span>Layer to insert</span>
                          <select
                            aria-label="Layer to insert"
                            value={insertKind}
                            onChange={(e) => {
                              if (isInsertionKind(e.target.value))
                                setInsertKind(e.target.value);
                            }}
                          >
                            <option value="linear">Learned linear layer</option>
                            <option value="adapter">Residual adapter</option>
                            <option value="ffn">Feed-forward network</option>
                            <option value="norm">RMSNorm</option>
                            <option value="transformer">
                              Transformer layer
                            </option>
                          </select>
                        </label>
                        <button
                          className="inspect-open"
                          onClick={insertOnConnection}
                        >
                          Insert independent layer <Plus size={15} />
                        </button>
                      </>
                    ) : (
                      <p>
                        Connect floating point features with a known width to
                        insert a learned layer here. Token IDs need an embedding
                        first.
                      </p>
                    )}
                  </div>
                ) : (
                  <>
                    <h2>Select a block</h2>
                    <p className="inspector-description">
                      Click a block to edit its settings. Click a connection to
                      insert a layer.
                    </p>
                    <div className="section-label">Whole graph</div>
                    <div className="stat-row">
                      <span>Trainable parameters</span>
                      <b>{format(report?.parameters || 0)}</b>
                    </div>
                    <div className="stat-row">
                      <span>Graph validation</span>
                      <b className={report?.valid ? "valid" : "invalid"}>
                        {report
                          ? report.valid
                            ? "Compatible"
                            : "Needs attention"
                          : validationError
                            ? "Unavailable"
                            : "Checking…"}
                      </b>
                    </div>
                  </>
                )}
              </div>
            </aside>
  );
}
