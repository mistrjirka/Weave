import React from "react";
import { Field } from "../Field";
import { inputSettingOwnership, containerParameterShadowReason } from "../../settingOwnership";
import { settingHelp, generalFieldDisabledReason } from "../../settingHelp";
import { attentionVariants, attentionVariant, attentionFieldDisabledReason, isAttentionVariant, transformerSettingKeys, wrappedTransformer, transformerParentScope, transformerResidual, setTransformerResidual, setTransformerSetting, setAttentionVariant } from "../../transformerControls";
import { bindingValue, previewParameters, graphVariableScope } from "../../graphVariables";
import { visiblePatternField, changePattern } from "../../patterns";
import { MaskMatrixEditor } from "../../previews";
import { parametersSchema } from "../../api";
import { newId as uid } from "../../ui";
import type { GraphNode, NodeData, JsonValue, Parameters, ValidationReport } from "../../types";

interface BlockSettingsProps {
  selected: GraphNode;
  transformer: Parameters | null;
  selectedAutoVocab: boolean;
  selectedVocabField: string | undefined;
  fieldVariables: Parameters;
  selectedFieldVariables: Parameters;
  variableScope: ReturnType<typeof graphVariableScope>;
  report: ValidationReport | null;
  path: string[];
  prefix: string;
  updateNode: (fn: (node: GraphNode) => void) => void;
  resizeExperts: (data: NodeData, value: JsonValue) => void;
}

export function BlockSettings({ selected, transformer, selectedAutoVocab, selectedVocabField, fieldVariables, selectedFieldVariables, variableScope, report, path, prefix, updateNode, resizeExperts }: BlockSettingsProps) {
  return <>
                    {transformer && wrappedTransformer(selected.data) && (
                      <p className="field-help">
                        These controls edit the{" "}
                        {wrappedTransformer(selected.data)?.data.label} inside
                        this loop. Opening that block shows the same settings.
                      </p>
                    )}
                    {!transformer &&
                      selected.data.body &&
                      (Object.keys(selected.data.params).some((key) =>
                        transformerSettingKeys.has(key),
                      ) ||
                        selected.data.body.nodes.filter(
                          (node) => node.data.role === "transformer",
                        ).length > 1) && (
                        <p className="field-help">
                          Configure each inner layer separately. Additional
                          container parameters affect only fields explicitly
                          linked with $name.
                        </p>
                      )}
                    {transformer && (
                      <label className="property-field">
                        <span>Attention type</span>
                        <select
                          aria-label="Attention type"
                          disabled={["heads", "kv_heads", "window"].some(
                            (key) =>
                              !!containerParameterShadowReason(
                                wrappedTransformer(selected.data)?.data ??
                                  selected.data,
                                key,
                              ),
                          )}
                          value={attentionVariant(
                            selected.data,
                            fieldVariables,
                          )}
                          onChange={(event) => {
                            if (isAttentionVariant(event.target.value)) {
                              const variant = event.target.value;
                              updateNode((node) =>
                                setAttentionVariant(
                                  node.data,
                                  variant,
                                  fieldVariables,
                                  uid,
                                ),
                              );
                            }
                          }}
                        >
                          {attentionVariants.map((variant) => (
                            <option key={variant.id} value={variant.id}>
                              {variant.label}
                            </option>
                          ))}
                        </select>
                        <small className="field-help">
                          Changes the editable attention graph inside this
                          container. Other layers are preserved.
                          {attentionVariant(selected.data, fieldVariables) ===
                            "sparse" &&
                            " The local-pattern window belongs to the Local attention window node inside; the additional attention window only adds a further restriction."}
                        </small>
                      </label>
                    )}
                    {transformer && transformerResidual(selected.data) && (
                      <label className="property-field">
                        <span>Attention residual</span>
                        <select
                          aria-label="Attention residual"
                          value={transformerResidual(selected.data) ?? "add"}
                          onChange={(event) => {
                            const value = event.target.value;
                            if (value === "add" || value === "gate")
                              updateNode((node) => setTransformerResidual(node.data, value));
                          }}
                        >
                          <option value="add">Add</option>
                          <option value="gate">Gated</option>
                        </select>
                      </label>
                    )}
                    <div className="section-label">Configuration</div>
                    {selected.data.kind === "moe" && (
                      <label className="property-field">
                        <span>Router scoring</span>
                        <select
                          aria-label="Router scoring"
                          value={
                            typeof selected.data.params.router === "string"
                              ? selected.data.params.router
                              : "softmax"
                          }
                          onChange={(event) =>
                            updateNode((n) => {
                              n.data.params.router = event.target.value;
                            })
                          }
                        >
                          <option value="softmax">Softmax top-k</option>
                          <option value="sigmoid">Sigmoid top-k</option>
                        </select>
                        <small className="field-help">
                          Both route each token to the top-k experts. Choose how
                          the router scores them.
                        </small>
                      </label>
                    )}
                    {selected.data.kind === "attention_pattern" &&
                      selected.data.params.pattern === "custom" &&
                      selected.data.params.matrix && (
                        <MaskMatrixEditor
                          matrix={selected.data.params.matrix}
                          onChange={(matrix) =>
                            updateNode((n) => {
                              n.data.params.matrix = matrix;
                            })
                          }
                        />
                      )}
                    {Object.entries(transformer ?? selected.data.params ?? {})
                      .filter(
                        ([k]) =>
                          k !== "source" &&
                          !(selectedAutoVocab && k === selectedVocabField) &&
                          !(k === "router" && selected.data.kind === "moe") &&
                          !(
                            k === "experts" &&
                            ["moe_router", "moe_combine"].includes(
                              selected.data.kind,
                            )
                          ) &&
                          (selected.data.kind !== "attention_pattern" ||
                            visiblePatternField(selected.data.params, k)) &&
                          (!selected.data.iteration ||
                            !["iterations", "shared"].includes(k)),
                      )
                      .map(([k, v]) => {
                        const inputOwnership = inputSettingOwnership(
                          selected.data.kind,
                          path.length,
                          k,
                          report?.shapes[prefix + selected.id],
                        );
                        const transformerOwner =
                          wrappedTransformer(selected.data)?.data ??
                          selected.data;
                        const localFfnLinked =
                          typeof transformerOwner.body?.variables?.ffn_width ===
                            "number" &&
                          transformerOwner.body.nodes.some(
                            (node) =>
                              node.data.kind === "ffn" &&
                              node.data.params.hidden === "$ffn_width",
                          );
                        return (
                          <React.Fragment key={k}>
                            <Field
                              name={k}
                              variables={selectedFieldVariables}
                              displayName={
                                k === "window" &&
                                transformer &&
                                attentionVariant(
                                  selected.data,
                                  fieldVariables,
                                ) === "sparse"
                                  ? "Additional attention window"
                                  : k === "gated" &&
                                      (transformer ||
                                        [
                                          "attention",
                                          "cross_attention",
                                        ].includes(selected.data.kind))
                                    ? "Attention output gate"
                                    : undefined
                              }
                              options={
                                selected.data.kind === "lane_join" &&
                                k === "mode"
                                  ? ["stack", "mean", "sum"]
                                  : undefined
                              }
                              value={
                                inputOwnership
                                  ? inputOwnership.value
                                  : (v ?? null)
                              }
                              help={settingHelp(selected.data.kind, k)}
                              disabledReason={
                                inputOwnership?.disabledReason ??
                                (transformer && k === "ffn_width"
                                  ? undefined
                                  : containerParameterShadowReason(
                                      wrappedTransformer(selected.data)?.data ??
                                        selected.data,
                                      k,
                                    )) ??
                                generalFieldDisabledReason(
                                  selected.data.kind,
                                  k,
                                  selected.data.params,
                                ) ??
                                (transformer ||
                                [
                                  "attention",
                                  "cross_attention",
                                  "rope",
                                ].includes(selected.data.kind)
                                  ? attentionFieldDisabledReason(
                                      k,
                                      previewParameters(
                                        transformer ?? selected.data.params,
                                        fieldVariables,
                                      ),
                                      transformer
                                        ? attentionVariant(
                                            selected.data,
                                            fieldVariables,
                                          )
                                        : undefined,
                                    )
                                  : undefined)
                              }
                              onChange={(next) =>
                                updateNode((n) => {
                                  if (
                                    n.data.kind === "moe" &&
                                    k === "experts"
                                  ) {
                                    resizeExperts(n.data, next);
                                    return;
                                  }
                                  if (
                                    transformer &&
                                    transformerSettingKeys.has(k)
                                  ) {
                                    setTransformerSetting(
                                      n.data,
                                      k,
                                      next,
                                      fieldVariables,
                                    );
                                    return;
                                  }
                                  n.data.params = parametersSchema.parse({
                                    ...n.data.params,
                                    [k]: next,
                                  });
                                  if (k === "pattern")
                                    n.data.params = changePattern(
                                      n.data.params,
                                      next,
                                    );
                                })
                              }
                            />
                            {transformer &&
                              k === "ffn_width" &&
                              localFfnLinked && (
                                <small className="field-help">
                                  Local variable inside this Transformer; its
                                  FFN uses <code>$ffn_width</code>.
                                </small>
                              )}
                            {typeof v === "number" &&
                              !["index", "experts"].includes(k) &&
                              typeof transformerParentScope(
                                selected.data,
                                variableScope.bindings,
                              )[
                                k === "hidden" && selected.data.kind === "ffn"
                                  ? "ffn_width"
                                  : k
                              ] === "number" &&
                              !inputOwnership &&
                              ((k === "ffn_width" && transformer) ||
                                !containerParameterShadowReason(
                                  wrappedTransformer(selected.data)?.data ??
                                    selected.data,
                                  k,
                                )) && (
                                <button
                                  className="small-button"
                                  onClick={() =>
                                    updateNode((node) => {
                                      const variable =
                                        k === "hidden" &&
                                        selected.data.kind === "ffn"
                                          ? "ffn_width"
                                          : k;
                                      if (
                                        transformer &&
                                        transformerSettingKeys.has(k)
                                      )
                                        setTransformerSetting(
                                          node.data,
                                          k,
                                          "$" + variable,
                                          fieldVariables,
                                        );
                                      else node.data.params[k] = "$" + variable;
                                    })
                                  }
                                >
                                  {transformer &&
                                  k === "ffn_width" &&
                                  localFfnLinked
                                    ? "Local value"
                                    : "Independent value"}{" "}
                                  · Link to parent $
                                  {k === "hidden" &&
                                  selected.data.kind === "ffn"
                                    ? "ffn_width"
                                    : k}{" "}
                                  (
                                  {String(
                                    transformerParentScope(
                                      selected.data,
                                      variableScope.bindings,
                                    )[
                                      k === "hidden" &&
                                      selected.data.kind === "ffn"
                                        ? "ffn_width"
                                        : k
                                    ],
                                  )}
                                  )
                                </button>
                              )}
                            {k === "width" && (
                              <p className="field-help">
                                Model width: the number of features in each
                                token representation. Connected layers must
                                agree on this size.
                              </p>
                            )}
                            {k === "hidden" && selected.data.kind === "ffn" && (
                              <p className="field-help">
                                Feed-forward hidden width: the intermediate
                                number of features inside the FFN. The FFN
                                projects back to the model width afterwards.
                              </p>
                            )}
                            {typeof v === "string" && v.startsWith("$") && (
                              <p className="binding-resolution field-help">
                                {v} ={" "}
                                {String(
                                  bindingValue(v, selectedFieldVariables) ??
                                    "undefined · add it in Graph variables",
                                )}
                              </p>
                            )}
                          </React.Fragment>
                        );
                      })}
  </>;
}
