import type { JsonValue, NodeData, TensorContract } from "./types";

export interface InheritedInputSetting {
  value: JsonValue;
  disabledReason: string;
}

/** Container interface inputs receive their contracts from the parent graph.
 * Their stored shape/dtype describe defaults, not editable runtime overrides.
 */
export function inputSettingOwnership(
  kind: string,
  pathLength: number,
  key: string,
  contract: TensorContract | null | undefined,
): InheritedInputSetting | undefined {
  if (
    kind !== "input" ||
    pathLength < 1 ||
    (key !== "shape" && key !== "dtype")
  )
    return undefined;

  if (!contract)
    return {
      value: null,
      disabledReason:
        "Inherited from the parent connection. Connect this container's input and resolve upstream errors to see its shape and dtype.",
    };

  return {
    value: key === "shape" ? [...contract.shape] : contract.dtype,
    disabledReason:
      "Inherited from the parent connection. Edit the originating input or upstream layers in the parent graph to change this value.",
  };
}

/** A body's local variables take precedence over its enclosing parameters. */
export function containerParameterShadowReason(
  data: Pick<NodeData, "body">,
  key: string,
): string | undefined {
  const variables = data.body?.variables;
  if (!variables || !Object.hasOwn(variables, key)) return undefined;
  return `Overridden by $${key} in this block's Graph variables. Edit or remove that local override inside the block.`;
}
