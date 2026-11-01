import test from "node:test";
import assert from "node:assert/strict";
import {
  containerParameterShadowReason,
  inputSettingOwnership,
} from "../frontend/src/settingOwnership.ts";
import type { TensorContract } from "../frontend/src/types.ts";

const contract: TensorContract = {
  shape: ["B", "T", 512],
  dtype: "float32",
};

test("nested interface settings display the connected tensor contract", () => {
  const shape = inputSettingOwnership("input", 1, "shape", contract);
  assert.deepEqual(shape?.value, ["B", "T", 512]);
  assert.match(shape?.disabledReason ?? "", /parent connection/);
  assert.deepEqual(inputSettingOwnership("input", 3, "dtype", contract), {
    value: "float32",
    disabledReason: shape?.disabledReason,
  });
  assert.notEqual(shape?.value, contract.shape);
});

test("unresolved nested contracts never display stale stored defaults", () => {
  for (const unresolved of [null, undefined])
    for (const key of ["shape", "dtype"]) {
      const setting = inputSettingOwnership("input", 1, key, unresolved);
      assert.equal(setting?.value, null);
      assert.match(setting?.disabledReason ?? "", /upstream errors/);
    }
});

test("root inputs, other fields, and other kinds retain their normal editors", () => {
  assert.equal(inputSettingOwnership("input", 0, "shape", contract), undefined);
  assert.equal(inputSettingOwnership("input", 0, "dtype", contract), undefined);
  for (const key of ["index", "optional", "width"])
    assert.equal(inputSettingOwnership("input", 2, key, contract), undefined);
  assert.equal(inputSettingOwnership("custom", 2, "shape", contract), undefined);
  assert.equal(inputSettingOwnership("output", 2, "dtype", contract), undefined);
});

test("only explicitly shadowed container parameters get override guidance", () => {
  const data = {
    body: { nodes: [], edges: [], variables: { heads: 4, dropout: 0 } },
  };
  assert.match(containerParameterShadowReason(data, "heads") ?? "", /\$heads/);
  assert.match(containerParameterShadowReason(data, "dropout") ?? "", /override/);
  assert.equal(containerParameterShadowReason(data, "ffn_width"), undefined);
  assert.equal(containerParameterShadowReason({}, "heads"), undefined);
  assert.equal(containerParameterShadowReason(data, "toString"), undefined);
});
