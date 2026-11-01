import test from "node:test";
import assert from "node:assert/strict";
import { parametersSchema } from "../frontend/src/api.ts";

test("boolean settings accept explicit parameter references and survive serialization", () => {
  const value = {
    causal: "$causal",
    rope: "$rope",
    backend: "$backend",
    rope_scaling: "$rope_scaling",
  };
  assert.deepEqual(
    parametersSchema.parse(JSON.parse(JSON.stringify(value))),
    value,
  );
  assert.equal(parametersSchema.safeParse({ causal: false }).success, true);
  assert.equal(parametersSchema.safeParse({ causal: "maybe" }).success, false);
  assert.equal(parametersSchema.safeParse({ causal: "$" }).success, false);
});
