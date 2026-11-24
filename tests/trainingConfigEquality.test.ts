import test from "node:test";
import assert from "node:assert/strict";
import { stableJSON } from "../frontend/src/trainingConfigEquality.ts";

test("config identity ignores nested object key order but preserves array order", () => {
  assert.equal(
    stableJSON({ placement: { b: 1, a: 0 }, devices: [0, 1] }),
    stableJSON({ devices: [0, 1], placement: { a: 0, b: 1 } }),
  );
  assert.notEqual(
    stableJSON({ devices: [0, 1] }),
    stableJSON({ devices: [1, 0] }),
  );
});
