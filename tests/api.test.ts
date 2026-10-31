import { test } from "node:test";
import assert from "node:assert/strict";
import { api } from "../frontend/src/api.ts";

for (const value of [false, 0, "", null]) {
  test(`API preserves the JSON body ${JSON.stringify(value)}`, async (t) => {
    let sent: RequestInit | undefined;
    t.mock.method(
      globalThis,
      "fetch",
      async (_input: string, init: RequestInit) => {
        sent = init;
        return new Response("{}", {
          headers: { "Content-Type": "application/json" },
        });
      },
    );
    await api("/test", value);
    assert.equal(sent?.method, "POST");
    assert.equal(sent?.body, JSON.stringify(value));
  });
}
