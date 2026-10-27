import test from "node:test";
import assert from "node:assert/strict";
import {
  readWorkspaceLocation,
  workspaceURL,
} from "../frontend/src/workspaceLocation.ts";

test("split model locations and open blocks survive URL round trips", () => {
  const state = readWorkspaceLocation(
    "?model=project-saved&block=stack&block=attention%2Fcustom&node=q&view=code&split=1&reference=example%3Aneedle1&refBlock=decoder&refNode=cross&ratio=62.5&library=models&panel=properties",
  );
  const url = workspaceURL(
    "http://localhost:8173/?unrelated=keep#workspace",
    state,
  );
  assert.deepEqual(readWorkspaceLocation(new URL(url).search), state);
  assert.equal(new URL(url).searchParams.get("unrelated"), "keep");
  assert.equal(new URL(url).hash, "#workspace");
});
test("invalid view state falls back safely and pane sizes are bounded", () => {
  const invalid = readWorkspaceLocation(
    "?view=bad&split=true&reference=javascript:bad&ratio=NaN&panel=bad",
  );
  assert.equal(invalid.tab, "graph");
  assert.equal(invalid.split, false);
  assert.equal(invalid.reference, null);
  assert.equal(invalid.panel, null);
  assert.equal(invalid.ratio, 50);
  assert.equal(readWorkspaceLocation("?ratio=500").ratio, 70);
  assert.equal(readWorkspaceLocation("?ratio=-12").ratio, 30);
});
test("closed split views omit reference details and default view settings", () => {
  const state = readWorkspaceLocation(
    "?model=transformer&split=1&reference=example:needle3&refBlock=stack&ratio=70",
  );
  state.split = false;
  const url = new URL(
    workspaceURL("http://localhost/?split=1&refBlock=old&view=code", state),
  );
  assert.equal(url.search, "?model=transformer");
});

test("mobile library drawer selection survives a reload location", () => {
  const location = readWorkspaceLocation(
    "?model=needle1&library=models&panel=library",
  );
  assert.equal(location.panel, "library");
  assert.equal(location.library, "models");
  assert.deepEqual(
    readWorkspaceLocation(
      new URL(workspaceURL("http://localhost/", location)).search,
    ),
    location,
  );
});
