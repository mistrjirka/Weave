import test from "node:test";
import assert from "node:assert/strict";
import { routeConnection, segmentBlocked } from "../frontend/src/routing.ts";
import type { Rectangle } from "../frontend/src/routing.ts";
const rect = (
  id: string,
  x: number,
  y = 0,
  width = 248,
  height = 220,
): Rectangle => ({ id, x, y, width, height });
function clear(
  route: ReturnType<typeof routeConnection>,
  rectangles: Rectangle[],
) {
  assert.equal(route.blocked, false);
  for (let i = 1; i < route.points.length; i++)
    assert.equal(
      segmentBlocked(route.points[i - 1], route.points[i], rectangles),
      false,
      `Segment ${i} crosses a node`,
    );
  assert(!/NaN|Infinity/.test(route.path));
}

test("residual jumps route outside every intervening card", () => {
  const rectangles = [
    rect("x", 0),
    rect("norm", 368),
    rect("attention", 736, 0, 248, 300),
    rect("out", 1104),
  ];
  const route = routeConnection(
    { x: 258.5, y: 69 },
    { x: 1093.5, y: 97 },
    rectangles,
    "x",
    "out",
  );
  clear(route, rectangles.slice(1, -1));
  assert(route.detour);
  assert(route.points.some((p) => p.y < 0 || p.y > 300));
});
test("backward connections and weight-sharing links avoid cards", () => {
  const rectangles = [rect("a", 0), rect("middle", 368), rect("b", 736)];
  clear(
    routeConnection(
      { x: 994.5, y: 69 },
      { x: -10.5, y: 97 },
      rectangles,
      "b",
      "a",
    ),
    rectangles,
  );
  const weight = routeConnection(
    { x: 136.4, y: 0 },
    { x: 847.6, y: 0 },
    rectangles,
    "a",
    "b",
    0,
    true,
  );
  clear(weight, rectangles);
  assert(weight.points.some((p) => p.y < 0));
});
test("legacy tight gaps remain routable", () => {
  const rectangles = [
    rect("x", 0),
    rect("norm", 260),
    rect("attention", 520),
    rect("out", 1040),
  ];
  const route = routeConnection(
    { x: 258.5, y: 69 },
    { x: 1029.5, y: 69 },
    rectangles,
    "x",
    "out",
  );
  clear(route, rectangles.slice(1, -1));
});
test("moving an unrelated node onto a connection reroutes it", () => {
  const a = rect("a", 0),
    b = rect("b", 900);
  const before = routeConnection(
    { x: 258.5, y: 69 },
    { x: 889.5, y: 69 },
    [a, b],
    "a",
    "b",
  );
  const obstacle = rect("obstacle", 450, -50);
  const after = routeConnection(
    { x: 258.5, y: 69 },
    { x: 889.5, y: 69 },
    [a, b, obstacle],
    "a",
    "b",
  );
  assert.notEqual(after.path, before.path);
  clear(after, [obstacle]);
});
test("an impossible overlap is flagged instead of claiming a clear route", () => {
  const rectangles = [rect("a", 0), rect("block", 240), rect("b", 700)];
  assert(
    routeConnection(
      { x: 258.5, y: 69 },
      { x: 689.5, y: 69 },
      rectangles,
      "a",
      "b",
    ).blocked,
  );
});
test("fan-out uses separate corridors without touching cards", () => {
  const rectangles = [rect("x", 0), rect("middle", 368), rect("out", 736)];
  const paths = new Set<string>();
  for (let lane = 0; lane < 3; lane++) {
    const route = routeConnection(
      { x: 258.5, y: 69 },
      { x: 725.5, y: 97 + lane * 28 },
      rectangles,
      "x",
      "out",
      lane,
    );
    clear(route, [rectangles[1]]);
    paths.add(route.path);
  }
  assert.equal(paths.size, 3);
});
