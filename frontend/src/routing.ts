/** Orthogonal routes through free corridors between measured node rectangles. */
import type { Node, XYPosition } from "@xyflow/react";

export interface Rectangle {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}
export interface Route {
  points: XYPosition[];
  path: string;
  label: XYPosition;
  detour: boolean;
  blocked: boolean;
}
type Axis = 0 | 1 | 2;
const CLEARANCE = 18;
const EPSILON = 0.05;

export function nodeRectangles(nodes: Node[]): Rectangle[] {
  return nodes
    .filter((n) => !n.hidden)
    .map((n) => ({
      id: n.id,
      ...n.position,
      width: n.measured?.width ?? n.width ?? (n.data.body ? 300 : 248),
      height: n.measured?.height ?? n.height ?? 240,
    }));
}
function inflate(r: Rectangle, padding: number): Rectangle {
  return {
    ...r,
    x: r.x - padding,
    y: r.y - padding,
    width: r.width + padding * 2,
    height: r.height + padding * 2,
  };
}
function inside(p: XYPosition, r: Rectangle) {
  return (
    p.x > r.x + EPSILON &&
    p.x < r.x + r.width - EPSILON &&
    p.y > r.y + EPSILON &&
    p.y < r.y + r.height - EPSILON
  );
}
export function segmentBlocked(
  a: XYPosition,
  b: XYPosition,
  rectangles: Rectangle[],
): boolean {
  return rectangles.some((r) =>
    Math.abs(a.y - b.y) < EPSILON
      ? a.y > r.y + EPSILON &&
        a.y < r.y + r.height - EPSILON &&
        Math.max(a.x, b.x) > r.x + EPSILON &&
        Math.min(a.x, b.x) < r.x + r.width - EPSILON
      : Math.abs(a.x - b.x) < EPSILON &&
        a.x > r.x + EPSILON &&
        a.x < r.x + r.width - EPSILON &&
        Math.max(a.y, b.y) > r.y + EPSILON &&
        Math.min(a.y, b.y) < r.y + r.height - EPSILON,
  );
}
function simplify(points: XYPosition[]) {
  const result: XYPosition[] = [];
  for (const p of points) {
    const last = result.at(-1),
      before = result.at(-2);
    if (
      last &&
      Math.abs(last.x - p.x) < EPSILON &&
      Math.abs(last.y - p.y) < EPSILON
    )
      continue;
    if (
      before &&
      last &&
      ((before.x === last.x && last.x === p.x) ||
        (before.y === last.y && last.y === p.y))
    )
      result.pop();
    result.push(p);
  }
  return result;
}

/** A tiny binary heap keeps searches proportional to the occupied part of the grid. */
class Queue {
  private items: { key: number; priority: number }[] = [];
  push(value: { key: number; priority: number }) {
    const items = this.items;
    items.push(value);
    let i = items.length - 1;
    while (i) {
      const parent = (i - 1) >> 1;
      if (items[parent].priority <= value.priority) break;
      items[i] = items[parent];
      i = parent;
    }
    items[i] = value;
  }
  pop() {
    const items = this.items,
      first = items[0],
      last = items.pop();
    if (!items.length || !last) return first;
    let i = 0;
    while (i * 2 + 1 < items.length) {
      let child = i * 2 + 1;
      if (
        child + 1 < items.length &&
        items[child + 1].priority < items[child].priority
      )
        child++;
      if (items[child].priority >= last.priority) break;
      items[i] = items[child];
      i = child;
    }
    items[i] = last;
    return first;
  }
  get length() {
    return this.items.length;
  }
}

function search(
  start: XYPosition,
  end: XYPosition,
  rectangles: Rectangle[],
  padding: number,
): XYPosition[] | null {
  const obstacles = rectangles.map((r) => inflate(r, padding));
  if (obstacles.some((r) => inside(start, r) || inside(end, r))) return null;
  const xs = [
    ...new Set([
      start.x,
      end.x,
      ...obstacles.flatMap((r) => [r.x, r.x + r.width]),
    ]),
  ].sort((a, b) => a - b);
  const ys = [
    ...new Set([
      start.y,
      end.y,
      ...obstacles.flatMap((r) => [r.y, r.y + r.height]),
    ]),
  ].sort((a, b) => a - b);
  const width = xs.length,
    height = ys.length;
  const coordinate = (index: number): XYPosition => ({
    x: xs[index % width],
    y: ys[Math.floor(index / width)],
  });
  const source = ys.indexOf(start.y) * width + xs.indexOf(start.x),
    target = ys.indexOf(end.y) * width + xs.indexOf(end.x);
  const distances = new Float64Array(width * height * 3).fill(Infinity),
    previous = new Int32Array(width * height * 3).fill(-1),
    closed = new Uint8Array(width * height * 3);
  const blocked = new Map<number, boolean>(),
    links = new Map<string, boolean>();
  const free = (index: number) => {
    let value = blocked.get(index);
    if (value === undefined) {
      value = obstacles.some((r) => inside(coordinate(index), r));
      blocked.set(index, value);
    }
    return !value;
  };
  const queue = new Queue();
  distances[source * 3] = 0;
  queue.push({ key: source * 3, priority: 0 });
  while (queue.length) {
    const item = queue.pop();
    if (!item || closed[item.key]) continue;
    const key = item.key,
      index = Math.floor(key / 3),
      axis = (key % 3) as Axis;
    closed[key] = 1;
    if (index === target) {
      const points: XYPosition[] = [];
      for (let k = key; k >= 0; k = previous[k])
        points.push(coordinate(Math.floor(k / 3)));
      return simplify(points.reverse());
    }
    const x = index % width,
      y = Math.floor(index / width),
      point = coordinate(index);
    for (const [nx, ny, direction] of [
      [x - 1, y, 1],
      [x + 1, y, 1],
      [x, y - 1, 2],
      [x, y + 1, 2],
    ]) {
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const next = ny * width + nx,
        nextKey = next * 3 + direction;
      if (closed[nextKey] || !free(next)) continue;
      const link = index < next ? index + ":" + next : next + ":" + index;
      let clear = links.get(link);
      const endpoint = coordinate(next);
      if (clear === undefined) {
        clear = !segmentBlocked(point, endpoint, obstacles);
        links.set(link, clear);
      }
      if (!clear) continue;
      const distance =
        distances[key] +
        Math.abs(point.x - endpoint.x) +
        Math.abs(point.y - endpoint.y) +
        (axis && axis !== direction ? 32 : 0);
      if (distance >= distances[nextKey]) continue;
      distances[nextKey] = distance;
      previous[nextKey] = key;
      queue.push({
        key: nextKey,
        priority:
          distance +
          Math.abs(end.x - endpoint.x) +
          Math.abs(end.y - endpoint.y),
      });
    }
  }
  return null;
}

export function roundedPath(points: XYPosition[], maxRadius = 8): string {
  if (!points.length) return "";
  let path = `M${points[0].x},${points[0].y}`;
  for (let i = 1; i < points.length - 1; i++) {
    const a = points[i - 1],
      b = points[i],
      c = points[i + 1];
    const before = Math.abs(a.x - b.x) + Math.abs(a.y - b.y),
      after = Math.abs(b.x - c.x) + Math.abs(b.y - c.y),
      radius = Math.min(maxRadius, before / 2, after / 2);
    const entry = {
        x: b.x + ((a.x - b.x) * radius) / before,
        y: b.y + ((a.y - b.y) * radius) / before,
      },
      exit = {
        x: b.x + ((c.x - b.x) * radius) / after,
        y: b.y + ((c.y - b.y) * radius) / after,
      };
    path += ` L${entry.x},${entry.y} Q${b.x},${b.y} ${exit.x},${exit.y}`;
  }
  const end = points.at(-1);
  return end ? path + ` L${end.x},${end.y}` : path;
}
function labelPosition(
  points: XYPosition[],
  rectangles: Rectangle[],
): XYPosition {
  const candidates = points
    .slice(1)
    .map((b, i) => {
      const a = points[i];
      return {
        point: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
        length: Math.abs(a.x - b.x) + Math.abs(a.y - b.y),
        horizontal: a.y === b.y,
      };
    })
    .sort(
      (a, b) =>
        (b.horizontal ? 80 : 0) +
        b.length -
        ((a.horizontal ? 80 : 0) + a.length),
    );
  return (
    candidates.find(
      (c) => !rectangles.some((r) => inside(c.point, inflate(r, 12))),
    )?.point ??
    candidates[0]?.point ??
    points[0]
  );
}

export function routeConnection(
  source: XYPosition,
  target: XYPosition,
  rectangles: Rectangle[],
  sourceId: string,
  targetId: string,
  lane = 0,
  weight = false,
): Route {
  const unrelated = rectangles.filter(
    (r) => r.id !== sourceId && r.id !== targetId,
  );
  const freeRay = (point: XYPosition, axis: "x" | "y", direction: number) => {
    let distance = Infinity;
    for (const r of unrelated) {
      const spans =
        axis === "x"
          ? point.y > r.y && point.y < r.y + r.height
          : point.x > r.x && point.x < r.x + r.width;
      const near =
        axis === "x"
          ? direction > 0
            ? r.x
            : r.x + r.width
          : direction > 0
            ? r.y
            : r.y + r.height;
      if (spans && (near - point[axis]) * direction >= 0)
        distance = Math.min(distance, (near - point[axis]) * direction);
    }
    return Math.max(
      0.25,
      Math.min(CLEARANCE + (lane % 4) * 10 + 12, distance / 2),
    );
  };
  const start = weight
    ? { x: source.x, y: source.y - freeRay(source, "y", -1) }
    : { x: source.x + freeRay(source, "x", 1), y: source.y };
  const end = weight
    ? { x: target.x, y: target.y - freeRay(target, "y", -1) }
    : { x: target.x - freeRay(target, "x", -1), y: target.y };
  const padding = CLEARANCE + (lane % 4) * 10;
  const stubsBlocked =
    segmentBlocked(source, start, unrelated) ||
    segmentBlocked(end, target, unrelated);
  let middle: XYPosition[] | null = null;
  // Clear adjacent connections need no pathfinding. Real skips use the same planner as arbitrary user wiring.
  for (const points of [
    [start, { x: end.x, y: start.y }, end],
    [start, { x: start.x, y: end.y }, end],
  ]) {
    if (
      points.slice(1).every(
        (p, i) =>
          !segmentBlocked(
            points[i],
            p,
            rectangles.map((r) => inflate(r, padding)),
          ),
      )
    ) {
      middle = points;
      break;
    }
  }
  let cornerRadius = Math.min(8, padding);
  // Tight hand-positioned cards can use smaller corridors; corners shrink with them.
  for (const clearance of [padding, 6, 2, 0.25]) {
    if (middle) break;
    middle = search(start, end, rectangles, clearance);
    if (middle) cornerRadius = Math.min(8, clearance);
  }
  const blocked = !middle || stubsBlocked;
  const points = simplify([source, ...(middle ?? [start, end]), target]);
  return {
    points,
    path: roundedPath(points, cornerRadius),
    label: labelPosition(points, rectangles),
    detour: points.length > 4 || target.x < source.x,
    blocked,
  };
}
