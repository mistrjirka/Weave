import test from "node:test";
import assert from "node:assert/strict";
import ELK from "elkjs/lib/elk.bundled.js";
import {
  lanePorts,
  outputPorts,
  portLabel,
  edgeContract,
} from "../frontend/src/lanePorts.ts";
import { nodePorts } from "../frontend/src/nodePorts.ts";
import { buildLayoutGraph } from "../frontend/src/layout-core.ts";
import type {
  Graph,
  GraphNode,
  ValidationReport,
} from "../frontend/src/types.ts";

const node = (id: string, kind: string): GraphNode => ({
  id,
  type: "architecture",
  position: { x: 0, y: 0 },
  measured: { width: 248, height: 300 },
  data: { kind, label: id, params: { lanes: 3 } },
});

test("lane count controls named ports with readable labels and bounded invalid settings", () => {
  const fork = node("fork", "lane_fork"),
    join = node("join", "lane_join");
  assert.deepEqual(outputPorts(fork.data), ["lane0", "lane1", "lane2"]);
  assert.deepEqual(nodePorts(join), outputPorts(fork.data));
  assert.equal(portLabel("lane2"), "lane 3");
  assert.deepEqual(lanePorts(33), []);
  assert.deepEqual(lanePorts(2.5), []);
  assert.deepEqual(lanePorts("$lanes"), []);
  join.data.params.lanes = 5;
  assert.equal(nodePorts(join).length, 5);
});

test("auto-arrange retains distinct source and target coordinates for lane wires", async () => {
  const graph: Graph = {
    nodes: [node("fork", "lane_fork"), node("join", "lane_join")],
    edges: [0, 1, 2].map((i) => ({
      id: `lane-${i}`,
      source: "fork",
      sourceHandle: `lane${i}`,
      target: "join",
      targetHandle: `lane${i}`,
    })),
  };
  const request = buildLayoutGraph(graph, new Map());
  assert.equal(new Set(request.edges!.map((edge) => edge.sources[0])).size, 3);
  const sourcePorts = request.children![0].ports!;
  assert.equal(new Set(sourcePorts.map((port) => port.y)).size, 3);
  const result = await new ELK().layout(request);
  assert.equal(result.edges!.length, 3);
  assert(result.edges!.every((edge) => edge.sections?.length));
});

test("edge contracts use the chosen output and do not fall back for invalid ports", () => {
  const lane = { shape: ["B", "T", 8], dtype: "float32" };
  const report: ValidationReport = {
    valid: true,
    errors: [],
    warnings: [],
    shapes: { fork: lane },
    port_shapes: { fork: { lane0: lane, lane1: lane } },
    counts: {},
    parameters: 0,
    outputs: [],
  };
  assert.deepEqual(
    edgeContract(report, "", { source: "fork", sourceHandle: "lane1" }),
    lane,
  );
  assert.equal(
    edgeContract(report, "", { source: "fork", sourceHandle: "out" }),
    undefined,
  );
});
