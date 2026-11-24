import type { Edge } from "@xyflow/react";
import { outputPorts, edgeContract } from "../lanePorts.ts";
import type { Graph, GraphNode, ValidationReport } from "../types";

const copy = <T>(value: T): T => structuredClone(value);

export function groupSelection(
  graph: Graph,
  selection: string | null,
  report: ValidationReport | null,
  prefix: string,
  uid: () => string,
  changeGraph: (change: (graph: Graph) => void) => void,
  setSelection: (id: string) => void,
  notify: (message: string) => void,
) {
  const ids = new Set(
    graph.nodes
      .filter((n: GraphNode) => n.selected)
      .map((n: GraphNode) => n.id),
  );
  if (!ids.size && selection) ids.add(selection);
  if (!ids.size) {
    notify("Ctrl/Cmd-click nodes or drag a selection box, then create a block");
    return;
  }
  const outgoing = graph.edges.filter(
    (e) => ids.has(e.source) && !ids.has(e.target),
  );
  const outputStreams = [
    ...new Map(
      outgoing.map((e) => [
        JSON.stringify([e.source, e.sourceHandle || "out"]),
        e,
      ]),
    ).values(),
  ];
  const outputIds = outputStreams.map((e) => e.source);
  if (outputIds.length > 1) {
    notify(
      "A reusable block needs one output stream. Select nodes with a single outgoing stream.",
    );
    return;
  }
  const incoming = graph.edges.filter(
    (e) => !ids.has(e.source) && ids.has(e.target),
  );
  const selectedNodes = graph.nodes.filter((n: GraphNode) => ids.has(n.id));
  if (
    selectedNodes.some((n: GraphNode) =>
      ["input", "output"].includes(n.data.kind),
    )
  ) {
    notify(
      "Select computation nodes; keep graph Input and Output nodes outside the block.",
    );
    return;
  }
  const dangling = selectedNodes.filter(
    (n: GraphNode) =>
      !graph.edges.some((e) => e.source === n.id && ids.has(e.target)),
  );
  const output = outputIds[0] || dangling[0]?.id;
  if (!output) {
    notify("Select a subgraph with an output");
    return;
  }
  const outputNode = selectedNodes.find((node) => node.id === output);
  const outputHandle =
    outputStreams[0]?.sourceHandle ||
    (outputNode ? outputPorts(outputNode.data)[0] : "out");
  if (
    !outputHandle ||
    (!outputStreams.length &&
      outputNode &&
      outputPorts(outputNode.data).length > 1)
  ) {
    notify(
      "Connect a single outgoing lane or include a Join lanes node before creating a block.",
    );
    return;
  }
  const id = uid();
  const minX = Math.min(...selectedNodes.map((n: GraphNode) => n.position.x)),
    minY = Math.min(...selectedNodes.map((n: GraphNode) => n.position.y));
  const body: Graph = {
    nodes: selectedNodes.map((n: GraphNode) => ({
      ...copy(n),
      selected: false,
      position: { x: n.position.x - minX + 300, y: n.position.y - minY + 80 },
    })),
    edges: copy(
      graph.edges.filter((e) => ids.has(e.source) && ids.has(e.target)),
    ),
  };
  const external = new Map<string, number>();
  const newIn: Edge[] = [];
  incoming.forEach((e) => {
    const key = JSON.stringify([e.source, e.sourceHandle || "out"]);
    if (!external.has(key)) {
      const i = external.size;
      external.set(key, i);
      const shape = edgeContract(report, prefix, e);
      body.nodes.push({
        id: "in" + i,
        type: "architecture",
        position: { x: 0, y: i * 230 },
        data: {
          kind: "input",
          label: "Input " + String.fromCharCode(97 + i),
          params: {
            index: i,
            shape: shape?.shape || ["B", "T", "$width"],
            dtype: shape?.dtype || "float32",
          },
        },
      });
      newIn.push({
        ...e,
        id: uid(),
        target: id,
        targetHandle: String.fromCharCode(97 + i),
      });
    }
    body.edges.push({
      ...e,
      id: uid(),
      source: "in" + external.get(key),
      sourceHandle: "out",
    });
  });
  body.nodes.push({
    id: "result",
    type: "architecture",
    position: {
      x: Math.max(...body.nodes.map((n: GraphNode) => n.position.x)) + 350,
      y: 120,
    },
    data: { kind: "output", label: "Block output", params: { index: 0 } },
  });
  body.edges.push({
    id: uid(),
    source: output,
    target: "result",
    targetHandle: "a",
    sourceHandle: outputHandle,
    type: "smoothstep",
  });
  const block: GraphNode = {
    id,
    type: "architecture",
    position: { x: minX, y: minY },
    data: { kind: "block", label: "Custom block", params: {}, body },
  };
  changeGraph((g) => {
    g.nodes = [...g.nodes.filter((n: GraphNode) => !ids.has(n.id)), block];
    g.edges = [
      ...g.edges.filter((e) => !ids.has(e.source) && !ids.has(e.target)),
      ...newIn,
      ...outgoing.map((e) => ({ ...e, source: id, sourceHandle: "out" })),
    ];
  });
  setSelection(id);
  notify("Reusable block created · open to edit its internals");
}
