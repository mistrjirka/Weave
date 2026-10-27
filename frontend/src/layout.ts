import ELK from "elkjs/lib/elk-api";
import ElkWorker from "elkjs/lib/elk-worker.min.js?worker";
import type { Graph } from "./types";
import { buildLayoutGraph, applyLayout } from "./layout-core";
import type { LayoutLabel } from "./layout-core";

let engine: InstanceType<typeof ELK> | undefined;
export async function arrangeGraph(
  graph: Graph,
  labels: Map<string, LayoutLabel>,
): Promise<Graph> {
  engine ??= new ELK({ workerFactory: () => new ElkWorker() });
  return applyLayout(
    graph,
    await engine.layout(buildLayoutGraph(graph, labels)),
  );
}
