import { useEffect, useState } from "react";
import { graphSchema } from "../api";
import type { NodeData } from "../types";

export function useCustomBlocks() {
  const [customBlocks, setCustomBlocks] = useState<NodeData[]>([]);
  useEffect(() => {
    try {
      const parsed: unknown = JSON.parse(localStorage.getItem("weave.blocks") || "[]");
      const result = graphSchema.safeParse({
        nodes: Array.isArray(parsed) ? parsed.map((data, i) => ({ id: String(i), type: "architecture", position: { x: 0, y: 0 }, data })) : [],
        edges: [],
      });
      if (result.success) setCustomBlocks(result.data.nodes.map((node) => node.data));
    } catch {}
  }, []);
  useEffect(() => {
    try { localStorage.setItem("weave.blocks", JSON.stringify(customBlocks)); } catch {}
  }, [customBlocks]);
  return { customBlocks, setCustomBlocks };
}
