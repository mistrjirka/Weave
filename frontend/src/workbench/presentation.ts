import { Activity, ArrowRight, Box, Braces, CircleAlert, Code2, Database, GitBranch, Layers3, LayoutGrid, RotateCcw, Shapes, Sparkles, Workflow, type LucideIcon } from "lucide-react";

export const icons: Record<string, LucideIcon> = {
  input: Braces,
  output: ArrowRight,
  embedding: Database,
  attention: GitBranch,
  cross_attention: GitBranch,
  ffn: Layers3,
  ngram: Workflow,
  lookup: Database,
  loop: RotateCcw,
  block: Box,
  norm: Activity,
  head: LayoutGrid,
  mask: Shapes,
  select: Sparkles,
  custom: Code2,
  reference: CircleAlert,
  attention_pattern: LayoutGrid,
  pattern_union: GitBranch,
  pattern_intersection: GitBranch,
};
export const family = (kind: string) =>
  ["embedding", "lookup", "engram_lookup", "memory_fuse"].includes(kind)
    ? "mint"
    : [
          "ngram",
          "hadamard",
          "engram_hash",
          "attention_pattern",
          "pattern_union",
          "pattern_intersection",
        ].includes(kind)
      ? "amber"
      : [
            "attention",
            "cross_attention",
            "ffn",
            "block",
            "monarch",
            "sdpa",
            "rope",
            "split_heads",
            "merge_heads",
            "repeat_heads",
            "causal_conv",
          ].includes(kind)
        ? "violet"
        : [
              "loop",
              "lane_fork",
              "lane_split",
              "lane_join",
              "lane_expand",
              "lane_reduce",
              "routing",
              "lane_mix",
              "lane_merge",
            ].includes(kind)
          ? "blue"
          : ["mask", "select"].includes(kind)
            ? "coral"
            : ["reference"].includes(kind)
              ? "red"
              : "neutral";
