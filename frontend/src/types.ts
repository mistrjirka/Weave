import type { Node, Edge } from "@xyflow/react";

export type JsonValue =
  string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
export type Dimension = string | number;
export interface Parameters {
  [key: string]: JsonValue | undefined;
  shape?: Dimension[];
  input_shapes?: Dimension[][];
  orders?: number[];
  global_tokens?: number[];
  matrix?: boolean[][];
  index?: number;
  optional?: boolean;
  pattern?: string;
  dtype?: string;
  source?: string;
  shared?: boolean;
  causal?: boolean;
  window?: Dimension;
  global_stride?: Dimension;
  block_size?: Dimension;
  neighbor_blocks?: Dimension;
  seed?: Dimension;
  density?: Dimension;
  iterations?: Dimension;
  heads?: Dimension;
  kv_heads?: Dimension;
  lanes?: Dimension;
  tie?: string;
}
export interface NodeData extends Record<string, unknown> {
  kind: string;
  label: string;
  params: Parameters;
  body?: Graph;
  share?: string;
  role?: string;
  description?: string;
  iteration?: { index: number; total: number };
}
export type GraphNode = Node<NodeData, "architecture">;
export interface Graph {
  layout?: import("./layout-core").GraphLayout;
  nodes: GraphNode[];
  edges: Edge[];
}
export interface TensorContract {
  shape: Dimension[];
  dtype: string;
}
export interface ValidationIssue {
  node?: string | null;
  message: string;
}
export interface ValidationReport {
  valid: boolean;
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
  shapes: Record<string, TensorContract | null>;
  counts: Record<string, number>;
  parameters: number;
  outputs: (TensorContract | null)[];
}
export interface TokenizerInfo {
  mask_id?: number | null;
  id: string;
  repo?: string | null;
  class?: string;
  size: number;
  capacity: number;
  base_size: number;
  special_tokens: Record<string, { token: string; id: number | null }>;
}
export interface Project {
  version: number;
  id: string;
  name: string;
  description: string;
  config: Record<string, number>;
  graph: Graph;
  tokenizer?: TokenizerInfo | null;
  references?: {
    title: string;
    url: string;
    notes?: string;
    revision?: string;
    config?: Record<string, JsonValue>;
  }[];
  updated?: string;
}
export interface CatalogEntry {
  kind: string;
  label: string;
  description: string;
  category: string;
  params: Parameters;
  role?: string;
}
export interface NodeLocation {
  node: GraphNode;
  path: string;
}
export interface DisplayData extends NodeData {
  _ports: string[];
  _optional: string[];
  _role: string;
  _groupLabel?: string;
  _shape?: TensorContract | null;
  _count?: number;
  _error?: boolean;
  _open: (id: string) => void;
}
export type DisplayNode = Node<DisplayData, "architecture">;
export interface TokenPreview {
  tokens: string[];
  ids: number[];
  shape: number[];
}
