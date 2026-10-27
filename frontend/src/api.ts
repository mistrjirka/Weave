import { z } from "zod";
import type {
  JsonValue,
  Graph,
  Project,
  Parameters,
  ValidationReport,
  TokenizerInfo,
  TokenPreview,
} from "./types";

const jsonValue: z.ZodType<JsonValue> = z.lazy(() =>
  z.union([
    z.string(),
    z.number(),
    z.boolean(),
    z.null(),
    z.array(jsonValue),
    z.record(z.string(), jsonValue),
  ]),
);
const dimension = z.union([z.string(), z.number()]);
export const parametersSchema: z.ZodType<Parameters> = z
  .object({
    shape: z.array(dimension).optional(),
    input_shapes: z.array(z.array(dimension)).optional(),
    orders: z.array(z.number()).optional(),
    global_tokens: z.array(z.number()).optional(),
    matrix: z.array(z.array(z.boolean())).optional(),
    index: z.number().optional(),
    optional: z.boolean().optional(),
    pattern: z.string().optional(),
    dtype: z.string().optional(),
    source: z.string().optional(),
    shared: z.boolean().optional(),
    causal: z.boolean().optional(),
    window: dimension.optional(),
    global_stride: dimension.optional(),
    block_size: dimension.optional(),
    neighbor_blocks: dimension.optional(),
    seed: dimension.optional(),
    density: dimension.optional(),
    iterations: dimension.optional(),
    heads: dimension.optional(),
    kv_heads: dimension.optional(),
    lanes: dimension.optional(),
    tie: z.string().optional(),
  })
  .catchall(jsonValue);
export const graphSchema: z.ZodType<Graph> = z.lazy(() =>
  z.object({
    layout: z
      .object({
        engine: z.literal("elk"),
        signature: z.string(),
        routes: z.record(
          z.string(),
          z.object({
            points: z.array(z.object({ x: z.number(), y: z.number() })),
            label: z.object({ x: z.number(), y: z.number() }).optional(),
          }),
        ),
      })
      .optional(),
    nodes: z.array(
      z
        .object({
          id: z.string(),
          type: z.literal("architecture"),
          position: z.object({ x: z.number(), y: z.number() }),
          data: z
            .object({
              kind: z.string(),
              label: z.string(),
              params: parametersSchema,
              body: graphSchema.optional(),
              share: z.string().optional(),
              role: z.string().optional(),
              description: z.string().optional(),
              iteration: z
                .object({ index: z.number(), total: z.number() })
                .optional(),
            })
            .passthrough(),
        })
        .passthrough(),
    ),
    edges: z.array(
      z
        .object({
          id: z.string(),
          source: z.string(),
          target: z.string(),
          sourceHandle: z.string().nullable().optional(),
          targetHandle: z.string().nullable().optional(),
          type: z.string().optional(),
        })
        .passthrough(),
    ),
  }),
);
export const tokenizerSchema: z.ZodType<TokenizerInfo> = z
  .object({
    id: z.string(),
    repo: z.string().nullable().optional(),
    class: z.string().optional(),
    size: z.number(),
    capacity: z.number(),
    base_size: z.number(),
    mask_id: z.number().nullable().optional(),
    special_tokens: z.record(
      z.string(),
      z.object({ token: z.string(), id: z.number().nullable() }),
    ),
  })
  .passthrough();
export const projectSchema: z.ZodType<Project> = z.object({
  version: z.literal(1),
  id: z.string(),
  name: z.string(),
  description: z.string(),
  config: z.record(z.string(), z.number()),
  graph: graphSchema,
  tokenizer: tokenizerSchema.nullable().optional(),
  references: z
    .array(
      z.object({
        title: z.string(),
        url: z.string(),
        revision: z.string().optional(),
        notes: z.string().optional(),
        config: z.record(z.string(), jsonValue).optional(),
      }),
    )
    .optional(),
  updated: z.string().optional(),
});
const issue = z.object({
  node: z.string().nullable().optional(),
  port: z.string().nullable().optional(),
  message: z.string(),
  code: z.string().optional(),
  title: z.string().optional(),
  suggestion: z.string().optional(),
  source: z.string().url().optional(),
});
const contract = z
  .object({ shape: z.array(dimension), dtype: z.string() })
  .passthrough();
export const reportSchema: z.ZodType<ValidationReport> = z.object({
  valid: z.boolean(),
  errors: z.array(issue),
  warnings: z.array(issue),
  shapes: z.record(z.string(), contract.nullable()),
  counts: z.record(z.string(), z.number()),
  parameters: z.number(),
  outputs: z.array(contract.nullable()),
});
export const codeSchema = z.object({
  report: reportSchema,
  source: z.string(),
});
export const unrollSchema = z.object({
  graph: graphSchema,
  first_id: z.string(),
  group: z.string().nullable(),
});
export const previewSchema: z.ZodType<TokenPreview> = z.object({
  tokens: z.array(z.string()),
  ids: z.array(z.number()),
  shape: z.array(z.number()),
});
export const projectsSchema = z.array(projectSchema);
export const savedProjectsSchema = z.array(
  z.object({
    id: z.string(),
    name: z.string(),
    description: z.string().default(""),
    updated: z.string().optional(),
  }),
);

export async function responseData(response: Response): Promise<unknown> {
  const data: unknown = await response.json();
  if (!response.ok) {
    const error = z.object({ detail: z.unknown() }).safeParse(data);
    throw new Error(
      error.success
        ? typeof error.data.detail === "string"
          ? error.data.detail
          : JSON.stringify(error.data.detail)
        : `Request failed (${response.status})`,
    );
  }
  return data;
}
export async function api(
  path: string,
  body?: unknown,
  method?: string,
): Promise<unknown> {
  return responseData(
    await fetch("/api" + path, {
      method: method || (body ? "POST" : "GET"),
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    }),
  );
}

export function describeError(error: unknown): string {
  if (error instanceof z.ZodError) {
    const issue = error.issues[0];
    return issue
      ? `${issue.path.join(".") || "Value"}: ${issue.message}`
      : "Invalid value";
  }
  return error instanceof Error ? error.message : String(error);
}
