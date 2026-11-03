import { z } from "zod";

export const toolsSchema = z
  .array(
    z
      .object({
        type: z.literal("function"),
        function: z
          .object({
            name: z
              .string()
              .min(1)
              .max(64)
              .regex(/^[^\x00-\x1f\x7f]+$/),
            parameters: z.record(z.string(), z.json()).optional(),
          })
          .passthrough(),
      })
      .passthrough(),
  )
  .max(128)
  .refine(
    (tools) =>
      new Set(tools.map((tool) => tool.function.name)).size === tools.length,
    "Tool names must be unique.",
  );
export const callSchema = z.object({
  name: z.string(),
  arguments: z.record(z.string(), z.json()),
  valid: z.boolean(),
  errors: z.array(z.string()),
});
export const parsedSchema = z.object({
  kind: z.string(),
  calls: z.array(callSchema),
  errors: z.array(z.string()),
});
export const messageSchema = z.object({
  role: z.enum(["user", "assistant", "tool"]),
  content: z.string(),
  turn_id: z.string().optional(),
  included_in_context: z.boolean().optional(),
  context_reason: z.string().nullable().optional(),
  simulated: z.boolean().optional(),
  parsed: parsedSchema.optional(),
  tool_call_id: z.string().optional(),
  name: z.string().optional(),
  tool_calls: z
    .array(
      z.object({
        id: z.string(),
        type: z.literal("function"),
        function: z.object({
          name: z.string(),
          arguments: z.record(z.string(), z.json()),
        }),
      }),
    )
    .default([]),
  generation: z
    .object({
      temperature: z.number().optional(),
      seed: z.number().optional(),
      decoder: z.string().nullable().optional(),
      tools: toolsSchema.nullable().optional(),
      system: z.string().optional(),
    })
    .passthrough()
    .optional(),
});
export const previewSchema = z.object({
  state: z.enum(["loading", "ready", "error"]),
  request_id: z.string(),
  error: z.string().optional(),
  prompt_tokens: z.number().optional(),
  history_tokens: z.number().optional(),
  reserved_reply_tokens: z.number().optional(),
  remaining_tokens: z.number().optional(),
  context_length: z.number().optional(),
});
export const suiteSchema = z.object({
  id: z.string(),
  state: z.string(),
  total: z.number(),
  results: z.array(
    z.object({
      index: z.number(),
      name: z.string(),
      output: z.string(),
      error: z.string().optional(),
      exact_match: z.boolean().nullable(),
      parse_valid: z.boolean(),
      tool_match: z.boolean().nullable().optional(),
      arguments_match: z.boolean().nullable().optional(),
      parsed: parsedSchema.optional(),
      settings: z.record(z.string(), z.json()),
    }),
  ),
});
export const optionsSchema = z.object({
  tools: toolsSchema.nullable().default(null),
  decoder: z.string().nullable().default(null),
  system: z.string().default(""),
  chat_template: z.string().default(""),
  mode: z.enum(["chat", "completion"]).default("chat"),
});
export const sessionSchema = optionsSchema.extend({
  state: z.enum([
    "unloaded",
    "loading",
    "ready",
    "generating",
    "testing",
    "error",
  ]),
  checkpoint: z.string().nullable(),
  device: z.string(),
  messages: z.array(messageSchema),
  error: z.string().nullable(),
  has_chat_template: z.boolean(),
  finish_reason: z.string().nullable(),
  loading_stage: z.string().nullable().default(null),
  loading_started: z.number().nullable().default(null),
  decoder_label: z.string().nullable().default(null),
  conversation_settings: optionsSchema.nullable().default(null),
  context_preview: previewSchema.nullable().default(null),
  idle_expires: z.number().nullable().default(null),
  notice: z.string().nullable().default(null),
  test_suite: suiteSchema.nullable().default(null),
});
export const presetSchema = z.object({
  id: z.string(),
  name: z.string(),
  tools: toolsSchema,
  system_template: z.string(),
});
export const catalogSchema = z.object({
  supports_cuda: z.boolean().default(false),
  checkpoints: z.array(
    z.object({
      path: z.string(),
      step: z.number(),
      validation_loss: z.number().nullable(),
      available: z.boolean(),
      task_metrics: z.record(z.string(), z.number()).default({}),
    }),
  ),
  recommended_checkpoint: z.string().nullable(),
  recommendation: z.string(),
  problem: z.string().nullable(),
  session: sessionSchema,
  preferences: optionsSchema.optional(),
  decoders: z
    .array(
      z.object({
        id: z.string(),
        label: z.string(),
        available: z.boolean(),
        schema_profile: z.string().default("generic"),
      }),
    )
    .default([]),
  presets: z.array(presetSchema).default([]),
});
export const diagnosticsSchema = z.object({
  errors: z.array(z.string()),
  warnings: z.array(z.string()),
});
export type ChatSession = z.infer<typeof sessionSchema>;
export type ChatMessage = z.infer<typeof messageSchema>;

export function liveSystem(template: string, now = new Date()) {
  const pad = (n: number) => String(n).padStart(2, "0");
  const stamp = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}T${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
  const weekday = new Intl.DateTimeFormat("en", { weekday: "long" }).format(
    now,
  );
  return template
    .replaceAll("{datetime}", stamp)
    .replaceAll("{weekday}", weekday);
}
export function downloadJsonl(filename: string, records: unknown[]) {
  const url = URL.createObjectURL(
    new Blob([records.map((r) => JSON.stringify(r)).join("\n") + "\n"], {
      type: "application/x-ndjson",
    }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
