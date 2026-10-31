import { z } from "zod";

export const tuningCandidateSchema = z.object({
  strategy: z.enum(["single", "ddp", "split"]),
  status: z.enum(["completed", "failed"]),
  devices: z.array(z.number()),
  precision: z.enum(["fp32", "fp16", "bf16"]),
  batch_sizes: z.array(z.number()),
  placement: z.record(z.string(), z.number()),
  global_batch_size: z.number(),
  sequence_length: z.number(),
  step_seconds: z.number().optional(),
  tokens_per_second: z.number().optional(),
  min_seconds: z.number().optional(),
  max_seconds: z.number().optional(),
  peak_memory_bytes: z.record(z.string(), z.number()).optional(),
  error: z.string().optional(),
});
export type TuningCandidate = z.infer<typeof tuningCandidateSchema>;
export const tuningJobSchema = z.object({
  id: z.string(),
  state: z.string(),
  message: z.string(),
  results: z.array(tuningCandidateSchema).default([]),
  logs: z.string().default(""),
});
export type TuningJob = z.infer<typeof tuningJobSchema>;
export const tuningListSchema = z.object({
  settings: z.object({ auto_apply: z.boolean() }),
  profiles: z.array(
    z.object({
      id: z.string(),
      created: z.number(),
      model_name: z.string(),
      model_signature: z.string(),
      devices: z.array(z.number()),
      python: z.string(),
      sequence_length: z.number(),
      global_batch_size: z.number(),
      initial_speed_ratios: z.array(z.number()),
      initial_estimate: z
        .enum(["standalone", "equal-fallback"])
        .default("standalone"),
      duration_seconds: z.number().optional(),
      best: z.record(z.string(), tuningCandidateSchema),
    }),
  ),
  jobs: z.array(
    z.object({ id: z.string(), state: z.string(), message: z.string() }),
  ),
});
