import { z } from "zod";
import type { TrainingEvent } from "./trainingApi";

const applicationSchema = z.object({
  applied: z.boolean(),
  message: z.string(),
  profile: z
    .object({
      id: z.string(),
      strategy: z.string(),
      sequence_length: z.number(),
      batch_sizes: z.record(z.string(), z.number()),
      placement: z.record(z.string(), z.number()),
    })
    .nullable(),
});
export function GPUTuningStatus({ events }: { events: TrainingEvent[] }) {
  const event = events.filter((item) => item.kind === "gpu_tuning").at(-1);
  if (!event) return null;
  const result = applicationSchema.safeParse(event.payload);
  if (!result.success) return null;
  const { applied, profile, message } = result.data;
  return (
    <section
      className="training-server-files"
      aria-label="GPU tuning allocation"
    >
      <strong>
        {applied ? "Saved GPU tuning applied" : "Saved GPU tuning skipped"}
      </strong>
      {applied && profile && (
        <p>
          {Object.entries(
            profile.strategy === "ddp"
              ? profile.batch_sizes
              : profile.placement,
          )
            .map(([key, value]) =>
              profile.strategy === "ddp"
                ? `GPU ${key}: ${value} sequences per microbatch`
                : `${key} → GPU ${value}`,
            )
            .join(" · ")}
        </p>
      )}
      {message && <p>{message}</p>}
      {profile && (
        <small>
          Benchmarked at {profile.sequence_length.toLocaleString()} tokens per
          sequence. Other lengths and data pipelines can change performance.
        </small>
      )}
    </section>
  );
}
