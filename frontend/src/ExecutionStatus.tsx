import type { RunDetail } from "./trainingApi";

export function ExecutionStatus({ detail }: { detail: RunDetail }) {
  const actual = detail.events
    .filter(
      (event) => event.kind === "configuration" && event.channel === "training",
    )
    .at(-1)?.payload;
  if (
    detail.run.recipe === "custom" &&
    (!actual || !Array.isArray(actual.devices))
  )
    return (
      <p className="training-muted" aria-label="Effective training devices">
        The custom script has not reported its applied device settings.
      </p>
    );
  const devices = Array.isArray(actual?.devices)
    ? actual.devices.filter(
        (device): device is number => typeof device === "number",
      )
    : detail.config.devices;
  const strategy =
    (actual?.strategy ?? detail.config.strategy) === "auto"
      ? devices.length > 1
        ? "ddp"
        : "single"
      : String(actual?.strategy ?? detail.config.strategy);
  const precision = String(
    actual?.precision ?? detail.config.precision,
  ).toUpperCase();
  const progress = detail.events
    .filter((event) => event.kind === "progress")
    .at(-1)?.payload;
  return (
    <p
      className="training-execution-status"
      aria-label="Effective training devices"
    >
      <strong>
        {strategy === "ddp"
          ? "DDP"
          : strategy === "split"
            ? "Split model"
            : "Single device"}
      </strong>
      {" · "}
      {devices.length
        ? devices.map((device) => `GPU ${device}`).join(" + ")
        : "CPU"}
      {" · "}
      {precision}
      {typeof progress?.world_size === "number" &&
        strategy === "ddp" &&
        ` · ${progress.world_size} workers`}
    </p>
  );
}
