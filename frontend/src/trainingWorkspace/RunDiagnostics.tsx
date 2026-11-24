import type { RunDetail } from "../trainingApi";
import { TrainingChart } from "../TrainingChart";
import { TrainingMessages } from "../TrainingMessages";

const bytes = (value: number) =>
  value >= 1024 ** 3
    ? `${(value / 1024 ** 3).toFixed(2)} GB`
    : `${(value / 1024 ** 2).toFixed(1)} MB`;

export function RunDiagnostics({
  detail,
  view,
  artifactURL,
}: {
  detail: RunDetail;
  view: "hardware" | "logs";
  artifactURL: (path: string) => string;
}) {
  const latestHardware = detail.hardware.at(-1);
  return (
    <>
      {view === "hardware" && (
        <>
          <p className="training-muted">
            GPU readings are device-wide and can include other jobs.
            Host-process RAM is reported separately.
          </p>
          <div className="training-hardware-grid">
            {latestHardware?.gpus.map((gpu) => (
              <article className="training-sample" key={gpu.uuid}>
                <h4>
                  GPU {gpu.index} · {gpu.name}
                </h4>
                <p>
                  {gpu.utilization == null
                    ? "Utilization unavailable"
                    : `${gpu.utilization}% utilization`}
                </p>
                {gpu.memory_total_mb != null && gpu.memory_used_mb != null ? (
                  <>
                    <meter
                      aria-label={`GPU ${gpu.index} VRAM used`}
                      min={0}
                      max={gpu.memory_total_mb ?? 1}
                      value={gpu.memory_used_mb ?? 0}
                    />
                    <p>
                      {((gpu.memory_used_mb ?? 0) / 1024).toFixed(1)} /{" "}
                      {((gpu.memory_total_mb ?? 0) / 1024).toFixed(1)} GB VRAM
                    </p>
                  </>
                ) : (
                  <p>VRAM readings unavailable</p>
                )}
                <p>
                  {gpu.temperature ?? "—"} °C · {gpu.power_watts ?? "—"} W
                </p>
              </article>
            ))}
          </div>
          {latestHardware?.process_rss_bytes !== undefined && (
            <p>
              Training launcher RAM: {bytes(latestHardware.process_rss_bytes)} ·
              Host CPU: {latestHardware.cpu_percent?.toFixed(0) ?? "—"}%
            </p>
          )}
          <div className="training-charts">
            {latestHardware?.gpus.map((gpu) => (
              <TrainingChart
                key={gpu.uuid}
                metric={`gpu/${gpu.index}/utilization`}
                axis="time"
                smooth={false}
                events={detail.hardware.map((sample, index) => ({
                  version: 1,
                  id: String(index),
                  time: sample.time,
                  kind: "metrics",
                  step: index,
                  tokens: 0,
                  payload: {
                    [`gpu/${gpu.index}/utilization`]: sample.gpus.find(
                      (g) => g.uuid === gpu.uuid,
                    )?.utilization,
                  },
                }))}
              />
            ))}
          </div>
          {!latestHardware && (
            <p>
              No hardware samples received. Terminal-launched packages report
              script events; host monitoring requires the local supervisor.
            </p>
          )}
        </>
      )}
      {view === "logs" && (
        <>
          <TrainingMessages key={detail.run.id} detail={detail} expanded />
          <h4>Training output</h4>
          <pre className="training-log" tabIndex={0}>
            {detail.logs || "No console output yet."}
          </pre>
          {detail.evaluation_logs && (
            <>
              <h4>Evaluation worker</h4>
              <pre className="training-log" tabIndex={0}>
                {detail.evaluation_logs}
              </pre>
            </>
          )}
          <a href={artifactURL("output.log")}>Download full training log</a>
          {detail.events
            .filter(
              (event) =>
                event.kind === "profile" &&
                typeof event.payload.path === "string",
            )
            .map((event) => (
              <p key={event.id}>
                <a href={artifactURL(String(event.payload.path))}>
                  Download performance trace
                </a>{" "}
                · Open in Perfetto or Chrome tracing
              </p>
            ))}
        </>
      )}
    </>
  );
}
