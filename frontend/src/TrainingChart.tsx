import { useState } from "react";
import type { TrainingEvent } from "./trainingApi";

export function TrainingChart({
  events,
  comparison = [],
  metric,
  axis,
  smooth,
}: {
  events: TrainingEvent[];
  comparison?: TrainingEvent[];
  metric: string;
  axis: "step" | "tokens" | "time";
  smooth: boolean;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const points = (items: TrainingEvent[]) => {
    const start = items[0]?.time ?? 0;
    let average: number | undefined;
    return items.flatMap((event) => {
      const value = event.payload[metric];
      if (
        event.kind !== "metrics" ||
        typeof value !== "number" ||
        !Number.isFinite(value)
      )
        return [];
      average = average === undefined ? value : average * 0.8 + value * 0.2;
      return [
        {
          x: axis === "time" ? event.time - start : event[axis],
          y: smooth ? average : value,
          raw: value,
          step: event.step,
        },
      ];
    });
  };
  const current = points(events),
    other = points(comparison),
    all = [...current, ...other];
  if (!all.length)
    return <div className="training-chart-empty">Waiting for {metric}…</div>;
  const x0 = Math.min(...all.map((p) => p.x)),
    x1 = Math.max(x0 + 1, ...all.map((p) => p.x));
  const minimum = Math.min(...all.map((p) => p.y));
  const maximum = Math.max(...all.map((p) => p.y));
  const padding =
    maximum === minimum
      ? Math.max(Math.abs(minimum) * 0.03, 1e-6)
      : (maximum - minimum) * 0.08;
  const y0 = minimum - padding,
    y1 = maximum + padding;
  const tickDigits = Math.min(
    8,
    Math.max(
      3,
      2 +
        Math.ceil(
          Math.log10(Math.max(Math.abs(y0), Math.abs(y1), 1e-10) / (y1 - y0)),
        ),
    ),
  );
  const scaleX = (x: number) => 64 + ((x - x0) / (x1 - x0)) * 464;
  const scaleY = (y: number) => 150 - ((y - y0) / (y1 - y0)) * 124;
  const path = (data: typeof current) =>
    data
      .map((p, i) => `${i ? "L" : "M"}${scaleX(p.x)},${scaleY(p.y)}`)
      .join(" ");
  const active =
    hover === null
      ? current.at(-1)
      : current.reduce<(typeof current)[number] | undefined>(
          (closest, point) =>
            !closest ||
            Math.abs(point.x - (x0 + hover * (x1 - x0))) <
              Math.abs(closest.x - (x0 + hover * (x1 - x0)))
              ? point
              : closest,
          undefined,
        );
  return (
    <figure className="training-chart">
      <figcaption>
        <strong>{metric}</strong>
        <span>
          {active
            ? `${active.raw.toPrecision(4)} · step ${active.step}`
            : "Comparison"}
        </span>
      </figcaption>
      <svg
        viewBox="0 0 550 180"
        role="img"
        aria-label={`${metric} by ${axis}. ${current.length} reported points.`}
        onPointerMove={(event) => {
          const r = event.currentTarget.getBoundingClientRect();
          setHover(
            Math.max(
              0,
              Math.min(
                1,
                (((event.clientX - r.left) / r.width) * 550 - 64) / 464,
              ),
            ),
          );
        }}
        onPointerLeave={() => setHover(null)}
      >
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line
              x1="64"
              x2="528"
              y1={26 + t * 124}
              y2={26 + t * 124}
              className="training-chart-grid"
            />
            <text x="2" y={30 + t * 124}>
              {(y1 - t * (y1 - y0)).toPrecision(tickDigits)}
            </text>
          </g>
        ))}
        <text x="64" y="174">
          {x0.toFixed(0)}
        </text>
        <text x="480" y="174">
          {x1.toFixed(0)}
        </text>
        <path d={path(other)} className="training-chart-compare" />
        <path d={path(current)} className="training-chart-line" />
        {active && (
          <circle
            cx={scaleX(active.x)}
            cy={scaleY(active.y)}
            r="4"
            className="training-chart-point"
          />
        )}
      </svg>
      <details>
        <summary>Reported values</summary>
        <table>
          <thead>
            <tr>
              <th>Step</th>
              <th>{axis}</th>
              <th>Raw value</th>
            </tr>
          </thead>
          <tbody>
            {current.slice(-20).map((p, i) => (
              <tr key={i}>
                <td>{p.step}</td>
                <td>{p.x.toFixed(1)}</td>
                <td>{p.raw.toPrecision(5)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
