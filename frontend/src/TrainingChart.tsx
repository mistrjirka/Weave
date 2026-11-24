import {
  runContract,
  counterLabel,
  type TrainingAxis,
} from "./trainingContract";
import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import type { TrainingEvent } from "./trainingApi";
import {
  formatMetric,
  metricPresentation,
  metricDomain,
  metricScope,
} from "./trainingMetrics";

const noEvents: TrainingEvent[] = [];

function chartPoints(
  items: TrainingEvent[],
  metric: string,
  axis: TrainingAxis,
  smooth: boolean,
) {
  const start = items[0]?.time ?? 0;
  const counter = axis.startsWith("counter:") ? axis.slice(8) : null;
  const points: { x: number; y: number; raw: number; step: number }[] = [];
  let average: number | undefined;
  for (const event of items) {
    const value = event.payload[metric];
    if (
      event.kind !== "metrics" ||
      typeof value !== "number" ||
      !Number.isFinite(value)
    )
      continue;
    const x =
      counter !== null
        ? event.counters?.[counter]
        : axis === "time"
          ? event.time - start
          : axis === "tokens"
            ? event.tokens
            : event.step;
    if (typeof x !== "number" || !Number.isFinite(x)) continue;
    average = average === undefined ? value : average * 0.8 + value * 0.2;
    points.push({
      x,
      y: smooth ? average : value,
      raw: value,
      step: event.step,
    });
  }
  return points;
}

export function TrainingChart({
  events,
  comparison = noEvents,
  metric,
  axis,
  smooth,
}: {
  events: TrainingEvent[];
  comparison?: TrainingEvent[];
  metric: string;
  axis: TrainingAxis;
  smooth: boolean;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const contract = useMemo(() => runContract(events), [events]);
  const presentation = metricPresentation(metric, contract);
  const scope = metricScope(metric);
  const title = (
    <div className="training-metric-title">
      <strong title={metric}>{presentation.label}</strong>
      {scope && <small>{scope}</small>}
    </div>
  );
  const axisLabel = axis.startsWith("counter:")
    ? counterLabel(axis.slice(8))
    : axis === "step"
      ? contract.step_unit
      : axis;
  const chartStyle: CSSProperties & { "--chart-color": string } = {
    "--chart-color": presentation.color,
  };
  const current = useMemo(
    () => chartPoints(events, metric, axis, smooth),
    [events, metric, axis, smooth],
  );
  const other = useMemo(
    () => chartPoints(comparison, metric, axis, smooth),
    [comparison, metric, axis, smooth],
  );
  const plot = useMemo(() => {
    if (!current.length && !other.length) return null;
    let x0 = Infinity,
      x1 = -Infinity,
      minimum = Infinity,
      maximum = -Infinity;
    for (const series of [current, other]) {
      for (const point of series) {
        x0 = Math.min(x0, point.x);
        x1 = Math.max(x1, point.x);
        minimum = Math.min(minimum, point.y);
        maximum = Math.max(maximum, point.y);
      }
    }
    x1 = Math.max(x0 + 1, x1);
    const [y0, y1] = metricDomain(metric, minimum, maximum, contract);
    const scaleX = (x: number) => 64 + ((x - x0) / (x1 - x0)) * 464;
    const scaleY = (y: number) => 150 - ((y - y0) / (y1 - y0)) * 124;
    const path = (data: typeof current) =>
      data
        .map((p, i) => `${i ? "L" : "M"}${scaleX(p.x)},${scaleY(p.y)}`)
        .join(" ");
    return {
      x0,
      x1,
      y0,
      y1,
      scaleX,
      scaleY,
      currentPath: path(current),
      otherPath: path(other),
    };
  }, [current, other, metric, contract]);
  if (!plot)
    return (
      <figure
        className="training-chart training-chart-empty"
        style={chartStyle}
      >
        <figcaption>{title}</figcaption>
        {presentation.empty ? (
          <p>{presentation.empty}</p>
        ) : (
          <div className="training-chart-placeholder">
            <svg viewBox="0 0 550 180" aria-hidden="true">
              {[26, 88, 150].map((y) => (
                <line
                  key={y}
                  x1="64"
                  x2="528"
                  y1={y}
                  y2={y}
                  className="training-chart-grid"
                />
              ))}
              <line
                x1="64"
                x2="64"
                y1="26"
                y2="150"
                className="training-chart-grid"
              />
            </svg>
            <p>
              <span>Waiting for the first datapoints</span>
            </p>
          </div>
        )}
      </figure>
    );
  const { x0, x1, y0, y1, scaleX, scaleY, currentPath, otherPath } = plot;
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
    <figure className="training-chart" style={chartStyle}>
      <figcaption>
        {title}
        <span>
          {active
            ? `${formatMetric(active.raw)} · ${formatMetric(active.step)} ${contract.step_unit}`
            : "Comparison"}
        </span>
      </figcaption>
      {current.length === 1 && (
        <p className="training-chart-note">One reported point · no trend yet</p>
      )}
      <svg
        viewBox="0 0 550 180"
        role="img"
        aria-label={`${metric} by ${axisLabel}. ${current.length} reported points.`}
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
              {formatMetric(y1 - t * (y1 - y0), tickDigits)}
            </text>
          </g>
        ))}
        <text x="64" y="174">
          {formatMetric(x0)}
        </text>
        <text x="480" y="174">
          {formatMetric(x1)}
        </text>
        <path d={otherPath} className="training-chart-compare" />
        {other.length === 1 && (
          <circle
            cx={scaleX(other[0].x)}
            cy={scaleY(other[0].y)}
            r="4"
            className="training-chart-compare-point"
          />
        )}
        <path d={currentPath} className="training-chart-line" />
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
              <th>{axisLabel}</th>
              <th>Raw value</th>
            </tr>
          </thead>
          <tbody>
            {current.slice(-20).map((p, i) => (
              <tr key={i}>
                <td>{p.step}</td>
                <td>{p.x.toFixed(1)}</td>
                <td>
                  {p.raw.toLocaleString("en", {
                    notation: "standard",
                    maximumSignificantDigits: 10,
                  })}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
