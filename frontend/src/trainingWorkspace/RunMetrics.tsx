import { useMemo } from "react";
import {
  counterLabel,
  isTrainingAxis,
  namedCounters,
  runContract,
  type TrainingAxis,
} from "../trainingContract";
import { metricsForDisplay } from "../trainingMetrics";
import { TrainingChart } from "../TrainingChart";
import { TrainingMessages } from "../TrainingMessages";
import { FP16Status } from "../FP16Status";
import type { Project } from "../types";
import type { RunDetail, TrainingRun } from "../trainingApi";
type Props = {
  detail: RunDetail;
  comparison: RunDetail | null;
  runs: TrainingRun[];
  project: Project;
  inspectBlocks: boolean | undefined;
  active: boolean;
  axis: TrainingAxis;
  setAxis: (axis: TrainingAxis) => void;
  smooth: boolean;
  setSmooth: (smooth: boolean) => void;
  comparisonId: string;
  setComparisonId: (id: string) => void;
};
export function RunMetrics({
  detail,
  comparison,
  runs,
  project,
  inspectBlocks,
  active,
  axis,
  setAxis,
  smooth,
  setSmooth,
  comparisonId,
  setComparisonId,
}: Props) {
  const runId = detail.run.id;
  const events = detail.events;
  const contract = useMemo(() => runContract(events), [events]);
  const counters = useMemo(() => namedCounters(events), [events]);
  const metrics = useMemo(
    () =>
      metricsForDisplay(
        events,
        Boolean(detail.run.code_first),
        contract,
        detail.run.recipe === "custom",
      ),
    [events, contract, detail.run.code_first, detail.run.recipe],
  );
  return (
    <>
      <details className="training-chart-options">
        <summary>Chart options</summary>
        <div className="training-chart-controls">
          <label>
            Horizontal axis
            <select
              value={axis}
              onChange={(e) => {
                if (isTrainingAxis(e.target.value)) setAxis(e.target.value);
              }}
            >
              <option value="step">{contract.step_unit}</option>
              <option value="tokens">Tokens processed</option>
              <option value="time">Elapsed seconds</option>
              {Object.keys(counters).map((name) => (
                <option key={name} value={`counter:${name}`}>
                  {counterLabel(name)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Compare run
            <select
              value={comparisonId}
              onChange={(e) => setComparisonId(e.target.value)}
            >
              <option value="">No comparison</option>
              {runs
                .filter((r) => r.id !== runId)
                .map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name} · {new Date(r.created).toLocaleString()}
                  </option>
                ))}
            </select>
          </label>
          <label className="training-inline-check">
            <input
              type="checkbox"
              checked={smooth}
              onChange={(e) => setSmooth(e.target.checked)}
            />
            Smooth curves (raw values retained)
          </label>
        </div>
      </details>
      <div className="training-charts">
        {metrics.map((metric) => (
          <TrainingChart
            key={metric}
            events={detail.events}
            comparison={comparison?.events}
            origin={detail.first_event_time}
            comparisonOrigin={comparison?.first_event_time}
            metric={metric}
            axis={axis}
            smooth={smooth}
          />
        ))}
      </div>
      {!!Object.keys(detail.module_nodes).length && inspectBlocks && (
        <details>
          <summary>Locate blocks in the graph</summary>
          {Object.entries(detail.module_nodes)
            .filter(([node]) => project.graph.nodes.some((n) => n.id === node))
            .map(([node, name]) => (
              <p key={node}>
                <a
                  href={`?model=${encodeURIComponent(project.id)}&node=${encodeURIComponent(node)}`}
                >
                  {name} · current graph
                </a>
              </p>
            ))}
        </details>
      )}
      {detail.events.length >= 10000 && (
        <p className="training-muted">
          Showing the latest 10,000 events. Full event history is preserved in
          the run and snapshots.
        </p>
      )}
      {!detail.events.some((event) => event.kind === "metrics") && (
        <p className="training-muted">
          {active
            ? "No training metrics yet. Check Logs for data preparation and startup progress."
            : detail.status.state === "prepared"
              ? "Start the prepared package to receive training metrics."
              : "No training metrics were recorded for this run."}
        </p>
      )}
      <TrainingMessages key={detail.run.id} detail={detail} />
      <FP16Status events={detail.events} />
    </>
  );
}
