import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Gauge, X } from "lucide-react";
import { z } from "zod";
import type { Project } from "./types";
import { Dialog } from "./ui";
import {
  capabilitySchema,
  hardwareSchema,
  trainingRequest,
  type TrainingGPU,
} from "./trainingApi";
import {
  tuningJobSchema,
  tuningListSchema,
  type TuningJob,
  type TuningCandidate,
} from "./gpuTuningApi";
import "./training.css";

const active = new Set(["starting", "running"]);
const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error);
function allocation(result: TuningCandidate) {
  return result.strategy === "ddp"
    ? result.devices
        .map(
          (device, rank) =>
            `GPU ${device}: ${result.batch_sizes[rank]} (${Math.round((100 * result.batch_sizes[rank]) / result.global_batch_size)}%)`,
        )
        .join(" · ")
    : result.strategy === "split"
      ? Object.entries(result.placement)
          .map(([block, device]) => `${block} → GPU ${device}`)
          .join(" · ")
      : `GPU ${result.devices[0]}`;
}

export function GPUTuning({ project }: { project: Project }) {
  const [open, setOpen] = useState(false);
  const [python, setPython] = useState("");
  const [gpus, setGPUs] = useState<TrainingGPU[]>([]);
  const [devices, setDevices] = useState<number[]>([]);
  const [precision, setPrecision] = useState("fp32");
  const [length, setLength] = useState(128);
  const [batch, setBatch] = useState(8);
  const [steps, setSteps] = useState(5);
  const [candidates, setCandidates] = useState(4);
  const [list, setList] = useState<z.infer<typeof tuningListSchema> | null>(
    null,
  );
  const [job, setJob] = useState<TuningJob | null>(null);
  const [enabled, setEnabled] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const jobRequest = useRef(0);
  useEffect(() => {
    if (!open) return;
    let current = true;
    const version = jobRequest.current;
    Promise.all([
      trainingRequest("/capabilities", capabilitySchema),
      trainingRequest("/hardware", hardwareSchema),
      trainingRequest("/gpu-tuning", tuningListSchema),
    ])
      .then(([capabilities, hardware, saved]) => {
        if (!current || version !== jobRequest.current) return;
        setEnabled(capabilities.enabled);
        setPython((value) => value || capabilities.python);
        setGPUs(hardware.gpus);
        setDevices((value) =>
          value.length
            ? value
            : hardware.gpus.slice(0, 2).map((gpu) => gpu.index),
        );
        setList(saved);
        const running = saved.jobs.find((item) => active.has(item.state));
        if (running)
          void trainingRequest(`/gpu-tuning/${running.id}`, tuningJobSchema)
            .then((next) => {
              if (current && version === jobRequest.current) setJob(next);
            })
            .catch((e) => {
              if (current && version === jobRequest.current)
                setError(errorMessage(e));
            });
      })
      .catch((e) => {
        if (current && version === jobRequest.current)
          setError(errorMessage(e));
      });
    return () => {
      current = false;
    };
  }, [open]);
  useEffect(() => {
    if (!open || !job || !active.has(job.state)) return;
    let cancelled = false;
    const version = jobRequest.current;
    const timer = setInterval(() => {
      void trainingRequest(`/gpu-tuning/${job.id}`, tuningJobSchema)
        .then((next) => {
          if (cancelled || version !== jobRequest.current) return;
          setJob(next);
          if (!active.has(next.state))
            void trainingRequest("/gpu-tuning", tuningListSchema)
              .then((next) => {
                if (!cancelled && version === jobRequest.current) setList(next);
              })
              .catch((e) => {
                if (!cancelled && version === jobRequest.current)
                  setError(errorMessage(e));
              });
        })
        .catch((e) => {
          if (!cancelled && version === jobRequest.current)
            setError(errorMessage(e));
        });
    }, 1500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [job?.id, job?.state, open]);
  async function start() {
    const version = ++jobRequest.current;
    setError("");
    setBusy(true);
    try {
      const next = await trainingRequest("/gpu-tuning", tuningJobSchema, {
        project,
        python,
        devices,
        precision,
        sequence_length: length,
        global_batch_size: batch,
        measure_steps: steps,
        max_candidates: candidates,
      });
      if (version === jobRequest.current) setJob(next);
    } catch (error) {
      if (version === jobRequest.current) setError(errorMessage(error));
    } finally {
      if (version === jobRequest.current) setBusy(false);
    }
  }
  const running = !!job && active.has(job.state);
  return (
    <>
      <button aria-label="Weave settings" onClick={() => setOpen(true)}>
        <Gauge size={15} aria-hidden="true" />
        <span>Weave settings</span>
      </button>
      {open &&
        createPortal(
          <Dialog
            label="Weave settings"
            className="gpu-tuning-dialog training-workspace"
            onClose={() => setOpen(false)}
          >
            <header className="training-section-heading">
              <h2>GPU performance tuning</h2>
              <button aria-label="Close dialog" onClick={() => setOpen(false)}>
                <X size={18} />
              </button>
            </header>
            <p>
              Host-wide profiles · benchmark <strong>{project.name}</strong>{" "}
              using an isolated copy with random weights. Your scripts,
              checkpoints and training data are not changed.
            </p>
            <label className="gpu-tuning-toggle">
              <input
                type="checkbox"
                checked={list?.settings.auto_apply ?? true}
                disabled={!list || busy}
                onChange={(event) => {
                  const auto_apply = event.target.checked;
                  void trainingRequest(
                    "/gpu-tuning/settings",
                    z.object({ auto_apply: z.boolean() }),
                    { auto_apply },
                    "PUT",
                  )
                    .then((settings) =>
                      setList((current) =>
                        current ? { ...current, settings } : current,
                      ),
                    )
                    .catch((e) => setError(errorMessage(e)));
                }}
              />
              Automatically load matching profiles for multi-GPU training
            </label>
            <p className="training-muted">
              Profiles match model, GPUs, Python/PyTorch, precision and global
              microbatch. Explicit block placements take precedence. Custom
              loaders must opt into unequal DDP batches.
            </p>
            <section className="training-card">
              <h3>Benchmark workload</h3>
              <div className="training-form-grid">
                <label>
                  Benchmark Python
                  <input
                    value={python}
                    onChange={(e) => setPython(e.target.value)}
                    disabled={running}
                  />
                </label>
                <label>
                  Benchmark precision
                  <select
                    value={precision}
                    onChange={(e) => setPrecision(e.target.value)}
                    disabled={running}
                  >
                    <option value="fp32">FP32</option>
                    <option value="fp16">FP16</option>
                    <option value="bf16">BF16 (native support required)</option>
                  </select>
                </label>
                <label>
                  Tokens per sequence
                  <input
                    type="number"
                    min={2}
                    max={32768}
                    value={length}
                    onChange={(e) => setLength(Number(e.target.value))}
                    disabled={running}
                  />
                </label>
                <label>
                  Global microbatch (all GPUs combined)
                  <input
                    type="number"
                    min={Math.max(1, devices.length)}
                    max={1024}
                    value={batch}
                    onChange={(e) => setBatch(Number(e.target.value))}
                    disabled={running}
                  />
                </label>
                <label>
                  Measured updates per candidate
                  <input
                    type="number"
                    min={3}
                    max={50}
                    value={steps}
                    onChange={(e) => setSteps(Number(e.target.value))}
                    disabled={running}
                  />
                </label>
                <label>
                  Candidates per multi-GPU strategy
                  <input
                    type="number"
                    min={1}
                    max={12}
                    value={candidates}
                    onChange={(e) => setCandidates(Number(e.target.value))}
                    disabled={running}
                  />
                </label>
              </div>
              <fieldset>
                <legend>Benchmark GPUs</legend>
                {gpus.length ? (
                  gpus.map((gpu) => (
                    <label className="gpu-tuning-toggle" key={gpu.uuid}>
                      <input
                        type="checkbox"
                        checked={devices.includes(gpu.index)}
                        disabled={running}
                        onChange={(event) =>
                          setDevices((current) =>
                            event.target.checked
                              ? [...current, gpu.index].sort((a, b) => a - b)
                              : current.filter((index) => index !== gpu.index),
                          )
                        }
                      />
                      GPU {gpu.index}: {gpu.name} ·{" "}
                      {gpu.memory_total_mb
                        ? `${Math.round(gpu.memory_total_mb / 1024)} GB`
                        : "memory unknown"}
                    </label>
                  ))
                ) : (
                  <p>
                    No NVIDIA GPUs reported. Select a CUDA-enabled Python
                    environment and check its GPU visibility.
                  </p>
                )}
              </fieldset>
              <p className="training-muted">
                Measures forward, backward and AdamW after warmup, including DDP
                synchronization and split transfers. Starts with standalone
                speed ratios, then tests nearby allocations. Split currently
                uses FP32 and sequential block placement. Data loading, real
                text distribution and checkpoint overhead are excluded; variable
                sequence lengths can change the best allocation. Run while the
                GPUs are idle for useful results.
              </p>
              <button
                className="primary"
                disabled={
                  busy ||
                  running ||
                  !enabled ||
                  !devices.length ||
                  !python ||
                  batch < devices.length
                }
                onClick={() => void start()}
              >
                Benchmark & autotune
              </button>
              {!enabled && (
                <p>
                  Enable local training in the server installation settings to
                  run benchmarks.
                </p>
              )}
            </section>
            {error && (
              <p role="alert" className="training-alert">
                {error}
              </p>
            )}
            {job && (
              <section
                className="training-card"
                aria-label="GPU tuning progress"
              >
                <div className="training-section-heading">
                  <h3>{job.state}</h3>
                  {running && (
                    <button
                      onClick={() =>
                        void trainingRequest(
                          `/gpu-tuning/${job.id}/stop`,
                          tuningJobSchema,
                          {},
                        )
                          .then(setJob)
                          .catch((e) => setError(errorMessage(e)))
                      }
                    >
                      Stop benchmark
                    </button>
                  )}
                </div>
                <p role="status">{job.message}</p>
                {running && <progress aria-label="GPU benchmark progress" />}
                <div className="gpu-tuning-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Strategy</th>
                        <th>Allocation</th>
                        <th>Precision</th>
                        <th>Update time</th>
                        <th>Tokens/s</th>
                      </tr>
                    </thead>
                    <tbody>
                      {job.results.map((result, index) => (
                        <tr key={index}>
                          <td>{result.strategy}</td>
                          <td>
                            {allocation(result)}
                            {result.error && (
                              <details>
                                <summary>Candidate failed</summary>
                                <pre>{result.error}</pre>
                              </details>
                            )}
                          </td>
                          <td>{result.precision}</td>
                          <td>
                            {result.step_seconds
                              ? `${(1000 * result.step_seconds).toFixed(1)} ms`
                              : "—"}
                          </td>
                          <td>
                            {result.tokens_per_second?.toLocaleString(
                              undefined,
                              { maximumFractionDigits: 0 },
                            ) ?? "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {job.logs && (
                  <details>
                    <summary>Benchmark logs</summary>
                    <pre>{job.logs}</pre>
                  </details>
                )}
              </section>
            )}
            <section className="training-card">
              <h3>Saved profiles</h3>
              {!list?.profiles.length && <p>No completed benchmarks yet.</p>}
              {list?.profiles.map((profile) => (
                <details key={profile.id}>
                  <summary>
                    {profile.model_name} · GPUs {profile.devices.join(", ")} ·{" "}
                    {profile.sequence_length} tokens · global batch{" "}
                    {profile.global_batch_size}
                  </summary>
                  <p>
                    {profile.initial_estimate === "equal-fallback"
                      ? "Equal fallback shares (standalone trial failed): "
                      : "Initial standalone shares: "}
                    {profile.initial_speed_ratios
                      .map((value) => `${Math.round(100 * value)}%`)
                      .join(" / ")}
                  </p>
                  {Object.entries(profile.best).map(([strategy, result]) => (
                    <p key={strategy}>
                      <strong>
                        {strategy} · {result.precision}
                      </strong>
                      : {allocation(result)} ·{" "}
                      {((result.step_seconds ?? 0) * 1000).toFixed(1)} ms/update
                    </p>
                  ))}
                  <small>
                    {profile.python} ·{" "}
                    {new Date(profile.created * 1000).toLocaleString()}
                  </small>
                </details>
              ))}
            </section>
          </Dialog>,
          document.body,
        )}
    </>
  );
}
