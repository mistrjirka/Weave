import { z } from "zod";
import { detailSchema, trainingRequest, type RunDetail } from "../trainingApi";

type Props = {
  detail: RunDetail;
  runId: string;
  busy: boolean;
  reportBaseURL: string;
  setReportBaseURL: (value: string) => void;
  task: (action: () => Promise<unknown>) => Promise<void>;
  setNotice: (notice: string) => void;
  setDetail: (detail: RunDetail) => void;
};

export function RunReporting({
  detail,
  runId,
  busy,
  reportBaseURL,
  setReportBaseURL,
  task,
  setNotice,
  setDetail,
}: Props) {
  return (
    <>
      {detail.run.execution === "monitor" && (
        <details className="training-reporting-options">
          <summary>Independent training & reporting</summary>
          <section
            className="training-server-files"
            aria-label="Independent training reporting"
          >
            <p>
              Download the training package and start it yourself. It contains
              the network, training script, configuration and reporting.json.
              Reports are sent in the background; a disconnected server does not
              stop training.
            </p>
            <label>
              Reporting server URL
              <input
                type="url"
                value={reportBaseURL}
                onChange={(event) => setReportBaseURL(event.target.value)}
              />
              <small>
                Localhost works when training and Weave run on the same
                computer, or through an SSH tunnel. On another computer, use
                this server’s reachable address (for example {location.origin}).
              </small>
            </label>
            <button
              disabled={busy}
              onClick={() =>
                void task(async () => {
                  await trainingRequest(
                    `/runs/${encodeURIComponent(runId)}/reporting`,
                    z.object({ url: z.string(), token: z.string() }),
                    { base_url: reportBaseURL },
                  );
                  setNotice(
                    "Reporting connection saved. Download the training package again to use this address.",
                  );
                  setDetail(
                    await trainingRequest(
                      `/runs/${encodeURIComponent(runId)}`,
                      detailSchema,
                    ),
                  );
                })
              }
            >
              Save reporting address
            </button>
            <small>
              Packaged endpoint: <code>{detail.reporting_url}</code>
            </small>
            <small>
              Connection file on server:{" "}
              <code>{detail.directory}/reporting.json</code>. A scoped reporting
              key is included automatically; no training login is needed.
            </small>
            <details>
              <summary>
                Launch command, reporting hooks and remote files
              </summary>
              <pre className="training-log">
                {detail?.run.code_first
                  ? "python train.py"
                  : "python train.py --config config.json"}
              </pre>
              <p>
                Run from the extracted package folder. Edit train.py for that
                computer’s data paths, devices and Python environment. Custom
                scripts report loss with run.log();
                run.watch_optimizer(optimizer) can track learning rates
                automatically. NVIDIA GPU readings are collected when nvidia-smi
                is available.
              </p>
              <p>
                Remote checkpoint files and datasets stay on the training
                computer. Server snapshots preserve the received history and
                packaged code; copy remote weights separately to back them up.
              </p>
            </details>
            {detail.reporting && (
              <small>
                Last report received:{" "}
                {new Date(
                  detail.reporting.last_received * 1000,
                ).toLocaleString()}
                . This is the last report time, not a process liveness check.
              </small>
            )}
          </section>
        </details>
      )}
      {detail.run.execution === "monitor" && (
        <details>
          <summary>Run this package in your terminal</summary>
          <p>
            On this host, keep the working directory and reporting directory
            pointed at the prepared run:
          </p>
          <pre className="training-log">
            cd {JSON.stringify(detail.directory)}
            {"\n"}CUDA_DEVICE_ORDER=PCI_BUS_ID WEAVE_RUN_DIR=
            {JSON.stringify(detail.directory)}{" "}
            {JSON.stringify(detail.config.python)}{" "}
            {!detail.run.code_first && detail.config.strategy === "ddp"
              ? `-m torch.distributed.run --standalone --nproc_per_node=${detail.config.devices.length} `
              : ""}
            {detail.run.code_first
              ? "train.py"
              : "train.py --config config.json"}
          </pre>
          <p>
            This command is displayed for copying; Weave does not execute shell
            text.
          </p>
        </details>
      )}
    </>
  );
}
