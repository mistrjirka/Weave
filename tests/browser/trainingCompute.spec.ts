import { expect, test, type APIRequestContext } from "@playwright/test";
import { readFileSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import { detailSchema, stageSchema } from "../../frontend/src/trainingApi";

async function unlockIfConfigured(request: APIRequestContext, token?: string) {
  const capabilities = await (
    await request.get("/api/training/capabilities")
  ).json();
  if (capabilities.auth_required) {
    if (!token)
      throw new Error("WEAVE_TRAINING_TOKEN is required by this server");
    await request.post("/api/training/session", { data: { token } });
  }
  return capabilities;
}

const tinyProject = (id: string) => ({
  version: 1,
  id,
  name: `Browser DDP ${id}`,
  description: "Temporary DDP launch verification fixture",
  config: { width: 8, vocab: 8, heads: 2, ffn_width: 16 },
  graph: {
    nodes: [
      {
        id: "ids",
        type: "architecture",
        position: { x: 0, y: 0 },
        data: {
          kind: "input",
          label: "Tokens",
          params: { index: 0, shape: ["B", "T"], dtype: "int64" },
        },
      },
      {
        id: "emb",
        type: "architecture",
        position: { x: 220, y: 0 },
        data: {
          kind: "embedding",
          label: "Embedding",
          params: { width: 8, vocab: 8 },
        },
      },
      {
        id: "head",
        type: "architecture",
        position: { x: 440, y: 0 },
        data: { kind: "head", label: "Vocabulary", params: { out: 8 } },
      },
      {
        id: "out",
        type: "architecture",
        position: { x: 660, y: 0 },
        data: { kind: "output", label: "Logits", params: {} },
      },
    ],
    edges: [
      {
        id: "input-embedding",
        source: "ids",
        target: "emb",
        targetHandle: "a",
      },
      {
        id: "embedding-head",
        source: "emb",
        target: "head",
        targetHandle: "a",
      },
      { id: "head-output", source: "head", target: "out", targetHandle: "a" },
    ],
  },
});

test("GUI DDP override survives Start and runs two CUDA ranks with CPU async evaluation", async ({
  page,
  request,
}) => {
  test.setTimeout(240000);
  const capabilities = await unlockIfConfigured(
    request,
    process.env.WEAVE_TRAINING_TOKEN,
  );
  test.skip(!capabilities.enabled, "Local training is disabled on this server");
  const environment = await (
    await request.get(
      `/api/training/environment?python=${encodeURIComponent(capabilities.python)}`,
    )
  ).json();
  test.skip(
    !environment.cuda_available || environment.device_count < 2,
    "Requires two visible CUDA GPUs in the configured training Python",
  );

  const id = `project-ddp-browser-${Date.now()}`;
  const project = tinyProject(id);
  let projectPath: string | null = null;
  let runDirectory: string | null = null;
  let runId: string | null = null;
  await request.put(`/api/projects/${id}`, { data: project });

  try {
    await page.goto(`/?model=${id}&view=training`);
    await expect(
      page.getByRole("region", { name: "Training workspace", exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Create training stage", exact: true })
      .click();

    const editor = page.getByRole("textbox", {
      name: "Training Python script",
      exact: true,
    });
    await expect(editor).toContainText("TrainingConfig(", { timeout: 30000 });
    const templates = await (
      await request.get("/api/training/templates")
    ).json();
    const dataLoader = `from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    import torch.distributed as dist
    class TinyTokens(torch.utils.data.Dataset):
        def __init__(self, count):
            self.count = count
        def __len__(self):
            return self.count
        def __getitem__(self, index):
            return {"input_ids": torch.tensor([1, 2, 3, 4]), "labels": torch.tensor([2, 3, 4, 5])}
    train = TinyTokens(4)
    replicas = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    sampler = DistributedSampler(train, num_replicas=replicas, rank=rank, shuffle=False)
    return DataLoader(train, batch_size=1, sampler=sampler), DataLoader(TinyTokens(2), batch_size=1)`;
    const source = String(templates.scripts.causal)
      .replace(
        'raise NotImplementedError("Add your data loaders here.")',
        dataLoader,
      )
      .replace("micro_batch_size=16", "micro_batch_size=1")
      .replace("max_epochs=3", "max_epochs=None")
      .replace("max_optimizer_steps=None", "max_optimizer_steps=2")
      .replace("checkpoint_every=250", "checkpoint_every=1")
      .replace("eval_every=100", "eval_every=1")
      .replace("eval_batches=5", "eval_batches=1")
      .replace("tokenizer=load_tokenizer,", "tokenizer=None,");
    await editor.fill(source);
    await editor.press("Control+s");
    await expect(page.locator(".training-notice")).toContainText(
      "Saved server file",
      { timeout: 30000 },
    );

    await page
      .getByRole("button", { name: "Training devices", exact: true })
      .click();
    const computeMode = page.getByRole("combobox", {
      name: "Training device settings",
    });
    await expect(computeMode).toHaveValue("script");
    await computeMode.selectOption("override");
    const strategy = page.getByRole("combobox", { name: "GPU strategy" });
    await strategy.selectOption("ddp");
    const devices = page.getByRole("group", { name: "Training devices" });
    await devices.getByRole("checkbox", { name: /GPU 0/ }).check();
    await devices.getByRole("checkbox", { name: /GPU 1/ }).check();
    await expect(
      devices.getByRole("checkbox", { name: /GPU 0/ }),
    ).toBeChecked();
    await expect(
      devices.getByRole("checkbox", { name: /GPU 1/ }),
    ).toBeChecked();
    await page
      .getByRole("combobox", { name: "Precision", exact: true })
      .selectOption("fp16");
    const evaluationMode = page.getByRole("combobox", {
      name: /Validation execution/,
    });
    await evaluationMode.selectOption("async");
    const evaluationDevice = page.getByRole("combobox", {
      name: /Validation device/,
    });
    await expect(evaluationDevice).toBeEnabled();
    await evaluationDevice.selectOption("cpu");
    await page
      .getByRole("button", { name: "Save stages", exact: true })
      .click();
    await expect(page.locator(".training-notice")).toContainText("Saved", {
      timeout: 30000,
    });

    const plan = await (await request.get(`/api/training/plans/${id}`)).json();
    const savedStage = stageSchema.parse(plan.stages[0]);
    projectPath = savedStage.project_path ?? null;
    expect(savedStage.config.compute_mode).toBe("override");
    expect(savedStage.config.strategy).toBe("ddp");
    expect(savedStage.config.devices).toEqual([0, 1]);
    expect(savedStage.config.precision).toBe("fp16");
    expect(savedStage.config.evaluation_mode).toBe("async");
    expect(savedStage.config.evaluation_device).toBe("cpu");

    const launchRequestPromise = page.waitForRequest(
      (candidate) =>
        candidate.method() === "POST" &&
        candidate.url().endsWith("/api/training/runs"),
      { timeout: 30000 },
    );
    await page
      .getByRole("button", { name: "Start training stage", exact: true })
      .click();
    const launchRequest = await launchRequestPromise;
    const submitted = launchRequest.postDataJSON();
    expect(submitted.stage.config.compute_mode).toBe("override");
    expect(submitted.stage.config.strategy).toBe("ddp");
    expect(submitted.stage.config.devices).toEqual([0, 1]);
    expect(submitted.stage.config.precision).toBe("fp16");
    expect(submitted.stage.config.evaluation_mode).toBe("async");
    expect(submitted.stage.config.evaluation_device).toBe("cpu");

    await expect(
      page.getByRole("region", { name: "Run dashboard", exact: true }),
    ).toBeVisible({ timeout: 30000 });
    runId = new URL(page.url()).searchParams.get("run");
    expect(runId).toBeTruthy();
    await expect
      .poll(
        async () => {
          const response = await request.get(`/api/training/runs/${runId}`);
          return (await response.json()).status.state;
        },
        { timeout: 180000 },
      )
      .toMatch(/^(completed|failed)$/);
    const detail = detailSchema.parse(
      await (await request.get(`/api/training/runs/${runId}`)).json(),
    );
    runDirectory = detail.directory;
    expect(detail.status.state, detail.logs.slice(-3000)).toBe("completed");
    expect(detail.run.code_first).toBeTruthy();
    expect(detail.run.strategy).toBe("ddp");
    expect(detail.config.strategy).toBe("ddp");
    expect(detail.config.devices).toEqual([0, 1]);
    expect(detail.config.precision).toBe("fp16");
    expect(detail.config.evaluation_mode).toBe("async");
    expect(detail.config.evaluation_device).toBe("cpu");

    const frozen = JSON.parse(
      readFileSync(resolve(runDirectory, "weave_support/launch.json"), "utf8"),
    );
    expect(frozen.strategy).toBe("ddp");
    expect(frozen.devices).toEqual([0, 1]);
    expect(frozen.precision).toBe("fp16");
    expect(frozen.evaluation_mode).toBe("async");
    expect(frozen.evaluation_device).toBe("cpu");
    expect(
      detail.events.some(
        (event) => event.kind === "progress" && event.payload.world_size === 2,
      ),
    ).toBeTruthy();
    expect(
      detail.events.some(
        (event) =>
          event.kind === "evaluation" &&
          event.payload.state === "completed" &&
          event.payload.device === "cpu",
      ),
    ).toBeTruthy();
  } finally {
    if (runId && runDirectory)
      rmSync(runDirectory, { recursive: true, force: true });
    if (projectPath && projectPath.endsWith(id))
      rmSync(projectPath, { recursive: true, force: true });
    rmSync(resolve("data/training/plans", `${id}.json`), { force: true });
    await request.delete(`/api/projects/${id}`);
  }
});
