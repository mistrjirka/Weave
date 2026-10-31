import { test, expect } from "@playwright/test";
import { existsSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import { detailSchema, stageSchema } from "../../frontend/src/trainingApi";

test("code-owned configuration, training hooks, provenance and snapshots", async ({
  page,
  context,
}) => {
  test.setTimeout(240000);
  const id = `project-hooks-browser-${Date.now()}`;
  const project = {
    version: 1,
    id,
    name: `Code hooks browser ${Date.now()}`,
    description: "Temporary browser verification",
    config: { vocab: 8, width: 8, heads: 2, ffn_width: 16 },
    graph: {
      nodes: [
        {
          id: "ids",
          type: "architecture",
          position: { x: 0, y: 0 },
          data: {
            kind: "input",
            label: "Token IDs",
            params: { index: 0, shape: ["B", "T"], dtype: "int64" },
          },
        },
        {
          id: "emb",
          type: "architecture",
          position: { x: 200, y: 0 },
          data: {
            kind: "embedding",
            label: "Embedding",
            params: { vocab: 8, width: 8 },
          },
        },
        {
          id: "head",
          type: "architecture",
          position: { x: 400, y: 0 },
          data: { kind: "head", label: "Vocabulary", params: { out: 8 } },
        },
        {
          id: "out",
          type: "architecture",
          position: { x: 600, y: 0 },
          data: { kind: "output", label: "Logits", params: {} },
        },
      ],
      edges: [
        { id: "a", source: "ids", target: "emb", targetHandle: "a" },
        { id: "b", source: "emb", target: "head", targetHandle: "a" },
        { id: "c", source: "head", target: "out", targetHandle: "a" },
      ],
    },
  };
  const data = resolve("data", `${id}.jsonl`);
  writeFileSync(
    data,
    Array.from({ length: 10 }, () =>
      JSON.stringify({ input_ids: [1, 2, 3, 4, 1, 2, 3, 4, 1] }),
    ).join("\n"),
  );
  let projectPath: string | null = null;
  let runDirectory: string | null = null;
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  let releaseInitialSave: () => void = () => {};
  const initialSaveGate = new Promise<void>((resolve) => {
    releaseInitialSave = resolve;
  });
  let initialSavePending = true;
  await page.route("**/api/training/workspaces/**", async (route) => {
    if (route.request().method() === "PUT" && initialSavePending) {
      initialSavePending = false;
      await initialSaveGate;
    }
    await route.continue();
  });
  await context.request.put(`/api/projects/${id}`, { data: project });
  try {
    await page.goto(`/?model=${id}&view=training`);
    await page
      .getByRole("button", { name: "Create training stage", exact: true })
      .click();
    const editor = page.getByRole("textbox", {
      name: "Training Python script",
      exact: true,
    });
    await expect(editor).toContainText("TrainingConfig(", { timeout: 30000 });
    await expect(
      page.getByRole("button", { name: "Launch settings", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("combobox", {
        name: "Insert reporting hook",
        exact: true,
      }),
    ).toHaveCount(0);
    await page.setViewportSize({ width: 390, height: 844 });
    const deviceShortcut = page.getByRole("button", {
      name: "Training devices",
      exact: true,
    });
    await expect(deviceShortcut).toBeInViewport();
    await deviceShortcut.click();
    const deviceHeading = page.getByRole("heading", {
      name: "Training devices and validation",
      exact: true,
    });
    await expect(deviceHeading).toBeInViewport();
    await expect(deviceHeading).toBeFocused();
    await page.screenshot({ path: "data/training-device-settings-phone.png" });
    await page.setViewportSize({ width: 1600, height: 1000 });
    const computeMode = page.getByRole("combobox", {
      name: "Training device settings",
    });
    await expect(computeMode).toHaveValue("script");
    await expect(
      page.getByRole("combobox", { name: "GPU strategy" }),
    ).toHaveCount(0);
    await computeMode.selectOption("override");
    const strategy = page.getByRole("combobox", { name: "GPU strategy" });
    await expect(strategy).toBeVisible();
    if (
      (await strategy
        .locator('option[value="ddp"]')
        .getAttribute("disabled")) === null
    ) {
      await strategy.selectOption("ddp");
      const devices = page.getByRole("group", { name: "Training devices" });
      await expect(
        devices.getByRole("checkbox", { name: /GPU 0/ }),
      ).toBeChecked();
      await expect(
        devices.getByRole("checkbox", { name: /GPU 1/ }),
      ).toBeChecked();
      await devices.getByRole("checkbox", { name: /GPU 1/ }).uncheck();
      await expect(strategy).toHaveValue("single");
      await devices.getByRole("checkbox", { name: /GPU 0/ }).uncheck();
      await expect(
        devices.getByRole("checkbox", { name: "CPU", exact: true }),
      ).toBeChecked();
      await expect(
        page.getByRole("combobox", { name: "Precision", exact: true }),
      ).toHaveValue("fp32");
      await strategy.selectOption("ddp");
      await devices.getByRole("checkbox", { name: /GPU 0/ }).check();
      await devices.getByRole("checkbox", { name: /GPU 1/ }).check();
      await expect(
        devices.getByRole("checkbox", { name: /GPU 0/ }),
      ).toBeChecked();
      await expect(
        devices.getByRole("checkbox", { name: /GPU 1/ }),
      ).toBeChecked();
      await strategy.selectOption("split");
      await expect(
        page.getByRole("combobox", { name: "Precision", exact: true }),
      ).toHaveValue("fp32");
      await expect(
        page
          .getByRole("combobox", { name: "Precision", exact: true })
          .locator('option[value="fp16"]'),
      ).toHaveAttribute("disabled", "");
    }
    await strategy.selectOption("single");
    await page
      .getByRole("group", { name: "Training devices" })
      .getByRole("checkbox", { name: "CPU", exact: true })
      .check();
    await expect(page.getByRole("combobox", { name: "Run on" })).toBeVisible();
    await expect(
      page.getByRole("combobox", { name: "GPU strategy", exact: true }),
    ).toHaveCount(0);
    await expect(
      page.getByRole("textbox", { name: /Training dataset path/ }),
    ).toHaveCount(0);
    await page
      .getByRole("button", { name: "Python script", exact: true })
      .click();
    const templates = await (
      await context.request.get("/api/training/templates")
    ).json();
    expect(String(templates.scripts.causal)).not.toContain("DATA_PATTERN");
    expect(String(templates.scripts.causal)).not.toContain("encode_record");
    expect(String(templates.scripts.causal)).not.toContain("sequence_length");
    const dataHook = `from torch.utils.data import DataLoader
    from weave_support.data_stream import TokenStream
    from weave_support.token_cache import cache_documents
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast
    import time
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    def read_documents():
        time.sleep(10)
        yield "hello"
        yield "hello"
    cache_documents(
        "browser-document-cache", sources=[${JSON.stringify(data)}], tokenizer=tokenizer,
        read_documents=read_documents, policy={"reader": "browser-v1"}, total_documents=2,
    )
    def encode(row):
        return row["input_ids"], row["input_ids"]
    dataset = TokenStream(${JSON.stringify(data)}, encode, seq_len=4)
    return DataLoader(dataset, batch_size=config.micro_batch_size), [{"input_ids": torch.tensor([[1, 2, 3, 4]]), "labels": torch.tensor([[2, 3, 4, 1]])}]`;
    const source = String(templates.scripts.causal)
      .replace(
        'raise NotImplementedError("Add your data loaders here.")',
        dataHook,
      )
      .replace("micro_batch_size=16", "micro_batch_size=2")
      .replace("max_epochs=3", "max_epochs=None")
      .replace("max_optimizer_steps=None", "max_optimizer_steps=2")
      .replace("checkpoint_every=250", "checkpoint_every=1")
      .replace("eval_every=50", "eval_every=1")
      .replace("tokenizer=load_tokenizer,", "tokenizer=None,");
    await editor.fill(source);
    await editor.press("Control+s");
    releaseInitialSave();
    await expect(
      page.getByRole("button", { name: "Save file", exact: true }),
    ).toBeEnabled({ timeout: 30000 });
    await expect(page.locator(".training-notice")).toContainText(
      "Saved server file",
    );
    await page
      .getByRole("button", { name: "Launch settings", exact: true })
      .click();
    const validationMode = page.getByRole("combobox", {
      name: "Validation execution",
      exact: false,
    });
    const validationDevice = page.getByRole("combobox", {
      name: "Validation device",
      exact: false,
    });
    await expect(validationMode).toHaveValue("script");
    await expect(validationDevice).toBeDisabled();
    await validationMode.selectOption("async");
    await expect(validationDevice).toBeEnabled();
    if (await validationDevice.locator('option[value="1"]').count()) {
      await validationDevice.selectOption("1");
      await expect(validationDevice).toHaveValue("1");
    }
    await validationDevice.selectOption("cpu");
    await page
      .getByRole("button", { name: "Python script", exact: true })
      .click();
    await expect(editor).toContainText("async_evaluation=False");
    await page.getByRole("button", { name: "Save file", exact: true }).click();
    await expect(
      page.getByRole("button", { name: "Save file", exact: true }),
    ).toBeEnabled({ timeout: 30000 });
    const plan = await (
      await context.request.get(`/api/training/plans/${id}`)
    ).json();
    const saved = stageSchema.parse(plan.stages[0]);
    expect(saved.config.compute_mode).toBe("override");
    expect(saved.config.devices).toEqual([]);
    expect(saved.config.evaluation_mode).toBe("async");
    expect(saved.config.evaluation_device).toBe("cpu");
    projectPath = saved.project_path ?? null;
    expect(projectPath).toContain("code-hooks-browser-");
    expect(saved.script_path && existsSync(saved.script_path)).toBeTruthy();
    expect(saved.config_path).toContain("weave_support/launch.json");
    expect(
      existsSync(resolve(saved.script_path!, "..", "config.json")),
    ).toBeFalsy();
    await page
      .getByRole("button", { name: "Start training stage", exact: true })
      .click();
    await expect(
      page.getByRole("region", { name: "Run dashboard", exact: true }),
    ).toBeVisible({ timeout: 30000 });
    const preparation = page.getByRole("region", {
      name: "Data preparation progress",
      exact: true,
    });
    await expect(preparation).toContainText("0 / 2", { timeout: 60000 });
    await expect(preparation.getByRole("progressbar")).toBeVisible();
    await page.screenshot({
      path: "data/training-cache-preparation.png",
      fullPage: true,
    });
    await expect(
      page.locator(".training-dashboard .training-section-heading").first(),
    ).toContainText("completed", { timeout: 180000 });
    const runId = new URL(page.url()).searchParams.get("run");
    const detail = detailSchema.parse(
      await (await context.request.get(`/api/training/runs/${runId}`)).json(),
    );
    runDirectory = detail.directory;
    const runEndpoint = `**/api/training/runs/${runId}`;
    await page.route(runEndpoint, (route) =>
      route.fulfill({
        status: 500,
        contentType: "text/plain",
        body: "Internal Server Error",
      }),
    );
    await expect(page.getByRole("alert")).toContainText(
      "Live updates interrupted",
    );
    await expect(page.getByRole("alert")).toContainText("HTTP 500");
    await expect(page.getByRole("alert")).not.toContainText("Unexpected token");
    await expect(page.getByTestId("run-freshness")).toContainText(
      "Showing last successful update",
    );
    await page.unroute(runEndpoint);
    await expect(page.getByRole("alert")).toHaveCount(0, { timeout: 15000 });
    await expect(page.getByTestId("run-freshness")).toContainText(
      /Live monitoring|Run data refreshed/,
    );
    expect(detail.run.code_first).toBeTruthy();
    expect(detail.datasets["train-0"].path).toBe(data);
    expect(detail.datasets["train-0"].sha256).toBeUndefined();
    expect(
      detail.events.some((event) => event.kind === "data_trace"),
    ).toBeTruthy();
    await page.getByText("Run files & diagnostics", { exact: true }).click();
    const counters = page.getByLabel("Training work counters");
    await expect(counters).toBeVisible();
    await expect(counters).toContainText("Epoch tracking");
    await expect(counters).toContainText("Microbatches processed");
    await expect(
      counters.locator("div").filter({ hasText: "Optimizer updates" }),
    ).toContainText("2");
    await expect(
      counters.locator("div").filter({ hasText: "Microbatches processed" }),
    ).toContainText("4");
    await expect(
      counters.locator("div").filter({ hasText: "Input tokens processed" }),
    ).toContainText("32");
    await expect(
      page.getByRole("status", { name: "Background validation status" }),
    ).toContainText("completed");
    await expect(
      page.getByRole("status", { name: "Background validation status" }),
    ).toContainText("CPU");
    await page
      .getByRole("button", { name: "Checkpoints & snapshots", exact: true })
      .click();
    await expect(page.locator(".training-checkpoint")).toHaveCount(2);
    await page
      .getByRole("button", { name: "Create snapshot", exact: true })
      .click();
    await expect(
      page.getByRole("link", { name: /Download snapshot/ }),
    ).toBeVisible({ timeout: 15000 });
    await page.getByText("Dataset provenance", { exact: true }).click();
    await expect(page.locator(".training-dashboard")).toContainText(
      "content hash not requested",
    );
    await page.screenshot({
      path: "data/training-code-hooks-desktop.png",
      fullPage: true,
    });
    await page.reload();
    await expect(
      page.getByText(`Stage setup · ${saved.name}`, { exact: true }),
    ).toBeVisible();
    await expect(editor).not.toBeVisible();
    await page
      .getByText(`Stage setup · ${saved.name}`, { exact: true })
      .click();
    await expect(editor).toContainText("TrainingConfig(", { timeout: 30000 });
    await page.setViewportSize({ width: 390, height: 844 });
    expect(
      await page
        .locator(".training-workspace")
        .evaluate((el) => el.scrollWidth <= el.clientWidth + 1),
    ).toBeTruthy();
    await page.screenshot({
      path: "data/training-code-hooks-phone.png",
      fullPage: true,
    });
    // A terminal stop must not look like an active process. Reopen the next
    // launch controls without launching anything or changing the real run.
    await page.route(runEndpoint, async (route) => {
      const response = await route.fetch();
      const stopped = detailSchema.parse(await response.json());
      stopped.status.state = "stopped";
      await route.fulfill({ response, json: stopped });
    });
    await page.reload();
    await expect(
      page.getByRole("status", { name: "Training stopped", exact: true }),
    ).toBeVisible();
    await expect(
      page.locator(".training-setup-card > details"),
    ).toHaveAttribute("open", "");
    await expect(
      page.getByRole("button", { name: "Start training stage", exact: true }),
    ).toBeVisible();
    await expect(page.getByTestId("run-freshness")).toContainText(
      "Run data refreshed",
    );
    await expect(
      page.getByRole("button", { name: "Pause training", exact: true }),
    ).toHaveCount(0);
    expect(errors).toEqual([]);
  } finally {
    releaseInitialSave();
    const activeRun = new URL(page.url()).searchParams.get("run");
    await page.goto("about:blank");
    if (activeRun) {
      const response = await context.request.get(
        `/api/training/runs/${activeRun}`,
      );
      if (response.ok()) {
        const current = detailSchema.parse(await response.json());
        runDirectory = current.directory;
        if (
          ["starting", "running", "preparing"].includes(current.status.state)
        ) {
          await context.request.post(
            `/api/training/runs/${activeRun}/commands`,
            { data: { kind: "kill" } },
          );
          await expect
            .poll(
              async () => {
                const result = await context.request.get(
                  `/api/training/runs/${activeRun}`,
                );
                return String((await result.json()).status.state);
              },
              { timeout: 15000 },
            )
            .not.toMatch(/^(starting|running)$/);
        }
      }
    }
    await context.request.delete(`/api/projects/${id}`);
    rmSync(data, { force: true });
    if (
      projectPath &&
      JSON.parse(readFileSync(resolve(projectPath, "project.json"), "utf8"))
        .id === id
    )
      rmSync(projectPath, { recursive: true, force: true });
    if (
      runDirectory &&
      JSON.parse(readFileSync(resolve(runDirectory, "run.json"), "utf8"))
        .project_id === id
    )
      rmSync(runDirectory, {
        recursive: true,
        force: true,
        maxRetries: 5,
        retryDelay: 100,
      });
  }
});
