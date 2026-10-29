import { test, expect, type APIRequestContext } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { detailSchema } from "../../frontend/src/trainingApi";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

async function unlockIfConfigured(request: APIRequestContext) {
  const capabilities = await (
    await request.get("/api/training/capabilities")
  ).json();
  if (capabilities.auth_required) {
    if (!process.env.WEAVE_TRAINING_TOKEN)
      throw new Error(
        "Set WEAVE_TRAINING_TOKEN for tests against an explicitly token-protected server",
      );
    await request.post("/api/training/session", {
      data: { token: process.env.WEAVE_TRAINING_TOKEN },
    });
  }
}
const project = (id: string) => ({
  version: 1,
  id,
  name: "Training UI acceptance",
  description: "Temporary training acceptance fixture",
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
        position: { x: 250, y: 0 },
        data: {
          kind: "embedding",
          label: "Embedding",
          params: { width: 8, vocab: 8 },
        },
      },
      {
        id: "head",
        type: "architecture",
        position: { x: 500, y: 0 },
        data: { kind: "head", label: "Vocabulary", params: { out: 8 } },
      },
      {
        id: "out",
        type: "architecture",
        position: { x: 750, y: 0 },
        data: { kind: "output", label: "Logits", params: {} },
      },
    ],
    edges: [
      { id: "one", source: "ids", target: "emb", targetHandle: "a" },
      { id: "two", source: "emb", target: "head", targetHandle: "a" },
      { id: "three", source: "head", target: "out", targetHandle: "a" },
    ],
  },
});

// Plain LAN HTTP lacks randomUUID even though localhost exposes it.
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(Crypto.prototype, "randomUUID", {
      value: undefined,
      configurable: true,
    });
  });
});

test("training stages, real run, charts, snapshots, and reload", async ({
  page,
  context,
}) => {
  test.setTimeout(120000);
  const id = `project-training-${Date.now()}`;
  await unlockIfConfigured(context.request);
  await context.request.put(`/api/projects/${id}`, { data: project(id) });
  const dataset = resolve("data", `${id}.jsonl`);
  writeFileSync(
    dataset,
    Array.from({ length: 10 }, () =>
      JSON.stringify({ input_ids: [1, 2, 3, 4, 1, 2, 3, 4] }),
    ).join("\n"),
  );
  try {
    await page.goto(`/?model=${id}&view=training`);
    await expect(
      page.getByRole("heading", { name: "Training", exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Create training stage", exact: true })
      .click();
    await page
      .getByRole("textbox", { name: "Stage name", exact: true })
      .fill("Pretraining acceptance");
    await page
      .getByRole("button", { name: "Preset settings", exact: true })
      .click();
    await page
      .getByRole("textbox", { name: "Training dataset path", exact: false })
      .fill(dataset);
    await page
      .getByRole("button", { name: "Preview tokens & loss mask", exact: true })
      .click();
    await expect(page.locator(".training-data-preview")).toContainText(
      "7 supervised target tokens",
    );
    await page
      .getByText("Optimizer, schedule, and evaluation", { exact: true })
      .click();
    await page
      .getByRole("spinbutton", { name: "Total optimizer steps", exact: false })
      .fill("4");
    await page
      .getByRole("spinbutton", { name: "Checkpoint interval", exact: false })
      .fill("2");
    await page
      .getByRole("button", { name: "Save stages", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Duplicate stage", exact: true })
      .click();
    await expect(page.locator(".training-stage")).toHaveCount(2);
    await page
      .getByRole("button", { name: "Move stage up", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Delete stage", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Undo stage deletion", exact: true })
      .click();
    await expect(page.locator(".training-stage")).toHaveCount(2);
    await page
      .getByRole("button", { name: "Add training stage", exact: true })
      .click();
    await expect(page.locator(".training-stage")).toHaveCount(3);
    await page
      .getByRole("button", { name: "Delete stage", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Python script", exact: true })
      .click();
    await page
      .getByRole("button", {
        name: "Open fresh template as another stage",
        exact: true,
      })
      .click();
    await expect(page.locator(".training-stage")).toHaveCount(3);
    await page
      .getByRole("button", { name: "Start training stage", exact: true })
      .click();
    await expect(
      page.getByRole("region", { name: "Run dashboard", exact: true }),
    ).toBeVisible({ timeout: 45000 });
    await expect(
      page.locator(".training-dashboard .training-section-heading").first(),
    ).toContainText("completed", { timeout: 60000 });
    await expect(page.locator(".training-chart")).toHaveCount(4);
    await page
      .getByRole("button", { name: "Create stage from this run", exact: true })
      .click();
    await expect(page.locator(".training-stage")).toHaveCount(4);
    await page.getByRole("button", { name: "Hardware", exact: true }).click();
    await expect(page.locator(".training-dashboard")).toContainText(
      "GPU readings are device-wide",
    );
    await page.getByRole("button", { name: "Logs", exact: true }).click();
    await expect(page.locator(".training-log")).toBeVisible();
    await page
      .getByRole("button", { name: "Generated examples", exact: true })
      .click();
    await expect(page.locator(".training-dashboard")).toContainText(
      "Add fixed evaluation prompts",
    );
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
    const snapshotLink = page.getByRole("link", { name: /Download snapshot/ });
    const download = await context.request.get(
      (await snapshotLink.getAttribute("href"))!,
    );
    expect(download.ok()).toBeTruthy();
    expect((await download.body()).length).toBeGreaterThan(1000);
    const url = page.url();
    expect(url).toContain("view=training");
    expect(url).toContain("run=");
    expect(url).toContain("stage=");
    await page.reload();
    await expect(
      page.getByRole("region", { name: "Run dashboard", exact: true }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Metrics", exact: true }).click();
    await expect(page.locator(".training-chart")).toHaveCount(4);
    await page
      .getByRole("combobox", { name: "Horizontal axis", exact: true })
      .selectOption("tokens");
    await page.getByRole("checkbox", { name: /Smooth curves/ }).check();
    await page.screenshot({
      path: "data/training-desktop.png",
      fullPage: true,
    });
    const violations = await new AxeBuilder({ page })
      .include(".training-workspace")
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(
      violations.violations.map((v) => ({
        id: v.id,
        nodes: v.nodes.map((n) => n.failureSummary),
      })),
    ).toEqual([]);
  } finally {
    await context.request.delete(`/api/projects/${id}`);
  }
});

test("training workspace fits black theme and phone", async ({
  page,
  context,
}) => {
  await unlockIfConfigured(context.request);
  const id = `project-training-responsive-${Date.now()}`;
  await context.request.put(`/api/projects/${id}`, { data: project(id) });
  try {
    await page.goto(`/?model=${id}&view=training`);
    await page
      .getByRole("button", { name: "Create training stage", exact: true })
      .click();
    await expect(
      page.getByRole("textbox", {
        name: "Training Python script",
        exact: true,
      }),
    ).toBeVisible();
    await page
      .getByRole("combobox", { name: "Color theme", exact: true })
      .selectOption("black");
    const audit = await new AxeBuilder({ page })
      .include(".training-workspace")
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(
      audit.violations.map((v) => ({
        id: v.id,
        nodes: v.nodes.map((n) => n.failureSummary),
      })),
    ).toEqual([]);
    await page.screenshot({ path: "data/training-black.png", fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await expect(
      page.getByRole("textbox", { name: "Stage name", exact: true }),
    ).toBeVisible();
    expect(
      await page
        .locator(".training-workspace")
        .evaluate((el) => el.scrollWidth <= el.clientWidth + 1),
    ).toBeTruthy();
    await page
      .getByRole("button", { name: "Python script", exact: true })
      .click();
    await expect(
      page.getByRole("textbox", {
        name: "Training Python script",
        exact: true,
      }),
    ).toBeVisible();
    await page.screenshot({ path: "data/training-phone.png", fullPage: true });
  } finally {
    await context.request.delete(`/api/projects/${id}`);
  }
});

test("real server scripts sync with SSH edits and use a highlighted Python editor", async ({
  page,
  context,
}) => {
  test.setTimeout(90000);
  await unlockIfConfigured(context.request);
  const id = `project-training-files-${Date.now()}`;
  await context.request.put(`/api/projects/${id}`, { data: project(id) });
  try {
    await page.goto(`/?model=${id}&view=training`);
    await page
      .getByRole("button", { name: "Create training stage", exact: true })
      .click();
    const editor = page.getByRole("textbox", {
      name: "Training Python script",
      exact: true,
    });
    await expect(editor).toBeVisible();
    await expect(page.locator(".training-server-files")).toContainText(
      "server paths",
    );
    await expect(
      page
        .locator(".training-server-path")
        .filter({ has: page.getByText("Python script", { exact: true }) })
        .locator("code"),
    ).toContainText("/train.py");
    const path = (
      await page
        .locator(".training-server-path")
        .filter({ has: page.getByText("Python script", { exact: true }) })
        .locator("code")
        .innerText()
    ).trim();
    expect(existsSync(path)).toBeTruthy();
    await expect(page.locator(".cm-gutters")).toBeVisible();
    expect(await page.locator(".cm-line span").count()).toBeGreaterThan(0);
    await expect(page.locator(".training-python-environment")).toContainText(
      /Python \d+\.\d+/,
      { timeout: 35000 },
    );
    await page.screenshot({
      path: "data/training-script-editor.png",
      fullPage: true,
    });
    await editor.fill("print('browser draft')\n");
    // Old browser drafts predate on-disk workspace paths. Keep their content.
    await page.evaluate((projectId) => {
      const key = `weave:training:${projectId}`;
      const draft = JSON.parse(localStorage.getItem(key) ?? "{}");
      for (const item of draft.stages ?? []) {
        delete item.script_path;
        delete item.config_path;
      }
      localStorage.setItem(key, JSON.stringify(draft));
    }, id);
    await page.reload();
    await expect(editor).toContainText("browser draft");
    await expect(
      page.getByRole("button", { name: "Reload server files", exact: true }),
    ).toBeEnabled();
    await expect(page.locator(".training-server-files")).toContainText(
      "not saved",
    );
    writeFileSync(path, "print('SSH edit')\n");
    await expect(page.locator(".training-server-files")).toContainText(
      "Files changed on the server",
      { timeout: 12000 },
    );
    await page.getByRole("button", { name: "Save file", exact: true }).click();
    await expect(
      page.locator(".training-workspace > [role=alert]"),
    ).toContainText("changed on the server");
    expect(readFileSync(path, "utf8")).toBe("print('SSH edit')\n");
    await page
      .getByRole("button", { name: "Reload server files", exact: true })
      .click();
    await expect(editor).toContainText("SSH edit");
    await editor.fill("print('saved through browser')\n");
    await editor.press("Control+s");
    await expect
      .poll(() => readFileSync(path, "utf8"))
      .toBe("print('saved through browser')\n");
    writeFileSync(path, "print('another SSH edit')\n");
    await expect(editor).toContainText("another SSH edit", { timeout: 12000 });
    await page
      .getByRole("combobox", { name: "Insert reporting hook", exact: true })
      .selectOption("Log scalar metrics");
    await expect(editor).toContainText("run.log");
    await editor.press("Control+s");
    await expect.poll(() => readFileSync(path, "utf8")).toContain("run.log");
    const configPath = (
      await page
        .locator(".training-server-path")
        .filter({ has: page.getByText("Preset values", { exact: true }) })
        .locator("code")
        .innerText()
    ).trim();
    const config = JSON.parse(readFileSync(configPath, "utf8"));
    config.learning_rate = 0.0123;
    writeFileSync(configPath, JSON.stringify(config));
    await page
      .getByRole("button", { name: "Preset settings", exact: true })
      .click();
    await page
      .getByText("Optimizer, schedule, and evaluation", { exact: true })
      .click();
    await expect(
      page.getByRole("spinbutton", { name: "Learning rate", exact: true }),
    ).toHaveValue("0.0123", { timeout: 12000 });
  } finally {
    await context.request.delete(`/api/projects/${id}`);
  }
});

test("independent training has a localhost reporting address and an editable connection file", async ({
  page,
  context,
}) => {
  const id = `project-training-report-${Date.now()}`;
  await unlockIfConfigured(context.request);
  await context.request.put(`/api/projects/${id}`, { data: project(id) });
  try {
    await page.goto(`/?model=${id}&view=training`);
    await page
      .getByRole("button", { name: "Create training stage", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Preset settings", exact: true })
      .click();
    await page
      .getByRole("combobox", { name: "Training recipe" })
      .selectOption("custom");
    await page
      .getByRole("combobox", { name: "Run on" })
      .selectOption("monitor");
    await page
      .getByRole("button", { name: "Prepare training package", exact: true })
      .click();
    const reporting = page.getByRole("region", {
      name: "Independent training reporting",
    });
    await expect(reporting).toBeVisible();
    await expect(
      reporting.getByRole("textbox", { name: "Reporting server URL" }),
    ).toHaveValue("http://localhost:8173");
    await expect(reporting).toContainText("/api/training/reports/run-");
    await reporting
      .getByRole("textbox", { name: "Reporting server URL" })
      .fill("http://10.8.0.8:8173");
    await reporting
      .getByRole("button", { name: "Save reporting address", exact: true })
      .click();
    await expect(reporting).toContainText(
      "http://10.8.0.8:8173/api/training/reports/",
    );
    const runId = new URL(page.url()).searchParams.get("run");
    const detail = await (
      await context.request.get(`/api/training/runs/${runId}`)
    ).json();
    const connection = JSON.parse(
      readFileSync(`${detail.directory}/reporting.json`, "utf8"),
    );
    expect(connection.url).toBe(
      `http://10.8.0.8:8173/api/training/reports/${runId}`,
    );
    expect(connection.token.length).toBeGreaterThan(20);
    await page.reload();
    await expect(
      page.getByRole("region", { name: "Independent training reporting" }),
    ).toContainText(connection.url);
    const audit = await new AxeBuilder({ page })
      .include(".training-workspace")
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(
      audit.violations.map((item) => ({
        id: item.id,
        nodes: item.nodes.map((node) => ({
          html: node.html,
          summary: node.failureSummary,
        })),
      })),
    ).toEqual([]);
    await reporting.scrollIntoViewIfNeeded();
    await page.screenshot({
      path: "data/training-independent-reporting.png",
      fullPage: true,
    });
  } finally {
    await context.request.delete(`/api/projects/${id}`);
  }
});

test("simple custom reporting infers steps, learning rates and progress", async ({
  page,
  context,
}) => {
  test.setTimeout(120000);
  await unlockIfConfigured(context.request);
  const id = `project-simple-reporting-${Date.now()}`;
  await context.request.put(`/api/projects/${id}`, { data: project(id) });
  try {
    await page.goto(`/?model=${id}&view=training`);
    await page
      .getByRole("button", { name: "Create training stage", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Preset settings", exact: true })
      .click();
    await page
      .getByRole("combobox", { name: "Training recipe" })
      .selectOption("custom");
    await page.getByRole("checkbox", { name: "CPU", exact: true }).check();
    await page
      .getByRole("button", { name: "Python script", exact: true })
      .click();
    const editor = page.getByRole("textbox", {
      name: "Training Python script",
      exact: true,
    });
    await expect(editor).toContainText(
      "with Run(optimizer=optimizer, total_steps=total_steps)",
    );
    await editor.fill(`import torch
from torch.nn import functional as F
from model import Model
from weave_training import Run

torch.set_num_threads(1)
model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
with Run(optimizer=optimizer, total_steps=2) as run:
    for batch in [torch.tensor([[1, 2, 3, 4]])] * 2:
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 8), batch[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        run.log_loss(loss)
`);
    await page.getByRole("button", { name: "Save file", exact: true }).click();
    await page
      .getByRole("button", { name: "Start training stage", exact: true })
      .click();
    await expect(
      page.locator(".training-dashboard .training-section-heading").first(),
    ).toContainText("completed", { timeout: 90000 });
    const runId = new URL(page.url()).searchParams.get("run");
    const detail = detailSchema.parse(
      await (await context.request.get(`/api/training/runs/${runId}`)).json(),
    );
    const lossEvents = detail.events.filter(
      (event) => event.kind === "metrics" && "train/loss" in event.payload,
    );
    const lrEvents = detail.events.filter(
      (event) => event.kind === "metrics" && "lr/group_0" in event.payload,
    );
    const progress = detail.events.filter((event) => event.kind === "progress");
    expect(lossEvents.map((event) => event.step)).toEqual([1, 2]);
    expect(lrEvents).toHaveLength(2);
    expect(progress.at(-1)?.payload).toMatchObject({ completed: 2, total: 2 });
    expect(detail.events.some((event) => event.kind === "warning")).toBeFalsy();
    await expect(page.locator(".training-chart")).toHaveCount(2);
  } finally {
    await context.request.delete(`/api/projects/${id}`);
  }
});
