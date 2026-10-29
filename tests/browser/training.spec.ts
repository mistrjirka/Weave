import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

const token = () =>
  process.env.WEAVE_TRAINING_TOKEN ??
  readFileSync("data/training/operator.token", "utf8").trim();
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
  await context.request.post("/api/training/session", {
    data: { token: token() },
  });
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
  await context.request.post("/api/training/session", {
    data: { token: token() },
  });
  const id = `project-training-responsive-${Date.now()}`;
  await context.request.put(`/api/projects/${id}`, { data: project(id) });
  try {
    await page.goto(`/?model=${id}&view=training`);
    await page
      .getByRole("button", { name: "Create training stage", exact: true })
      .click();
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
