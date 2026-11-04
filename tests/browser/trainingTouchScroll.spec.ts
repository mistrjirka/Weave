import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { projectSchema } from "../../frontend/src/api";
import { templateSchema } from "../../frontend/src/trainingApi";
import type { RunDetail, TrainingRun } from "../../frontend/src/trainingApi";

test.use({
  viewport: { width: 390, height: 844 },
  hasTouch: true,
  isMobile: true,
});

async function trainingFixture(page: Page, completed = false) {
  await page.goto("/");
  await expect(page.locator('.react-flow__node[data-id="ids"]')).toBeVisible();
  const project = projectSchema.parse(
    await page.evaluate(() =>
      JSON.parse(localStorage.getItem("weave.project")!),
    ),
  );
  const template = templateSchema.parse(
    await (await page.request.get("/api/training/templates")).json(),
  );
  const stage = {
    id: "touch-stage",
    name: "Touch stage",
    config: template.config,
    script: template.scripts.causal,
  };
  const run: TrainingRun = {
    id: "run-touch",
    project_id: project.id,
    stage_id: stage.id,
    name: stage.name,
    created: new Date().toISOString(),
    recipe: "causal",
    strategy: "ddp",
    execution: "monitor",
    parent: null,
    warnings: [],
    model: { modules: [], split_supported: false, split_reasons: [] },
  };
  const detail: RunDetail = {
    run,
    status: { state: completed ? "completed" : "running", updated: 1793753560 },
    events: completed
      ? [
          {
            version: 1,
            id: "final-progress",
            kind: "progress",
            time: 1793753500,
            step: 207,
            tokens: 0,
            channel: "training",
            payload: { completed: 828, total: 1000, unit: "rollout groups" },
          },
        ]
      : [],
    hardware: [],
    logs: "",
    evaluation_logs: "",
    diagnostics: [],
    snapshots: [],
    disk_bytes: 0,
    source_data_bytes: 0,
    config: stage.config,
    module_nodes: {},
    directory: "/tmp/touch-fixture",
    script: stage.script,
    datasets: {},
  };
  const earlier = {
    ...stage,
    id: "earlier-stage",
    name: "Earlier stage",
    script: stage.script + "\n# Unsaved draft\n",
  };
  if (completed) {
    await page.evaluate(
      ({ projectId, draft }) => {
        localStorage.setItem(
          `weave:training:${projectId}`,
          JSON.stringify({
            version: 1,
            project_id: projectId,
            stages: [draft],
          }),
        );
      },
      { projectId: project.id, draft: earlier },
    );
  }
  await page.route("**/api/training/plans/*", (route) =>
    route.fulfill({
      json: {
        version: 1,
        project_id: project.id,
        stages: completed
          ? [{ ...earlier, script: stage.script }, stage]
          : [stage],
      },
    }),
  );
  await page.route("**/api/training/runs?project_id=*", (route) =>
    route.fulfill({ json: [run] }),
  );
  await page.route("**/api/training/runs/run-touch", (route) =>
    route.fulfill({ json: detail }),
  );
  await page.route("**/api/training/workspaces/*/stages/touch-stage", (route) =>
    route.fulfill({ json: stage }),
  );
  return `/?model=${encodeURIComponent(project.id)}&stage=touch-stage&run=run-touch`;
}

for (const width of [390, 1440])
  test(`finished run shows completion instead of an active partial progress bar at ${width}px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 1000 });
    const base = await trainingFixture(page, true);
    const link = new URL(base, "http://127.0.0.1:8173");
    link.searchParams.delete("stage");
    link.searchParams.set("view", "training");
    await page.goto(link.pathname + link.search);
    const completed = page.getByRole("status", {
      name: "Training completed",
      exact: true,
    });
    await expect(completed).toBeVisible();
    await expect(
      page.getByRole("combobox", { name: "Training stage", exact: true }),
    ).toHaveValue("touch-stage");
    const restoredDraft = await page.evaluate(() =>
      localStorage.getItem(
        `weave:training:${new URL(location.href).searchParams.get("model")}`,
      ),
    );
    expect(restoredDraft).toContain("# Unsaved draft");
    await expect(completed).toContainText("Run finished");
    await expect(completed.locator("time")).toHaveAttribute(
      "datetime",
      new Date(1793753560000).toISOString(),
    );
    await expect(
      page.getByRole("progressbar", { name: "Training progress", exact: true }),
    ).toHaveCount(0);
    await expect(
      page.getByText("Last reported progress · 828 / 1,000 rollout groups", {
        exact: true,
      }),
    ).toBeVisible();
    await expect(
      page.getByLabel("Training finish estimate", { exact: true }),
    ).toHaveCount(0);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  });

async function swipeUp(page: Page) {
  const cdp = await page.context().newCDPSession(page);
  try {
    await cdp.send("Input.dispatchTouchEvent", {
      type: "touchStart",
      touchPoints: [{ x: 12, y: 650 }],
    });
    for (let y = 630; y >= 220; y -= 20) {
      await cdp.send("Input.dispatchTouchEvent", {
        type: "touchMove",
        touchPoints: [{ x: 12, y }],
      });
      await page.waitForTimeout(16);
    }
    await cdp.send("Input.dispatchTouchEvent", {
      type: "touchEnd",
      touchPoints: [],
    });
  } finally {
    await cdp.detach();
  }
}

for (const mode of ["copied link", "tab switch"] as const)
  test(`phone Training accepts touch scrolling after ${mode} with properties open`, async ({
    page,
  }) => {
    const base = await trainingFixture(page);
    if (mode === "copied link")
      await page.goto(base + "&view=training&panel=properties&node=ids");
    else {
      await page.setViewportSize({ width: 1440, height: 900 });
      await page.goto(base + "&panel=properties&node=ids");
      await page.getByRole("button", { name: "Training", exact: true }).click();
      await page.setViewportSize({ width: 390, height: 844 });
    }
    await expect(
      page.getByRole("region", { name: "Run dashboard", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Close side panel", exact: true }),
    ).toHaveCount(0);
    await expect
      .poll(() => new URL(page.url()).searchParams.get("panel"))
      .toBeNull();
    const scroller = page.locator("section.training-workspace");
    await swipeUp(page);
    await expect
      .poll(() => scroller.evaluate((el) => el.scrollTop))
      .toBeGreaterThan(100);
    const position = await scroller.evaluate((el) => el.scrollTop);
    await page.waitForTimeout(2500); // live polling must keep the user's scroll position
    expect(
      await scroller.evaluate((el) => el.scrollTop),
    ).toBeGreaterThanOrEqual(position - 2);
    await page.screenshot({
      path: `test-results/training-touch-${mode.replaceAll(" ", "-")}.png`,
      fullPage: true,
    });
  });
