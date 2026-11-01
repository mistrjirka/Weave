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

async function trainingFixture(page: Page) {
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
    status: { state: "running" },
    events: [],
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
  await page.route("**/api/training/plans/*", (route) =>
    route.fulfill({
      json: { version: 1, project_id: project.id, stages: [stage] },
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
