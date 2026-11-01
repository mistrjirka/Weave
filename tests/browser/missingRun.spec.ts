import { expect, test } from "@playwright/test";
import { rmSync } from "node:fs";
import {
  detailSchema,
  runSchema,
  stageSchema,
  templateSchema,
} from "../../frontend/src/trainingApi";

test("missing run links recover on mobile, while valid history and transient failures stay selected", async ({
  page,
  context,
}) => {
  test.setTimeout(90000);
  const id = `project-missing-run-${Date.now()}`;
  const project = {
    version: 1,
    id,
    name: "Missing run recovery test",
    description: "Temporary fixture",
    config: { width: 8, vocab: 8, heads: 2, ffn_width: 16 },
    graph: {
      nodes: [
        {
          id: "ids",
          kind: "input",
          params: { index: 0, shape: ["B", "T"], dtype: "int64" },
        },
        { id: "emb", kind: "embedding", params: { width: 8, vocab: 8 } },
        { id: "head", kind: "head", params: { out: 8 } },
        { id: "out", kind: "output", params: {} },
      ].map(({ id, kind, params }, index) => ({
        id,
        type: "architecture",
        position: { x: index * 200, y: 0 },
        data: { kind, params, label: kind },
      })),
      edges: [
        ["ids", "emb"],
        ["emb", "head"],
        ["head", "out"],
      ].map(([source, target]) => ({
        id: `${source}-${target}`,
        source,
        target,
        targetHandle: "a",
      })),
    },
  };
  let directory: string | undefined;
  let projectPath: string | null | undefined;
  try {
    await context.request.put(`/api/projects/${id}`, { data: project });
    const templates = templateSchema.parse(
      await (await context.request.get("/api/training/templates")).json(),
    );
    const stage = stageSchema.parse({
      id: "stage-ui",
      name: "Pretraining",
      script: templates.scripts.causal,
      config: {
        ...templates.config,
        execution: "monitor",
        devices: [],
        precision: "fp32",
      },
    });
    const saved = stageSchema.parse(
      await (
        await context.request.put(
          `/api/training/workspaces/${id}/stages/${stage.id}`,
          { data: { project, stage } },
        )
      ).json(),
    );
    projectPath = saved.project_path;
    const run = runSchema.parse(
      await (
        await context.request.post("/api/training/runs", {
          data: { project, stage: saved },
        })
      ).json(),
    );
    const detail = detailSchema.parse(
      await (await context.request.get(`/api/training/runs/${run.id}`)).json(),
    );
    directory = detail.directory;
    const older = { ...run, status: { state: "stopped" } };
    const latest = {
      ...older,
      id: `${run.id}-latest`,
      status: { state: "running" },
    };
    let available = [latest, older];
    let responseStatus = 200;
    await page.route(`**/api/training/runs?project_id=${id}`, (route) =>
      route.fulfill({ json: available }),
    );
    await page.route(`**/api/training/runs/${run.id}`, (route) =>
      route.fulfill({
        status: responseStatus,
        json:
          responseStatus === 200
            ? { ...detail, run: older, status: older.status }
            : { detail: "Training run not found" },
      }),
    );
    await page.route(`**/api/training/runs/${latest.id}`, (route) =>
      route.fulfill({
        json: { ...detail, run: latest, status: latest.status },
      }),
    );
    await page.route(`**/api/training/runs/${latest.id}/events`, (route) =>
      route.fulfill({
        contentType: "text/event-stream",
        body: ": connected\n\n",
      }),
    );
    await page.setViewportSize({ width: 390, height: 844 });
    const base = `/?model=${id}&view=training&stage=${stage.id}`;
    await page.goto(`${base}&run=run-`);
    await expect(page).toHaveURL(
      (url) => url.searchParams.get("run") === latest.id,
    );
    await expect(
      page.getByRole("status").filter({ hasText: /run is unavailable/ }),
    ).toBeVisible();
    await expect(
      page.getByText("Loading selected run…", { exact: true }),
    ).not.toBeVisible();
    await expect(page.getByText(/Live updates interrupted/)).not.toBeVisible();

    // A deliberately selected historical run must stay selected.
    await page.goto(`${base}&run=${run.id}`);
    await expect(
      page.getByText("Loading selected run…", { exact: true }),
    ).not.toBeVisible();
    await expect(page).toHaveURL(
      (url) => url.searchParams.get("run") === run.id,
    );
    responseStatus = 500;
    await expect(page.getByText(/Live updates interrupted/)).toBeVisible({
      timeout: 15000,
    });
    await expect(page).toHaveURL(
      (url) => url.searchParams.get("run") === run.id,
    );

    // If the selected run really disappears, recover during polling too.
    available = [latest];
    responseStatus = 404;
    await expect(page).toHaveURL(
      (url) => url.searchParams.get("run") === latest.id,
      {
        timeout: 15000,
      },
    );
    await expect(page.getByText(/Live updates interrupted/)).not.toBeVisible();

    // An empty project should offer setup, rather than wait forever.
    available = [];
    await page.goto(`${base}&run=run-`);
    await expect(page).not.toHaveURL(/[?&]run=/);
    await expect(
      page.getByRole("status").filter({ hasText: "No saved runs were found" }),
    ).toBeVisible();
    await expect(
      page.getByText("Loading selected run…", { exact: true }),
    ).not.toBeVisible();
    await expect(
      page.getByRole("textbox", { name: "Stage name", exact: true }),
    ).toBeVisible();
  } finally {
    await page.goto("about:blank");
    await context.request.delete(`/api/projects/${id}`);
    if (directory) rmSync(directory, { recursive: true, force: true });
    if (projectPath) rmSync(projectPath, { recursive: true, force: true });
  }
});
