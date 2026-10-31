import { expect, test } from "@playwright/test";
import { rmSync } from "node:fs";
import {
  detailSchema,
  stageSchema,
  templateSchema,
  runSchema,
} from "../../frontend/src/trainingApi";

test("launch saves first, reports failures beside Start, and selects the new run; hardware has meaningful empty states", async ({
  page,
  context,
}) => {
  test.setTimeout(90000);
  const id = `project-launch-feedback-${Date.now()}`;
  const project = {
    version: 1,
    id,
    name: id,
    description: "Temporary launch feedback test",
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
  const templates = templateSchema.parse(
    await (await context.request.get("/api/training/templates")).json(),
  );
  const stage = stageSchema.parse({
    id: "launch-feedback",
    name: "Old stage",
    script: templates.scripts.causal,
    config: {
      ...templates.config,
      execution: "monitor",
      devices: [],
      precision: "fp32",
    },
  });
  const directories: string[] = [];
  let projectPath: string | null | undefined;
  let releaseSave: () => void = () => {};
  let releaseLaunch: () => void = () => {};
  try {
    expect(
      (
        await context.request.put(`/api/projects/${id}`, { data: project })
      ).ok(),
    ).toBeTruthy();
    const endpoint = `/api/training/workspaces/${id}/stages/${stage.id}`;
    const saved = stageSchema.parse(
      await (
        await context.request.put(endpoint, { data: { project, stage } })
      ).json(),
    );
    projectPath = saved.project_path;
    const oldRun = runSchema.parse(
      await (
        await context.request.post("/api/training/runs", {
          data: { project, stage: saved },
        })
      ).json(),
    );
    directories.push(
      detailSchema.parse(
        await (
          await context.request.get(`/api/training/runs/${oldRun.id}`)
        ).json(),
      ).directory,
    );
    await page.goto(
      `/?model=${id}&view=training&stage=${stage.id}&run=${oldRun.id}`,
    );
    await expect(
      page.getByRole("region", { name: "Run dashboard", exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Training devices", exact: true })
      .click();
    await page
      .getByRole("textbox", { name: "Stage name", exact: true })
      .fill("Edited before Start");
    const calls: string[] = [];
    let mode: "save-failure" | "launch-failure" | "success" = "save-failure";
    const saveGate = new Promise<void>((resolve) => {
      releaseSave = resolve;
    });
    const launchGate = new Promise<void>((resolve) => {
      releaseLaunch = resolve;
    });
    await page.route(`**${endpoint}`, async (route) => {
      if (route.request().method() !== "PUT") return route.continue();
      calls.push("save");
      if (mode === "save-failure") {
        await saveGate;
        return route.fulfill({
          status: 409,
          json: {
            detail: "train.py changed on the server. Reload the server files.",
          },
        });
      }
      const response = await route.fetch();
      await route.fulfill({ response });
    });
    await page.route("**/api/training/runs", async (route) => {
      if (route.request().method() !== "POST") return route.continue();
      calls.push("launch");
      if (mode === "launch-failure") {
        await launchGate;
        return route.fulfill({
          status: 422,
          json: { detail: "Device configuration rejected for testing." },
        });
      }
      const response = await route.fetch();
      const run = runSchema.parse(await response.json());
      directories.push(
        detailSchema.parse(
          await (
            await context.request.get(`/api/training/runs/${run.id}`)
          ).json(),
        ).directory,
      );
      await route.fulfill({ response });
    });
    await page.setViewportSize({ width: 390, height: 844 });
    const start = page.locator(".training-launch button.primary");
    await start.click();
    await expect(start).toBeDisabled();
    await expect(start).toHaveText("Saving stage…");
    await expect(
      page.getByRole("region", { name: "Run dashboard", exact: true }),
    ).toHaveCount(0);
    expect(calls).toEqual(["save"]);
    releaseSave();
    const error = page.getByRole("alert", { name: "Training launch failed" });
    await expect(error).toContainText("train.py changed on the server");
    await expect(error).toBeInViewport();
    await expect(start).toBeEnabled();
    expect(calls).toEqual(["save"]);
    mode = "launch-failure";
    await start.click();
    await expect(start).toHaveText("Starting training…");
    await expect(start).toBeDisabled();
    expect(calls).toEqual(["save", "save", "launch"]);
    releaseLaunch();
    await expect(error).toContainText("Device configuration rejected");
    await expect(error).toBeInViewport();
    await expect(
      page.getByRole("region", { name: "Run dashboard", exact: true }),
    ).toContainText("Previous run");
    mode = "success";
    await start.click();
    await expect
      .poll(() => new URL(page.url()).searchParams.get("run"), {
        timeout: 30000,
      })
      .not.toBe(oldRun.id);
    const newRun = new URL(page.url()).searchParams.get("run");
    const detail = detailSchema.parse(
      await (await context.request.get(`/api/training/runs/${newRun}`)).json(),
    );
    expect(detail.run.name).toBe("Edited before Start");
    expect(calls).toEqual(["save", "save", "launch", "save", "launch"]);
    const diskStage = stageSchema.parse(
      await (await context.request.get(endpoint)).json(),
    );
    expect(diskStage.name).toBe("Edited before Start");
    await expect(error).toHaveCount(0);
    await expect(
      page.getByRole("region", { name: "Run dashboard", exact: true }),
    ).toBeInViewport();
    await expect(page.getByLabel("Latest training metrics")).toHaveCount(0);
    await expect(
      page.getByText(
        "Start the prepared package to receive training metrics.",
        { exact: true },
      ),
    ).toBeVisible();

    await page.route(`**/api/training/runs/${newRun}`, async (route) => {
      const fixture = structuredClone(detail);
      fixture.hardware = [
        {
          time: Date.now() / 1000,
          gpus: [0, 1].map((index) => ({
            index,
            uuid: `fixture-gpu-${index}`,
            name: `Fixture GPU ${index}`,
            utilization: index === 0 ? 0 : null,
            memory_total_mb: null,
            memory_used_mb: null,
            temperature: null,
            power_watts: null,
          })),
        },
      ];
      await route.fulfill({ json: fixture });
    });
    await page.reload();
    await page.getByRole("button", { name: "Hardware", exact: true }).click();
    const idle = page
      .locator(".training-chart")
      .filter({ hasText: "GPU 0 utilization" });
    await expect(idle.locator('svg text[x="2"]')).toHaveText([
      "100",
      "50",
      "0",
    ]);
    const empty = page
      .locator(".training-chart")
      .filter({ hasText: "GPU 1 utilization" });
    await expect(empty).toContainText("No utilization readings received");
    await expect(empty.locator("svg")).toHaveCount(0);
  } finally {
    releaseSave();
    releaseLaunch();
    await page.unrouteAll({ behavior: "wait" });
    await page.goto("about:blank");
    await context.request.delete(`/api/projects/${id}`);
    if (projectPath?.includes(id))
      rmSync(projectPath, { recursive: true, force: true });
    for (const directory of directories)
      rmSync(directory, { recursive: true, force: true });
  }
});
