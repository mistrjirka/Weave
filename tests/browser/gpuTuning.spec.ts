import { test, expect } from "@playwright/test";
import { rmSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import {
  capabilitySchema,
  pythonEnvironmentSchema,
} from "../../frontend/src/trainingApi";
import { tuningListSchema } from "../../frontend/src/gpuTuningApi";

test("global GPU tuner measures strategies, persists a profile and fits on a phone", async ({
  page,
  request,
}) => {
  test.setTimeout(240000);
  const stamp = Date.now();
  const id = `project-tuning-browser-${stamp}`;
  const project = {
    version: 1,
    id,
    name: `GPU tuning browser ${stamp}`,
    description: "Temporary tuning verification",
    config: { vocab: 8, width: 8 },
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
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const capability = capabilitySchema.parse(
    await (await request.get("/api/training/capabilities")).json(),
  );
  test.skip(
    !capability.enabled,
    "Local training must be enabled for real GPU benchmarks",
  );
  const environment = pythonEnvironmentSchema.parse(
    await (
      await request.get(
        `/api/training/environment?python=${encodeURIComponent(capability.python)}`,
      )
    ).json(),
  );
  test.skip(
    !environment.cuda_available || environment.device_count < 2,
    "Requires two CUDA GPUs in the selected training interpreter",
  );
  let jobId: string | null = null;
  await request.put(`/api/projects/${id}`, { data: project });
  try {
    await page.goto(`/?model=${id}`);
    await page
      .getByRole("button", { name: "Weave settings", exact: true })
      .click();
    const dialog = page.getByRole("dialog", {
      name: "Weave settings",
      exact: true,
    });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByLabel("Benchmark Python")).not.toHaveValue("");
    await dialog.getByLabel("Tokens per sequence", { exact: true }).fill("8");
    await dialog
      .getByLabel("Global microbatch (all GPUs combined)", { exact: true })
      .fill("4");
    await dialog
      .getByLabel("Measured updates per candidate", { exact: true })
      .fill("3");
    await dialog
      .getByLabel("Candidates per multi-GPU strategy", { exact: true })
      .fill("1");
    const started = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/training/gpu-tuning") &&
        response.request().method() === "POST",
    );
    await dialog
      .getByRole("button", { name: "Benchmark & autotune", exact: true })
      .click();
    const launched = await started;
    if (launched.ok()) jobId = String((await launched.json()).id);
    const progress = dialog.getByRole("region", {
      name: "GPU tuning progress",
      exact: true,
    });
    await expect(progress.getByRole("progressbar")).toBeVisible({
      timeout: 30000,
    });
    await expect(
      progress.getByRole("heading", { name: "completed", exact: true }),
    ).toBeVisible({ timeout: 180000 });
    const list = tuningListSchema.parse(
      await (await request.get("/api/training/gpu-tuning")).json(),
    );
    const profile = list.profiles.find(
      (item) => item.model_name === project.name,
    );
    expect(profile).toBeTruthy();
    jobId = profile!.id;
    expect(Object.keys(profile!.best).sort()).toEqual([
      "ddp",
      "single",
      "split",
    ]);
    await expect(progress.locator("tbody tr")).toHaveCount(4);
    await dialog.screenshot({ path: "data/gpu-tuning-desktop.png" });
    await page.setViewportSize({ width: 390, height: 844 });
    expect(
      await dialog.evaluate(
        (element) => element.scrollWidth <= element.clientWidth + 1,
      ),
    ).toBeTruthy();
    await page.screenshot({ path: "data/gpu-tuning-phone.png" });
    await dialog
      .getByRole("button", { name: "Close dialog", exact: true })
      .click();
    await expect(dialog).toHaveCount(0);
    await page.reload();
    await page
      .getByRole("button", { name: "Weave settings", exact: true })
      .click();
    await expect(
      page.getByRole("dialog").getByText(new RegExp(`${project.name} · GPUs`)),
    ).toBeVisible();
    expect(errors).toEqual([]);
  } finally {
    if (!jobId) {
      const list = tuningListSchema.parse(
        await (await request.get("/api/training/gpu-tuning")).json(),
      );
      jobId =
        list.profiles.find((item) => item.model_name === project.name)?.id ??
        null;
    }
    if (jobId) {
      await request.post(`/api/training/gpu-tuning/${jobId}/stop`, {
        data: {},
      });
      const directory = resolve("data/training/gpu-tuning", jobId);
      if (existsSync(directory))
        rmSync(directory, { recursive: true, force: true });
    }
    await page.goto("about:blank");
    await request.delete(`/api/projects/${id}`);
    if (capability.projects_directory)
      rmSync(
        resolve(capability.projects_directory, `gpu-tuning-browser-${stamp}`),
        { recursive: true, force: true },
      );
  }
});

test("GPU settings fields stay readable on desktop and phone", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByRole("button", { name: "Weave settings", exact: true })
    .click();
  const dialog = page.getByRole("dialog", {
    name: "Weave settings",
    exact: true,
  });
  await expect(dialog.getByLabel("Benchmark Python")).not.toHaveValue("");
  expect((await dialog.boundingBox())!.width).toBeGreaterThan(850);
  await dialog.screenshot({ path: "data/gpu-tuning-settings-desktop.png" });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await dialog.evaluate(
      (element) => element.scrollWidth <= element.clientWidth + 1,
    ),
  ).toBeTruthy();
  expect(
    (await dialog.getByLabel("Benchmark Python").boundingBox())!.width,
  ).toBeGreaterThan(220);
  await page.screenshot({ path: "data/gpu-tuning-settings-phone.png" });
  await dialog
    .getByRole("button", { name: "Close dialog", exact: true })
    .click();
  await expect(dialog).toHaveCount(0);
});
