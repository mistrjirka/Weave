import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { rmSync } from "node:fs";
import {
  detailSchema,
  stageSchema,
  templateSchema,
} from "../../frontend/src/trainingApi";

for (const width of [1440, 390, 3840]) {
  test(`stage setup shows effective settings and keeps monitoring separate at ${width}px`, async ({
    page,
    request,
  }) => {
    test.setTimeout(150000);
    const id = `project-workflow-${width}-${Date.now()}`;
    const project = {
      version: 1,
      id,
      name: `Stage workflow ${width} ${Date.now()}`,
      description: "Temporary workflow fixture",
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
          data: { kind, label: kind, params },
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
      await (await request.get("/api/training/templates")).json(),
    );
    const script = templates.scripts.causal
      .replace("devices=[]", "devices=[1]")
      .replace('precision="fp32"', 'precision="fp16"')
      .replace('compile_mode="off"', 'compile_mode="reduce-overhead"')
      .replace("fused_optimizer=False", "fused_optimizer=True")
      .replace("profile_steps=0", "profile_steps=3")
      .replace("async_evaluation=False", "async_evaluation=True")
      .replace('evaluation_device="cpu"', "evaluation_device=0");
    let folder: string | null = null;
    const runFolders: string[] = [];
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    try {
      expect(
        (await request.put(`/api/projects/${id}`, { data: project })).ok(),
      ).toBeTruthy();
      const stages = [];
      for (const [index, source] of [
        script,
        templates.scripts.causal,
      ].entries()) {
        const stage = stageSchema.parse({
          id: `stage-${index}`,
          name: index === 0 ? "Pretraining" : "Fine tuning",
          config: { ...templates.config, execution: "monitor" },
          script: source,
        });
        const response = await request.put(
          `/api/training/workspaces/${id}/stages/${stage.id}`,
          { data: { project, stage } },
        );
        expect(response.ok(), await response.text()).toBeTruthy();
        const saved = stageSchema.parse(await response.json());
        folder = saved.project_path ?? null;
        stages.push(saved);
      }
      const oldResponse = await request.post("/api/training/runs", {
        data: { project, stage: stages[0] },
      });
      expect(oldResponse.ok(), await oldResponse.text()).toBeTruthy();
      const oldRun = await oldResponse.json();
      runFolders.push(
        detailSchema.parse(
          await (await request.get(`/api/training/runs/${oldRun.id}`)).json(),
        ).directory,
      );
      await page.route("**/api/training/hardware", (route) =>
        route.fulfill({
          json: {
            scope: "device-wide",
            gpus: [0, 1].map((index) => ({
              index,
              uuid: `workflow-gpu-${index}`,
              name: `Workflow GPU ${index}`,
              memory_total_mb: 24576,
              memory_used_mb: 1024,
              utilization: 0,
              temperature: null,
              power_watts: null,
            })),
          },
        }),
      );
      await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 });
      await page.goto(
        `/?model=${id}&view=training&stage=stage-0&run=${oldRun.id}&training_panel=setup`,
      );
      const setup = page.getByRole("region", {
        name: "Stage setup",
        exact: true,
      });
      await expect(setup).toBeVisible({ timeout: 30000 });
      const devices = page.getByRole("group", {
        name: "Choose CPU or GPUs",
        exact: true,
      });
      await expect(
        devices.getByRole("checkbox", { name: /GPU 1/ }),
      ).toBeChecked();
      await expect(
        devices.getByRole("checkbox", { name: /^CPU/ }),
      ).not.toBeChecked();
      await expect(page.getByLabel("Precision", { exact: true })).toHaveValue(
        "fp16",
      );
      await expect(
        page.getByLabel("Validation execution", { exact: true }),
      ).toHaveValue("async");
      await expect(
        page.getByLabel("Validation device", { exact: true }),
      ).toHaveValue("0");
      await expect(
        page.getByRole("region", { name: "Run dashboard", exact: true }),
      ).toHaveCount(0);
      await expect(
        page.getByText("Use train.py settings", { exact: true }),
      ).toHaveCount(0);
      await expect(
        page.getByText("Use GUI overrides", { exact: true }),
      ).toHaveCount(0);
      const accessibility = await new AxeBuilder({ page })
        .include(".training-workspace")
        .withTags(["wcag2a", "wcag2aa"])
        .analyze();
      expect(accessibility.violations).toEqual([]);
      if (width === 1440) {
        for (const theme of ["dark", "black", "light"]) {
          await page
            .getByRole("combobox", { name: "Color theme", exact: true })
            .selectOption(theme);
          const audit = await new AxeBuilder({ page })
            .include(".training-workspace")
            .withTags(["wcag2a", "wcag2aa"])
            .analyze();
          expect(audit.violations).toEqual([]);
        }
      }
      await page.screenshot({
        path: `test-results/training-setup-${width}.png`,
      });
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth,
        ),
      ).toBeTruthy();
      if (width === 390) {
        expect(
          await page
            .getByLabel("Precision", { exact: true })
            .evaluate((el) => getComputedStyle(el).fontSize),
        ).toBe("16px");
      }
      await page
        .getByRole("button", { name: "Python script", exact: true })
        .click();
      await expect(
        page.getByRole("textbox", {
          name: "Training Python script",
          exact: true,
        }),
      ).toContainText("devices=[1]");
      await expect(page.locator(".training-script-location")).toContainText(
        "train.py",
      );
      await expect(
        page.getByRole("region", {
          name: "Compute configuration",
          exact: true,
        }),
      ).toHaveCount(0);
      await page.reload();
      await expect(
        page.getByRole("textbox", {
          name: "Training Python script",
          exact: true,
        }),
      ).toBeVisible();
      await page.getByRole("button", { name: "Setup", exact: true }).click();
      await devices.getByRole("checkbox", { name: /^CPU/ }).check();
      await expect(
        devices.getByRole("checkbox", { name: /^CPU/ }),
      ).toBeChecked();
      await expect(page.getByLabel("Precision", { exact: true })).toHaveValue(
        "fp32",
      );
      await page
        .getByLabel("GPU strategy", { exact: true })
        .selectOption("ddp");
      await expect(
        devices.getByRole("checkbox", { name: /GPU 0/ }),
      ).toBeChecked();
      await expect(
        devices.getByRole("checkbox", { name: /GPU 1/ }),
      ).toBeChecked();
      await page.getByLabel("Precision", { exact: true }).selectOption("fp16");
      await page
        .getByLabel("Validation device", { exact: true })
        .selectOption("cpu");
      await page
        .getByText("Performance and profiling", { exact: true })
        .click();
      await expect(page.getByLabel("Compilation", { exact: true })).toHaveValue(
        "reduce-overhead",
      );
      await expect(
        page.getByLabel("Compilation", { exact: true }),
      ).toBeEnabled();
      await page
        .getByLabel("Compilation", { exact: true })
        .selectOption("default");
      await expect(
        page.getByLabel("Fused AdamW (CUDA only)", { exact: true }),
      ).not.toBeChecked();
      await page
        .getByRole("button", { name: "Save stage", exact: true })
        .click();
      await expect(page.locator(".training-notice")).toContainText(
        "Saved server file",
      );
      const saved = stageSchema.parse(
        await (
          await request.get(`/api/training/workspaces/${id}/stages/stage-0`)
        ).json(),
      );
      expect(saved.script).toBe(script);
      expect(saved.config).toMatchObject({
        compute_mode: "override",
        strategy: "ddp",
        devices: [0, 1],
        precision: "fp16",
        evaluation_mode: "async",
        evaluation_device: "cpu",
        performance_mode: "override",
        compile_mode: "default",
        fused_optimizer: false,
        profile_steps: 3,
      });
      await page
        .getByLabel("Training stage", { exact: true })
        .selectOption("stage-1");
      await expect(
        devices.getByRole("checkbox", { name: /^CPU/ }),
      ).toBeChecked();
      await expect(
        page.getByRole("button", { name: "Monitor", exact: true }),
      ).toBeDisabled();
      await expect(
        page.getByRole("region", { name: "Run dashboard", exact: true }),
      ).toHaveCount(0);
      await page
        .getByLabel("Training stage", { exact: true })
        .selectOption("stage-0");
      await expect(page.getByLabel("Precision", { exact: true })).toHaveValue(
        "fp16",
      );
      await page.getByRole("button", { name: "Monitor", exact: true }).click();
      await expect(
        page.getByRole("region", { name: "Run dashboard", exact: true }),
      ).toBeVisible();
      await expect(setup).toHaveCount(0);
      await page.screenshot({
        path: `test-results/training-monitor-${width}.png`,
      });
      await page.getByRole("button", { name: "Setup", exact: true }).click();
      if (width === 1440) {
        await page.route(
          `**/api/training/runs/${oldRun.id}/event-records?*`,
          async (route) => {
            const after = new URL(route.request().url()).searchParams.get(
              "after",
            );
            await route.fulfill({
              json: {
                events: after
                  ? []
                  : [
                      {
                        version: 1,
                        id: "parent-checkpoint",
                        time: Date.now() / 1000,
                        kind: "checkpoint",
                        step: 10,
                        tokens: 0,
                        payload: { path: "checkpoints/example.pt" },
                      },
                    ],
                next_after: "parent-checkpoint",
              },
            });
          },
        );
        await page
          .getByText("Run environment and starting checkpoint", { exact: true })
          .click();
        await page
          .getByRole("combobox", { name: /^Starting state/ })
          .selectOption("weights");
        await page
          .getByRole("combobox", { name: /^Parent run/ })
          .selectOption(oldRun.id);
        await expect(
          page.locator("#training-checkpoint-paths option"),
        ).toHaveAttribute("value", "checkpoints/example.pt");
        await expect(
          page.getByRole("region", { name: "Run dashboard", exact: true }),
        ).toHaveCount(0);
        await page
          .getByRole("combobox", { name: /^Starting state/ })
          .selectOption("random");
      }
      await page.getByLabel("Precision", { exact: true }).selectOption("fp32");
      const requests: string[] = [];
      page.on("request", (request) => {
        if (
          request.method() === "PUT" &&
          request.url().endsWith("/stages/stage-0")
        )
          requests.push("save");
        if (
          request.method() === "POST" &&
          request.url().endsWith("/api/training/runs")
        )
          requests.push("launch");
      });
      const launched = page.waitForResponse(
        (response) =>
          response.request().method() === "POST" &&
          response.url().endsWith("/api/training/runs"),
      );
      await page
        .getByRole("button", { name: "Prepare training package", exact: true })
        .click();
      const launchedResponse = await launched;
      expect(launchedResponse.ok(), await launchedResponse.text()).toBeTruthy();
      const newRun = await launchedResponse.json();
      const detail = detailSchema.parse(
        await (await request.get(`/api/training/runs/${newRun.id}`)).json(),
      );
      runFolders.push(detail.directory);
      expect(requests).toEqual(["save", "launch"]);
      expect(detail.config.devices).toEqual([0, 1]);
      expect(detail.config.precision).toBe("fp32");
      expect(detail.script).toBe(script);
      await expect(
        page.getByRole("region", { name: "Run dashboard", exact: true }),
      ).toBeVisible();
      await expect(setup).toHaveCount(0);
      expect(new URL(page.url()).searchParams.get("training_panel")).toBe(
        "monitor",
      );
      expect(errors).toEqual([]);
    } finally {
      await page.goto("about:blank");
      await request.delete(`/api/projects/${id}`, { maxRetries: 2 });
      if (folder) rmSync(folder, { recursive: true, force: true });
      for (const folder of runFolders)
        rmSync(folder, { recursive: true, force: true });
    }
  });
}
