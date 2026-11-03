import { expect, test } from "@playwright/test";
import { readFileSync, rmSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import {
  detailSchema,
  stageSchema,
  templateSchema,
} from "../../frontend/src/trainingApi";

for (const width of [1440, 390]) {
  test(`custom loop metrics, artifacts, controls and checkpoint resume work at ${width}px`, async ({
    page,
    request,
  }) => {
    test.setTimeout(180000);
    await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 });
    const id = `project-custom-loop-${width}-${Date.now()}`;
    const project = {
      version: 1,
      id,
      name: `Custom loop ${width} ${Date.now()}`,
      description: "Temporary integration fixture",
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
    const template = templateSchema.parse(
      await (await request.get("/api/training/templates")).json(),
    );
    const stage = stageSchema.parse({
      id: "policy",
      name: "Policy updates",
      script: readFileSync("tests/fixtures/custom_run.py", "utf8"),
      config: {
        ...template.config,
        recipe: "custom",
        devices: [],
        strategy: "single",
        precision: "fp32",
        max_steps: 1000,
      },
    });
    let projectPath: string | null = null;
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    try {
      expect(
        (await request.put(`/api/projects/${id}`, { data: project })).ok(),
      ).toBeTruthy();
      const saved = await request.put(
        `/api/training/workspaces/${id}/stages/policy`,
        { data: { project, stage } },
      );
      expect(saved.ok(), await saved.text()).toBeTruthy();
      projectPath = stageSchema.parse(await saved.json()).project_path ?? null;
      await page.goto(
        `/?model=${id}&view=training&stage=policy&training_panel=setup`,
      );
      await expect(
        page.getByRole("region", {
          name: "Compute configuration",
          exact: true,
        }),
      ).toBeVisible();
      await expect(
        page.getByLabel("GPU strategy", { exact: true }),
      ).toBeDisabled();
      await expect(
        page.getByLabel("Precision", { exact: true }),
      ).toBeDisabled();
      await expect(
        page.getByLabel("Validation execution", { exact: true }),
      ).toBeDisabled();
      await expect(
        page.getByText("Optimizer, schedule, and evaluation", { exact: true }),
      ).toHaveCount(0);
      await page
        .getByRole("button", { name: "Start training stage", exact: true })
        .click();
      const dashboard = page.getByRole("region", {
        name: "Run dashboard",
        exact: true,
      });
      await expect(
        dashboard.getByRole("term").filter({ hasText: /^Task success$/ }),
      ).toBeVisible({ timeout: 45000 });
      await expect(
        dashboard.getByRole("term").filter({ hasText: /^Mean reward$/ }),
      ).toBeVisible();
      await expect(
        page.getByRole("region", { name: "Training workspace", exact: true }),
      ).toContainText("policy updates");
      const overview = page.getByRole("definition");
      expect(await overview.count()).toBeGreaterThan(0);
      await page.getByText("Chart options", { exact: true }).click();
      await page
        .getByRole("combobox", { name: "Horizontal axis", exact: true })
        .selectOption("counter:episodes");
      await expect(
        page.getByRole("img", { name: /^eval\/success by Episodes/ }),
      ).toBeVisible();
      await page
        .getByRole("button", { name: "Tables & artifacts", exact: true })
        .click();
      await expect(
        page.getByRole("region", { name: "Tables and artifacts", exact: true }),
      ).toContainText("Example task");
      const download = page.getByRole("link", {
        name: "Download full JSONL table",
        exact: true,
      });
      const data = await request.get((await download.getAttribute("href"))!);
      expect(data.ok()).toBeTruthy();
      expect(JSON.parse((await data.text()).trim()).success).toBe(true);
      const audit = await new AxeBuilder({ page })
        .include(".training-dashboard")
        .withTags(["wcag2a", "wcag2aa"])
        .analyze();
      expect(audit.violations).toEqual([]);
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBeTruthy();
      await page.screenshot({
        path: `test-results/custom-tables-${width}.png`,
      });
      await page
        .getByRole("button", { name: "Pause training", exact: true })
        .click();
      await expect(
        page.getByRole("status", { name: "Training stopped", exact: true }),
      ).toBeVisible({ timeout: 30000 });
      await page
        .getByRole("button", { name: "Resume training", exact: true })
        .click();
      const dialog = page.getByRole("dialog", {
        name: "Resume custom training",
        exact: true,
      });
      await expect(dialog).toContainText(/recorded/i);
      await dialog
        .getByRole("button", { name: "Resume from checkpoint", exact: true })
        .click();
      await expect(dialog).toHaveCount(0);
      await expect(
        dashboard.getByRole("term").filter({ hasText: /^Task success$/ }),
      ).toBeVisible({ timeout: 45000 });
      await page
        .getByRole("button", { name: "Pause training", exact: true })
        .click();
      await expect(
        page.getByRole("status", { name: "Training stopped", exact: true }),
      ).toBeVisible({ timeout: 30000 });
      expect(errors).toEqual([]);
    } finally {
      await page.goto("about:blank");
      const runs = await (
        await request.get(`/api/training/runs?project_id=${id}`)
      ).json();
      const folders: string[] = [];
      for (const run of runs) {
        const response = await request.get(`/api/training/runs/${run.id}`);
        if (!response.ok()) continue;
        const detail = detailSchema.parse(await response.json());
        folders.push(detail.directory);
        if (
          ["starting", "running", "preparing", "evaluating"].includes(
            detail.status.state,
          )
        )
          await request.post(`/api/training/runs/${run.id}/commands`, {
            data: { kind: "kill" },
          });
      }
      // Wait until the supervisor acknowledges any cleanup kill before deleting its files.
      await expect
        .poll(
          async () => {
            const active = await (
              await request.get("/api/training/active-runs")
            ).json();
            return active.filter(
              (run: { project_id: string }) => run.project_id === id,
            ).length;
          },
          { timeout: 20000 },
        )
        .toBe(0);
      await request.delete(`/api/projects/${id}`);
      for (const folder of folders)
        rmSync(folder, { recursive: true, force: true });
      if (projectPath) rmSync(projectPath, { recursive: true, force: true });
    }
  });
}
