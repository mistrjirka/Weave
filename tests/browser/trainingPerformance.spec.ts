import { expect, test } from "@playwright/test";
import { rmSync } from "node:fs";
import { resolve } from "node:path";
import {
  stageSchema,
  templateSchema,
  planSchema,
} from "../../frontend/src/trainingApi";

for (const width of [1440, 390]) {
  test(`performance launch settings persist without rewriting Python at ${width}px`, async ({
    page,
    request,
  }) => {
    test.setTimeout(90000);
    const id = `project-performance-${width}-${Date.now()}`;
    const project = {
      version: 1,
      id,
      name: "Performance settings test",
      description: "Temporary browser fixture",
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
        ].map(({ id, kind, params }, i) => ({
          id,
          type: "architecture",
          position: { x: i * 200, y: 0 },
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
      await (await request.get("/api/training/templates")).json(),
    );
    const stage = stageSchema.parse({
      id: "performance",
      name: "Performance",
      config: { ...templates.config, execution: "monitor" },
      script: templates.scripts.causal,
    });
    let folder: string | null = null;
    try {
      expect(
        (await request.put(`/api/projects/${id}`, { data: project })).ok(),
      ).toBeTruthy();
      const saved = stageSchema.parse(
        await (
          await request.put(
            `/api/training/workspaces/${id}/stages/${stage.id}`,
            { data: { project, stage } },
          )
        ).json(),
      );
      folder = saved.project_path ?? null;
      await page.setViewportSize({ width, height: 900 });
      await page.goto(`/?model=${id}&view=training&stage=${stage.id}`);
      await page.getByRole("button", { name: "Setup", exact: true }).click();
      await page
        .getByText("Performance and profiling", { exact: true })
        .click();
      await expect(
        page.getByLabel("Compilation", { exact: true }),
      ).toBeEnabled();
      const gpus = page
        .getByRole("group", { name: "Choose CPU or GPUs" })
        .getByRole("checkbox", { name: /GPU/ });
      const hasGPU = (await gpus.count()) > 0;
      if (hasGPU) await gpus.first().check();
      await page
        .getByLabel("Compilation", { exact: true })
        .selectOption("reduce-overhead");
      if (hasGPU)
        await page
          .getByLabel("Fused AdamW (CUDA only)", { exact: true })
          .check();
      else
        await expect(
          page.getByLabel("Fused AdamW (CUDA only)", { exact: true }),
        ).toBeDisabled();
      await page
        .getByLabel("Profile updates (0 disables capture)", { exact: true })
        .fill("3");
      await page
        .getByRole("button", { name: "Save stage", exact: true })
        .click();
      await expect(page.locator(".training-notice")).toContainText("Saved", {
        timeout: 30000,
      });
      const plan = planSchema.parse(
        await (await request.get(`/api/training/plans/${id}`)).json(),
      );
      expect(plan.stages[0].script).toBe(stage.script);
      expect(plan.stages[0].config).toMatchObject({
        performance_mode: "override",
        compile_mode: "reduce-overhead",
        fused_optimizer: hasGPU,
        profile_steps: 3,
      });
      await page.reload();
      await page.getByRole("button", { name: "Setup", exact: true }).click();
      await page
        .getByText("Performance and profiling", { exact: true })
        .click();
      await expect(page.getByLabel("Compilation", { exact: true })).toHaveValue(
        "reduce-overhead",
      );
      await page.screenshot({
        path: `data/performance-options-${width}.png`,
        fullPage: true,
      });
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth > window.innerWidth,
        ),
      ).toBe(false);
    } finally {
      await request.delete(`/api/projects/${id}`);
      rmSync(resolve("data/training/plans", `${id}.json`), { force: true });
      if (folder) rmSync(folder, { recursive: true, force: true });
    }
  });
}
