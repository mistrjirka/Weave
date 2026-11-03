import { expect, test } from "@playwright/test";
import { rmSync } from "node:fs";
import { stageSchema, templateSchema } from "../../frontend/src/trainingApi";
import AxeBuilder from "@axe-core/playwright";

test("copy a saved script from another project, preview, cancel or explicitly save it", async ({
  page,
  context,
}) => {
  test.setTimeout(90000);
  const stamp = Date.now();
  const ids = [`project-copy-source-${stamp}`, `project-copy-target-${stamp}`];
  const directories: string[] = [];
  const config = { width: 8, vocab: 8, heads: 2, ffn_width: 16 };
  const graph = {
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
  };
  const templates = templateSchema.parse(
    await (await context.request.get("/api/training/templates")).json(),
  );
  try {
    for (const [index, id] of ids.entries()) {
      const project = {
        version: 1,
        id,
        name: index === 0 ? "Copy source fixture" : "Copy target fixture",
        description: "Temporary script-copy test",
        config,
        graph,
      };
      await context.request.put(`/api/projects/${id}`, { data: project });
      const stage = stageSchema.parse({
        id: "pretraining",
        name: "Pretraining",
        config: { ...templates.config, execution: "monitor" },
        script:
          templates.scripts.causal +
          `\n# ${index === 0 ? "SOURCE_SERVER_SCRIPT" : "TARGET_ORIGINAL_SCRIPT"}\n`,
      });
      const response = await context.request.put(
        `/api/training/workspaces/${id}/stages/pretraining`,
        { data: { project, stage } },
      );
      expect(response.ok(), await response.text()).toBeTruthy();
      const saved = stageSchema.parse(await response.json());
      if (saved.project_path) directories.push(saved.project_path);
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(`/?model=${ids[1]}&view=training&stage=pretraining`);
    await page
      .getByRole("button", { name: "Python script", exact: true })
      .click();
    const copy = page.getByRole("button", {
      name: "Copy script from project…",
      exact: true,
    });
    await expect(copy).toBeVisible();
    await copy.click();
    const dialog = page.getByRole("dialog", {
      name: "Copy training script",
      exact: true,
    });
    await dialog
      .getByRole("combobox", { name: "Source project", exact: true })
      .selectOption(ids[0]);
    await expect(
      dialog.getByLabel("Source training script", { exact: true }),
    ).toContainText("SOURCE_SERVER_SCRIPT");
    expect(
      await dialog.evaluate((el) => el.scrollWidth <= el.clientWidth + 1),
    ).toBeTruthy();
    await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
    await page.locator(".cm-content").click();
    await page.keyboard.press("Control+End");
    await expect(page.locator(".cm-content")).toContainText(
      "TARGET_ORIGINAL_SCRIPT",
    );
    await copy.click();
    await dialog
      .getByRole("combobox", { name: "Source project", exact: true })
      .selectOption(ids[0]);
    await expect(
      dialog.getByLabel("Source training script", { exact: true }),
    ).toContainText("SOURCE_SERVER_SCRIPT");
    const audit = await new AxeBuilder({ page })
      .include(".training-script-copy-dialog")
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(
      audit.violations.map((v) => ({
        id: v.id,
        nodes: v.nodes.map((n) => n.failureSummary),
      })),
    ).toEqual([]);
    await page.screenshot({
      path: "/tmp/weave-copy-script-phone.png",
      fullPage: true,
    });
    await dialog
      .getByRole("button", { name: "Replace editor script", exact: true })
      .click();
    await page.locator(".cm-content").click();
    await page.keyboard.press("Control+End");
    await expect(page.locator(".cm-content")).toContainText(
      "SOURCE_SERVER_SCRIPT",
    );
    const read = async (id: string) =>
      stageSchema.parse(
        await (
          await context.request.get(
            `/api/training/workspaces/${id}/stages/pretraining`,
          )
        ).json(),
      );
    expect((await read(ids[1])).script).toContain("TARGET_ORIGINAL_SCRIPT");
    await page.getByRole("button", { name: "Save file", exact: true }).click();
    await expect
      .poll(async () => (await read(ids[1])).script)
      .toContain("SOURCE_SERVER_SCRIPT");
    expect((await read(ids[0])).script).toContain("SOURCE_SERVER_SCRIPT");
  } finally {
    await page.goto("about:blank");
    for (const id of ids) await context.request.delete(`/api/projects/${id}`);
    for (const path of directories)
      rmSync(path, { recursive: true, force: true });
  }
});
