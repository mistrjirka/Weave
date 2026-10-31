import { expect, test } from "@playwright/test";
import { rmSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import {
  detailSchema,
  runSchema,
  stageSchema,
  templateSchema,
} from "../../frontend/src/trainingApi";

test("failed resume offers a previous-script retry or saved edits without cluttering the dashboard", async ({
  page,
  context,
}) => {
  test.setTimeout(90000);
  const id = `project-resume-recovery-${Date.now()}`;
  const project = {
    version: 1,
    id,
    name: "Resume recovery UI test",
    description: "Temporary UI fixture",
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
  let directory: string | null = null;
  let projectPath: string | null | undefined = null;
  try {
    await context.request.put(`/api/projects/${id}`, { data: project });
    const templates = templateSchema.parse(
      await (await context.request.get("/api/training/templates")).json(),
    );
    const stage = stageSchema.parse({
      id: "resume-ui",
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
    const failed = {
      ...detail,
      run: { ...run, execution: "local" },
      status: { state: "failed" },
      restart_context: {
        source_run: run.id,
        directory: detail.directory,
        checkpoint: "checkpoints/step-20.pt",
        step: 20,
        exact_resume: true,
        edited_script_path: saved.script_path ?? "train.py",
      },
      resume_problem: {
        message:
          "size mismatch for embedding.weight: checkpoint [8, 8], edited model [8, 16]",
        checkpoint: "checkpoints/step-20.pt",
        recovery:
          "Fix the stage script and retry, or resume with the previous script.",
      },
    };
    await page.route(`**/api/training/runs/${run.id}`, (route) =>
      route.fulfill({ json: failed }),
    );
    const next = {
      ...run,
      id: `run-resume-ui-${Date.now()}`,
      execution: "local",
      parent: {
        run_id: run.id,
        checkpoint: "checkpoints/step-20.pt",
        sha256: "test",
      },
    };
    await page.route(`**/api/training/runs/${next.id}`, (route) =>
      route.fulfill({
        json: { ...detail, run: next, status: { state: "starting" } },
      }),
    );
    const submissions: unknown[] = [];
    await page.route(
      `**/api/training/runs/${run.id}/restart`,
      async (route) => {
        const body: unknown = route.request().postDataJSON();
        submissions.push(body);
        await route.fulfill({ json: next });
      },
    );
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(`/?model=${id}&view=training&run=${run.id}`);
    await expect(
      page.getByRole("alert").filter({ hasText: "Checkpoint resume failed." }),
    ).toBeVisible();
    await expect(
      page.getByText("size mismatch for embedding.weight", { exact: false }),
    ).not.toBeVisible();
    await page.getByText("Compatibility details", { exact: true }).click();
    await expect(
      page.getByText("size mismatch for embedding.weight", { exact: false }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Retry resume", exact: true })
      .click();
    const dialog = page.getByRole("dialog", {
      name: "Resume training",
      exact: true,
    });
    const scripts = dialog.getByRole("combobox", {
      name: "Resume script",
      exact: true,
    });
    await expect(scripts).toHaveValue("previous");
    await expect(dialog).toContainText("Update 20");
    await scripts.selectOption("edited");
    await expect(dialog).toContainText("Save editor changes first");
    expect(
      await dialog.evaluate((el) => el.scrollWidth <= el.clientWidth + 1),
    ).toBeTruthy();
    const audit = await new AxeBuilder({ page })
      .include(".training-validation-dialog")
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(
      audit.violations.map((v) => ({
        id: v.id,
        nodes: v.nodes.map((n) => n.failureSummary),
      })),
    ).toEqual([]);
    await page.screenshot({
      path: "/tmp/weave-resume-recovery-phone.png",
      fullPage: true,
    });
    await dialog
      .getByRole("button", { name: "Resume from checkpoint", exact: true })
      .click();
    await expect(page).toHaveURL(new RegExp(next.id));
    expect(submissions).toEqual([
      {
        script_source: "edited",
        validation: {
          mode: "inline",
          device: "cpu",
          eval_every: 50,
          eval_batches: 5,
        },
      },
    ]);
    const commands: unknown[] = [];
    await page.route(
      `**/api/training/runs/${next.id}/commands`,
      async (route) => {
        const body: unknown = route.request().postDataJSON();
        commands.push(body);
        await route.fulfill({ json: { id: "pause-request" } });
      },
    );
    await page
      .getByRole("button", { name: "Pause training", exact: true })
      .click();
    await expect(
      page.getByRole("button", { name: "Pausing…", exact: true }),
    ).toBeDisabled();
    expect(commands).toEqual([{ kind: "stop", include_data: false }]);
    await expect(
      page.getByRole("button", { name: "Metrics", exact: true }),
    ).toHaveAttribute("aria-pressed", "true");
  } finally {
    await page.goto("about:blank");
    await context.request.delete(`/api/projects/${id}`);
    if (directory) rmSync(directory, { recursive: true, force: true });
    if (projectPath) rmSync(projectPath, { recursive: true, force: true });
  }
});
