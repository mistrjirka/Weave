import { test, expect } from "@playwright/test";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import AxeBuilder from "@axe-core/playwright";
import { detailSchema, runSchema } from "../../frontend/src/trainingApi";

test("resume a frozen checkpoint and apply live validation from the phone UI", async ({
  page,
  context,
}) => {
  test.setTimeout(300000);
  page.setDefaultTimeout(15000);
  const temporary = mkdtempSync(join(tmpdir(), "weave-live-validation-"));
  const id = `project-live-validation-${Date.now()}`;
  const generated = spawnSync(
    ".venv/bin/python",
    [
      "-c",
      `
import json,os,sys,shutil
from pathlib import Path
root=Path(sys.argv[1]); os.environ['WEAVE_PROJECTS_DIR']=str(root/'projects')
sys.path.insert(0,'tests')
from test_training_hooks import filled_starter,make_hook_run,execute
from backend.training.service import events
from backend.training.weave_training import atomic_json
gate=root/'continue'; gate.touch()
script=filled_starter(root/'rows.jsonl').replace('def training_step(model, batch, device):', 'def training_step(model, batch, device):\\n    import time\\n    from pathlib import Path\\n    while not Path('+repr(str(gate))+').exists(): time.sleep(0.05)')
service,source,stage=make_hook_run(root,script=script)
execute(source)
gate.unlink()
rows=events(source)
first=next(e for e in rows if e['kind']=='checkpoint')
for row in rows:
 if row['kind']=='capabilities': row['payload'].pop('live_validation',None)
rows=[e for e in rows if e['kind']!='checkpoint' or e['id']==first['id']]
(source/'events.jsonl').write_text(''.join(json.dumps(e)+'\\n' for e in rows))
for index in source.glob('event-index.sqlite*'): index.unlink()
project=json.loads((source/'weave_support/project.json').read_text()); project.update(id=sys.argv[2],name='Live validation verification')
atomic_json(source/'weave_support/project.json',project)
info=json.loads((source/'run.json').read_text()); info.update(project_id=sys.argv[2],execution='local')
atomic_json(source/'run.json',info)
for name in ('config.json','stage.json'):
 data=json.loads((source/name).read_text())
 (data['config'] if name=='stage.json' else data)['execution']='local'
 atomic_json(source/name,data)
atomic_json(source/'status.json',{'state':'stopped'})
destination=Path('data/training/runs')/source.name
shutil.copytree(source,destination)
(root/'fixture.json').write_text(json.dumps({'project':project,'run':source.name,'directory':str(destination.resolve()),'script':script,'gate':str(gate)}))
`,
      temporary,
      id,
    ],
    { encoding: "utf8", timeout: 120000 },
  );
  expect(generated.status, generated.stderr + generated.stdout).toBe(0);
  const fixture = z
    .object({
      project: z.record(z.string(), z.unknown()),
      run: z.string(),
      directory: z.string(),
      script: z.string(),
      gate: z.string(),
    })
    .parse(JSON.parse(readFileSync(join(temporary, "fixture.json"), "utf8")));
  const createdRuns = new Set([fixture.run]);
  let projectPath: string | null = null;
  try {
    await context.request.put(`/api/projects/${id}`, { data: fixture.project });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(`/?model=${id}&view=training&run=${fixture.run}`);
    await expect(
      page.getByRole("region", { name: "Run dashboard", exact: true }),
    ).toBeVisible({ timeout: 30000 });
    await page
      .getByRole("button", { name: "Resume training", exact: true })
      .click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toContainText("needs a checkpoint restart");
    await dialog.locator("summary").filter({ hasText: "Validation ·" }).click();
    await expect(
      dialog.getByRole("combobox", {
        name: "Run validation device",
        exact: true,
      }),
    ).toBeDisabled();
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
    const restartResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith(`/runs/${fixture.run}/restart`) &&
        response.request().method() === "POST",
      { timeout: 30000 },
    );
    await dialog
      .getByRole("button", { name: "Resume from checkpoint", exact: true })
      .click();
    const response = await restartResponse;
    expect(response.ok(), await response.text()).toBeTruthy();
    const restarted = runSchema.parse(await response.json());
    createdRuns.add(restarted.id);
    await expect(page).toHaveURL(new RegExp(restarted.id), { timeout: 15000 });
    await expect
      .poll(
        async () =>
          detailSchema.parse(
            await (
              await context.request.get(`/api/training/runs/${restarted.id}`)
            ).json(),
          ).validation_control?.supported,
        { timeout: 90000 },
      )
      .toBe(true);
    const details = detailSchema.parse(
      await (
        await context.request.get(`/api/training/runs/${restarted.id}`)
      ).json(),
    );
    expect(details.script).toBe(fixture.script);
    expect(details.run.parent?.run_id).toBe(fixture.run);
    await page
      .getByRole("button", { name: "Validation settings", exact: true })
      .click();
    await expect(dialog).not.toContainText("needs a checkpoint restart");
    await dialog
      .getByRole("combobox", { name: "Run validation execution", exact: true })
      .selectOption("async");
    await expect(
      dialog.getByRole("combobox", {
        name: "Run validation device",
        exact: true,
      }),
    ).toBeEnabled();
    await dialog
      .getByRole("combobox", { name: "Run validation device", exact: true })
      .selectOption("cpu");
    await dialog
      .getByRole("spinbutton", { name: "Validation interval", exact: true })
      .fill("1");
    await dialog
      .getByRole("spinbutton", {
        name: "Maximum validation batches",
        exact: true,
      })
      .fill("1");
    await dialog
      .getByRole("button", { name: "Apply to running training", exact: true })
      .click();
    await expect(dialog).toContainText("Waiting for the trainer");
    await page.screenshot({
      path: "data/live-validation-phone.png",
      fullPage: true,
    });
    writeFileSync(fixture.gate, "continue");
    await expect(dialog).toContainText("Applied at optimizer update", {
      timeout: 30000,
    });
    await dialog
      .getByRole("button", { name: "Close validation settings", exact: true })
      .click();
    await expect
      .poll(
        async () =>
          detailSchema.parse(
            await (
              await context.request.get(`/api/training/runs/${restarted.id}`)
            ).json(),
          ).status.state,
        { timeout: 90000 },
      )
      .toBe("completed");
    const final = detailSchema.parse(
      await (
        await context.request.get(`/api/training/runs/${restarted.id}`)
      ).json(),
    );
    expect(final.validation_control?.current).toEqual({
      mode: "async",
      device: "cpu",
      eval_every: 1,
      eval_batches: 1,
    });
    expect(
      final.events.some(
        (event) =>
          event.kind === "evaluation" &&
          event.payload.state === "completed" &&
          event.payload.device === "cpu",
      ),
    ).toBeTruthy();
    expect(readFileSync(join(fixture.directory, "train.py"), "utf8")).toBe(
      fixture.script,
    );
  } finally {
    writeFileSync(fixture.gate, "continue");
    await page.goto("about:blank");
    for (const run of createdRuns) {
      const detail = detailSchema.parse(
        await (await context.request.get(`/api/training/runs/${run}`)).json(),
      );
      if (
        ["running", "starting", "preparing", "evaluating"].includes(
          detail.status.state,
        )
      ) {
        await context.request.post(`/api/training/runs/${run}/commands`, {
          data: { kind: "kill" },
        });
        await expect
          .poll(
            async () =>
              detailSchema.parse(
                await (
                  await context.request.get(`/api/training/runs/${run}`)
                ).json(),
              ).status.state,
            { timeout: 15000 },
          )
          .not.toMatch(/^(running|starting|preparing|evaluating)$/);
      }
      rmSync(detail.directory, { recursive: true, force: true });
    }
    const root = String(
      (await (await context.request.get("/api/training/capabilities")).json())
        .projects_directory,
    );
    const lookup = spawnSync(
      ".venv/bin/python",
      [
        "-c",
        "from pathlib import Path\nimport json,sys\nfor p in Path(sys.argv[1]).glob('*/project.json'):\n if json.loads(p.read_text()).get('id')==sys.argv[2]: print(p.parent)",
        root,
        id,
      ],
      { encoding: "utf8" },
    );
    projectPath = lookup.stdout.trim() || null;
    await context.request.delete(`/api/projects/${id}`);
    if (projectPath) rmSync(projectPath, { recursive: true, force: true });
    rmSync(temporary, { recursive: true, force: true });
  }
});
