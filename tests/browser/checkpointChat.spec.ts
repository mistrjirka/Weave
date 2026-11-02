import { test, expect } from "@playwright/test";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import AxeBuilder from "@axe-core/playwright";

test("chat with the recommended checkpoint and continue raw text without changing train.py", async ({
  page,
  context,
}) => {
  test.setTimeout(240000);
  page.setDefaultTimeout(15000);
  const temporary = mkdtempSync(join(tmpdir(), "weave-chat-browser-"));
  const id = `project-chat-browser-${Date.now()}`;
  const result = spawnSync(
    ".venv/bin/python",
    [
      "-c",
      `
import sys, json, shutil
from pathlib import Path
sys.path.insert(0, 'tests')
from test_checkpoint_chat import make_chat_run
root=Path(sys.argv[1])
service, source=make_chat_run(root)
project=json.loads((source/'weave_support/project.json').read_text())
project.update(id=sys.argv[2], name='Checkpoint chat verification')
info=json.loads((source/'run.json').read_text())
info['project_id']=sys.argv[2]
(source/'run.json').write_text(json.dumps(info))
destination=Path('data/training/runs')/source.name
shutil.copytree(source,destination)
(root/'fixture.json').write_text(json.dumps({'project':project,'run':source.name,'directory':str(destination.resolve())}))
`,
      temporary,
      id,
    ],
    {
      cwd: process.cwd(),
      env: { ...process.env, USE_TORCH: "1" },
      encoding: "utf8",
      timeout: 120000,
    },
  );
  expect(result.status, result.stderr).toBe(0);
  const fixture = JSON.parse(
    readFileSync(join(temporary, "fixture.json"), "utf8"),
  );
  const sourcePath = resolve(fixture.directory, "train.py");
  const original = readFileSync(sourcePath, "utf8");
  let projectDirectory = "";
  const projectsRoot = String(
    (await (await context.request.get("/api/training/capabilities")).json())
      .projects_directory,
  );
  try {
    await context.request.put(`/api/projects/${id}`, { data: fixture.project });
    await page.goto(`/?model=${id}&view=training&run=${fixture.run}`);
    await page
      .getByRole("button", { name: "Chat & text completion", exact: true })
      .click();
    const panel = page.getByRole("region", {
      name: "Checkpoint chat",
      exact: true,
    });
    const select = panel.getByRole("combobox", { name: "Checkpoint" });
    await expect(select).toHaveValue("checkpoints/first.pt");
    await expect(panel).toContainText("Lowest recorded validation loss");
    await panel
      .getByRole("button", { name: "Load checkpoint", exact: true })
      .click();
    await expect(
      panel.getByRole("status", { name: "Chat model status" }),
    ).toContainText("ready", { timeout: 90000 });
    await expect(
      panel.getByRole("button", { name: "Checkpoint loaded", exact: true }),
    ).toBeDisabled();
    await select.selectOption("checkpoints/latest.pt");
    await expect(
      panel.getByRole("button", { name: "Load checkpoint", exact: true }),
    ).toBeEnabled();
    await select.selectOption("checkpoints/first.pt");
    await expect(
      panel.getByRole("button", { name: "Checkpoint loaded", exact: true }),
    ).toBeDisabled();
    // A slow checkpoint catalog must not block loading/error status polling.
    const chatURL = `/api/training/runs/${fixture.run}/chat`;
    const readySession = await (
      await context.request.get(chatURL + "/session")
    ).json();
    let phase: "loading" | "error" | "ready" | "network-error" = "loading";
    let releaseCatalog: () => void = () => {};
    const catalogGate = new Promise<void>((resolve) => {
      releaseCatalog = resolve;
    });
    await page.route("**" + chatURL, async (route) => {
      await catalogGate;
      await route.continue();
    });
    await page.route("**" + chatURL + "/session", async (route) => {
      if (phase === "network-error") {
        await route.fulfill({
          status: 503,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Status temporarily unavailable" }),
        });
        return;
      }
      await route.fulfill({
        json: {
          ...readySession,
          state: phase,
          loading_stage: phase === "loading" ? "Moving model to cuda:0" : null,
          loading_started: phase === "loading" ? Date.now() / 1000 - 12 : null,
          error:
            phase === "error"
              ? "CUDA out of memory: choose another device"
              : null,
        },
      });
    });
    try {
      await panel
        .getByRole("button", { name: "Refresh checkpoints", exact: true })
        .click();
      await expect(
        panel.getByRole("status", { name: "Checkpoint loading progress" }),
      ).toContainText("Moving model to cuda:0");
      await expect(
        panel.getByRole("status", { name: "Checkpoint loading progress" }),
      ).toContainText("elapsed");
      phase = "error";
      await expect(
        panel.getByRole("alert", { name: "Checkpoint error" }),
      ).toContainText("CUDA out of memory");
      phase = "network-error";
      await expect(
        panel.getByRole("alert", { name: "Checkpoint error" }),
      ).toContainText("Retrying automatically");
      phase = "ready";
      await expect(
        panel.getByRole("status", { name: "Chat model status" }),
      ).toContainText("ready");
      await expect(
        panel.getByRole("alert", { name: "Checkpoint error" }),
      ).toHaveCount(0);
    } finally {
      releaseCatalog();
      await page.unroute("**" + chatURL);
      await page.unroute("**" + chatURL + "/session");
    }
    await panel.getByText("Generation settings", { exact: true }).click();
    await expect(
      panel.getByRole("spinbutton", {
        name: "Repetition penalty",
        exact: true,
      }),
    ).toHaveValue("1");
    await panel
      .getByRole("spinbutton", { name: "Maximum new tokens", exact: true })
      .fill("3");
    await panel.getByRole("spinbutton", { name: /^Temperature\b/ }).fill("0");
    await panel
      .getByRole("textbox", { name: "Message or prompt", exact: true })
      .fill("hello");
    await panel.getByRole("button", { name: "Send", exact: true }).click();
    await expect(
      panel.getByRole("log", { name: "Chat conversation" }),
    ).toContainText("world world world");
    await expect(panel).toContainText("new-token limit reached");
    expect(readFileSync(sourcePath, "utf8")).toBe(original);
    await panel
      .getByRole("button", { name: "New conversation", exact: true })
      .click();
    await panel
      .getByRole("combobox", { name: "Prompt format", exact: false })
      .selectOption("completion");
    await expect(
      panel.getByRole("textbox", { name: "System message", exact: true }),
    ).toBeDisabled();
    await panel
      .getByRole("textbox", { name: "Message or prompt", exact: true })
      .fill("hello world");
    await panel.getByRole("button", { name: "Send", exact: true }).click();
    await expect(
      panel.getByRole("log", { name: "Chat conversation" }),
    ).toContainText("world world world");
    await expect(
      panel.getByRole("log", { name: "Chat conversation" }).locator("article"),
    ).toHaveCount(2);

    // Exercise the tool editor against explicit session responses. Actual
    // worker/module isolation and masking are covered by test_chat_decoders.py.
    const toolCatalog = Array.from({ length: 7 }, (_, index) => ({
      type: "function",
      function: {
        name: `tool_${index}`,
        parameters: {
          type: "object",
          properties: { value: { type: "string" } },
          required: ["value"],
          additionalProperties: false,
        },
      },
    }));
    const realCatalog = await (await context.request.get(chatURL)).json();
    let configuredSession = await (
      await context.request.get(chatURL + "/session")
    ).json();
    const applied: unknown[] = [];
    await page.route("**" + chatURL, (route) =>
      route.fulfill({
        json: {
          ...realCatalog,
          session: configuredSession,
          decoders: [
            {
              id: "xml-tool-grammar",
              label: "XML tool grammar",
              available: true,
            },
          ],
        },
      }),
    );
    await page.route("**" + chatURL + "/session", (route) =>
      route.fulfill({ json: configuredSession }),
    );
    await page.route("**" + chatURL + "/settings", async (route) => {
      const body = route.request().postDataJSON();
      applied.push(body);
      configuredSession = {
        ...configuredSession,
        ...body,
        decoder_label: body.decoder ? "XML tool grammar" : null,
      };
      await route.fulfill({ json: configuredSession });
    });
    try {
      await panel
        .getByRole("button", { name: "Refresh checkpoints", exact: true })
        .click();
      await panel
        .getByText("Tools & constrained decoding", { exact: true })
        .click();
      const toolsEditor = panel.getByRole("textbox", {
        name: "Tool catalog JSON",
        exact: false,
      });
      const apply = panel.getByRole("button", {
        name: "Apply tool settings",
        exact: true,
      });
      await toolsEditor.fill("not JSON");
      await expect(panel.getByRole("alert")).toContainText("Invalid JSON");
      await expect(apply).toBeDisabled();
      await panel
        .getByLabel("Upload tool catalog", { exact: false })
        .setInputFiles({
          name: "seven-tools.json",
          mimeType: "application/json",
          buffer: Buffer.from(JSON.stringify(toolCatalog)),
        });
      await panel
        .getByRole("combobox", { name: "Decoder", exact: false })
        .selectOption("xml-tool-grammar");
      await expect(panel).toContainText("7 tools in draft");
      await expect(panel).toContainText("Unapplied changes");
      await expect(apply).toBeEnabled();
      await apply.click();
      await expect(panel).toContainText("Tool settings applied.");
      await expect(panel.getByLabel("Active tool settings")).toContainText(
        "XML tool grammar",
      );
      await expect(panel.getByLabel("Active tool settings")).toContainText(
        "7 tools loaded",
      );
      await expect(
        panel.getByRole("button", { name: "Checkpoint loaded", exact: true }),
      ).toBeDisabled();
      expect(applied[0]).toEqual({
        tools: toolCatalog,
        decoder: "xml-tool-grammar",
      });

      // Reload restores the applied draft from session state, including tools.
      await page.reload();
      await page
        .getByRole("button", { name: "Chat & text completion", exact: true })
        .click();
      await panel
        .getByText("Tools & constrained decoding", { exact: true })
        .click();
      await expect(toolsEditor).toHaveValue(
        JSON.stringify(toolCatalog, null, 2),
      );
      await expect(
        panel.getByRole("combobox", { name: "Decoder", exact: false }),
      ).toHaveValue("xml-tool-grammar");
      await expect(apply).toBeDisabled();
      await page.setViewportSize({ width: 390, height: 844 });
      await panel.scrollIntoViewIfNeeded();
      expect(
        await panel.evaluate(
          (element) => element.scrollWidth <= element.clientWidth + 1,
        ),
      ).toBeTruthy();
      const toolAudit = await new AxeBuilder({ page })
        .include(".checkpoint-chat")
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze();
      expect(
        toolAudit.violations.map((v) => ({
          id: v.id,
          nodes: v.nodes.map((n) => n.failureSummary),
        })),
      ).toEqual([]);
      await page.screenshot({
        path: "data/checkpoint-chat-tools-phone.png",
        fullPage: true,
      });
      await page.setViewportSize({ width: 1600, height: 1000 });
      await page.screenshot({
        path: "data/checkpoint-chat-tools-desktop.png",
        fullPage: true,
      });
      await toolsEditor.fill("");
      await panel
        .getByRole("combobox", { name: "Decoder", exact: false })
        .selectOption("");
      await apply.click();
      await expect(panel.getByLabel("Active tool settings")).toContainText(
        "0 tools loaded",
      );
    } finally {
      await page.unroute("**" + chatURL);
      await page.unroute("**" + chatURL + "/session");
      await page.unroute("**" + chatURL + "/settings");
    }
    await page.screenshot({
      path: "data/checkpoint-chat-desktop.png",
      fullPage: true,
    });
    const audit = await new AxeBuilder({ page })
      .include(".checkpoint-chat")
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(
      audit.violations.map((v) => ({
        id: v.id,
        nodes: v.nodes.map((n) => n.failureSummary),
      })),
    ).toEqual([]);
    await page.setViewportSize({ width: 390, height: 844 });
    await panel.scrollIntoViewIfNeeded();
    expect(
      await panel.evaluate(
        (element) => element.scrollWidth <= element.clientWidth + 1,
      ),
    ).toBeTruthy();
    await page.screenshot({
      path: "data/checkpoint-chat-phone.png",
      fullPage: true,
    });
    await page.setViewportSize({ width: 1600, height: 1000 });
    await panel
      .getByRole("button", { name: "Unload model", exact: true })
      .click();
    await expect(
      panel.getByRole("status", { name: "Chat model status" }),
    ).toContainText("unloaded");
    await select.selectOption("checkpoints/latest.pt");
    await panel
      .getByRole("button", { name: "Load checkpoint", exact: true })
      .click();
    await expect(
      panel.getByRole("status", { name: "Chat model status" }),
    ).toContainText("ready", { timeout: 90000 });
    await panel
      .getByRole("textbox", { name: "Message or prompt", exact: true })
      .fill("hello");
    await panel.getByRole("button", { name: "Send", exact: true }).click();
    await expect(panel).toContainText("end-of-sequence token");
  } finally {
    await context.request.delete(`/api/training/runs/${fixture.run}/chat`);
    await page.goto("about:blank");
    await context.request.delete(`/api/projects/${id}`);
    // The project folder is separate from immutable run exports.
    const lookup = spawnSync(
      ".venv/bin/python",
      [
        "-c",
        `from pathlib import Path
import json,sys
for p in Path(sys.argv[2]).glob('*/project.json'):
 if json.loads(p.read_text()).get('id')==sys.argv[1]: print(p.parent)`,
        id,
        projectsRoot,
      ],
      { encoding: "utf8" },
    );
    projectDirectory = lookup.stdout.trim();
    if (projectDirectory)
      rmSync(projectDirectory, { recursive: true, force: true });
    rmSync(fixture.directory, {
      recursive: true,
      force: true,
      maxRetries: 5,
      retryDelay: 100,
    });
    rmSync(temporary, { recursive: true, force: true });
  }
});
