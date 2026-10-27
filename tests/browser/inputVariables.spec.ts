import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { projectSchema } from "../../frontend/src/api";
import type { Graph, GraphNode } from "../../frontend/src/types";
import { openInputTokenizer, openExample } from "./helpers";
const document = async (page: Page) =>
  projectSchema.parse(
    await page.evaluate(() =>
      JSON.parse(localStorage.getItem("weave.project") || "null"),
    ),
  );
const node = (
  id: string,
  kind: string,
  params: GraphNode["data"]["params"],
  x: number,
  body?: Graph,
): GraphNode => ({
  id,
  type: "architecture",
  position: { x, y: 80 },
  data: { kind, label: id, params, ...(body ? { body } : {}) },
});
const edge = (source: string, target: string) => ({
  id: source + target,
  source,
  target,
  sourceHandle: "out",
  targetHandle: "a",
});
const info = {
  id: "12345678901234567890",
  class: "TestTokenizer",
  size: 4,
  capacity: 6,
  base_size: 4,
  mask_id: 5,
  special_tokens: { mask_token: { token: "[MASK]", id: 5 } },
};
async function fixture(page: Page, graph?: Graph) {
  const templates = await (await page.request.get("/api/templates")).json();
  const project = projectSchema.parse(
    templates.find((p: { id: string }) => p.id === "blank"),
  );
  project.id = "variables-browser";
  project.name = "Variables example";
  project.config = { width: 8, vocab: 16, mask_id: 3 };
  project.graph = graph ?? {
    nodes: [
      node("x", "input", { dtype: "float32", shape: ["B", "T", 8] }, 0),
      node("linear", "linear", { out: 8 }, 360),
      node("out", "output", {}, 720),
    ],
    edges: [edge("x", "linear"), edge("linear", "out")],
  };
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await page
    .locator("input[type=file]")
    .first()
    .setInputFiles({
      name: "graph.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(project)),
    });
  await expect(page.locator(".project-title")).toHaveText(project.name);
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
}

test("tokenizer is configured on an input and sets connected vocabulary sizes automatically", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await expect(page.locator(".view-actions .tokenizer-chip")).toHaveCount(0);
  test.setTimeout(60000);
  const vocabBefore = (await document(page)).config.vocab;
  await openInputTokenizer(page);
  const bundle = {
    version: "1.0",
    truncation: null,
    padding: null,
    added_tokens: [
      {
        id: 5,
        content: "[MASK]",
        single_word: false,
        lstrip: false,
        rstrip: false,
        normalized: false,
        special: true,
      },
    ],
    normalizer: null,
    pre_tokenizer: { type: "Whitespace" },
    post_processor: null,
    decoder: null,
    model: {
      type: "WordLevel",
      vocab: { "[UNK]": 0, hello: 1, world: 2, "[MASK]": 5 },
      unk_token: "[UNK]",
    },
  };
  await page
    .locator("input[type=file]")
    .last()
    .setInputFiles([
      {
        name: "tokenizer.json",
        mimeType: "application/json",
        buffer: Buffer.from(JSON.stringify(bundle)),
      },
      {
        name: "tokenizer_config.json",
        mimeType: "application/json",
        buffer: Buffer.from(
          JSON.stringify({
            tokenizer_class: "PreTrainedTokenizerFast",
            unk_token: "[UNK]",
            mask_token: "[MASK]",
          }),
        ),
      },
    ]);
  await expect(page.locator(".tokenizer-result")).toBeVisible({
    timeout: 45000,
  });
  await expect(page.locator(".token-stats")).toContainText("6");
  await page.getByRole("button", { name: "Close dialog" }).click();
  await expect(page.locator(".input-tokenizer-settings")).toContainText(
    "Vocabulary capacity6",
  );
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await expect
    .poll(
      async () =>
        (await document(page)).graph.nodes[0].data.tokenizer?.capacity,
    )
    .toBe(6);
  const project = await document(page);
  expect(project.tokenizer).toBeUndefined();
  expect(project.config.vocab).toBe(vocabBefore);
  const report = await (
    await page.request.post("/api/validate", { data: project })
  ).json();
  expect(report.shapes["denoise/predict/head"].shape.at(-1)).toBe(6);
  expect(report.tokenizers["denoise/state"].capacity).toBe(6);
  await page
    .getByRole("button", { name: "Open loop body", exact: true })
    .click();
  await page.locator('.flow-wrap .react-flow__node[data-id="state"]').click();
  await expect(page.locator(".input-tokenizer-settings")).toContainText(
    "Inherited from the connected input",
  );
  await expect(
    page.locator(".input-tokenizer-settings .tokenizer-chip"),
  ).toBeDisabled();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export PyTorch" }).click();
  expect((await download).suggestedFilename()).toContain(".zip");
});

test("graph variables can be added, edited, bound, removed, and replaced by a number", async ({
  page,
}) => {
  await fixture(page);
  await page
    .getByRole("button", { name: "Graph variables", exact: true })
    .click();
  const dialog = page.getByRole("dialog", {
    name: "Graph variables",
    exact: true,
  });
  await dialog
    .getByRole("textbox", { name: "New variable name" })
    .fill("hidden_size");
  await dialog.getByRole("textbox", { name: "New variable value" }).fill("6");
  await dialog.getByRole("button", { name: "Add", exact: true }).click();
  await expect(
    dialog.getByRole("textbox", { name: "Value of $hidden_size", exact: true }),
  ).toHaveValue("6");
  await dialog.getByRole("button", { name: "Close dialog" }).click();
  await page.locator('.flow-wrap .react-flow__node[data-id="linear"]').click();
  const out = page.getByRole("textbox", { name: "out", exact: true });
  await out.fill("$hidden_size");
  await out.press("Enter");
  await expect(page.locator(".binding-resolution")).toContainText(
    "$hidden_size = 6",
  );
  await expect(page.locator(".shape-card")).toContainText("6");
  await page
    .getByRole("button", { name: "Graph variables", exact: true })
    .click();
  const value = dialog.getByRole("textbox", {
    name: "Value of $hidden_size",
    exact: true,
  });
  await value.fill("4");
  await value.press("Enter");
  await dialog.getByRole("button", { name: "Close dialog" }).click();
  await expect(page.locator(".shape-card")).toContainText("4");
  await page
    .getByRole("button", { name: "Graph variables", exact: true })
    .click();
  await dialog
    .getByRole("button", { name: "Delete $hidden_size", exact: true })
    .click();
  await dialog.getByRole("button", { name: "Close dialog" }).click();
  await expect(page.locator(".inspector .inline-error")).toContainText(
    "$hidden_size",
  );
  await expect(page.locator(".inspector")).toContainText(
    "Add it in Graph variables",
  );
  await out.fill("5");
  await out.press("Enter");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await expect
    .poll(
      async () =>
        (await document(page)).graph.nodes.find((node) => node.id === "linear")
          ?.data.params.out,
    )
    .toBe(5);
});

test("nested graph variables override inherited values without changing the parent", async ({
  page,
}) => {
  const body: Graph = {
    variables: { hidden_size: 3 },
    nodes: [
      node("x", "input", { dtype: "float32", shape: ["B", "T", 8] }, 0),
      node("linear", "linear", { out: "$hidden_size" }, 360),
      node("out", "output", {}, 720),
    ],
    edges: [edge("x", "linear"), edge("linear", "out")],
  };
  await fixture(page, {
    variables: { hidden_size: 7 },
    nodes: [
      node("x", "input", { dtype: "float32", shape: ["B", "T", 8] }, 0),
      node("body", "block", {}, 360, body),
      node("out", "output", {}, 720),
    ],
    edges: [edge("x", "body"), edge("body", "out")],
  });
  await page.getByRole("button", { name: "Open block", exact: true }).click();
  await page
    .getByRole("button", { name: "Graph variables", exact: true })
    .click();
  const dialog = page.getByRole("dialog", {
    name: "Graph variables",
    exact: true,
  });
  const value = dialog.getByRole("textbox", {
    name: "Value of $hidden_size",
    exact: true,
  });
  await expect(value).toHaveValue("3");
  await value.fill("5");
  await value.press("Enter");
  await dialog.getByRole("button", { name: "Close dialog" }).click();
  await expect
    .poll(
      async () =>
        (await document(page)).graph.nodes.find((node) => node.id === "body")
          ?.data.body?.variables?.hidden_size,
    )
    .toBe(5);
  const project = await document(page);
  expect(project.graph.variables?.hidden_size).toBe(7);
  const report = await (
    await page.request.post("/api/validate", { data: project })
  ).json();
  expect(report.shapes["body/linear"].shape.at(-1)).toBe(5);
  await page.reload();
  await expect(page.locator(".breadcrumbs")).toContainText("body");
  await page
    .getByRole("button", { name: "Graph variables", exact: true })
    .click();
  await expect(value).toHaveValue("5");
});

test("legacy tokenizer projects migrate onto token inputs and survive reload", async ({
  page,
}) => {
  const templates = await (await page.request.get("/api/templates")).json();
  const project = projectSchema.parse(
    templates.find((p: { id: string }) => p.id === "needle1"),
  );
  project.tokenizer = info;
  await page.addInitScript(
    (project) => localStorage.setItem("weave.project", JSON.stringify(project)),
    project,
  );
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await expect
    .poll(async () =>
      (await document(page)).graph.nodes
        .filter((node) => node.data.kind === "input")
        .map((node) => node.data.tokenizer?.capacity),
    )
    .toEqual([6, 6]);
  expect((await document(page)).tokenizer).toBeUndefined();
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  const report = await (
    await page.request.post("/api/validate", { data: await document(page) })
  ).json();
  expect(report.shapes.head.shape.at(-1)).toBe(6);
});

for (const theme of ["light", "black"] as const)
  test(`${theme} variables table fits on a phone and passes accessibility checks`, async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.addInitScript(
      (theme) => localStorage.setItem("weave.theme", theme),
      theme,
    );
    await page.goto("/");
    await expect(
      page.getByRole("button", { name: "Shapes compatible" }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Graph variables", exact: true })
      .click();
    const dialog = page.getByRole("dialog", {
      name: "Graph variables",
      exact: true,
    });
    await expect(dialog).toBeVisible();
    expect(
      await dialog.evaluate((node) => node.scrollWidth <= node.clientWidth + 1),
    ).toBe(true);
    const audit = await new AxeBuilder({ page })
      .include('[role="dialog"]')
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(audit.violations).toEqual([]);
    await page.screenshot({ path: `data/${theme}-phone-variables.png` });
  });

test("reference properties show tokenizer-resolved vocabulary bindings", async ({
  page,
}) => {
  const graph: Graph = {
    nodes: [
      node("ids", "input", { dtype: "int64", shape: ["B", "T"] }, 0),
      node("emb", "embedding", { width: 8, vocab: "$vocab" }, 360),
      node("head", "head", { out: "$vocab", tie: "emb" }, 720),
      node("out", "output", {}, 1080),
    ],
    edges: [edge("ids", "emb"), edge("emb", "head"), edge("head", "out")],
  };
  graph.nodes[0].data.tokenizer = info;
  await fixture(page, graph);
  const saved = await document(page);
  saved.id = "input-reference-browser";
  await page.request.put("/api/projects/" + saved.id, { data: saved });
  try {
    await page.reload();
    await expect(
      page.getByRole("button", { name: "Shapes compatible" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Split view", exact: true }).click();
    const right = page.getByRole("region", {
      name: "Reference graph",
      exact: true,
    });
    await right
      .getByRole("combobox", { name: "Reference model" })
      .selectOption("saved:" + saved.id);
    await expect(right.locator(".react-flow__node")).toHaveCount(4);
    await right.locator('.react-flow__node[data-id="emb"]').click();
    await expect(page.locator(".reference-details")).toContainText(
      "$vocab → 6",
    );
    await right
      .getByRole("button", { name: "Previous node", exact: true })
      .click();
    await expect(right.locator(".node-nav-current")).toHaveAttribute(
      "data-node-id",
      "ids",
    );
    await right.locator('.react-flow__node[data-id="ids"]').click();
    await expect(page.locator(".reference-details")).toContainText(
      "Vocabulary capacity6",
    );
  } finally {
    await page.request.delete("/api/projects/" + saved.id);
  }
});

test("a tokenizer load cannot attach its result to a different project", async ({
  page,
}) => {
  let release: () => void = () => {};
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/tokenizers/inspect", async (route) => {
    await pending;
    await route.fulfill({ json: info });
  });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await openInputTokenizer(page);
  await page
    .getByPlaceholder("e.g. gpt2 or /path/to/tokenizer")
    .fill("test-tokenizer");
  await page
    .getByRole("button", { name: "Load & inspect", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Inspecting…", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close dialog" }).click();
  await openExample(page, "Blank project");
  release();
  await expect(page.getByRole("status")).toContainText(
    "active project changed",
  );
  expect(
    (await document(page)).graph.nodes.some((node) => node.data.tokenizer),
  ).toBe(false);
});
