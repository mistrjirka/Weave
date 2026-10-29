import { test, expect } from "@playwright/test";

test("reference notes and layer schedules are visible without editing the document", async ({
  page,
}) => {
  await page.goto("/?split=1&reference=example:lfm-shellper-350m");
  const right = page.getByRole("region", {
    name: "Reference graph",
    exact: true,
  });
  await expect(
    right.getByRole("combobox", { name: "Reference model" }),
  ).toHaveValue("example:lfm-shellper-350m");
  await expect(right.locator(".react-flow__node")).toHaveCount(21);
  await expect(
    right.getByRole("complementary", {
      name: "Reference variables",
      exact: true,
    }),
  ).toContainText("$width");
  await right.getByRole("button", { name: "Next node", exact: true }).click();
  await right.getByRole("button", { name: "Next node", exact: true }).click();
  await expect(
    page.getByRole("complementary", { name: "Property inspector" }),
  ).toContainText("embedded CPUs");
  await right
    .getByRole("combobox", { name: "Reference model" })
    .selectOption("example:hammer21-05b");
  await expect(right.locator(".react-flow__node")).toHaveCount(29);
});

test("GPT-2 reference exposes its GELU block and learned positions", async ({
  page,
}) => {
  await page.goto("/?split=1&reference=example:gpt2");
  const right = page.getByRole("region", {
    name: "Reference graph",
    exact: true,
  });
  await expect(
    right.getByRole("combobox", { name: "Reference model" }),
  ).toHaveValue("example:gpt2");
  await expect(right.locator('.react-flow__node[data-id="pos"]')).toContainText(
    "Learned positions",
  );
  await right
    .getByRole("button", { name: "Open loop body", exact: true })
    .click();
  await expect(
    right.locator('.react-flow__node[data-id="gelu"]'),
  ).toContainText("GELU");
  await expect(right.locator('.react-flow__node[data-id="n1"]')).toContainText(
    "LayerNorm",
  );
});

test("new general purpose blocks are searchable and addable", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Blocks", exact: true }).click();
  const search = page.getByRole("textbox", { name: "Search blocks" });
  await search.fill("LayerNorm");
  await page.locator(".library-item").filter({ hasText: "LayerNorm" }).click();
  await expect(
    page.locator(".react-flow__node").filter({ hasText: "LayerNorm" }).first(),
  ).toBeVisible();
  await search.fill("GELU");
  await expect(
    page.locator(".library-item").filter({ hasText: "GELU" }),
  ).toBeVisible();
  await search.fill("convolution");
  await expect(
    page.locator(".library-item").filter({ hasText: "2D convolution" }),
  ).toBeVisible();
});

test("attention explanations are available in a read-only reference", async ({
  page,
}) => {
  await page.goto(
    "/?split=1&reference=example:gpt2&refBlock=stack&refNode=attn",
  );
  const inspector = page.getByRole("region", { name: "Reference properties" });
  await expect(inspector).toBeVisible();
  const help = inspector.getByRole("button", {
    name: "Explain causal",
    exact: true,
  });
  await help.click();
  await expect(inspector).toContainText("earlier tokens");
  await expect(inspector).toContainText("Reference · read-only");
});

test("Needle 3 exposes sourced notes and its two kinds of embedding tables", async ({
  page,
}) => {
  await page.goto("/?split=1&reference=example:needle3");
  const right = page.getByRole("region", {
    name: "Reference graph",
    exact: true,
  });
  const embedding = right.locator('.react-flow__node[data-id="emb"]');
  await expect(embedding).toContainText("embedding table");
  const note = right.getByLabel(
    "Design note for 20-layer Simple Attention Network",
  );
  await expect(note).toContainText("layer marked Engram");
  await expect(note.getByRole("link", { name: /Source/ })).toHaveAttribute(
    "href",
    "https://huggingface.co/Cactus-Compute/needle3",
  );
  for (let i = 0; i < 4; i++)
    await right.getByRole("button", { name: "Next node", exact: true }).click();
  await expect
    .poll(async () => {
      const box = await note.boundingBox();
      const variables = await right
        .getByLabel("Reference variables", { exact: true })
        .boundingBox();
      return box && variables ? box.y >= variables.y + variables.height : false;
    })
    .toBe(true);
  await right
    .getByRole("button", { name: "Hide reference design notes" })
    .click();
  await expect(note).toHaveCount(0);
  await right
    .getByRole("button", { name: "Show reference design notes" })
    .click();
  await expect(note).toHaveCount(1);
  await page.goto(
    "/?split=1&reference=example:needle3&refBlock=stack&refBlock=layer_3",
  );
  await expect(
    right.locator('.react-flow__node[data-id="lookup"]'),
  ).toContainText("Engram key/value tables");
});

test("editable design notes persist, can be hidden, and link only to web sources", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.locator('.react-flow__node[data-id="ids"]').first(),
  ).toBeVisible();
  const nodeCount = await page.locator(".react-flow__node").count();
  const edgeCount = await page.locator(".react-flow__edge").count();
  await page.locator('.react-flow__node[data-id="ids"]').first().click();
  const input = page.getByRole("textbox", { name: "Design note", exact: true });
  await input.fill("Remember the token input.");
  await page
    .getByRole("textbox", { name: "Design note source URL" })
    .fill("https://example.com/source");
  const note = page.getByLabel("Design note for Prompt tokens");
  await expect(note).toContainText("Remember the token input.");
  await expect(page.locator(".react-flow__node")).toHaveCount(nodeCount);
  await expect(page.locator(".react-flow__edge")).toHaveCount(edgeCount);
  await expect(note.getByRole("link", { name: /Source/ })).toHaveAttribute(
    "href",
    "https://example.com/source",
  );
  await page
    .getByRole("button", { name: "Hide design notes", exact: true })
    .click();
  await expect(note).toHaveCount(0);
  await page
    .getByRole("button", { name: "Show design notes", exact: true })
    .click();
  await expect(note).toHaveCount(1);
  await page.reload();
  await expect(note).toContainText("Remember the token input.");
});
