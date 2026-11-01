import { test, expect } from "@playwright/test";
import { projectSchema } from "../../frontend/src/api";

// Load a browser-only copy. These checks never save or launch the user's project.
test("Tokenizer vocabulary is automatic, absent from both variable tables, and manually overridable per embedding", async ({
  page,
}) => {
  const response = await page.request.get("/api/templates");
  const templates = await response.json();
  const project = projectSchema.parse(
    templates.find((item: { id: string }) => item.id === "transformer"),
  );
  project.id = "project-vocabulary-ownership-test";
  project.config.vocab = 1024;
  project.graph.nodes.find(
    (node) => node.data.kind === "input",
  )!.data.tokenizer = {
    id: "a".repeat(20),
    class: "WordLevel",
    capacity: 16000,
    size: 16000,
    base_size: 16000,
    special_tokens: {},
  };
  await page.addInitScript(
    (value) => localStorage.setItem("weave.project", JSON.stringify(value)),
    project,
  );
  await page.goto("/");
  await expect(
    page.getByRole("complementary", { name: "Canvas variables" }),
  ).toBeVisible();
  await expect(
    page.getByRole("textbox", { name: "Value of $vocab", exact: true }),
  ).toHaveCount(0);
  await page
    .getByRole("button", { name: "Graph variables", exact: true })
    .click();
  await expect(
    page.getByRole("textbox", { name: "Value of $vocab", exact: true }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  const embedding = project.graph.nodes.find(
    (node) => node.data.kind === "embedding",
  )!;
  await page.locator(`.react-flow__node[data-id="${embedding.id}"]`).click();
  const capacity = page.getByRole("region", {
    name: "Tokenizer vocabulary capacity",
  });
  await expect(capacity).toContainText("16,000");
  await expect(capacity).toContainText(
    "Automatic from the connected input tokenizer",
  );
  await expect(
    page.getByRole("textbox", { name: "vocab", exact: true }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "Set capacity manually" }).click();
  await expect(
    page.getByRole("textbox", { name: "vocab", exact: true }),
  ).toHaveValue("16000");
  await page
    .getByRole("button", { name: "Use input tokenizer capacity" })
    .click();
  await expect(capacity).toContainText("16,000");
});

test("loop and child share attention controls; primitive bindings are explicit and nested input shape is read-only", async ({
  page,
}) => {
  await page.goto("/?library=blocks");
  await page.getByRole("button", { name: "Blocks", exact: true }).click();
  await page
    .getByRole("textbox", { name: "Search blocks" })
    .fill("Looped transformer");
  await page
    .locator(".library-item")
    .filter({ hasText: "Looped transformer" })
    .click();
  await expect(page.locator(".inspector")).toContainText(
    "These controls edit the",
  );
  await page
    .getByRole("combobox", { name: "Attention type", exact: true })
    .selectOption("mqa");
  const loop = page.locator(".react-flow__node").filter({
    has: page.locator(".node-title").filter({ hasText: "Looped transformer" }),
  });
  await loop
    .getByRole("button", { name: "Open loop body", exact: true })
    .click();
  await page.locator('.react-flow__node[data-id="transformer_layer"]').click();
  await expect(
    page.getByRole("combobox", { name: "Attention type", exact: true }),
  ).toHaveValue("mqa");
  await page
    .getByRole("combobox", { name: "Attention type", exact: true })
    .selectOption("mha");
  await page
    .getByRole("button", { name: "Open internal graph", exact: true })
    .click();
  await page.locator('.react-flow__node[data-id="attn"]').click();
  await expect(page.locator(".inspector")).toContainText("$causal");
  await expect(page.locator(".inspector")).toContainText(
    "Linked to the enclosing parameter",
  );
  await page.locator('.react-flow__node[data-id="x"]').click();
  await expect(
    page.getByRole("textbox", { name: "shape", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("combobox", { name: "dtype", exact: true }),
  ).toBeDisabled();
  await expect(page.locator(".inspector")).toContainText(
    "Inherited from the parent connection",
  );
  await page.getByRole("button", { name: "Up one level", exact: true }).click();
  await page
    .getByRole("textbox", { name: "Search blocks" })
    .fill("Transformer layer");
  await page.getByTitle("Editable attention, residual paths and FFN").click();
  await page
    .getByRole("combobox", { name: "Attention type", exact: true })
    .selectOption("mqa");
  await page.getByRole("button", { name: "Up one level", exact: true }).click();
  await loop.click();
  await expect(
    page.getByRole("combobox", { name: "Attention type", exact: true }),
  ).toHaveCount(0);
  await expect(page.locator(".inspector")).toContainText(
    "Configure each inner layer separately",
  );
  await loop
    .getByRole("button", { name: "Open loop body", exact: true })
    .click();
  await page.locator('.react-flow__node[data-id="transformer_layer"]').click();
  await expect(
    page.getByRole("combobox", { name: "Attention type", exact: true }),
  ).toHaveValue("mha");
});
