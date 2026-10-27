import AxeBuilder from "@axe-core/playwright";
import { test, expect } from "@playwright/test";
import type { Locator, Page } from "@playwright/test";
import { projectSchema } from "../../frontend/src/api";

async function document(page: Page) {
  return projectSchema.parse(
    await page.evaluate(() =>
      JSON.parse(localStorage.getItem("weave.project") || "null"),
    ),
  );
}
async function focus(pane: Locator, id: string) {
  for (let index = 0; index < 80; index++) {
    if (
      (await pane.locator(".node-nav-current").getAttribute("data-node-id")) ===
      id
    )
      return;
    const next = pane.getByRole("button", { name: "Next node", exact: true });
    if (await next.isDisabled()) break;
    await next.click();
  }
  await expect(pane.locator(".node-nav-current")).toHaveAttribute(
    "data-node-id",
    id,
  );
}
const topology = (project: Awaited<ReturnType<typeof document>>) => ({
  nodes: project.graph.nodes.map((n) => ({
    id: n.id,
    data: n.data,
    position: n.position,
  })),
  edges: project.graph.edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    targetHandle: e.targetHandle,
  })),
});

test("editing and reference browsing have independent documents, shortcuts and viewports", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await expect
    .poll(async () => (await document(page)).graph.layout?.engine)
    .toBe("elk");
  await page.getByRole("button", { name: "Split view", exact: true }).click();
  const left = page.getByRole("region", {
    name: "Editable graph",
    exact: true,
  });
  const right = page.getByRole("region", {
    name: "Reference graph",
    exact: true,
  });
  await expect(right.locator(".react-flow__node")).toHaveCount(6);
  await focus(left, "mask");
  await page
    .getByRole("complementary", { name: "Property inspector" })
    .getByRole("textbox", { name: "Block name", exact: true })
    .fill("My editable mask");
  await expect
    .poll(
      async () =>
        (await document(page)).graph.nodes.find((n) => n.id === "mask")?.data
          .label,
    )
    .toBe("My editable mask");
  const before = topology(await document(page));
  await right
    .getByRole("combobox", { name: "Reference model" })
    .selectOption("example:needle1");
  await expect(right.locator(".react-flow__node")).toHaveCount(9);
  await focus(right, "decoder");
  await right
    .locator('.react-flow__node[data-id="decoder"]')
    .getByRole("button", { name: "Open loop body" })
    .click();
  await expect(right.getByLabel("Reference block path")).toContainText(
    "Decoder",
  );
  await focus(right, "attn");
  await right.locator('.react-flow__node[data-id="attn"]').click();
  await expect(
    page.getByRole("region", { name: "Reference properties" }),
  ).toContainText("heads");
  await page.keyboard.press("Delete");
  await page.keyboard.press("Control+z");
  expect(topology(await document(page))).toEqual(before);
  const viewport = await left
    .locator(".react-flow__viewport")
    .getAttribute("style");
  await right
    .getByRole("button", { name: "Reference zoom to 100 percent" })
    .click();
  expect(
    await left.locator(".react-flow__viewport").getAttribute("style"),
  ).toBe(viewport);
  await page
    .getByRole("button", { name: "Close inspector selection", exact: true })
    .click();
  const node = right.locator('.react-flow__node[data-id="attn"]');
  const position = await node.evaluate((element) => element.style.transform),
    box = await node.boundingBox();
  if (!box) throw new Error("Missing reference node");
  await page.mouse.move(box.x + box.width / 2, box.y + 20);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 45, box.y + 55, { steps: 6 });
  await page.mouse.up();
  expect(await node.evaluate((element) => element.style.transform)).toBe(
    position,
  );
  await right.getByRole("button", { name: "Close reference pane" }).click();
  await expect(right).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Split view", exact: true }),
  ).toBeFocused();
  expect(topology(await document(page))).toEqual(before);
  expect(errors).toEqual([]);
});

test("a copied URL restores models, nested blocks, split size and the active tab in a fresh browser context", async ({
  page,
  browser,
}) => {
  await page.goto(
    "/?model=transformer&block=stack&split=1&reference=example:needle1&refBlock=decoder&ratio=60",
  );
  const right = page.getByRole("region", {
    name: "Reference graph",
    exact: true,
  });
  await expect(page.locator(".editable-pane-heading")).toContainText(
    "Decoder transformer",
  );
  await expect(
    page
      .getByRole("region", { name: "Editable graph", exact: true })
      .locator(".breadcrumbs"),
  ).toContainText("Transformer layers");
  await expect(right.getByLabel("Reference block path")).toContainText(
    "Decoder",
  );
  await expect(
    page.getByRole("separator", { name: "Resize graph panes" }),
  ).toHaveAttribute("aria-valuenow", "60");
  await page.getByRole("button", { name: "PyTorch code", exact: true }).click();
  await expect
    .poll(() => new URL(page.url()).searchParams.get("view"))
    .toBe("code");
  const url = page.url();
  await page.reload();
  await expect(page.locator(".code-view")).toContainText(
    "class DecoderTransformer",
  );
  await expect(right.getByLabel("Reference block path")).toContainText(
    "Decoder",
  );
  const context = await browser.newContext();
  try {
    const fresh = await context.newPage();
    await fresh.goto(url);
    await expect(fresh.locator(".code-view")).toContainText(
      "class DecoderTransformer",
    );
    await expect(
      fresh
        .getByRole("region", { name: "Reference graph", exact: true })
        .getByLabel("Reference block path"),
    ).toContainText("Decoder");
  } finally {
    await context.close();
  }
});

test("saved reference snapshots and keyboard pane resizing do not modify the working graph", async ({
  page,
  request,
}) => {
  const templates = await (await request.get("/api/templates")).json();
  const saved = projectSchema.parse(
    templates.find((p: { id: string }) => p.id === "looped"),
  );
  saved.id = "project-split-test";
  saved.name = "Saved split reference";
  await request.put("/api/projects/" + saved.id, { data: saved });
  try {
    await page.goto("/?model=transformer&split=1&reference=saved:" + saved.id);
    const right = page.getByRole("region", {
      name: "Reference graph",
      exact: true,
    });
    await expect(
      right.getByRole("combobox", { name: "Reference model" }),
    ).toHaveValue("saved:" + saved.id);
    await expect(right.locator(".transformer-preview")).toHaveCount(3);
    const divider = page.getByRole("separator", { name: "Resize graph panes" });
    await divider.focus();
    await page.keyboard.press("ArrowRight");
    await expect(divider).toHaveAttribute("aria-valuenow", "55");
    await expect
      .poll(() => new URL(page.url()).searchParams.get("ratio"))
      .toBe("55");
    await page.keyboard.press("Enter");
    await expect(divider).toHaveAttribute("aria-valuenow", "50");
    await page.reload();
    await expect(right.locator(".transformer-preview")).toHaveCount(3);
    expect((await document(page)).id).toBe("transformer");
  } finally {
    await request.delete("/api/projects/" + saved.id);
  }
});

test("phone split view stacks two usable panes without horizontal overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/?model=transformer&split=1&reference=example:needle1");
  const left = page.getByRole("region", {
    name: "Editable graph",
    exact: true,
  });
  const right = page.getByRole("region", {
    name: "Reference graph",
    exact: true,
  });
  await expect(right.locator(".react-flow__node")).toHaveCount(9);
  const a = await left.boundingBox(),
    b = await right.boundingBox();
  expect(a && b && a.y + a.height <= b.y).toBe(true);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await right.scrollIntoViewIfNeeded();
  await right.getByRole("button", { name: "Next node", exact: true }).click();
  await expect(right.locator(".node-nav-current")).toHaveAttribute(
    "data-node-id",
    "target",
  );
  await page.screenshot({ path: "data/split-phone.png" });
});

test("missing references can be replaced without losing local work", async ({
  page,
}) => {
  await page.goto(
    "/?model=transformer&split=1&reference=saved:missing-project",
  );
  const right = page.getByRole("region", {
    name: "Reference graph",
    exact: true,
  });
  await expect(right.getByRole("status")).toContainText(
    "Could not open this reference",
  );
  await right
    .getByRole("combobox", { name: "Reference model" })
    .selectOption("example:thimble");
  await expect(right.locator(".react-flow__node")).toHaveCount(6);
  expect((await document(page)).id).toBe("transformer");
});

test("split view keeps both graph canvases accessible", async ({ page }) => {
  await page.goto("/?model=transformer&split=1&reference=example:needle1");
  await expect(
    page
      .getByRole("region", { name: "Reference graph", exact: true })
      .locator(".react-flow__node"),
  ).toHaveCount(9);
  const result = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(
    result.violations.map((v) => ({
      id: v.id,
      description: v.description,
      nodes: v.nodes.map((n) => n.target),
    })),
  ).toEqual([]);
  await page.screenshot({ path: "data/split-desktop.png" });
});

test("mobile model drawer and its tab restore from a shared URL", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/?model=transformer&library=models&panel=library");
  await expect(
    page.getByRole("button", { name: "Close library", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Reference architectures", exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Close library", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Close library", exact: true })
    .click();
  await expect
    .poll(() => new URL(page.url()).searchParams.get("panel"))
    .toBeNull();
});
