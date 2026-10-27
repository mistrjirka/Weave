import { test, expect } from "@playwright/test";
import type { Page, Locator } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { projectSchema } from "../../frontend/src/api";
import { openExample } from "./helpers";
const document = async (page: Page) =>
  projectSchema.parse(
    await page.evaluate(() =>
      JSON.parse(localStorage.getItem("weave.project") || "null"),
    ),
  );
async function dropOutput(page: Page, node: Locator) {
  await node.locator('.react-flow__handle.source[data-handleid="out"]').hover();
  const handle = await node
    .locator('.react-flow__handle.source[data-handleid="out"]')
    .boundingBox();
  const bounds = await page.locator(".flow-wrap").boundingBox();
  if (!handle || !bounds) throw new Error("Visible output and canvas required");
  const target = {
    x: Math.min(bounds.x + bounds.width - 35, handle.x + 160),
    y: bounds.y + Math.min(160, bounds.height / 3),
  };
  await page.mouse.move(
    handle.x + handle.width / 2,
    handle.y + handle.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(target.x, target.y, { steps: 20 });
  await page.mouse.up();
  await expect(
    page.getByRole("dialog", { name: "Add connected block", exact: true }),
  ).toBeVisible();
  return target;
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
});

test("weight groups are created and selected, and incompatible node types are excluded", async ({
  page,
}) => {
  await openExample(page, "Looped transformer");
  const root = page.locator(".flow-wrap");
  await root.locator('.react-flow__node[data-id="adapter_1"]').click();
  const group = page.getByRole("combobox", {
    name: "Weight group",
    exact: true,
  });
  await expect(
    page.getByRole("textbox", { name: "weight group", exact: true }),
  ).toHaveCount(0);
  await expect(group.locator("option")).toHaveCount(1);
  await page
    .getByRole("button", { name: "New weight group", exact: true })
    .click();
  await expect(group).not.toHaveValue("");
  const first = (await document(page)).graph.nodes.find(
    (node) => node.id === "adapter_1",
  )!;
  await root.locator('.react-flow__node[data-id="adapter_2"]').click();
  await expect(group.locator("option")).toHaveCount(2);
  await group.selectOption(first.data.share!);
  await expect
    .poll(
      async () =>
        (await document(page)).graph.nodes.find(
          (node) => node.id === "adapter_2",
        )?.data.share,
    )
    .toBe(first.data.share);
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await root.locator('.react-flow__node[data-id="norm"]').click();
  await expect(
    group.locator(`option[value="${first.data.share}"]`),
  ).toHaveCount(0);
  // Even a forged option cannot make the editor replace a norm with a linear layer.
  await group.evaluate((select: HTMLSelectElement, id) => {
    const option = new Option("Invalid type", id);
    select.append(option);
    select.value = id;
    select.dispatchEvent(new Event("change", { bubbles: true }));
  }, first.data.share!);
  await expect(page.getByRole("status")).toContainText("same node type");
  expect(
    (await document(page)).graph.nodes.find((node) => node.id === "norm")?.data
      .kind,
  ).toBe("norm");
  await root.locator('.react-flow__node[data-id="adapter_2"]').click();
  await group.selectOption("");
  await expect
    .poll(
      async () =>
        (await document(page)).graph.nodes.find(
          (node) => node.id === "adapter_2",
        )?.data.share,
    )
    .toBeUndefined();
});

test("dropping an output on empty space searches, adds and connects a node in one undo step", async ({
  page,
}) => {
  const before = await document(page);
  await dropOutput(
    page,
    page.locator('.flow-wrap .react-flow__node[data-id="ids"]'),
  );
  const picker = page.getByRole("dialog", {
    name: "Add connected block",
    exact: true,
  });
  const search = picker.getByRole("combobox", {
    name: "Search connected blocks",
  });
  await expect(search).toBeFocused();
  await search.fill("embedding");
  await expect(picker.getByRole("option")).toHaveCount(1);
  await search.press("Enter");
  await expect(picker).toHaveCount(0);
  await expect(page.locator(".flow-wrap .react-flow__node")).toHaveCount(
    before.graph.nodes.length + 1,
  );
  await expect
    .poll(async () => (await document(page)).graph.edges.length)
    .toBe(before.graph.edges.length + 1);
  const after = await document(page);
  const added = after.graph.nodes.find(
    (node) => !before.graph.nodes.some((original) => original.id === node.id),
  )!;
  expect(added.data.kind).toBe("embedding");
  expect(
    after.graph.edges.some(
      (edge) =>
        edge.source === "ids" &&
        edge.target === added.id &&
        edge.targetHandle === "a",
    ),
  ).toBe(true);
  await page.locator(".flow-wrap").focus();
  await page.keyboard.press("Control+z");
  await expect(page.locator(".flow-wrap .react-flow__node")).toHaveCount(
    before.graph.nodes.length,
  );
  await expect
    .poll(async () => (await document(page)).graph.edges.length)
    .toBe(before.graph.edges.length);
  await page.keyboard.press("Control+Shift+z");
  await expect(page.locator(".flow-wrap .react-flow__node")).toHaveCount(
    before.graph.nodes.length + 1,
  );
});

test("connection picker cancels with Escape and empty search leaves the graph unchanged", async ({
  page,
}) => {
  const before = await document(page);
  await dropOutput(
    page,
    page.locator('.flow-wrap .react-flow__node[data-id="ids"]'),
  );
  const search = page.getByRole("combobox", {
    name: "Search connected blocks",
  });
  await search.fill("not-a-real-block");
  await expect(
    page.getByRole("dialog", { name: "Add connected block" }),
  ).toContainText("No matching blocks");
  await search.press("Escape");
  await expect(
    page.getByRole("dialog", { name: "Add connected block" }),
  ).toHaveCount(0);
  expect((await document(page)).graph.nodes.length).toBe(
    before.graph.nodes.length,
  );
  expect((await document(page)).graph.edges.length).toBe(
    before.graph.edges.length,
  );
  await dropOutput(
    page,
    page.locator('.flow-wrap .react-flow__node[data-id="ids"]'),
  );
  const cancel = page.getByRole("button", {
    name: "Cancel adding block",
    exact: true,
  });
  await cancel.focus();
  await cancel.press("Enter");
  await expect(
    page.getByRole("dialog", { name: "Add connected block", exact: true }),
  ).toHaveCount(0);
  expect((await document(page)).graph.nodes.length).toBe(
    before.graph.nodes.length,
  );
});

for (const width of [1600, 390])
  test(`explicit Up controls return to the enclosing graph at ${width}px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 1000 });
    await expect(page.locator(".breadcrumbs")).toContainText("Model root");
    await expect(
      page.getByRole("button", { name: "Up one level", exact: true }),
    ).toHaveCount(0);
    await page
      .getByRole("button", { name: "Open loop body", exact: true })
      .click();
    await page.getByRole("button", { name: "Open block", exact: true }).click();
    const up = page.getByRole("button", { name: "Up one level", exact: true });
    await expect(up).toBeVisible();
    await up.click();
    await expect(page.locator(".breadcrumbs")).toContainText(
      "Iterative denoising",
    );
    await expect(page.locator(".breadcrumbs")).not.toContainText(
      "Token predictor",
    );
    await up.click();
    await expect(page.locator(".breadcrumbs")).toContainText("Model root");
    await expect(up).toHaveCount(0);
    await expect(
      page.locator(
        '.flow-wrap .react-flow__node[data-id="denoise"] .arch-node',
      ),
    ).toHaveClass(/selected/);
  });

test("reference Up control navigates independently and the picker is accessible", async ({
  page,
}) => {
  await page.getByRole("button", { name: "Split view", exact: true }).click();
  const right = page.getByRole("region", {
    name: "Reference graph",
    exact: true,
  });
  await right
    .getByRole("combobox", { name: "Reference model" })
    .selectOption("example:combined");
  await right
    .getByRole("button", { name: "Open loop body", exact: true })
    .click();
  await right
    .getByRole("button", { name: "Up one level", exact: true })
    .click();
  await expect(right.getByLabel("Reference block path")).toContainText(
    "Model root",
  );
  await page.getByRole("button", { name: "Split view", exact: true }).click();
  await dropOutput(
    page,
    page.locator('.flow-wrap .react-flow__node[data-id="ids"]'),
  );
  const audit = await new AxeBuilder({ page })
    .include(".connection-picker")
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(audit.violations).toEqual([]);
});

test("premade looped transformers expose settings and exchange their editable attention graph", async ({
  page,
}) => {
  await openExample(page, "Blank project");
  await page.getByRole("button", { name: "Blocks", exact: true }).click();
  await page
    .locator(".library-item")
    .filter({ hasText: "Premade recurrent transformer" })
    .click();
  const inspector = page.getByRole("complementary", {
    name: "Property inspector",
  });
  const heads = inspector.getByRole("textbox", { name: "heads", exact: true });
  await heads.fill("4");
  await heads.press("Enter");
  const ffn = inspector.getByRole("textbox", {
    name: "ffn width",
    exact: true,
  });
  await ffn.fill("48");
  await ffn.press("Enter");
  await inspector.getByRole("switch", { name: "causal", exact: true }).click();
  await inspector
    .getByRole("combobox", { name: "Attention type" })
    .selectOption("sparse");
  const project = await document(page);
  const loop = project.graph.nodes.find((node) => node.data.kind === "loop")!;
  expect(loop.data.params).toMatchObject({
    heads: 4,
    ffn_width: 48,
    iterations: 4,
    shared: true,
    causal: false,
  });
  expect(
    loop.data.body!.nodes.filter(
      (node) => node.data.role === "attention-layout",
    ),
  ).toHaveLength(3);
  expect(
    loop.data.body!.nodes.find((node) => node.data.kind === "attention")!.data
      .params,
  ).toMatchObject({ heads: "$heads", causal: false });
  await page
    .getByRole("button", { name: "Open loop body", exact: true })
    .click();
  await expect(
    page
      .locator(".flow-wrap .node-title")
      .filter({ hasText: "Local ∪ global" }),
  ).toBeVisible();
  const attentionNode = page.locator(
    '.flow-wrap .react-flow__node[data-id="attn"]',
  );
  await expect(attentionNode.locator(".attention-preview")).toContainText(
    "4 / 2",
  );
  await expect(attentionNode.locator(".attention-preview")).not.toContainText(
    "windowed",
  );
  await page.getByRole("button", { name: "Up one level", exact: true }).click();
  await inspector
    .getByRole("combobox", { name: "Attention type" })
    .selectOption("mqa");
  await expect
    .poll(
      async () =>
        (await document(page)).graph.nodes.find((node) => node.id === loop.id)!
          .data.params.kv_heads,
    )
    .toBe(1);
  await page.reload();
  await page
    .locator(`.flow-wrap .react-flow__node[data-id="${loop.id}"]`)
    .click();
  await expect(
    inspector.getByRole("combobox", { name: "Attention type" }),
  ).toHaveValue("mqa");
  await expect(heads).toHaveValue("4");
  expect(
    (await document(page)).graph.nodes
      .find((node) => node.id === loop.id)!
      .data.body!.nodes.some((node) => node.data.role === "attention-layout"),
  ).toBe(false);
});

for (const width of [1600, 390])
  test(`canvas variables are visible, editable and movable at ${width}px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 1000 });
    await page.getByTitle("Fit graph").click();
    const card = page.getByRole("complementary", {
      name: "Canvas variables",
      exact: true,
    });
    await expect(
      card.getByRole("textbox", { name: "Value of $width", exact: true }),
    ).toBeVisible();
    await card
      .getByRole("button", { name: "Add variable", exact: true })
      .click();
    await card
      .getByRole("textbox", { name: "New variable name", exact: true })
      .fill("adapter_width");
    await card
      .getByRole("textbox", { name: "New variable value", exact: true })
      .fill("64");
    await card.getByRole("button", { name: "Add", exact: true }).click();
    await expect
      .poll(async () => (await document(page)).config.adapter_width)
      .toBe(64);
    await expect(
      card.getByRole("textbox", {
        name: "Value of $adapter_width",
        exact: true,
      }),
    ).toHaveValue("64");
    const before = await card.boundingBox();
    const grip = await card.locator("header").boundingBox();
    if (!before || !grip) throw new Error("Variables card must be visible");
    await page.mouse.move(grip.x + 30, grip.y + grip.height / 2);
    await page.mouse.down();
    await page.mouse.move(
      grip.x + (width > 760 ? 600 : 50),
      grip.y + grip.height / 2 + 45,
      {
        steps: 10,
      },
    );
    await page.mouse.up();
    expect((await card.boundingBox())!.y).toBeGreaterThan(before.y + 20);
    const audit = await new AxeBuilder({ page })
      .include(".graph-variable-card")
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(audit.violations).toEqual([]);
    if (width > 760) {
      await page.setViewportSize({ width: 390, height: 1000 });
      await expect
        .poll(async () => {
          const moved = await card.boundingBox();
          const canvas = await page.locator(".flow-wrap").boundingBox();
          return (
            !!moved &&
            !!canvas &&
            moved.x >= canvas.x &&
            moved.x + moved.width <= canvas.x + canvas.width
          );
        })
        .toBe(true);
      await page.setViewportSize({ width, height: 1000 });
    }
    await page
      .getByRole("combobox", { name: "Color theme" })
      .selectOption("black");
    const darkAudit = await new AxeBuilder({ page })
      .include(".graph-variable-card")
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(darkAudit.violations).toEqual([]);
    await card
      .getByRole("button", { name: "Minimize canvas variables" })
      .click();
    await expect(card.getByRole("textbox")).toHaveCount(0);
    await card.getByRole("button", { name: "Expand canvas variables" }).click();
    await expect(
      card.getByRole("textbox", { name: "Value of $width", exact: true }),
    ).toBeVisible();
  });
