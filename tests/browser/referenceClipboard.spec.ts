import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import type { Page } from "@playwright/test";
import { projectSchema } from "../../frontend/src/api";

const document = async (page: Page) =>
  projectSchema.parse(
    await page.evaluate(() =>
      JSON.parse(localStorage.getItem("weave.project") || "null"),
    ),
  );
async function split(page: Page) {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Split view", exact: true }).click();
  const right = page.getByRole("region", {
    name: "Reference graph",
    exact: true,
  });
  await right
    .getByRole("combobox", { name: "Reference model" })
    .selectOption("example:combined");
  await expect(right.locator(".react-flow__node")).toHaveCount(4);
  await expect(
    right.getByRole("button", { name: "Auto arrange reference", exact: true }),
  ).toBeEnabled();
  await right
    .getByRole("button", { name: "Auto arrange reference", exact: true })
    .click();
  await expect(
    right.getByRole("button", { name: "Auto arrange reference", exact: true }),
  ).toBeEnabled();
  await right
    .getByRole("button", { name: "Fit reference graph", exact: true })
    .click();
  return {
    right,
    left: page.getByRole("region", { name: "Editable graph", exact: true }),
  };
}

test("reference multi-selection copies nested blocks and internal edges to the editable pane with undo", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const { right, left } = await split(page);
  const before = await document(page);
  await right.locator('.react-flow__node[data-id="mask"]').click();
  await right
    .locator('.react-flow__node[data-id="denoise"]')
    .click({ modifiers: ["Control"] });
  await expect(right.locator(".react-flow__node.selected")).toHaveCount(2);
  await page.keyboard.press("Control+c");
  await expect(right.getByRole("status")).toContainText("Copied");
  await left.locator(".flow-wrap").focus();
  await page.keyboard.press("Control+v");
  await expect(left.locator(".react-flow__node")).toHaveCount(
    before.graph.nodes.length + 2,
  );
  const after = await document(page);
  const copied = after.graph.nodes.filter(
    (node) => !before.graph.nodes.some((original) => original.id === node.id),
  );
  expect(copied.map((node) => node.data.kind).sort()).toEqual(["loop", "mask"]);
  expect(
    copied.find((node) => node.data.kind === "loop")!.data.body!.nodes.length,
  ).toBeGreaterThan(1);
  expect(
    after.graph.edges.filter(
      (edge) =>
        copied.some((node) => node.id === edge.source) &&
        copied.some((node) => node.id === edge.target),
    ),
  ).toHaveLength(1);
  await expect(right.locator(".react-flow__node")).toHaveCount(4);
  await expect(right.locator(".react-flow__node.selected")).toHaveCount(0);
  await page.keyboard.press("Control+z");
  await expect(left.locator(".react-flow__node")).toHaveCount(
    before.graph.nodes.length,
  );
  await page.keyboard.press("Control+Shift+z");
  await expect(left.locator(".react-flow__node")).toHaveCount(
    before.graph.nodes.length + 2,
  );
  const audit = await new AxeBuilder({ page })
    .include(".reference-pane")
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(audit.violations).toEqual([]);
  expect(errors).toEqual([]);
});

for (const width of [1600, 390])
  test(`reference copy button works without Clipboard API at ${width}px, including nested graphs`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 1000 });
    await page.addInitScript(() =>
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: undefined,
      }),
    );
    const { right, left } = await split(page);
    const controls = await right.locator(".react-flow__controls").boundingBox();
    expect(controls && controls.width < 60).toBe(true);
    const before = await document(page);
    await right
      .getByRole("button", { name: "Open loop body", exact: true })
      .click();
    await right
      .getByRole("button", { name: "Fit reference graph", exact: true })
      .click();
    await right.locator('.react-flow__node[data-id="predict"]').click();
    const close = page.getByRole("button", {
      name: "Close properties",
      exact: true,
    });
    if (await close.isVisible()) await close.click();
    await right
      .getByRole("button", { name: "Copy reference selection", exact: true })
      .click();
    await expect(right.getByRole("status")).toContainText("Copied");
    await left.locator(".flow-wrap").focus();
    await page.keyboard.press("Control+v");
    await expect(left.locator(".react-flow__node")).toHaveCount(
      before.graph.nodes.length + 1,
    );
    const copied = (await document(page)).graph.nodes.find(
      (node) => !before.graph.nodes.some((original) => original.id === node.id),
    )!;
    expect(copied.data.kind).toBe("block");
    expect(
      copied.data.body!.nodes.find((node) => node.data.kind === "embedding")!
        .data.params.width,
    ).toBe(128);
    await expect(right.locator(".react-flow__node")).toHaveCount(4);
    await right
      .getByRole("button", { name: "Fit reference graph", exact: true })
      .click();
    await right.locator(".react-flow__node").first().click();
    if (await close.isVisible()) await close.click();
    await page.keyboard.press("Delete");
    await page.keyboard.press("Control+v");
    await expect(right.locator(".react-flow__node")).toHaveCount(4);
    expect((await document(page)).graph.nodes.length).toBe(
      before.graph.nodes.length + 1,
    );
  });
