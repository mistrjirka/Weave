import { expect, test } from "@playwright/test";
import { cardStyleViolations } from "./cardStyles";

for (const viewport of [
  { width: 1600, height: 1000 },
  { width: 390, height: 844 },
]) {
  test(`MoE and embedding examples are navigable at ${viewport.width}px`, async ({
    page,
  }) => {
    await page.setViewportSize(viewport);
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/?model=embedding-table");
    await expect(
      page.locator('.react-flow__node[data-id="embedding"]'),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Shapes compatible" }),
    ).toBeVisible();
    await page.goto("/?model=moe-transformer&block=stack");
    await expect(
      page.locator('.react-flow__node[data-id="ffn"] .moe-preview'),
    ).toBeVisible();
    await expect.poll(() => cardStyleViolations(page)).toEqual([]);
    await page
      .getByRole("button", { name: "Open experts", exact: true })
      .click();
    await expect(
      page.locator('.react-flow__node[data-id="router"]'),
    ).toBeVisible();
    await expect(
      page.locator('.react-flow__node[data-id="combine"]'),
    ).toBeVisible();
    await expect(page.locator('[data-visual-role="moe-expert"]')).toHaveCount(
      4,
    );
    await expect.poll(() => cardStyleViolations(page)).toEqual([]);
    await page
      .locator('.react-flow__node[data-id="expert_1"]')
      .getByRole("button", { name: "Open block" })
      .click();
    await expect(
      page.locator('.react-flow__node[data-id="ffn"]'),
    ).toBeVisible();
    expect(errors).toEqual([]);
  });
}

test("changing expert count updates the graph and keeps expert bodies editable", async ({
  page,
}) => {
  await page.goto("/?model=moe-transformer&block=stack");
  await page.locator('.react-flow__node[data-id="ffn"]').click();
  const count = page.getByRole("textbox", { name: "experts", exact: true });
  await count.fill("3");
  await count.press("Enter");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Open experts", exact: true }).click();
  await expect(page.locator('[data-visual-role="moe-expert"]')).toHaveCount(3);
  await expect(
    page.locator('.react-flow__node[data-id="combine"] .target'),
  ).toHaveCount(4);
  await page
    .locator('.react-flow__node[data-id="expert_1"]')
    .getByRole("button", { name: "Open block" })
    .click();
  await page.locator('.react-flow__node[data-id="ffn"]').click();
  const hidden = page.getByRole("textbox", { name: "hidden", exact: true });
  await hidden.fill("64");
  await hidden.press("Enter");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
});
