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
    await page.goto("/?model=moe-transformer");
    await expect(
      page.locator('.react-flow__node[data-id="moe"] .moe-preview'),
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
  await page.goto("/?model=moe-transformer");
  await page.locator('.react-flow__node[data-id="moe"]').click();
  const router = page.getByRole("combobox", {
    name: "Router scoring",
    exact: true,
  });
  await router.selectOption("sigmoid");
  await expect(page.locator(".moe-preview")).toContainText("sigmoid router");
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

test("older MoE graphs without a router setting can choose one", async ({
  page,
  request,
}) => {
  const templates = await (await request.get("/api/templates")).json();
  const project = templates.find(
    (item: { id: string }) => item.id === "moe-transformer",
  );
  const moe = project.graph.nodes.find(
    (node: { data: { kind: string } }) => node.data.kind === "moe",
  );
  delete moe.data.params.router;
  await page.goto("/?model=blank");
  await page
    .locator('input[type="file"]')
    .first()
    .setInputFiles({
      name: "old-moe.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(project)),
    });
  await page.locator('.react-flow__node[data-id="moe"]').click();
  const router = page.getByRole("combobox", { name: "Router scoring" });
  await expect(router).toHaveValue("softmax");
  await router.selectOption("sigmoid");
  await expect(page.locator(".moe-preview")).toContainText("sigmoid router");
});
