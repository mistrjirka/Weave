import { test, expect } from "@playwright/test";
import type { Project } from "../../frontend/src/types";

async function document(
  page: import("@playwright/test").Page,
): Promise<Project> {
  return page.evaluate(() =>
    JSON.parse(localStorage.getItem("weave.project") || "null"),
  );
}
test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
});

test("insert an independent layer between shared transformer applications", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.getByRole("button", { name: "Models", exact: true }).click();
  await page
    .locator(".model-item")
    .filter({ hasText: "Looped transformer" })
    .click();
  await expect(page.locator(".weight-link-label")).toHaveCount(2);
  await expect(page.locator(".transformer-preview")).toHaveCount(3);
  await expect(page.locator(".adapter-preview")).toHaveCount(2);
  await expect
    .poll(async () => (await document(page)).graph.layout?.engine)
    .toBe("elk");
  await expect(
    page.getByRole("button", { name: "Auto arrange", exact: true }),
  ).toBeEnabled();
  // Allow the arrange action's fit animation to finish before reading screen coordinates.
  await expect
    .poll(async () => {
      const viewport = page.locator(".react-flow__viewport");
      const before = await viewport.getAttribute("style");
      await page.evaluate(
        () =>
          new Promise((resolve) =>
            requestAnimationFrame(() => requestAnimationFrame(resolve)),
          ),
      );
      return before === (await viewport.getAttribute("style"));
    })
    .toBe(true);
  const point = await page
    .locator(
      '.react-flow__edge[data-id="transformer_1-adapter_1-a"] .react-flow__edge-path',
    )
    .evaluate((element) => {
      const path = element as SVGPathElement,
        point = path
          .getPointAtLength(path.getTotalLength() / 2)
          .matrixTransform(path.getScreenCTM()!);
      return { x: point.x, y: point.y };
    });
  await page.mouse.click(point.x, point.y);
  await page.getByLabel("Layer to insert").selectOption("adapter");
  await page.getByRole("button", { name: "Insert independent layer" }).click();
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  const p = await document(page);
  const added = p.graph.nodes.find(
    (n) => n.data.label === "Independent residual adapter",
  );
  expect(added?.data.share).toBeUndefined();
  expect(added?.data.body?.nodes.some((n) => n.data.kind === "ffn")).toBe(true);
  expect(
    p.graph.nodes.filter((n) => n.data.share === "transformer_core"),
  ).toHaveLength(3);
  expect(
    p.graph.edges.some(
      (e) => e.source === "transformer_1" && e.target === added?.id,
    ),
  ).toBe(true);
  expect(
    p.graph.edges.some(
      (e) => e.source === added?.id && e.target === "adapter_1",
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
});

test("editing one shared body updates its applications and can be made independent", async ({
  page,
}) => {
  await page.getByRole("button", { name: "Models", exact: true }).click();
  await page
    .locator(".model-item")
    .filter({ hasText: "Looped transformer" })
    .click();
  await page.locator('.react-flow__node[data-id="transformer_1"]').dblclick();
  await page.locator('.react-flow__node[data-id="ffn"]').click();
  await page.getByRole("textbox", { name: "hidden", exact: true }).fill("512");
  await page
    .getByRole("textbox", { name: "hidden", exact: true })
    .press("Enter");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  let p = await document(page);
  const shared = p.graph.nodes.filter(
    (n) => n.data.share === "transformer_core",
  );
  expect(shared).toHaveLength(3);
  for (const n of shared)
    expect(
      n.data.body?.nodes.find((n) => n.id === "ffn")?.data.params.hidden,
    ).toBe(512);
  await page.getByRole("button", { name: "Model", exact: true }).click();
  await page.locator('.react-flow__node[data-id="transformer_2"]').click();
  await page
    .getByRole("button", { name: "Make this application independent" })
    .click();
  await expect(page.locator(".weight-link-label")).toHaveCount(1);
  p = await document(page);
  expect(
    p.graph.nodes.find((n) => n.id === "transformer_2")?.data.share,
  ).toBeUndefined();
});

test("compact loops expand into editable applications with explicit weight sharing", async ({
  page,
}) => {
  await page.getByRole("button", { name: "Open loop body" }).click();
  await page.getByRole("button", { name: "Open block", exact: true }).click();
  await page.locator('.react-flow__node[data-id="refine"]').click();
  await page
    .getByRole("button", { name: "Expand into editable applications" })
    .click();
  await expect(page.locator(".weight-link-label")).toHaveCount(2);
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  const p = await document(page);
  const denoise = p.graph.nodes.find((n) => n.id === "denoise")!;
  const predictor = denoise.data.body!.nodes.find((n) => n.id === "predict")!;
  expect(
    predictor.data.body!.nodes.filter((n) => n.data.iteration),
  ).toHaveLength(3);
});

test("sparse patterns are editable building blocks on the same canvas", async ({
  page,
}) => {
  await page.getByRole("button", { name: "Models", exact: true }).click();
  await page
    .locator(".model-item")
    .filter({ hasText: "Sparse attention composition" })
    .click();
  await page.getByRole("button", { name: "Open loop body" }).click();
  await expect(page.locator(".pattern-grid")).toHaveCount(2);
  await page.locator('.react-flow__node[data-id="local"]').click();
  await expect(
    page.getByRole("textbox", { name: "window", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("textbox", { name: "density", exact: true }),
  ).toHaveCount(0);
  await page.locator('select[name="pattern"]').selectOption("random");
  await expect(
    page.getByRole("textbox", { name: "density", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("textbox", { name: "window", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await page.screenshot({ path: "data/sparse-pattern-editor.png" });
});

test("add a shared application without adding transformer parameters", async ({
  page,
  request,
}) => {
  await page.getByRole("button", { name: "Models", exact: true }).click();
  await page
    .locator(".model-item")
    .filter({ hasText: "Looped transformer" })
    .click();
  const before = await document(page);
  const original = await (
    await request.post("/api/validate", { data: before })
  ).json();
  await page.locator('.react-flow__node[data-id="transformer_1"]').click();
  await page
    .getByRole("button", { name: "Add application with shared weights" })
    .click();
  await expect(page.locator(".weight-link-label")).toHaveCount(3);
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  const after = await document(page);
  const report = await (
    await request.post("/api/validate", { data: after })
  ).json();
  expect(report.parameters).toBe(original.parameters);
  expect(
    after.graph.nodes.filter((n) => n.data.share === "transformer_core"),
  ).toHaveLength(4);
});

test("custom attention cells can be edited and malformed matrices cannot crash the editor", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.getByRole("button", { name: "Models", exact: true }).click();
  await page
    .locator(".model-item")
    .filter({ hasText: "Sparse attention composition" })
    .click();
  await page.locator('.react-flow__node[data-id="ids"]').click();
  await page
    .getByRole("textbox", { name: "shape", exact: true })
    .fill('["B",4]');
  await page
    .getByRole("textbox", { name: "shape", exact: true })
    .press("Enter");
  await page.getByRole("button", { name: "Open loop body" }).click();
  await page.locator('.react-flow__node[data-id="local"]').click();
  await page.locator('select[name="pattern"]').selectOption("custom");
  const cell = page.getByRole("button", {
    name: "Query 1, key 0",
    exact: true,
  });
  await expect(cell).toHaveAttribute("aria-pressed", "true");
  await cell.click();
  await expect(cell).toHaveAttribute("aria-pressed", "false");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await page
    .getByRole("textbox", { name: "matrix", exact: true })
    .fill("[1,2]");
  await page
    .getByRole("textbox", { name: "matrix", exact: true })
    .press("Enter");
  await expect(page.getByRole("status")).toBeVisible();
  await expect(cell).toHaveAttribute("aria-pressed", "false");
  expect(errors).toEqual([]);
});
