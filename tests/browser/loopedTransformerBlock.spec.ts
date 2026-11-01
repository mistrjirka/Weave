import { test, expect } from "@playwright/test";
import { projectSchema } from "../../frontend/src/api";

test("new looped transformer opens into a single editable transformer layer", async ({
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
  const loopNode = page.locator(".react-flow__node").filter({
    has: page.locator(".node-title").filter({ hasText: "Looped transformer" }),
  });
  await expect(loopNode).toBeVisible();
  await expect
    .poll(async () => {
      return page.evaluate(() => {
        const value = localStorage.getItem("weave.project");
        return value
          ? JSON.parse(value).graph.nodes.some(
              (node: { data: { label: string } }) =>
                node.data.label === "Looped transformer",
            )
          : false;
      });
    })
    .toBe(true);
  const project = projectSchema.parse(
    await page.evaluate(() =>
      JSON.parse(localStorage.getItem("weave.project") || "null"),
    ),
  );
  const loop = project.graph.nodes.find(
    (node) => node.data.label === "Looped transformer",
  )!;
  expect(loop.data.body!.nodes.map((node) => node.data.kind)).toEqual([
    "input",
    "block",
    "output",
  ]);
  await loopNode
    .getByRole("button", { name: "Open loop body", exact: true })
    .click();
  await expect(
    page.locator(".node-title").filter({ hasText: "Transformer layer" }),
  ).toBeVisible();
  await expect(page.locator(".react-flow__node")).toHaveCount(3);
  await page.locator('.react-flow__node[data-id="transformer_layer"]').click();
  await page.getByRole("button", { name: "Open block", exact: true }).click();
  await expect(
    page.locator(".node-title").filter({ hasText: "SwiGLU feed-forward" }),
  ).toBeVisible();
  await expect(
    page.locator(".node-title").filter({ hasText: "Grouped-query attention" }),
  ).toBeVisible();
});
