import { expect, test } from "@playwright/test";
import { projectSchema } from "../../frontend/src/api";

test("layout, browsing and theme changes do not restart PyTorch generation, while parameter edits do", async ({
  page,
}) => {
  let compilations = 0;
  await page.route("**/api/code", (route) => {
    compilations++;
    return route.continue();
  });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible", exact: true }),
  ).toBeVisible();
  await expect
    .poll(
      async () =>
        projectSchema.parse(
          await page.evaluate(() =>
            JSON.parse(localStorage.getItem("weave.project") || "null"),
          ),
        ).graph.layout?.engine,
    )
    .toBe("elk");
  const initial = compilations;
  await page
    .getByRole("combobox", { name: "Color theme", exact: true })
    .selectOption("dark");
  await page.getByRole("button", { name: "Auto arrange", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Auto arrange", exact: true }),
  ).toBeEnabled();
  await page
    .getByRole("button", { name: "Open loop body", exact: true })
    .click();
  await page.getByRole("button", { name: "Open block", exact: true }).click();
  await expect(page.locator(".embedding-grid")).toHaveCount(2);
  await expect
    .poll(
      async () =>
        projectSchema
          .parse(
            await page.evaluate(() =>
              JSON.parse(localStorage.getItem("weave.project") || "null"),
            ),
          )
          .graph.nodes.find((n) => n.id === "denoise")
          ?.data.body?.nodes.find((n) => n.id === "predict")?.data.body?.layout
          ?.engine,
    )
    .toBe("elk");
  expect(compilations).toBe(initial);
  await page.locator('.react-flow__node[data-id="memory"]').click();
  const width = page.getByRole("textbox", { name: "width", exact: true });
  await width.fill("768");
  await width.press("Enter");
  await expect(page.locator(".validation-bar")).toContainText("graph issues");
  expect(compilations).toBe(initial + 1);
});
