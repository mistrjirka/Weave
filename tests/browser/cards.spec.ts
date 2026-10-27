import { expect, test } from "@playwright/test";
import { catalog } from "../../frontend/src/catalog";
import {
  cardStyleViolations,
  collectCardSamples,
  saveCardGallery,
} from "./cardStyles";

for (const theme of ["light", "black"]) {
  test(`every library card keeps its content inside the card in ${theme}`, async ({
    page,
  }) => {
    test.setTimeout(60000);
    await page.addInitScript(
      (value) => localStorage.setItem("weave.theme", value),
      theme,
    );
    await page.goto("/?model=blank");
    await expect(page.locator(".project-title")).toHaveText("Blank project");
    await expect(page.locator(".arch-node")).toHaveCount(2);
    const search = page.getByRole("textbox", {
      name: "Search blocks",
      exact: true,
    });
    for (const entry of catalog) {
      await search.fill(entry.label);
      await page
        .getByRole("button", {
          name: `${entry.label} ${entry.description}`,
          exact: true,
        })
        .click();
    }
    await expect(page.locator(".arch-node")).toHaveCount(catalog.length + 2);
    await expect(page.locator(".validation-bar")).toContainText("graph issues");
    await expect.poll(() => cardStyleViolations(page)).toEqual([]);
    const samples = new Map<string, string>();
    await collectCardSamples(page, samples);
    await saveCardGallery(page, samples, `catalog-${theme}`);
  });
}

test("a long denoising label wraps without pushing its preview into the card borders on a phone", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.addInitScript(() => localStorage.setItem("weave.theme", "black"));
  await page.goto("/?model=combined&node=denoise");
  await expect(
    page.getByRole("button", { name: "Shapes compatible", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Properties", exact: true }).click();
  await page
    .getByRole("textbox", { name: "Block name", exact: true })
    .fill(
      "Iterative denoising with a very long custom label and additional_context_that_needs_wrapping",
    );
  await page
    .getByRole("button", { name: "Close properties", exact: true })
    .click();
  await page.getByRole("button", { name: /^Focus current node:/ }).click();
  await expect.poll(() => cardStyleViolations(page)).toEqual([]);
  await page
    .locator('.react-flow__node[data-id="denoise"] .arch-node')
    .screenshot({ path: "data/card-denoising-long-label-phone.png" });
  await page
    .getByRole("button", { name: "Open loop body", exact: true })
    .click();
  await expect(page.locator(".predictor-preview")).toHaveCount(1);
  await expect.poll(() => cardStyleViolations(page)).toEqual([]);
});
