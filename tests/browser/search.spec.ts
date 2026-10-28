import { expect, test } from "@playwright/test";

for (const theme of ["light", "dark", "black"]) {
  for (const width of [1600, 390]) {
    test(`sidebar search works and keeps one focus ring in ${theme} at ${width}px`, async ({
      page,
    }) => {
      await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 });
      await page.addInitScript(
        (value) => localStorage.setItem("weave.theme", value),
        theme,
      );
      await page.goto("/?model=blank");
      if (width === 390)
        await page
          .getByRole("button", { name: "Add blocks", exact: true })
          .last()
          .click();
      const search = page.getByRole("textbox", {
        name: "Search blocks",
        exact: true,
      });
      await expect(search).toBeVisible();
      await search.fill("attention");
      await expect(page.locator(".library-item").first()).toBeVisible();
      await expect(
        page.getByRole("button", { name: "Clear search", exact: true }),
      ).toBeVisible();
      const focus = await search.evaluate((input) => {
        const field = input.closest(".library-search")!;
        const a = input.getBoundingClientRect(),
          b = field.getBoundingClientRect();
        return {
          inputOutline: getComputedStyle(input).outlineStyle,
          fieldOutline: getComputedStyle(field).outlineStyle,
          contained:
            a.left >= b.left &&
            a.right <= b.right &&
            a.top >= b.top &&
            a.bottom <= b.bottom,
        };
      });
      expect(focus).toEqual({
        inputOutline: "none",
        fieldOutline: "solid",
        contained: true,
      });
      await search.fill("no-such-building-block");
      await expect(
        page.getByRole("status").filter({ hasText: "No blocks match" }),
      ).toBeVisible();
      await page
        .getByRole("button", { name: "Clear search", exact: true })
        .click();
      await expect(search).toHaveValue("");
      await expect(search).toBeFocused();
      await page
        .locator(".sidebar-tabs")
        .getByRole("button", { name: "Projects", exact: true })
        .click();
      const projectSearch = page.getByRole("textbox", {
        name: "Search projects and examples",
        exact: true,
      });
      await projectSearch.fill("embedding");
      await expect(page.locator(".sidebar")).toContainText("Embedding table");
      await page
        .getByRole("button", { name: "Clear search", exact: true })
        .click();
      await expect(projectSearch).toBeFocused();
    });
  }
}
