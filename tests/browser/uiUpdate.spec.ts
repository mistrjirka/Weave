import { test, expect } from "@playwright/test";

test("offers a browser reload when a new UI is deployed", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Training", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Reload UI", exact: true }),
  ).toHaveCount(0);
  await page.route("http://127.0.0.1:8173/", (route) =>
    route.fulfill({
      contentType: "text/html",
      body: '<!doctype html><script type="module" src="/assets/new-release.js"></script>',
    }),
  );
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  const reload = page.getByRole("button", { name: "Reload UI", exact: true });
  await expect(reload).toBeVisible();
  await expect(reload).toHaveAttribute("title", /does not stop training/);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page
      .locator(".viewbar")
      .evaluate((element) => element.scrollWidth <= element.clientWidth + 1),
  ).toBeTruthy();
});
