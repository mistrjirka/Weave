import { expect, test } from "@playwright/test";

test("phone sidebar is reachable from both the graph and training workspace", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  const menu = page.getByRole("button", { name: "Open sidebar", exact: true });
  const sidebar = page.locator("#workspace-sidebar");
  await expect(menu).toBeVisible();
  await menu.click();
  await expect(sidebar).toBeVisible();
  await expect(menu).toHaveAttribute("aria-expanded", "true");
  await expect(
    sidebar.getByRole("button", { name: "Close library", exact: true }),
  ).toBeFocused();
  await sidebar
    .getByRole("button", { name: "Close library", exact: true })
    .click();
  await expect(sidebar).not.toBeVisible();
  await expect(menu).toBeFocused();
  await menu.click();
  await page.keyboard.press("Escape");
  await expect(sidebar).not.toBeVisible();
  await expect(menu).toBeFocused();
  await page.getByRole("button", { name: "Training", exact: true }).click();
  await expect(
    page.getByRole("region", { name: "Training workspace", exact: true }),
  ).toBeVisible();
  await menu.click();
  await expect(sidebar).toBeVisible();
  await expect(
    sidebar.getByRole("button", { name: "Your workspace", exact: false }),
  ).toBeVisible();
  await sidebar
    .getByRole("button", { name: "Architecture", exact: true })
    .click();
  await expect(sidebar).not.toBeVisible();
  for (const width of [320, 390]) {
    await page.setViewportSize({ width, height: 844 });
    expect(
      await page
        .locator(".topbar")
        .evaluate((el) => el.scrollWidth <= el.clientWidth + 1),
    ).toBeTruthy();
  }
  await page.screenshot({ path: "/tmp/weave-sidebar-phone.png" });
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(menu).not.toBeVisible();
});
