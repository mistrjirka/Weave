import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("global active training indicator lists multiple projects, refreshes and fits phone", async ({
  page,
}) => {
  const first = {
    id: "run-test-a",
    project_id: "project-a",
    project_name: "Alpha model",
    stage_id: "stage-a",
    name: "Pretraining",
    created: "2026-10-31T20:00:00Z",
    state: "running",
  };
  const second = {
    ...first,
    id: "run-test-b",
    project_id: "project-b",
    project_name: "Beta model",
    stage_id: "stage-b",
    state: "preparing",
  };
  let runs = [first];
  await page.route("**/api/training/active-runs", (route) =>
    route.fulfill({ json: runs }),
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  const badge = page.getByRole("button", {
    name: "Active training runs: 1",
    exact: true,
  });
  await expect(badge).toBeVisible();
  expect(
    await page
      .locator(".topbar")
      .evaluate((el) => el.scrollWidth <= el.clientWidth + 1),
  ).toBeTruthy();
  await badge.click();
  const dialog = page.getByRole("dialog", {
    name: "Active training runs",
    exact: true,
  });
  await expect(
    dialog.getByRole("link", { name: /Alpha model/ }),
  ).toHaveAttribute("href", /model=project-a.*run=run-test-a/);
  runs = [first, second];
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await expect(dialog.getByRole("link", { name: /Beta model/ })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Active training runs: 2", exact: true }),
  ).toHaveCount(1);
  expect(
    await dialog.evaluate((el) => el.scrollWidth <= el.clientWidth + 1),
  ).toBeTruthy();
  const audit = await new AxeBuilder({ page })
    .include(".active-training-dialog")
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(
    audit.violations.map((v) => ({
      id: v.id,
      nodes: v.nodes.map((n) => n.failureSummary),
    })),
  ).toEqual([]);
  await page.screenshot({
    path: "/tmp/weave-global-training-phone.png",
    fullPage: true,
  });
  runs = [];
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await expect(dialog).toContainText("No active training runs.");
  await dialog
    .getByRole("button", { name: "Close active training", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: /Active training runs:/ }),
  ).toHaveCount(0);
});
