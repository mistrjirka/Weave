import { test, expect } from "@playwright/test";

test("an old benchmark response cannot replace the job from a reopened dialog", async ({
  page,
}) => {
  let visits = 0;
  let resolveOld: (() => void) | undefined;
  let oldStarted = false;
  await page.route("**/api/training/capabilities", (route) =>
    route.fulfill({
      json: {
        enabled: true,
        authenticated: true,
        python: "/test/python",
        token_file: null,
        remote_launch: false,
      },
    }),
  );
  await page.route("**/api/training/hardware", (route) =>
    route.fulfill({ json: { gpus: [], scope: "test" } }),
  );
  await page.route("**/api/training/gpu-tuning", (route) => {
    visits++;
    return route.fulfill({
      json: {
        settings: { auto_apply: false },
        profiles: [],
        jobs: [
          {
            id: visits === 1 ? "old" : "current",
            state: "running",
            message: "benchmark",
          },
        ],
      },
    });
  });
  await page.route("**/api/training/gpu-tuning/old", async (route) => {
    oldStarted = true;
    await new Promise<void>((resolve) => {
      resolveOld = resolve;
    });
    await route.fulfill({
      json: {
        id: "old",
        state: "completed",
        message: "STALE benchmark result",
        results: [],
      },
    });
  });
  await page.route("**/api/training/gpu-tuning/current", (route) =>
    route.fulfill({
      json: {
        id: "current",
        state: "completed",
        message: "Current benchmark result",
        results: [],
      },
    }),
  );
  await page.goto("/?model=transformer");
  const open = page.getByRole("button", {
    name: "Weave settings",
    exact: true,
  });
  await open.click();
  const dialog = page.getByRole("dialog", {
    name: "Weave settings",
    exact: true,
  });
  await expect.poll(() => oldStarted).toBe(true);
  await dialog
    .getByRole("button", { name: "Close dialog", exact: true })
    .click();
  await open.click();
  await expect(dialog).toContainText("Current benchmark result");
  const oldResponse = page.waitForResponse((response) =>
    response.url().endsWith("/gpu-tuning/old"),
  );
  resolveOld?.();
  await oldResponse;
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      ),
  );
  await expect(dialog).not.toContainText("STALE benchmark result");
  await expect(dialog).toContainText("Current benchmark result");
});
