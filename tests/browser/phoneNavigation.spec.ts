import { expect, test, type Locator, type Page } from "@playwright/test";
import { stageSchema, templateSchema } from "../../frontend/src/trainingApi";

test.use({ hasTouch: true, isMobile: true });

async function swipeUp(page: Page, scroller: Locator) {
  const box = await scroller.boundingBox();
  if (!box) throw new Error("Scrollable panel is not visible");
  const x = box.x + 8;
  const start = Math.min(
    box.y + box.height - 24,
    page.viewportSize()!.height - 70,
  );
  const end = Math.max(box.y + 24, start - 300);
  const session = await page.context().newCDPSession(page);
  try {
    await session.send("Input.dispatchTouchEvent", {
      type: "touchStart",
      touchPoints: [{ x, y: start }],
    });
    for (let y = start - 15; y >= end; y -= 15) {
      await session.send("Input.dispatchTouchEvent", {
        type: "touchMove",
        touchPoints: [{ x, y }],
      });
      await page.waitForTimeout(16);
    }
    await session.send("Input.dispatchTouchEvent", {
      type: "touchEnd",
      touchPoints: [],
    });
    // Wait for native momentum scrolling to stop. A tap during a fling only
    // stops the scroll; Chromium intentionally suppresses its click event.
    await scroller.evaluate(
      (element) =>
        new Promise<void>((resolve) => {
          let previous = element.scrollTop;
          let stableFrames = 0;
          function check() {
            const current = element.scrollTop;
            stableFrames =
              Math.abs(current - previous) < 1 ? stableFrames + 1 : 0;
            previous = current;
            if (stableFrames >= 8) resolve();
            else requestAnimationFrame(check);
          }
          requestAnimationFrame(check);
        }),
    );
  } finally {
    await session.detach();
  }
}

async function noPageOverflow(page: Page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
}

for (const viewport of [
  { width: 320, height: 740 },
  { width: 390, height: 844 },
  { width: 844, height: 390 },
]) {
  test(`phone navigation, tokenizer, dialogs and stage setup work at ${viewport.width}×${viewport.height}`, async ({
    page,
    request,
  }) => {
    test.setTimeout(90000);
    await page.setViewportSize(viewport);
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/?model=transformer");
    await expect(
      page.getByRole("button", { name: "Shapes compatible", exact: true }),
    ).toBeVisible();
    const title = page.locator(".project-title");
    await expect(title.locator("span")).toHaveText("Decoder transformer");
    expect(
      await title.evaluate((el) => getComputedStyle(el).justifyContent),
    ).toBe("flex-start");
    await expect(
      page.getByRole("button", { name: "Open sidebar", exact: true }),
    ).toBeVisible();
    await expect(page.locator(".inspector")).not.toBeVisible();
    if (viewport.width === 390) {
      await page.setViewportSize({ width: 844, height: 390 });
      await expect(
        page.getByRole("button", {
          name: "Expand canvas variables",
          exact: true,
        }),
      ).toBeVisible();
      await expect(
        page.getByRole("button", { name: "Open sidebar", exact: true }),
      ).toBeVisible();
      await page.setViewportSize(viewport);
    }
    const minimizeVariables = page.getByRole("button", {
      name: "Minimize canvas variables",
      exact: true,
    });
    if (await minimizeVariables.isVisible()) await minimizeVariables.tap();
    await page.locator('.flow-wrap .react-flow__node[data-id="ids"]').tap();
    const inspector = page.getByRole("complementary", {
      name: "Property inspector",
    });
    await expect(inspector).toHaveClass(/is-open/);
    const tokenizer = inspector.getByRole("button", {
      name: "Select tokenizer",
      exact: true,
    });
    await expect(tokenizer.locator("span")).toBeVisible();
    expect(
      await tokenizer.evaluate((el) =>
        Number.parseFloat(getComputedStyle(el).fontSize),
      ),
    ).toBeGreaterThanOrEqual(14);
    await tokenizer.tap();
    const dialog = page.getByRole("dialog");
    await dialog
      .getByRole("button", { name: "Repository / path", exact: true })
      .tap();
    await noPageOverflow(page);
    await dialog
      .getByRole("button", { name: "Other project", exact: true })
      .tap();
    await noPageOverflow(page);
    await dialog
      .getByRole("button", { name: "Close dialog", exact: true })
      .tap();
    await inspector
      .getByRole("button", { name: "Close properties", exact: true })
      .tap();
    await page.getByRole("button", { name: "Open sidebar", exact: true }).tap();
    const library = page.locator("#workspace-sidebar");
    await library
      .getByRole("textbox", { name: "Search blocks", exact: true })
      .fill("attention");
    await expect(library.locator(".library-item").first()).toBeVisible();
    await library
      .getByRole("button", { name: "Close library", exact: true })
      .tap();
    await page.getByRole("button", { name: "New project", exact: true }).tap();
    await expect(
      dialog.getByRole("textbox", { name: "Project name", exact: true }),
    ).toBeVisible();
    await dialog
      .getByRole("button", { name: "Create project", exact: true })
      .scrollIntoViewIfNeeded();
    await expect(
      dialog.getByRole("button", { name: "Create project", exact: true }),
    ).toBeInViewport();
    await noPageOverflow(page);
    await dialog
      .getByRole("button", { name: "Close dialog", exact: true })
      .tap();
    await page
      .getByRole("button", { name: "Weave settings", exact: true })
      .tap();
    const close = dialog.getByRole("button", {
      name: "Close dialog",
      exact: true,
    });
    const headingBox = await dialog
      .getByRole("heading", { name: "GPU performance tuning", exact: true })
      .boundingBox();
    const closeBox = await close.boundingBox();
    expect(Math.abs(headingBox!.y - closeBox!.y)).toBeLessThan(10);
    await swipeUp(page, dialog);
    await expect
      .poll(() => dialog.evaluate((el) => el.scrollTop))
      .toBeGreaterThan(40);
    await close.tap();
    await expect(dialog).toHaveCount(0);
    await page.getByRole("button", { name: "PyTorch code", exact: true }).tap();
    await expect(page.locator(".code-view")).toBeVisible();
    await noPageOverflow(page);
    await page.getByRole("button", { name: "Sources", exact: true }).tap();
    await noPageOverflow(page);

    // In-memory stage fixtures let the touch walkthrough exercise setup without
    // saving user files, launching a job, or changing an existing conversation.
    const template = templateSchema.parse(
      await (await request.get("/api/training/templates")).json(),
    );
    const stage = stageSchema.parse({
      id: "phone-navigation",
      name: "Pretraining",
      config: { ...template.config, execution: "monitor" },
      script: template.scripts.causal,
      script_path: "/tmp/weave-phone/pretraining/train.py",
    });
    await page.route("**/api/training/plans/*", (route) =>
      route.fulfill({
        json: { version: 1, project_id: "transformer", stages: [stage] },
      }),
    );
    await page.route("**/api/training/runs?project_id=*", (route) =>
      route.fulfill({ json: [] }),
    );
    await page.route(
      "**/api/training/workspaces/*/stages/phone-navigation",
      (route) => route.fulfill({ json: stage }),
    );
    await page.getByRole("button", { name: "Training", exact: true }).tap();
    await expect(
      page.getByRole("region", { name: "Compute configuration", exact: true }),
    ).toBeVisible();
    const workspace = page.getByRole("region", {
      name: "Training workspace",
      exact: true,
    });
    await swipeUp(page, workspace);
    await expect
      .poll(() => workspace.evaluate((el) => el.scrollTop))
      .toBeGreaterThan(40);
    await page
      .getByRole("button", { name: "Python script", exact: true })
      .tap();
    await expect(
      page.getByRole("textbox", {
        name: "Training Python script",
        exact: true,
      }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Stages and run history", exact: true })
      .tap();
    await expect(
      dialog.getByRole("textbox", { name: "Stage name", exact: true }),
    ).toHaveValue("Pretraining");
    await dialog
      .getByRole("button", { name: "Close dialog", exact: true })
      .tap();
    await noPageOverflow(page);
    await page.screenshot({
      path: `test-results/phone-workflow-${viewport.width}.png`,
    });
    expect(errors).toEqual([]);
  });
}
