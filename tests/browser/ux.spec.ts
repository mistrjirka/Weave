import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { editProjectDescription } from "./helpers";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
});

test("LAN HTTP can add nodes, save projects and copy generated code", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  // Reproduce the browser APIs unavailable on plain HTTP LAN origins.
  await page.evaluate(() => {
    Object.defineProperty(crypto, "randomUUID", {
      value: undefined,
      configurable: true,
    });
    Object.defineProperty(navigator, "clipboard", {
      value: undefined,
      configurable: true,
    });
  });
  await page
    .locator(".library-item")
    .filter({ hasText: "Token IDs → learned vectors" })
    .click();
  await expect(page.locator(".react-flow__node")).toHaveCount(5);
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("Project saved");
  const id = await page.evaluate(
    () => JSON.parse(localStorage.getItem("weave.project")!).id,
  );
  await page.getByRole("button", { name: "PyTorch code", exact: true }).click();
  await page.getByRole("button", { name: "Copy code" }).click();
  await expect(page.getByRole("status")).toContainText(
    /Code copied|Use Export/,
  );
  expect(errors).toEqual([]);
  await page.request.delete("/api/projects/" + id);
});

test("property edits show errors and numeric overrides can become bindings again", async ({
  page,
}) => {
  await page.getByRole("button", { name: "Open loop body" }).click();
  await page.getByRole("button", { name: "Open block" }).click();
  await page.locator('.react-flow__node[data-id="memory"]').click();
  const width = page.getByRole("textbox", { name: "width", exact: true });
  await width.fill("256");
  await width.press("Enter");
  await expect(page.locator(".validation-bar")).toContainText("graph issues");
  await width.fill("invalid");
  await width.press("Enter");
  await expect(page.getByRole("alert")).toContainText("Not applied");
  await expect(width).toHaveAttribute("aria-invalid", "true");
  await width.fill("$width");
  await width.press("Enter");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          JSON.parse(
            localStorage.getItem("weave.project")!,
          ).graph.nodes[2].data.body.nodes[1].data.body.nodes.find(
            (n: any) => n.id === "memory",
          ).data.params.width,
      ),
    )
    .toBe("$width");
  await page.getByTitle("Undo (Ctrl+Z)").click();
  await expect(page.locator(".breadcrumbs")).toContainText("Token predictor");
});

test("dialogs close on Escape, trap keyboard focus and return focus to the opener", async ({
  page,
}) => {
  const opener = page.locator(".tokenizer-chip");
  await opener.click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await expect(
    dialog.getByRole("button", { name: "Close dialog" }),
  ).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  expect(
    await page.evaluate(
      () => !!document.activeElement?.closest("[role=dialog]"),
    ),
  ).toBe(true);
  await page.keyboard.press("Tab");
  await expect(
    dialog.getByRole("button", { name: "Close dialog" }),
  ).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(opener).toBeFocused();
});

test("switching architectures protects unsaved edits", async ({ page }) => {
  await editProjectDescription(page, "Edited draft description");
  await page.getByRole("button", { name: "Projects", exact: true }).click();
  await page
    .getByRole("button", { name: "Browse all examples", exact: true })
    .click();
  await page
    .getByRole("button", {
      name: "Create copy of Needle 1",
      exact: true,
    })
    .click();
  await page
    .getByRole("button", { name: "Create project", exact: true })
    .click();
  await expect(
    page.getByRole("dialog", { name: "Unsaved changes" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Keep editing" }).click();
  await expect(page.locator(".project-title")).toContainText(
    "Composed language model",
  );
  await page
    .getByRole("button", { name: "Browse all examples", exact: true })
    .click();
  await page
    .getByRole("button", {
      name: "Create copy of Needle 1",
      exact: true,
    })
    .click();
  await page
    .getByRole("button", { name: "Create project", exact: true })
    .click();
  await page.getByRole("button", { name: "Discard & create" }).click();
  await expect(page.locator(".project-title")).toContainText("Needle 1");
});

test("mobile users can open models, add blocks and edit their properties", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Add blocks", exact: true }).click();
  await expect(page.locator(".sidebar")).toBeVisible();
  await page.getByRole("textbox", { name: "Search blocks" }).fill("Python");
  await page
    .locator(".library-item")
    .filter({ hasText: "Python block" })
    .click();
  await expect(page.locator(".sidebar")).not.toBeVisible();
  await expect(page.locator(".inspector")).toBeVisible();
  await expect(
    page.getByRole("textbox", { name: "Python source" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close properties" }).click();
  await expect(page.locator(".inspector")).not.toBeVisible();
  await page.getByRole("button", { name: "Projects", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "View reference Needle 3", exact: true }),
  ).toBeVisible();
  await page.screenshot({ path: "data/ux-audit/after-phone-library.png" });
  await page.getByRole("button", { name: "Close library" }).click();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test("library search clears when switching from blocks to models", async ({
  page,
}) => {
  await page
    .getByRole("textbox", { name: "Search blocks" })
    .fill("nothingmatches");
  await expect(page.getByRole("status")).toContainText("No blocks match");
  await page.getByRole("button", { name: "Projects", exact: true }).click();
  await expect(
    page.getByRole("textbox", { name: "Search projects and examples" }),
  ).toHaveValue("");
  await expect(
    page.getByRole("button", { name: "View reference Needle 3", exact: true }),
  ).toBeVisible();
});

test("predictor cards do not overlap before or after automatic layout", async ({
  page,
}) => {
  await page.getByRole("button", { name: "Open loop body" }).click();
  await page.getByRole("button", { name: "Open block" }).click();
  async function overlaps() {
    return page.locator(".react-flow__node").evaluateAll((nodes) => {
      const boxes = nodes.map((n) => ({
        id: n.getAttribute("data-id"),
        box: n.getBoundingClientRect(),
      }));
      return boxes.flatMap((a, i) =>
        boxes
          .slice(i + 1)
          .filter(
            (b) =>
              Math.min(a.box.right, b.box.right) -
                Math.max(a.box.left, b.box.left) >
                1 &&
              Math.min(a.box.bottom, b.box.bottom) -
                Math.max(a.box.top, b.box.top) >
                1,
          )
          .map((b) => [a.id, b.id]),
      );
    });
  }
  await expect.poll(overlaps).toEqual([]);
  await page.getByRole("button", { name: "Auto arrange", exact: true }).click();
  await expect.poll(overlaps).toEqual([]);
});

test("a deleted saved project can be restored from the undo notification", async ({
  page,
}) => {
  const name = `Undo regression ${Date.now()}`;
  await page.locator(".project-title").click();
  await page
    .getByRole("textbox", { name: "Project name", exact: true })
    .fill(name);
  await page.getByRole("button", { name: "Close dialog", exact: true }).click();
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("Project saved");
  const id = await page.evaluate(
    () => JSON.parse(localStorage.getItem("weave.project")!).id,
  );
  try {
    await page.locator(".workspace-button").click();
    await page
      .getByRole("button", { name: "Delete " + name, exact: true })
      .click();
    await expect(page.getByRole("status")).toContainText("Project deleted");
    await page.getByRole("button", { name: "Undo", exact: true }).click();
    await expect(page.getByRole("status")).toContainText("Project restored");
    expect((await page.request.get("/api/projects/" + id)).status()).toBe(200);
  } finally {
    await page.request.delete("/api/projects/" + id);
  }
});

test("main workbench and tokenizer dialog pass automated accessibility checks", async ({
  page,
}) => {
  const builder = () =>
    new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21aa"]);
  expect((await builder().analyze()).violations).toEqual([]);
  await page.locator(".tokenizer-chip").click();
  expect((await builder().analyze()).violations).toEqual([]);
});
