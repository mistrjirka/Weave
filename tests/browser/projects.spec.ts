import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { projectSchema, projectsSchema } from "../../frontend/src/api";

async function currentProject(page: Page) {
  return projectSchema.parse(
    await page.evaluate(() =>
      JSON.parse(localStorage.getItem("weave.project") || "null"),
    ),
  );
}
async function openBrowser(page: Page) {
  const opener = page.locator(".workspace-button");
  if (!(await opener.isVisible()))
    await page.getByRole("button", { name: "Projects", exact: true }).click();
  await opener.click();
  return page.getByRole("dialog", { name: "Projects & examples", exact: true });
}
async function newForm(page: Page, name: string) {
  await page
    .locator(".topbar")
    .getByRole("button", { name: "New project", exact: true })
    .click();
  const dialog = page.getByRole("dialog", { name: "New project", exact: true });
  await dialog
    .getByRole("textbox", { name: "Project name", exact: true })
    .fill(name);
  return dialog;
}
test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible", exact: true }),
  ).toBeVisible();
});

test("create, recover, save and reopen a named blank project", async ({
  page,
}) => {
  const name = `Workflow blank ${Date.now()}`;
  let savedId = "";
  try {
    const form = await newForm(page, name);
    await expect(
      form.getByRole("combobox", { name: "Start from" }),
    ).toHaveValue("blank");
    await form
      .getByRole("button", { name: "Create project", exact: true })
      .click();
    await expect(page.locator(".project-title")).toHaveText(name);
    await expect(page.locator(".react-flow__node")).toHaveCount(2);
    const draft = await currentProject(page);
    expect(draft.id).toMatch(/^project-/);
    await expect(page.locator(".project-meta")).toContainText("Draft");
    page.once("dialog", (dialog) => dialog.accept());
    await page.reload();
    await expect(page.locator(".project-title")).toHaveText(name);
    await expect(page.locator(".project-meta")).toContainText("Draft");
    const draftBrowser = await openBrowser(page);
    await expect(draftBrowser.locator(".current-project-draft")).toContainText(
      name,
    );
    await draftBrowser
      .getByRole("button", { name: "Continue editing", exact: true })
      .click();
    expect((await currentProject(page)).id).toBe(draft.id);
    await page.getByRole("button", { name: "Save", exact: true }).click();
    await expect(page.getByRole("status")).toContainText("Project saved");
    savedId = (await currentProject(page)).id;
    expect(savedId).toBe(draft.id);
    await expect(page.locator(".project-meta")).toContainText(
      "Saved on server",
    );
    const browser = await openBrowser(page);
    await browser
      .getByRole("searchbox", { name: "Search saved projects" })
      .fill(name);
    expect(
      (await new AxeBuilder({ page }).include('[role="dialog"]').analyze())
        .violations,
    ).toEqual([]);
    await browser
      .locator(".open-saved-project")
      .filter({ hasText: name })
      .click();
    await expect(page.getByRole("dialog")).toHaveCount(0);
    expect((await currentProject(page)).id).toBe(savedId);
  } finally {
    if (savedId) await page.request.delete("/api/projects/" + savedId);
  }
});

test("viewing a reference preserves unsaved work and offers an explicit editable copy", async ({
  page,
}) => {
  const width = page.getByRole("textbox", { name: "width", exact: true });
  await width.fill("64");
  await width.press("Enter");
  const before = await currentProject(page);
  const browser = await openBrowser(page);
  await browser
    .getByRole("tab", { name: "Reference architectures", exact: true })
    .click();
  await browser
    .getByRole("button", { name: "View reference Needle 1", exact: true })
    .click();
  const reference = page.getByRole("region", {
    name: "Reference graph",
    exact: true,
  });
  await expect(
    reference.getByRole("combobox", { name: "Reference model" }),
  ).toHaveValue("example:needle1");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  expect((await currentProject(page)).id).toBe(before.id);
  expect((await currentProject(page)).config).toEqual(before.config);
  await expect(page.locator(".project-meta")).toContainText("Unsaved");
  await reference
    .getByRole("button", { name: "Use this example: Needle 1", exact: true })
    .click();
  const form = page.getByRole("dialog", { name: "New project", exact: true });
  await expect(form.getByRole("combobox", { name: "Start from" })).toHaveValue(
    "needle1",
  );
  await expect(form.getByRole("textbox", { name: "Project name" })).toHaveValue(
    "Needle 1 copy",
  );
  await form.getByRole("button", { name: "Cancel", exact: true }).click();
  expect((await currentProject(page)).id).toBe(before.id);
});

test("using an example creates an independent named draft, retaining its source references", async ({
  page,
}) => {
  const templates = projectsSchema.parse(
    await (await page.request.get("/api/templates")).json(),
  );
  const starter = templates.find((p) => p.id === "needle1");
  if (!starter) throw new Error("Needle example is required");
  const browser = await openBrowser(page);
  await browser
    .getByRole("tab", { name: "Reference architectures", exact: true })
    .click();
  await browser
    .getByRole("button", {
      name: "Use as starting point: Needle 1",
      exact: true,
    })
    .click();
  const form = page.getByRole("dialog", { name: "New project", exact: true });
  await form
    .getByRole("textbox", { name: "Project name" })
    .fill("My attention experiment");
  await form
    .getByRole("button", { name: "Create project", exact: true })
    .click();
  await expect(page.locator(".project-title")).toHaveText(
    "My attention experiment",
  );
  const draft = await currentProject(page);
  expect(draft.id).not.toBe(starter.id);
  expect(draft.references).toEqual(starter.references);
  expect(draft.config).toEqual(starter.config);
  expect(draft.graph.nodes.map((n) => n.id)).toEqual(
    starter.graph.nodes.map((n) => n.id),
  );
  await expect(page.locator(".project-meta")).toContainText("Draft");
});

test("a failed save never discards work, and Save & create keeps the old project", async ({
  page,
}) => {
  let savedId = "";
  try {
    const width = page.getByRole("textbox", { name: "width", exact: true });
    await width.fill("64");
    await width.press("Enter");
    const form = await newForm(page, "Second network");
    await form
      .getByRole("button", { name: "Create project", exact: true })
      .click();
    const guard = page.getByRole("dialog", {
      name: "Unsaved changes",
      exact: true,
    });
    await expect(guard).toContainText("Second network");
    await expect(
      guard.getByRole("button", { name: "Close dialog" }),
    ).toBeFocused();
    await page.route("**/api/projects/*", async (route) => {
      if (route.request().method() === "PUT")
        await route.fulfill({
          status: 500,
          json: { detail: "Save failed for workflow test" },
        });
      else await route.continue();
    });
    await guard
      .getByRole("button", { name: "Save & create", exact: true })
      .click();
    await expect(page.getByRole("status")).toContainText(
      "Save failed for workflow test",
    );
    await expect(guard).toBeVisible();
    expect((await currentProject(page)).config.width).toBe(64);
    await page.unroute("**/api/projects/*");
    const savedRequest = page.waitForRequest(
      (request) =>
        request.method() === "PUT" && request.url().includes("/api/projects/"),
    );
    await guard
      .getByRole("button", { name: "Save & create", exact: true })
      .click();
    savedId = projectSchema.parse((await savedRequest).postDataJSON()).id;
    await expect(page.locator(".project-title")).toHaveText("Second network");
    const old = projectSchema.parse(
      await (await page.request.get("/api/projects/" + savedId)).json(),
    );
    expect(old.config.width).toBe(64);
    expect((await currentProject(page)).id).not.toBe(savedId);
  } finally {
    if (savedId) await page.request.delete("/api/projects/" + savedId);
  }
});

test("browser tabs, new-project starter, keyboard navigation and form errors have clear state", async ({
  page,
}) => {
  await page.goto("/?workspace=examples&model=transformer");
  let browser = page.getByRole("dialog", {
    name: "Projects & examples",
    exact: true,
  });
  const examples = browser.getByRole("tab", {
    name: "Reference architectures",
    exact: true,
  });
  await expect(examples).toHaveAttribute("aria-selected", "true");
  await examples.focus();
  await page.keyboard.press("Home");
  await expect(
    browser.getByRole("tab", { name: /^Your projects/ }),
  ).toBeFocused();
  await page.keyboard.press("End");
  await expect(examples).toBeFocused();
  await browser
    .getByRole("searchbox", { name: "Search reference architectures" })
    .fill("Needle 3");
  await expect(browser.locator(".example-project-card")).toHaveCount(1);
  await browser
    .getByRole("button", {
      name: "Use as starting point: Needle 3",
      exact: true,
    })
    .click();
  await expect
    .poll(() => new URL(page.url()).searchParams.get("workspace"))
    .toBe("new");
  await page.reload();
  const form = page.getByRole("dialog", { name: "New project", exact: true });
  await expect(form.getByRole("combobox", { name: "Start from" })).toHaveValue(
    "needle3",
  );
  const name = form.getByRole("textbox", { name: "Project name", exact: true });
  await name.fill("   ");
  await form
    .getByRole("button", { name: "Create project", exact: true })
    .click();
  await expect(name).toBeFocused();
  await expect(form.getByRole("alert")).toContainText(
    "Give your project a name",
  );
  await name.fill("My named network");
  await form
    .getByRole("combobox", { name: "Start from" })
    .selectOption("looped");
  await expect(name).toHaveValue("My named network");
});

for (const width of [1600, 390, 320]) {
  test(`project entry points and examples are readable and accessible at ${width}px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.route("**/api/projects", (route) => route.fulfill({ json: [] }));
    await page.reload();
    const browser = await openBrowser(page);
    await expect(
      browser.getByRole("heading", { name: "Your next network starts here" }),
    ).toBeVisible();
    await browser
      .getByRole("tab", { name: "Reference architectures", exact: true })
      .click();
    await expect(
      browser.getByRole("button", {
        name: "View reference Needle 1",
        exact: true,
      }),
    ).toBeVisible();
    const audit = await new AxeBuilder({ page })
      .include('[role="dialog"]')
      .analyze();
    expect(audit.violations).toEqual([]);
    await page.screenshot({ path: `data/project-browser-${width}.png` });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await browser
      .getByRole("button", { name: "New project", exact: true })
      .click();
    const form = page.getByRole("dialog", { name: "New project", exact: true });
    await expect(
      form.getByRole("button", { name: "Create project", exact: true }),
    ).toBeVisible();
    expect(
      (await new AxeBuilder({ page }).include('[role="dialog"]').analyze())
        .violations,
    ).toEqual([]);
    await page.screenshot({ path: `data/new-project-${width}.png` });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  });
}
