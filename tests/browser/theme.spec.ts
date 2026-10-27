import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { projectSchema } from "../../frontend/src/api";

const root = (page: Page) => page.locator("html");
const picker = (page: Page) =>
  page.getByRole("combobox", { name: "Color theme", exact: true });
async function ready(page: Page) {
  await expect(
    page.getByRole("button", { name: "Shapes compatible", exact: true }),
  ).toBeVisible();
}
async function audit(page: Page, selector?: string) {
  let builder = new AxeBuilder({ page }).withTags([
    "wcag2a",
    "wcag2aa",
    "wcag21a",
    "wcag21aa",
  ]);
  if (selector) builder = builder.include(selector);
  const result = await builder.analyze();
  expect(
    result.violations.map((v) => ({
      id: v.id,
      nodes: v.nodes.map((n) => ({ html: n.html, summary: n.failureSummary })),
    })),
  ).toEqual([]);
}

test("theme choices persist without changing the network or its URL", async ({
  page,
}) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto("/");
  await ready(page);
  const document = () =>
    page.evaluate(() =>
      JSON.parse(localStorage.getItem("weave.project") || "null"),
    );
  const before = projectSchema.parse(await document());
  const url = page.url();
  await picker(page).selectOption("dark");
  await expect(root(page)).toHaveAttribute("data-theme", "dark");
  expect(await page.evaluate(() => localStorage.getItem("weave.theme"))).toBe(
    "dark",
  );
  expect(page.url()).toBe(url);
  const after = projectSchema.parse(await document());
  expect(after.id).toBe(before.id);
  expect(after.config).toEqual(before.config);
  expect(after.graph.nodes.map((n) => n.data)).toEqual(
    before.graph.nodes.map((n) => n.data),
  );
  await expect(page.locator(".project-meta")).not.toContainText("Unsaved");
  await page.reload();
  await ready(page);
  await expect(picker(page)).toHaveValue("dark");
  await expect(root(page)).toHaveCSS("color-scheme", "dark");
  await picker(page).selectOption("black");
  await page.emulateMedia({ colorScheme: "light" });
  await page.reload();
  await ready(page);
  await expect(picker(page)).toHaveValue("black");
  await expect(root(page)).toHaveCSS("color-scheme", "dark");
  await picker(page).selectOption("light");
  await page.reload();
  await ready(page);
  await expect(picker(page)).toHaveValue("light");
  await expect(root(page)).toHaveCSS("color-scheme", "light");
});

test("system follows OS changes while an explicit preference takes precedence", async ({
  page,
}) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto("/");
  await ready(page);
  await expect(picker(page)).toHaveValue("system");
  await expect(root(page)).toHaveAttribute("data-theme", "dark");
  await page.emulateMedia({ colorScheme: "light" });
  await expect(root(page)).toHaveAttribute("data-theme", "light");
  await picker(page).selectOption("light");
  await page.emulateMedia({ colorScheme: "dark" });
  await expect(root(page)).toHaveAttribute("data-theme", "light");
  await picker(page).selectOption("system");
  await expect(root(page)).toHaveAttribute("data-theme", "dark");
});

test("theme changes synchronize across open tabs", async ({
  page,
  context,
}) => {
  await page.goto("/");
  await ready(page);
  const other = await context.newPage();
  await other.goto("/");
  await ready(other);
  await picker(page).selectOption("dark");
  await expect(root(other)).toHaveAttribute("data-theme", "dark");
  await expect(picker(other)).toHaveValue("dark");
  await picker(other).selectOption("black");
  await expect(root(page)).toHaveAttribute("data-theme", "black");
  await expect(picker(page)).toHaveValue("black");
  await picker(other).selectOption("light");
  await expect(root(page)).toHaveAttribute("data-theme", "light");
  await expect(picker(page)).toHaveValue("light");
});

for (const theme of ["light", "dark", "black"]) {
  test(`saved ${theme} theme is applied before the app bundle runs`, async ({
    page,
  }) => {
    await page.emulateMedia({
      colorScheme: theme === "dark" ? "light" : "dark",
    });
    await page.addInitScript(
      (value) => localStorage.setItem("weave.theme", value),
      theme,
    );
    await page.route("**/assets/index-*.js", (route) => route.abort());
    await page.goto("/");
    await expect(root(page)).toHaveAttribute("data-theme", theme);
    await expect(root(page)).toHaveCSS(
      "background-color",
      {
        light: "rgb(247, 249, 245)",
        dark: "rgb(16, 26, 21)",
        black: "rgb(17, 17, 19)",
      }[theme],
    );
    await expect(page.locator(".app-shell")).toHaveCount(0);
  });
}

test("blocked local storage still allows a session theme choice", async ({
  page,
}) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.addInitScript(() => {
    Object.defineProperty(Storage.prototype, "getItem", {
      value: () => {
        throw new DOMException("Storage blocked", "SecurityError");
      },
    });
    Object.defineProperty(Storage.prototype, "setItem", {
      value: () => {
        throw new DOMException("Storage blocked", "SecurityError");
      },
    });
  });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await ready(page);
  await picker(page).selectOption("dark");
  await expect(root(page)).toHaveAttribute("data-theme", "dark");
  expect(errors).toEqual([]);
});

for (const { width, theme } of ["dark", "black"].flatMap((theme) =>
  [1600, 390, 320].map((width) => ({ width, theme })),
)) {
  test(`${theme} editor, nested diagrams, reference pane and dialogs are readable at ${width}px`, async ({
    page,
  }) => {
    test.setTimeout(60000);
    await page.setViewportSize({ width, height: 1000 });
    if (theme === "black")
      await page.addInitScript(() =>
        localStorage.setItem("weave.theme", "black"),
      );
    await page.emulateMedia({ colorScheme: "dark" });
    await page.goto("/");
    await ready(page);
    await audit(page);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({ path: `data/${theme}-editor-${width}.png` });
    await page
      .getByRole("button", { name: "1 design note", exact: true })
      .click();
    await audit(page, ".checks-drawer");
    await page.getByRole("button", { name: "Close validation issues" }).click();
    await page
      .getByRole("button", { name: "Open loop body", exact: true })
      .click();
    await page.getByRole("button", { name: "Open block", exact: true }).click();
    await audit(page);
    await page.screenshot({ path: `data/${theme}-diagrams-${width}.png` });
    await page.getByRole("button", { name: "Split view", exact: true }).click();
    await expect(
      page
        .getByRole("region", { name: "Reference graph", exact: true })
        .locator(".react-flow__node"),
    ).not.toHaveCount(0);
    await audit(page);
    await page.screenshot({ path: `data/${theme}-split-${width}.png` });
    await page.goto("/?workspace=examples");
    const browser = page.getByRole("dialog", {
      name: "Projects & examples",
      exact: true,
    });
    await expect(browser).toBeVisible();
    await audit(page, '[role="dialog"]');
    await page.screenshot({ path: `data/${theme}-projects-${width}.png` });
    await browser
      .getByRole("button", { name: "New project", exact: true })
      .click();
    const form = page.getByRole("dialog", { name: "New project", exact: true });
    await expect(form).toBeVisible();
    await audit(page, '[role="dialog"]');
    await form.getByRole("button", { name: "Cancel", exact: true }).click();
    await page
      .getByRole("button", { name: "Select tokenizer", exact: false })
      .click();
    await audit(page, '[role="dialog"]');
    await page
      .getByRole("button", { name: "Close dialog", exact: true })
      .click();
    await page
      .getByRole("button", { name: "PyTorch code", exact: true })
      .click();
    await expect(page.locator(".code-view")).toContainText("nn.Module");
    await audit(page, ".code-view");
    if (width < 600) {
      const code = page.getByRole("region", {
        name: "Generated PyTorch code",
        exact: true,
      });
      await code.focus();
      await page.keyboard.press("ArrowRight");
      await expect
        .poll(() => code.evaluate((element) => element.scrollLeft))
        .toBeGreaterThan(0);
    }
  });
}
