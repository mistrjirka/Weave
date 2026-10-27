import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import {
  projectSchema,
  projectsSchema,
  codeSchema,
} from "../../frontend/src/api";
import { localizeLayerDefaults } from "../../frontend/src/parameterScopes";
import { openExample } from "./helpers";

const currentProject = async (page: Page) =>
  projectSchema.parse(
    await page.evaluate(() =>
      JSON.parse(localStorage.getItem("weave.project") || "null"),
    ),
  );

test("materializing per-block settings preserves every example's PyTorch code, shapes and parameters", async ({
  request,
}) => {
  const templates = projectsSchema.parse(
    await (await request.get("/api/templates")).json(),
  );
  for (const project of templates) {
    const original = codeSchema.parse(
      await (await request.post("/api/code", { data: project })).json(),
    );
    const before = original.report;
    const local = localizeLayerDefaults(project);
    const localized = codeSchema.parse(
      await (await request.post("/api/code", { data: local })).json(),
    );
    const after = localized.report;
    expect(localized.source, project.name).toBe(original.source);
    expect(after.valid, project.name).toBe(before.valid);
    expect(after.errors, project.name).toEqual(before.errors);
    expect(after.shapes, project.name).toEqual(before.shapes);
    expect(after.parameters, project.name).toEqual(before.parameters);
    expect(after.counts, project.name).toEqual(before.counts);
    expect(localizeLayerDefaults(local), project.name).toEqual(local);
  }
});

for (const width of [3840, 1920, 1280]) {
  test(`dark workspace fits at ${width}px, with a usable divider and compact reference library`, async ({
    page,
  }) => {
    test.setTimeout(60000);
    await page.setViewportSize({ width, height: width === 3840 ? 2160 : 1080 });
    await page.emulateMedia({ colorScheme: "dark", reducedMotion: "reduce" });
    await page.goto("/");
    await expect(
      page.getByRole("button", { name: "Shapes compatible", exact: true }),
    ).toBeVisible();
    const inspector = page.getByRole("complementary", {
      name: "Property inspector",
    });
    await expect(inspector).toContainText("Trainable parameters");
    await expect(inspector).toContainText("Graph validation");
    await expect(inspector.locator("input, select")).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: /Select tokenizer/ }),
    ).toHaveCount(1);
    await page.getByRole("button", { name: "Split view", exact: true }).click();
    const divider = page.getByRole("separator", { name: "Resize graph panes" });
    await expect(divider).toBeVisible();
    const canvas = page.locator(".canvas-comparison");
    expect(
      await divider.evaluate((e) => getComputedStyle(e).backgroundColor),
    ).not.toBe(
      await canvas
        .locator(".flow-wrap")
        .first()
        .evaluate((e) => getComputedStyle(e).backgroundColor),
    );
    const before = (await currentProject(page)).graph.nodes.map((n) => n.data);
    await divider.focus();
    await page.keyboard.press("ArrowRight");
    await expect(divider).toHaveAttribute("aria-valuenow", "55");
    await page.keyboard.press("Enter");
    await expect(divider).toHaveAttribute("aria-valuenow", "50");
    if (width === 3840) {
      // A wide comparison must open with both complete graphs, not a lone first node.
      for (const pane of ["Editable graph", "Reference graph"]) {
        await expect
          .poll(() =>
            page
              .getByRole("region", { name: pane, exact: true })
              .locator(".flow-wrap")
              .evaluate((host) => {
                const viewport = host.getBoundingClientRect();
                const nodes = [...host.querySelectorAll(".react-flow__node")];
                return (
                  nodes.length > 0 &&
                  nodes.every((node) => {
                    const box = node.getBoundingClientRect();
                    return (
                      box.left >= viewport.left &&
                      box.right <= viewport.right &&
                      box.top >= viewport.top &&
                      box.bottom <= viewport.bottom
                    );
                  })
                );
              }),
          )
          .toBe(true);
      }
    }
    const overflow = await page.evaluate(() => {
      const selectors = [
        ".sidebar",
        ".inspector",
        ".topbar",
        ".viewbar",
        ".canvas-toolbar",
        ".reference-heading",
      ];
      return selectors.flatMap((selector) =>
        [...document.querySelectorAll(selector)]
          .filter((e) => e.scrollWidth > e.clientWidth + 1)
          .map(() => selector),
      );
    });
    expect(overflow).toEqual([]);
    expect((await currentProject(page)).graph.nodes.map((n) => n.data)).toEqual(
      before,
    );
    await page.screenshot({ path: `data/workspace-refined-${width}.png` });
    await page
      .locator(".sidebar-tabs")
      .getByRole("button", { name: "Projects", exact: true })
      .click();
    await expect(page.locator(".sidebar .example-project-card")).toHaveCount(0);
    await expect(
      page.getByRole("button", {
        name: "View reference Needle 3",
        exact: true,
      }),
    ).toBeVisible();
    await page.screenshot({
      path: `data/workspace-library-refined-${width}.png`,
    });
    await page
      .getByRole("button", { name: "Browse all examples", exact: true })
      .click();
    await expect(
      page.getByRole("tab", { name: "Reference architectures", exact: true }),
    ).toHaveAttribute("aria-selected", "true");
    await page
      .getByRole("button", { name: "Close dialog", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Open project settings", exact: true })
      .click();
    const settings = page.getByRole("dialog", {
      name: "Project settings",
      exact: true,
    });
    await expect(
      settings.getByRole("textbox", { name: "heads", exact: true }),
    ).toHaveCount(0);
    await expect(
      settings.getByRole("textbox", { name: "ffn width", exact: true }),
    ).toHaveCount(0);
    await expect(settings.getByRole("textbox")).toHaveCount(2);
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(
      results.violations.map((v) => ({
        id: v.id,
        nodes: v.nodes.map((n) => n.failureSummary),
      })),
    ).toEqual([]);
  });
}

test("transformer settings are local by default and survive selection and reload", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible", exact: true }),
  ).toBeVisible();
  await openExample(page, "Looped transformer");
  await expect
    .poll(async () => (await currentProject(page)).graph.layout?.engine)
    .toBe("elk");
  await page.locator('.react-flow__node[data-id="transformer_2"]').click();
  await page
    .getByRole("button", {
      name: "Make this application independent",
      exact: true,
    })
    .click();
  const heads = page.getByRole("textbox", { name: "heads", exact: true });
  await expect(heads).toHaveValue("8");
  await heads.fill("2");
  await heads.press("Enter");
  await page.locator('.react-flow__node[data-id="transformer_1"]').click();
  await expect(heads).toHaveValue("8");
  await page.locator('.react-flow__node[data-id="transformer_2"]').click();
  await expect(heads).toHaveValue("2");
  page.once("dialog", (dialog) => dialog.accept());
  await page.reload();
  await expect(
    page.getByRole("textbox", { name: "heads", exact: true }),
  ).toHaveValue("2");
  await expect
    .poll(
      async () =>
        (await currentProject(page)).graph.nodes.find(
          (n) => n.id === "transformer_2",
        )?.data.params.heads,
    )
    .toBe(2);
  const project = await currentProject(page);
  expect(
    project.graph.nodes.find((n) => n.id === "transformer_1")?.data.params
      .heads,
  ).toBe(8);
  expect(
    project.graph.nodes.find((n) => n.id === "transformer_3")?.data.params
      .heads,
  ).toBe(8);
  await expect(
    page.getByRole("button", { name: "Shapes compatible", exact: true }),
  ).toBeVisible();
});

for (const { width, theme } of [
  { width: 1600, theme: "light" },
  { width: 1600, theme: "black" },
  { width: 390, theme: "black" },
]) {
  test(`reference selection owns the shared inspector and clears the editable selection at ${width}px in ${theme}`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 1000 });
    await page.addInitScript(
      (value) => localStorage.setItem("weave.theme", value),
      theme,
    );
    await page.goto("/?split=1");
    await expect(
      page.getByRole("button", { name: "Shapes compatible", exact: true }),
    ).toBeVisible();
    await expect
      .poll(async () => (await currentProject(page)).graph.layout?.engine)
      .toBe("elk");
    const left = page.getByRole("region", {
      name: "Editable graph",
      exact: true,
    });
    const right = page.getByRole("region", {
      name: "Reference graph",
      exact: true,
    });
    const inspector = page.getByRole("complementary", {
      name: "Property inspector",
      exact: true,
    });
    // Position the right viewport, then select the left block to reproduce the reported bug.
    await right.getByRole("button", { name: "Next node", exact: true }).click();
    await expect(right.locator(".node-nav-current")).toHaveAttribute(
      "data-node-id",
      "emb",
    );
    await left.locator('.react-flow__node[data-id="ids"]').click();
    await expect(
      inspector.getByRole("textbox", { name: "Block name", exact: true }),
    ).toHaveValue("Prompt tokens");
    if (width < 760)
      await page
        .getByRole("button", { name: "Close properties", exact: true })
        .click();
    const computation = (
      project: Awaited<ReturnType<typeof currentProject>>,
    ) => ({
      config: project.config,
      nodes: project.graph.nodes.map(({ id, position, data }) => ({
        id,
        position,
        data,
      })),
      edges: project.graph.edges.map(
        ({ id, source, target, targetHandle }) => ({
          id,
          source,
          target,
          targetHandle,
        }),
      ),
    });
    const before = computation(await currentProject(page));
    await right.locator('.react-flow__node[data-id="emb"]').click();
    const properties = inspector.getByRole("region", {
      name: "Reference properties",
      exact: true,
    });
    await expect(properties).toContainText("Token embedding");
    await expect(properties).toContainText("$width → 128");
    await expect(properties).toContainText("Reference · read-only");
    await expect(properties.locator("input, select, textarea")).toHaveCount(0);
    await expect(left.locator(".arch-node.selected")).toHaveCount(0);
    await expect(right.locator(".arch-node.selected")).toHaveCount(1);
    await expect
      .poll(() => new URL(page.url()).searchParams.get("node"))
      .toBeNull();
    await expect
      .poll(() => new URL(page.url()).searchParams.get("refNode"))
      .toBe("emb");
    await properties.focus();
    await page.keyboard.press("Delete");
    await page.keyboard.press("Control+z");
    expect(computation(await currentProject(page))).toEqual(before);
    const audit = await new AxeBuilder({ page })
      .include(".inspector")
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(
      audit.violations.map((v) => ({
        id: v.id,
        nodes: v.nodes.map((n) => n.failureSummary),
      })),
    ).toEqual([]);
    await page.screenshot({
      path: `data/reference-inspector-${theme}-${width}.png`,
    });
    await page.reload();
    await expect(
      inspector.getByRole("region", {
        name: "Reference properties",
        exact: true,
      }),
    ).toContainText("Token embedding");
    if (width < 760)
      await page
        .getByRole("button", { name: "Close properties", exact: true })
        .click();
    await left.locator('.react-flow__node[data-id="ids"]').click();
    await expect(
      inspector.getByRole("textbox", { name: "Block name", exact: true }),
    ).toHaveValue("Prompt tokens");
    await expect(right.locator(".arch-node.selected")).toHaveCount(0);
    await expect
      .poll(() => new URL(page.url()).searchParams.get("refNode"))
      .toBeNull();
  });
}
