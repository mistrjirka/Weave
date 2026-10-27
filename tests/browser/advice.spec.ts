import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import type { Graph, GraphNode, Project } from "../../frontend/src/types";

function fixture(broken = false): Project {
  const block = (
    id: string,
    kind: string,
    params: GraphNode["data"]["params"] = {},
  ): GraphNode => ({
    id,
    type: "architecture",
    position: { x: 0, y: 0 },
    data: { label: id, kind, params },
  });
  const wire = (source: string, target: string, targetHandle = "a") => ({
    id: `${source}-${target}`,
    source,
    target,
    sourceHandle: "out",
    targetHandle,
  });
  const input = () =>
    block("x", "input", { index: 0, shape: ["B", "T", 128], dtype: "float32" });
  const body: Graph = {
    nodes: [
      input(),
      block("down", "linear", { out: 1 }),
      block("activation", "silu"),
      block("up", "linear", { out: broken ? 64 : 128 }),
      block("residual", "add"),
      block("out", "output"),
    ],
    edges: [
      wire("x", "down"),
      wire("down", "activation"),
      wire("activation", "up"),
      wire("up", "residual"),
      wire("x", "residual", "b"),
      wire("residual", "out"),
    ],
  };
  const composite = block("body", "block");
  body.nodes.forEach((node, index) => {
    node.position = { x: index * 320, y: 100 };
  });
  composite.data.body = body;
  return {
    version: 1,
    id: "design-helper-test",
    name: "Design helper test",
    description: "Test fixture",
    config: { width: 128, heads: 8, vocab: 128, ffn_width: 512 },
    graph: {
      nodes: [
        input(),
        { ...composite, position: { x: 400, y: 0 } },
        { ...block("out", "output"), position: { x: 800, y: 0 } },
      ],
      edges: [wire("x", "body"), wire("body", "out")],
    },
  };
}

for (const width of [1600, 390]) {
  test(`design helper explains, locates and exports a valid bottleneck at ${width}px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 1000 });
    await page.addInitScript(
      (project) =>
        localStorage.setItem("weave.project", JSON.stringify(project)),
      fixture(),
    );
    await page.goto("/");
    await expect(
      page.getByRole("button", { name: "Shapes compatible", exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "1 design note", exact: true })
      .click();
    const drawer = page.locator(".checks-drawer");
    await expect(drawer).toContainText("These never block export");
    await expect(drawer).toContainText("128 features to 1");
    await expect(drawer).toContainText("heuristic");
    await expect(
      drawer.getByRole("link", { name: "Read the reference" }),
    ).toHaveAttribute("href", /^https:\/\//);
    const accessibility = await new AxeBuilder({ page })
      .include(".checks-drawer")
      .analyze();
    expect(accessibility.violations).toEqual([]);
    await page.screenshot({ path: `data/design-advice-${width}.png` });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await drawer
      .getByRole("button", { name: "Locate block body/down", exact: true })
      .click();
    await expect(page.locator(".breadcrumbs")).toContainText("body");
    await page.getByRole("button", { name: "Close validation issues" }).click();
    if (width < 600)
      await page
        .getByRole("button", { name: "Properties", exact: true })
        .click();
    await expect(
      page.getByRole("textbox", { name: "Block name", exact: true }),
    ).toHaveValue("down");
    await expect(page.locator(".inspector .design-advice")).toContainText(
      "Very narrow hidden bottleneck",
    );
    if (width < 600)
      await page
        .getByRole("button", { name: "Close properties", exact: true })
        .click();
    const download = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export PyTorch", exact: true })
      .click();
    expect((await download).suggestedFilename()).toMatch(/\.zip$/);
  });
}

test("shape errors preserve the nested node, input port, expected dimensions and repair", async ({
  page,
}) => {
  await page.addInitScript(
    (project) => localStorage.setItem("weave.project", JSON.stringify(project)),
    fixture(true),
  );
  await page.goto("/");
  await expect(page.locator(".validation-bar")).toContainText("graph issues");
  await page.locator(".validation-bar button.invalid").click();
  const mismatch = page
    .locator(".checks-drawer .check-error")
    .filter({ hasText: "body/residual" });
  await expect(mismatch).toContainText("input b");
  await expect(mismatch).toContainText("64");
  await expect(mismatch).toContainText("128");
  await mismatch.click();
  await expect(page.locator(".breadcrumbs")).toContainText("body");
  await expect(
    page.getByRole("textbox", { name: "Block name", exact: true }),
  ).toHaveValue("residual");
  await expect(
    page.locator('.react-flow__node[data-id="residual"] .port-error'),
  ).toHaveAttribute("data-handleid", "b");
  await page.getByRole("button", { name: "Close validation issues" }).click();
  await page.getByTitle("Fit graph", { exact: true }).click();
  await page.locator('.react-flow__node[data-id="up"]').click();
  const out = page.getByRole("textbox", { name: "out", exact: true });
  await out.fill("128");
  await out.press("Enter");
  await expect(
    page.getByRole("button", { name: "Shapes compatible", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".arch-node.has-error")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "1 design note", exact: true }),
  ).toBeVisible();
});

for (const kind of ["scale", "attention_pattern"]) {
  test(`${kind} advice survives API parsing and identifies its block`, async ({
    page,
  }) => {
    const project = fixture();
    const operation = project.graph.nodes[1];
    if (!operation) throw new Error("Fixture operation is required");
    delete operation.data.body;
    operation.data.kind = kind;
    operation.data.params =
      kind === "scale"
        ? { factor: 0 }
        : {
            pattern: "custom",
            matrix: [
              [true, false],
              [false, false],
            ],
          };
    if (kind === "attention_pattern") {
      const input = project.graph.nodes[0];
      if (!input) throw new Error("Fixture input is required");
      input.data.params.shape = ["B", 2, 128];
    }
    await page.addInitScript(
      (document) =>
        localStorage.setItem("weave.project", JSON.stringify(document)),
      project,
    );
    await page.goto("/");
    await expect(
      page.getByRole("button", { name: "Shapes compatible", exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "1 design note", exact: true })
      .click();
    const note = page.locator(
      `.checks-drawer [data-rule="${kind === "scale" ? "zero-scale" : "empty-attention-rows"}"]`,
    );
    await expect(note).toContainText(
      kind === "scale" ? "zero gradients" : "zero-based indices 1",
    );
    await note
      .getByRole("button", { name: "Locate block body", exact: true })
      .click();
    await page.getByRole("button", { name: "Close validation issues" }).click();
    await expect(page.locator(".inspector .design-advice")).toHaveCount(1);
    if (kind === "scale") {
      const factor = page.getByRole("textbox", { name: "factor", exact: true });
      await factor.fill("0.25");
      await factor.press("Enter");
      await expect(page.locator(".inspector .design-advice")).toHaveCount(0);
      await expect(page.locator(".advice-count")).toHaveCount(0);
    }
  });
}
