import { expect, test } from "@playwright/test";
import { projectsSchema } from "../../frontend/src/api";
import { catalog } from "../../frontend/src/catalog";
import type { GraphNode } from "../../frontend/src/types";

const phoneWidths = [360, 390, 430];

async function inspectSelectedNode(
  page: import("@playwright/test").Page,
  id: string,
) {
  const openInspector = page.locator(".inspector.is-open");
  if (await openInspector.count())
    await openInspector
      .getByRole("button", { name: "Close properties" })
      .click();
  await page.locator(`.flow-wrap .react-flow__node[data-id="${id}"]`).click();
  const inspector = page.getByRole("complementary", {
    name: "Property inspector",
  });
  await expect(inspector).toHaveClass(/is-open/);
  return inspector;
}

async function checkInspector(
  inspector: import("@playwright/test").Locator,
  failures: string[],
  label: string,
) {
  const measurements = await inspector.evaluate((root) => {
    const px = (el: Element) =>
      Number.parseFloat(getComputedStyle(el).fontSize);
    const inputs = [
      ...root.querySelectorAll<HTMLElement>(
        'input:not([type="checkbox"]):not([type="radio"]), select, textarea',
      ),
    ];
    const help = [
      ...root.querySelectorAll<HTMLElement>(
        ".field-help, .property-field small, .setting-explanation",
      ),
    ];
    const labels = [
      ...root.querySelectorAll<HTMLElement>(
        ".property-field > span, .setting-heading, .setting-boolean label",
      ),
    ];
    return {
      width: root.clientWidth,
      scrollWidth: root.scrollWidth,
      tooSmallHelp: help
        .filter((el) => px(el) < 13)
        .map((el) => `${el.textContent?.trim()}:${px(el)}`),
      tooSmallLabels: labels
        .filter((el) => px(el) < 14)
        .map((el) => `${el.textContent?.trim()}:${px(el)}`),
      tooSmallInputs: inputs
        .filter((el) => px(el) < 16)
        .map(
          (el) =>
            `${el.getAttribute("aria-label") || el.getAttribute("name") || el.tagName}:${px(el)}`,
        ),
    };
  });
  if (measurements.scrollWidth > measurements.width)
    failures.push(
      `${label}: inspector horizontal overflow ${measurements.scrollWidth}/${measurements.width}`,
    );
  failures.push(
    ...measurements.tooSmallHelp.map((item) => `${label}: help <13px ${item}`),
  );
  failures.push(
    ...measurements.tooSmallLabels.map(
      (item) => `${label}: label <14px ${item}`,
    ),
  );
  failures.push(
    ...measurements.tooSmallInputs.map(
      (item) => `${label}: control <16px ${item}`,
    ),
  );
}

test("phone graph settings stay readable and usable at 360, 390, and 430px", async ({
  page,
  request,
}) => {
  const failures: string[] = [];
  for (const width of phoneWidths) {
    await page.setViewportSize({ width, height: 844 });
    // These catalog examples cover tokenizer/input, embedding, loop/block and
    // MoE settings without saving or mutating any project.
    for (const [model, id] of [
      ["transformer", "ids"],
      ["embedding-table", "embedding"],
      ["transformer", "stack"],
      ["moe-transformer", "moe"],
    ]) {
      await page.goto(`/?model=${model}`);
      await expect(
        page.locator(`.react-flow__node[data-id="${id}"]`),
      ).toBeVisible();
      const inspector = await inspectSelectedNode(page, id);
      await checkInspector(inspector, failures, `${width}px ${model}/${id}`);
    }

    const sparse = projectsSchema
      .parse(await (await request.get("/api/templates")).json())
      .find((project) => project.id === "sparse");
    const localPattern = sparse?.graph.nodes
      .find((node) => node.id === "stack")
      ?.data.body?.nodes.find((node) => node.id === "local");
    if (!sparse || !localPattern)
      throw new Error("Sparse attention fixture is unavailable");
    localPattern.data.params.pattern = "custom";
    localPattern.data.params.matrix = Array.from({ length: 18 }, (_, row) =>
      Array.from({ length: 18 }, (_, column) => column <= row),
    );
    await page.goto("/");
    await page.evaluate(
      (project) =>
        localStorage.setItem("weave.project", JSON.stringify(project)),
      sparse,
    );
    await page.goto("/?model=sparse");
    const loop = await inspectSelectedNode(page, "stack");
    await checkInspector(loop, failures, `${width}px sparse/loop`);
    await loop
      .getByRole("button", { name: "Open internal graph", exact: true })
      .click();
    const attention = await inspectSelectedNode(page, "local");
    await checkInspector(attention, failures, `${width}px sparse/custom mask`);
    const viewport = page.getByRole("region", { name: "Attention mask cells" });
    const cells = viewport.locator("button");
    await expect(cells.first()).toBeVisible();
    const cellSize = await cells.first().evaluate((el) => {
      const rect = el.getBoundingClientRect();
      return { width: rect.width, height: rect.height };
    });
    if (Math.min(cellSize.width, cellSize.height) < 32)
      failures.push(
        `${width}px mask cell ${cellSize.width}×${cellSize.height}px (expected 32×32)`,
      );
    if (!(await viewport.evaluate((el) => el.scrollWidth > el.clientWidth)))
      failures.push(`${width}px mask is not horizontally scrollable`);

    const compactInputs = page.locator(".graph-variables.compact input");
    if (await compactInputs.count()) {
      if (
        (await compactInputs
          .first()
          .evaluate((el) => Number.parseFloat(getComputedStyle(el).fontSize))) <
        16
      )
        failures.push(`${width}px compact variable input under 16px`);
    }
  }
  expect(failures, failures.join("\n")).toEqual([]);
});

test("every catalog element exposes readable settings on a phone", async ({
  page,
  request,
}) => {
  test.setTimeout(120000);
  await page.setViewportSize({ width: 390, height: 844 });
  const templates = projectsSchema.parse(
    await (await request.get("/api/templates")).json(),
  );
  const blank = templates.find((project) => project.id === "blank");
  if (!blank) throw new Error("Blank fixture is unavailable");
  const allNodes: GraphNode[] = [];
  function collect(nodes: GraphNode[]) {
    for (const node of nodes) {
      allNodes.push(node);
      if (node.data.body) collect(node.data.body.nodes);
    }
  }
  for (const project of templates) collect(project.graph.nodes);
  const failures: string[] = [];
  await page.goto("/?model=blank");
  for (const entry of catalog) {
    const example = allNodes.find(
      (node) => node.data.kind === entry.kind && node.data.role === entry.role,
    );
    const project = structuredClone(blank);
    project.id = "mobile-controls-audit";
    project.name = "Mobile controls audit";
    project.graph = {
      nodes: [
        {
          id: "control",
          type: "architecture",
          position: { x: 0, y: 0 },
          data: {
            ...structuredClone(example?.data),
            kind: entry.kind,
            role: entry.role,
            label: entry.label,
            params: structuredClone(entry.params),
          },
        },
      ],
      edges: [],
    };
    await page.evaluate(
      (value) => localStorage.setItem("weave.project", JSON.stringify(value)),
      project,
    );
    await page.goto(
      "/?model=mobile-controls-audit&node=control&panel=properties",
    );
    const inspector = page.getByRole("complementary", {
      name: "Property inspector",
    });
    await expect(
      inspector.getByRole("textbox", { name: "Block name", exact: true }),
    ).toHaveValue(entry.label);
    await checkInspector(inspector, failures, `${entry.kind}/${entry.label}`);
  }
  expect(failures, failures.join("\n")).toEqual([]);
});
