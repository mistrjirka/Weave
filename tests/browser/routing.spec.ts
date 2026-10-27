import { openExample } from "./helpers";
import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";
import type { Graph, Project, GraphNode } from "../../frontend/src/types";
import { projectSchema, projectsSchema } from "../../frontend/src/api";

async function currentGraph(page: Page, path: string[]): Promise<Graph> {
  const project = projectSchema.parse(
    await page.evaluate(() =>
      JSON.parse(localStorage.getItem("weave.project") || "null"),
    ),
  );
  let g = project.graph;
  for (const id of path) {
    const body = g.nodes.find((n) => n.id === id)?.data.body;
    if (!body) throw new Error("Missing scope " + id);
    g = body;
  }
  return g;
}
async function violations(page: Page, graph: Graph) {
  return page.evaluate((g) => {
    const nodes = [
      ...document.querySelectorAll<HTMLElement>(".react-flow__node"),
    ].map((n) => ({ id: n.dataset.id, box: n.getBoundingClientRect() }));
    const overlaps = nodes.flatMap((a, i) =>
      nodes
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
        .map((b) => `${a.id}/${b.id}`),
    );
    const edges = [...g.edges];
    const groups = new Map<string, string>();
    for (const n of g.nodes) {
      const share = n.data.share;
      if (share) {
        const first = groups.get(share);
        if (first)
          edges.push({
            id: `weights:${share}:${n.id}`,
            source: first,
            target: n.id,
          });
        else groups.set(share, n.id);
      }
    }
    const hits: string[] = [];
    for (const edge of document.querySelectorAll<SVGGElement>(
      ".react-flow__edge",
    )) {
      const data = edges.find((e) => e.id === edge.dataset.id),
        path = edge.querySelector<SVGPathElement>(".react-flow__edge-path");
      if (!data || !path) continue;
      const matrix = path.getScreenCTM(),
        length = path.getTotalLength();
      if (!matrix) continue;
      const visited = new Set<string>();
      const obstacles = nodes.filter(
        (n) => n.id !== data.source && n.id !== data.target,
      );
      for (let i = 0; i <= length; i += 3) {
        const point = path.getPointAtLength(i).matrixTransform(matrix);
        for (const n of obstacles) {
          if (
            !visited.has(n.id || "") &&
            point.x > n.box.left + 0.7 &&
            point.x < n.box.right - 0.7 &&
            point.y > n.box.top + 0.7 &&
            point.y < n.box.bottom - 0.7
          ) {
            hits.push(`${data.id} through ${n.id}`);
            visited.add(n.id || "");
          }
        }
      }
    }
    const labels = [
      ...document.querySelectorAll<HTMLElement>('[data-label-visible="true"]'),
    ].map((label) => ({
      id: label.dataset.edgeLabel,
      box: label.getBoundingClientRect(),
    }));
    function intersects(a: DOMRect, b: DOMRect) {
      return (
        Math.min(a.right, b.right) - Math.max(a.left, b.left) > 0.7 &&
        Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 0.7
      );
    }
    const labelHits = labels.flatMap((label) =>
      nodes
        .filter((node) => intersects(label.box, node.box))
        .map((node) => `${label.id} over ${node.id}`),
    );
    const labelOverlaps = labels.flatMap((label, i) =>
      labels
        .slice(i + 1)
        .filter((other) => intersects(label.box, other.box))
        .map((other) => `${label.id}/${other.id}`),
    );
    return {
      labelHits,
      labelOverlaps,
      overlaps,
      hits,
      obstructed: document.querySelectorAll(".route-obstructed").length,
    };
  }, graph);
}
async function openNode(page: Page, node: GraphNode, mobile: boolean) {
  if (mobile) {
    for (let i = 0; i < 80; i++) {
      if (
        (await page
          .locator(".node-nav-current")
          .getAttribute("data-node-id")) === node.id
      )
        break;
      await page
        .getByRole("button", { name: "Next node", exact: true })
        .click();
    }
    await expect(page.locator(".node-nav-current")).toHaveAttribute(
      "data-node-id",
      node.id,
    );
  }
  await page
    .locator(`.react-flow__node[data-id="${node.id}"] .open-block`)
    .click();
}
function scopes(project: Project) {
  const result: { path: string[]; label: string }[] = [];
  function visit(g: Graph, path: string[]) {
    result.push({ path, label: project.name + "/" + path.join("/") });
    for (const n of g.nodes)
      if (n.data.body) visit(n.data.body, [...path, n.id]);
  }
  visit(project.graph, []);
  return result;
}

for (const { mobile, dark } of [
  { mobile: false, dark: false },
  { mobile: true, dark: false },
  { mobile: false, dark: true },
  { mobile: true, dark: true },
])
  test(`every example and nested graph has unobstructed arrows on ${mobile ? "phone" : "desktop"}${dark ? " in dark theme" : ""}`, async ({
    page,
    request,
  }) => {
    test.setTimeout(300000);
    if (dark)
      await page.addInitScript(() =>
        localStorage.setItem("weave.theme", "dark"),
      );
    await page.setViewportSize(
      mobile ? { width: 390, height: 844 } : { width: 1600, height: 1000 },
    );
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    const templates = projectsSchema.parse(
      await (await request.get("/api/templates")).json(),
    );
    await page.goto("/");
    await expect(
      page.getByRole("button", { name: "Shapes compatible" }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Show edge labels", exact: true })
      .click();
    let count = 0;
    for (const template of templates) {
      await openExample(page, template.name);
      // Traverse by clicking each scope, checking all authored variants including every Needle layer.
      async function visit(path: string[]) {
        await expect
          .poll(async () => (await currentGraph(page, path)).layout?.engine, {
            message: "ELK layout: " + template.name + "/" + path.join("/"),
            timeout: 15000,
          })
          .toBe("elk");
        await expect(
          page.getByRole("button", { name: "Auto arrange", exact: true }),
        ).toBeEnabled();
        const graph = await currentGraph(page, path);
        if (dark) {
          await expect(page.locator("html")).toHaveAttribute(
            "data-theme",
            "dark",
          );
          const brightCards = await page.locator(".arch-node").evaluateAll(
            (cards) =>
              cards.filter((card) => {
                const color =
                  getComputedStyle(card)
                    .backgroundColor.match(/[\d.]+/g)
                    ?.map(Number) || [];
                return (
                  color.length >= 3 && Math.min(...color.slice(0, 3)) > 100
                );
              }).length,
          );
          expect(
            brightCards,
            template.name + "/" + path.join("/") + " dark cards",
          ).toBe(0);
        }
        if (graph.edges.length)
          await expect
            .poll(() => page.locator('[data-label-visible="true"]').count())
            .toBeGreaterThan(0);
        await expect
          .poll(() => violations(page, graph), {
            message: template.name + "/" + path.join("/"),
            timeout: 5000,
          })
          .toEqual({
            overlaps: [],
            hits: [],
            obstructed: 0,
            labelHits: [],
            labelOverlaps: [],
          });
        await expect
          .poll(async () => ({
            visible: await page.locator('[data-label-visible="true"]').count(),
            total: await page.locator("[data-edge-label]").count(),
          }))
          .toMatchObject({
            visible: await page.locator("[data-edge-label]").count(),
          });
        if (!path.length)
          await page.screenshot({
            path: `data/routing-audit/elk-${template.id}-${mobile ? "phone" : "desktop"}${dark ? "-dark" : ""}.png`,
          });
        count++;
        for (const node of graph.nodes.filter((n) => n.data.body)) {
          await openNode(page, node, mobile);
          await visit([...path, node.id]);
          await page.locator(".breadcrumbs button").nth(path.length).click();
        }
      }
      await visit([]);
    }
    expect(count).toBe(templates.reduce((sum, p) => sum + scopes(p).length, 0));
    expect(errors).toEqual([]);
  });

test("phone navigation stays readable and its controls do not cover the focused block", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Next node", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Next node", exact: true }).click();
  await page.getByRole("button", { name: "Next node", exact: true }).click();
  await expect(page.locator(".node-nav-current")).toHaveAttribute(
    "data-node-id",
    "denoise",
  );
  await expect
    .poll(async () =>
      Number(
        (await page.getByLabel("Zoom to 100 percent").innerText()).replace(
          "%",
          "",
        ),
      ),
    )
    .toBeGreaterThanOrEqual(80);
  const node = page.locator('.react-flow__node[data-id="denoise"]');
  const box = await node.boundingBox(),
    nav = await page.locator(".mobile-node-nav").boundingBox();
  expect(box && nav && box.y + box.height < nav.y).toBe(true);
  for (const selector of [".react-flow__controls", ".react-flow__minimap"]) {
    const control = await page.locator(selector).boundingBox();
    expect(control && nav && control.y + control.height < nav.y).toBe(true);
  }
  await node.getByRole("button", { name: "Open loop body" }).click();
  await expect(page.locator(".breadcrumbs")).toContainText(
    "Iterative denoising",
  );
  await page.getByRole("button", { name: "Fit graph", exact: true }).click();
  await expect(page.locator(".react-flow__node")).toHaveCount(4);
  await page.getByRole("button", { name: "Focus current node" }).click();
  await expect
    .poll(async () =>
      Number(
        (await page.getByLabel("Zoom to 100 percent").innerText()).replace(
          "%",
          "",
        ),
      ),
    )
    .toBeGreaterThanOrEqual(80);
  await page.screenshot({ path: "data/routing-audit/phone-navigation.png" });
});

test("dragging cards recomputes their connected and bypass paths", async ({
  page,
}) => {
  await page.goto("/");
  await openExample(page, "Decoder transformer");
  await page.getByRole("button", { name: "Open loop body" }).click();
  await page
    .getByRole("button", { name: "Show edge labels", exact: true })
    .click();
  const node = page.locator('.react-flow__node[data-id="ffn"]');
  const box = await node.boundingBox();
  expect(box).not.toBeNull();
  if (!box) return;
  await page.mouse.move(box.x + box.width / 2, box.y + 20);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 - 90, box.y + 160, { steps: 12 });
  await page.mouse.up();
  await expect
    .poll(async () => violations(page, await currentGraph(page, ["stack"])))
    .toEqual({
      overlaps: [],
      hits: [],
      obstructed: 0,
      labelHits: [],
      labelOverlaps: [],
    });
  await page.screenshot({ path: "data/routing-audit/drag-rerouting.png" });
});

test("expanded attention routes Q/K/V branches, output gates, and optional masks", async ({
  page,
}) => {
  await page.goto("/");
  await openExample(page, "Thimble");
  await page.getByRole("button", { name: "Open loop body" }).click();
  await page.locator('.react-flow__node[data-id="attn"]').click();
  await page.getByRole("button", { name: "Edit attention internals" }).click();
  await expect
    .poll(async () =>
      violations(page, await currentGraph(page, ["stack", "attn"])),
    )
    .toEqual({
      overlaps: [],
      hits: [],
      obstructed: 0,
      labelHits: [],
      labelOverlaps: [],
    });
  await page.getByRole("button", { name: "Auto arrange", exact: true }).click();
  await expect
    .poll(async () =>
      violations(page, await currentGraph(page, ["stack", "attn"])),
    )
    .toEqual({
      overlaps: [],
      hits: [],
      obstructed: 0,
      labelHits: [],
      labelOverlaps: [],
    });
  await page.screenshot({ path: "data/routing-audit/expanded-attention.png" });
});

test("restored graph coordinates stay unchanged until Auto arrange is requested", async ({
  page,
  request,
}) => {
  const templates = projectsSchema.parse(
    await (await request.get("/api/templates")).json(),
  );
  const project = templates.find((p) => p.id === "transformer");
  expect(project).toBeDefined();
  if (!project) return;
  for (const n of project.graph.nodes)
    n.position = { x: n.position.x + 37, y: n.position.y + 19 };
  await page.addInitScript(
    (p) => localStorage.setItem("weave.project", JSON.stringify(p)),
    project,
  );
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  let graph = await currentGraph(page, []);
  expect(graph.nodes.map((n) => n.position)).toEqual(
    project.graph.nodes.map((n) => n.position),
  );
  await page.getByRole("button", { name: "Auto arrange", exact: true }).click();
  await expect
    .poll(async () => {
      graph = await currentGraph(page, []);
      return graph.nodes.map((n) => n.position);
    })
    .not.toEqual(project.graph.nodes.map((n) => n.position));
  expect(graph.edges.map((e) => [e.source, e.target, e.targetHandle])).toEqual(
    project.graph.edges.map((e) => [e.source, e.target, e.targetHandle]),
  );
  await page.getByTitle("Undo (Ctrl+Z)").click();
  await expect
    .poll(async () =>
      (await currentGraph(page, [])).nodes.map((n) => n.position),
    )
    .toEqual(project.graph.nodes.map((n) => n.position));
});

test("long tensor labels stay clear at multiple zoom levels after Auto arrange", async ({
  page,
  request,
}) => {
  const templates = projectsSchema.parse(
    await (await request.get("/api/templates")).json(),
  );
  const project = templates.find((p) => p.id === "blank");
  if (!project) throw new Error("Missing blank template");
  project.graph.nodes[0].data.params.shape = [
    "Batch_dimension_with_a_long_name",
    "Sequence_dimension_with_a_long_name",
    128,
  ];
  project.graph.nodes[0].data.params.dtype = "float32";
  project.graph.edges = [
    {
      id: "long-label",
      source: "ids",
      sourceHandle: "out",
      target: "out",
      targetHandle: "a",
    },
  ];
  await page.addInitScript(
    (p) => localStorage.setItem("weave.project", JSON.stringify(p)),
    project,
  );
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Show edge labels", exact: true })
    .click();
  await page.getByRole("button", { name: "Auto arrange", exact: true }).click();
  await expect
    .poll(async () => (await currentGraph(page, [])).layout?.engine)
    .toBe("elk");
  await expect(page.locator('[data-edge-label="long-label"]')).toHaveAttribute(
    "data-label-visible",
    "true",
  );
  for (const zoom of ["Zoom to 100 percent", "Zoom In", "Zoom Out"]) {
    if (zoom === "Zoom to 100 percent") await page.getByLabel(zoom).click();
    else await page.getByRole("button", { name: zoom, exact: true }).click();
    await expect
      .poll(async () => violations(page, await currentGraph(page, [])))
      .toEqual({
        overlaps: [],
        hits: [],
        obstructed: 0,
        labelHits: [],
        labelOverlaps: [],
      });
  }
  await page.screenshot({ path: "data/routing-audit/long-labels.png" });
});

test("starting edits cancels pending initial arrangement", async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await expect
    .poll(async () => (await currentGraph(page, [])).layout?.engine)
    .toBe("elk");
  let release: () => void = () => {};
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/code", async (route) => {
    await gate;
    await route.continue();
  });
  await openExample(page, "Blank project");
  await page.getByRole("button", { name: "Blocks", exact: true }).click();
  await page
    .locator(".library-item")
    .filter({ hasText: "Token IDs → learned vectors" })
    .click();
  const before = (await currentGraph(page, [])).nodes.map((n) => n.position);
  const validated = page.waitForResponse("**/api/code");
  release();
  await validated;
  await expect(
    page.getByRole("button", { name: "2 graph issues" }),
  ).toBeVisible();
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  await expect(
    page.getByRole("button", { name: "Auto arrange", exact: true }),
  ).toBeEnabled();
  expect((await currentGraph(page, [])).nodes.map((n) => n.position)).toEqual(
    before,
  );
  expect((await currentGraph(page, [])).layout).toBeUndefined();
});
