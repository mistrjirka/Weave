import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";
import { projectSchema } from "../../frontend/src/api";
const nodes = (page: Page) => page.locator(".flow-wrap .react-flow__node");
const selected = (page: Page) =>
  page.locator(".flow-wrap .react-flow__node.selected");
const document = async (page: Page) =>
  projectSchema.parse(
    await page.evaluate(() =>
      JSON.parse(localStorage.getItem("weave.project") || "null"),
    ),
  );

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await expect
    .poll(async () => (await document(page)).graph.layout?.engine)
    .toBe("elk");
});

test("Ctrl and Command clicks extend and toggle selection, without stale highlights", async ({
  page,
}) => {
  await nodes(page)
    .filter({
      has: page
        .locator(".node-title")
        .filter({ hasText: /^Append masked tokens$/ }),
    })
    .click();
  await page
    .locator('.flow-wrap .react-flow__node[data-id="denoise"]')
    .click({ modifiers: ["Control"] });
  await expect(selected(page)).toHaveCount(2);
  await expect(page.locator(".inspector-heading")).toContainText(
    "2 blocks selected",
  );
  await page
    .locator('.flow-wrap .react-flow__node[data-id="denoise"]')
    .click({ modifiers: ["Control"] });
  await expect(selected(page)).toHaveCount(1);
  await expect(page.locator(".flow-wrap .arch-node.selected")).toHaveCount(1);
  await page
    .locator('.flow-wrap .react-flow__node[data-id="denoise"]')
    .click({ modifiers: ["Meta"] });
  await expect(selected(page)).toHaveCount(2);
});

test("native Ctrl C/V copies connected nodes and nested blocks, with independent IDs and undo", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const before = await document(page);
  await page.locator('.flow-wrap .react-flow__node[data-id="mask"]').click();
  await page
    .locator('.flow-wrap .react-flow__node[data-id="denoise"]')
    .click({ modifiers: ["Control"] });
  await page.keyboard.press("Control+c");
  await expect(page.getByRole("status")).toContainText("copied");
  await page.keyboard.press("Control+v");
  await expect(nodes(page)).toHaveCount(before.graph.nodes.length + 2);
  await expect(selected(page)).toHaveCount(2);
  await expect
    .poll(async () => (await document(page)).graph.nodes.length)
    .toBe(before.graph.nodes.length + 2);
  const after = await document(page);
  const copies = after.graph.nodes.filter(
    (node) => !before.graph.nodes.some((original) => original.id === node.id),
  );
  expect(copies.map((node) => node.data.label)).toEqual([
    "Append masked tokens",
    "Iterative denoising",
  ]);
  expect(after.graph.edges).toHaveLength(before.graph.edges.length + 1);
  const edge = after.graph.edges.find(
    (edge) => edge.source === copies[0].id && edge.target === copies[1].id,
  );
  expect(edge).toBeDefined();
  expect(copies[1].data.body?.nodes.length).toBe(
    before.graph.nodes.find((node) => node.id === "denoise")!.data.body?.nodes
      .length,
  );
  await page.keyboard.press("Control+v");
  await expect(nodes(page)).toHaveCount(before.graph.nodes.length + 4);
  await expect
    .poll(async () => (await document(page)).graph.nodes.length)
    .toBe(before.graph.nodes.length + 4);
  const twice = await document(page);
  expect(twice.graph.nodes.at(-1)!.position).not.toEqual(copies[1].position);
  await page.keyboard.press("Control+z");
  await expect(nodes(page)).toHaveCount(before.graph.nodes.length + 2);
  await page.keyboard.press("Control+Shift+z");
  await expect(nodes(page)).toHaveCount(before.graph.nodes.length + 4);
  await page.keyboard.press("Delete");
  await expect(nodes(page)).toHaveCount(before.graph.nodes.length + 2);
  await page.keyboard.press("Control+z");
  await expect(nodes(page)).toHaveCount(before.graph.nodes.length + 4);
  await expect
    .poll(async () => (await document(page)).graph.edges.length)
    .toBe(before.graph.edges.length + 2);
  expect(errors).toEqual([]);
});

test("dragging a rectangle selects nodes and their copied internal edges", async ({
  page,
}) => {
  const before = await document(page);
  const pane = page.locator(".flow-wrap");
  const box = await pane.boundingBox();
  if (!box) throw new Error("Canvas is not visible");
  const boxes = await nodes(page).evaluateAll((nodes) =>
    nodes.map((node) => {
      const b = node.getBoundingClientRect();
      return { x: b.x, y: b.y, right: b.right, bottom: b.bottom };
    }),
  );
  const x = Math.max(box.x + 8, Math.min(...boxes.map((b) => b.x)) - 15);
  const y = Math.max(box.y + 8, Math.min(...boxes.map((b) => b.y)) - 15);
  const right = Math.min(
    box.x + box.width - 8,
    Math.max(...boxes.map((b) => b.right)) + 15,
  );
  const bottom = Math.min(
    box.y + box.height - 8,
    Math.max(...boxes.map((b) => b.bottom)) + 15,
  );
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(right, bottom, { steps: 25 });
  await page.mouse.up();
  await expect(selected(page)).toHaveCount(before.graph.nodes.length);
  const selectionBox = await page
    .locator(".flow-wrap .react-flow__nodesselection-rect")
    .boundingBox();
  if (!selectionBox) throw new Error("Selection box is not visible");
  await page.mouse.move(selectionBox.x + 5, selectionBox.y + 5);
  await page.mouse.down();
  await page.mouse.move(selectionBox.x + 35, selectionBox.y + 35, {
    steps: 12,
  });
  await page.mouse.up();
  await expect
    .poll(async () => (await document(page)).graph.nodes[0].position.x)
    .not.toBe(before.graph.nodes[0].position.x);
  await page.keyboard.press("Control+z");
  await expect
    .poll(async () =>
      (await document(page)).graph.nodes.map((node) => node.position),
    )
    .toEqual(before.graph.nodes.map((node) => node.position));
  await expect(page.getByTitle("Undo (Ctrl+Z)")).toBeDisabled();
  await page.keyboard.press("Control+c");
  await page.keyboard.press("Control+v");
  await expect(nodes(page)).toHaveCount(before.graph.nodes.length * 2);
  await expect
    .poll(async () => (await document(page)).graph.edges.length)
    .toBe(before.graph.edges.length * 2);
});

test("text editing and unrelated clipboard text keep their standard behavior", async ({
  page,
}) => {
  const count = await nodes(page).count();
  await page.locator('.flow-wrap .react-flow__node[data-id="mask"]').click();
  await page.keyboard.press("Control+c");
  const name = page.getByRole("textbox", { name: "Block name", exact: true });
  await name.fill("Ordinary text");
  await name.press("Control+a");
  await name.press("Control+c");
  await name.press("End");
  await name.press("Control+v");
  await expect(name).toHaveValue("Ordinary textOrdinary text");
  await page.locator(".flow-wrap").focus();
  await page.keyboard.press("Control+v");
  await expect(nodes(page)).toHaveCount(count);
});

test("pasting in the read-only reference leaves both documents unchanged", async ({
  page,
}) => {
  await page.locator('.flow-wrap .react-flow__node[data-id="mask"]').click();
  await page.keyboard.press("Control+c");
  await page.getByRole("button", { name: "Split view", exact: true }).click();
  const left = await nodes(page).count();
  const right = page.getByRole("region", {
    name: "Reference graph",
    exact: true,
  });
  const count = await right.locator(".react-flow__node").count();
  await right.locator(".react-flow__node").first().click();
  await page.keyboard.press("Control+v");
  await expect(nodes(page)).toHaveCount(left);
  await expect(right.locator(".react-flow__node")).toHaveCount(count);
});

test("node clipboard works without the secure Clipboard API and survives reload", async ({
  page,
}) => {
  await page.addInitScript(() =>
    Object.defineProperty(navigator, "clipboard", {
      value: undefined,
      configurable: true,
    }),
  );
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  const count = await nodes(page).count();
  await page.locator('.flow-wrap .react-flow__node[data-id="mask"]').click();
  await page.keyboard.press("Control+c");
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await page.locator(".flow-wrap").focus();
  await page.keyboard.press("Control+v");
  await expect(nodes(page)).toHaveCount(count + 1);
});

test("Space-drag pans, and Duplicate/Delete apply to the full selection", async ({
  page,
}) => {
  const count = await nodes(page).count();
  const viewport = page.locator(".flow-wrap .react-flow__viewport");
  const before = await viewport.getAttribute("style");
  const box = await page.locator(".flow-wrap").boundingBox();
  if (!box) throw new Error("Canvas is not visible");
  await page.keyboard.down("Space");
  await page.mouse.move(box.x + 50, box.y + 50);
  await page.mouse.down();
  await page.mouse.move(box.x + 130, box.y + 120, { steps: 12 });
  await page.mouse.up();
  await page.keyboard.up("Space");
  expect(await viewport.getAttribute("style")).not.toBe(before);
  await expect(selected(page)).toHaveCount(0);
  await page.getByTitle("Fit graph").click();
  await page.waitForTimeout(300);
  await page.locator('.flow-wrap .react-flow__node[data-id="mask"]').click();
  await page
    .locator('.flow-wrap .react-flow__node[data-id="denoise"]')
    .click({ modifiers: ["Control"] });
  await page
    .getByRole("button", { name: "Duplicate selected", exact: true })
    .click();
  await expect(nodes(page)).toHaveCount(count + 2);
  await expect(selected(page)).toHaveCount(2);
  await page
    .getByRole("button", { name: "Delete selected", exact: true })
    .click();
  await expect(nodes(page)).toHaveCount(count);
  await page.getByTitle("Undo (Ctrl+Z)").click();
  await expect(nodes(page)).toHaveCount(count + 2);
});

test("pasting inside a loop changes only that nested graph", async ({
  page,
}) => {
  const before = await document(page);
  const loop = before.graph.nodes.find((node) => node.id === "denoise")!;
  await page.locator('.flow-wrap .react-flow__node[data-id="mask"]').click();
  await page.keyboard.press("Control+c");
  await page
    .getByRole("button", { name: "Open loop body", exact: true })
    .click();
  await expect(nodes(page)).toHaveCount(loop.data.body!.nodes.length);
  await page.locator(".flow-wrap").focus();
  await page.keyboard.press("Control+v");
  await expect(nodes(page)).toHaveCount(loop.data.body!.nodes.length + 1);
  await expect
    .poll(
      async () =>
        (await document(page)).graph.nodes.find(
          (node) => node.id === "denoise",
        )!.data.body!.nodes.length,
    )
    .toBe(loop.data.body!.nodes.length + 1);
  expect((await document(page)).graph.nodes).toHaveLength(
    before.graph.nodes.length,
  );
  await page.keyboard.press("Control+z");
  await expect(nodes(page)).toHaveCount(loop.data.body!.nodes.length);
});

test("touch swipes still pan on a phone instead of starting a selection box", async ({
  browser,
}) => {
  const context = await browser.newContext({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
  });
  const page = await context.newPage();
  try {
    await page.goto("/");
    await expect(
      page.getByRole("button", { name: "Shapes compatible" }),
    ).toBeVisible();
    await page.waitForTimeout(500);
    const viewport = page.locator(".flow-wrap .react-flow__viewport");
    const before = await viewport.getAttribute("style");
    const box = await page.locator(".flow-wrap").boundingBox();
    if (!box) throw new Error("Canvas is not visible");
    const session = await context.newCDPSession(page);
    // Start below the variables card, clear of the phone zoom controls.
    const x = box.x + 20,
      y = box.y + box.height * 0.6;
    await session.send("Input.dispatchTouchEvent", {
      type: "touchStart",
      touchPoints: [{ x, y }],
    });
    for (let step = 1; step <= 10; step++)
      await session.send("Input.dispatchTouchEvent", {
        type: "touchMove",
        touchPoints: [{ x: x + step * 5, y: y + step * 7 }],
      });
    await session.send("Input.dispatchTouchEvent", {
      type: "touchEnd",
      touchPoints: [],
    });
    await expect(viewport).not.toHaveAttribute("style", before!);
    await expect(selected(page)).toHaveCount(0);
    await expect(page.locator(".react-flow__selection")).toHaveCount(0);
  } finally {
    await context.close();
  }
});
