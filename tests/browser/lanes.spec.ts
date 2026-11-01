import { test, expect } from "@playwright/test";
import { projectSchema } from "../../frontend/src/api";

async function openExample(page: import("@playwright/test").Page) {
  const values = await (await page.request.get("/api/templates")).json();
  const project = projectSchema.parse(
    values.find((value: { id: string }) => value.id === "residual-lanes"),
  );
  await page.addInitScript(
    (value) => localStorage.setItem("weave.project", JSON.stringify(value)),
    project,
  );
  await page.goto("/?library=blocks");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  return project;
}

test("explicit lane wires render, arrange and offer compatible nodes from each output", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openExample(page);
  const fork = page.locator('.react-flow__node[data-id="fork"]');
  const split = page.locator('.react-flow__node[data-id="split"]');
  const join = page.locator('.react-flow__node[data-id="join"]');
  await expect(fork.locator(".react-flow__handle.source")).toHaveCount(4);
  await expect(split.locator(".react-flow__handle.source")).toHaveCount(4);
  await expect(join.locator(".react-flow__handle.target")).toHaveCount(4);
  await expect(fork.locator(".output-port-name")).toHaveText([
    "lane 1",
    "lane 2",
    "lane 3",
    "lane 4",
  ]);
  await expect(page.locator(".react-flow__edge")).toHaveCount(15);
  await page
    .getByTitle("Auto arrange: reduce crossings and reserve label space")
    .click();
  await expect(
    page.getByRole("button", { name: "Auto arrange", exact: true }),
  ).toBeEnabled();
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await page.screenshot({ path: "/tmp/weave-lanes-desktop.png" });
  // Each source port can open the type-filtered node picker on empty canvas.
  await fork.click();
  const from = fork.locator(
    '.react-flow__handle.source[data-handleid="lane2"]',
  );
  const rect = await from.boundingBox();
  if (!rect) throw new Error("Lane output must be rendered");
  const canvas = await page.locator(".flow-wrap").first().boundingBox();
  if (!canvas) throw new Error("Canvas must be rendered");
  await page.mouse.move(rect.x + rect.width / 2, rect.y + rect.height / 2);
  await page.mouse.down();
  await page.mouse.move(canvas.x + canvas.width - 60, canvas.y + 55, {
    steps: 20,
  });
  await page.mouse.up();
  await expect(
    page.getByRole("combobox", { name: "Search connected blocks" }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("lane count updates sockets and join mode has the correct choices", async ({
  page,
}) => {
  await openExample(page);
  const join = page.locator('.react-flow__node[data-id="join"]');
  await join.click();
  const mode = page.getByRole("combobox", { name: "mode", exact: true });
  await expect(mode.locator("option")).toHaveText(["stack", "mean", "sum"]);
  await page.getByLabel("lanes", { exact: true }).fill("5");
  await page.getByLabel("lanes", { exact: true }).press("Tab");
  await expect(join.locator(".react-flow__handle.target")).toHaveCount(5);
  await expect(page.locator(".inspector")).toContainText(
    "Input lane 5 is unconnected",
  );
  await page.screenshot({ path: "/tmp/weave-lanes-count.png" });
});

test("lane graph is usable on a phone without document overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openExample(page);
  await expect(
    page.locator(
      '.react-flow__node[data-id="split"] .react-flow__handle.source',
    ),
  ).toHaveCount(4);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: "Next node", exact: true }).click();
  await expect(
    page.getByRole("button", {
      name: /Focus current node: Fork four residual lanes/,
    }),
  ).toBeVisible();
  await page.screenshot({ path: "/tmp/weave-lanes-phone.png" });
});
