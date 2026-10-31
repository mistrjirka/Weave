import { test, expect } from "@playwright/test";
import { projectSchema } from "../../frontend/src/api";

test("an incompatible edge shows an actionable error and undo clears it", async ({
  page,
}) => {
  const project = projectSchema.parse({
    version: 1,
    id: "project-connection-error-test",
    name: "Connection test",
    description: "",
    config: { width: 16, vocab: 32, heads: 2, ffn_width: 32 },
    graph: {
      nodes: [
        {
          id: "ids",
          type: "architecture",
          position: { x: 0, y: 0 },
          data: {
            kind: "input",
            label: "Token IDs",
            params: { dtype: "int64", shape: ["B", "T"], index: 0 },
          },
        },
        {
          id: "linear",
          type: "architecture",
          position: { x: 380, y: 0 },
          data: { kind: "linear", label: "Projection", params: { out: 16 } },
        },
        {
          id: "out",
          type: "architecture",
          position: { x: 760, y: 0 },
          data: { kind: "output", label: "Output", params: {} },
        },
      ],
      edges: [
        {
          id: "original",
          source: "ids",
          target: "out",
          sourceHandle: "out",
          targetHandle: "a",
        },
      ],
    },
  });
  await page.addInitScript(
    (value) => localStorage.setItem("weave.project", JSON.stringify(value)),
    project,
  );
  await page.goto("/");
  const from = page.locator(
    '.flow-wrap .react-flow__node[data-id="ids"] .react-flow__handle.source',
  );
  const to = page.locator(
    '.flow-wrap .react-flow__node[data-id="linear"] .react-flow__handle.target',
  );
  await expect(from).toBeVisible();
  await expect(to).toBeVisible();
  // Wait for the initial fit animation before reading handle positions.
  await page.waitForTimeout(600);
  const start = await from.boundingBox();
  const end = await to.boundingBox();
  if (!start || !end) throw new Error("Connection handles must be visible");
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
  await page.mouse.down();
  await page.mouse.move(end.x + end.width / 2, end.y + end.height / 2, {
    steps: 20,
  });
  await page.mouse.up();
  const alert = page
    .getByRole("alert")
    .filter({ hasText: "Connection error:" });
  await expect(alert).toBeVisible();
  await expect(alert).toContainText("Token IDs → Projection (input a)");
  await expect(alert).toContainText(/float|feature/i);
  await page.getByTitle("Undo (Ctrl+Z)").click();
  await expect(alert).toHaveCount(0);
});
