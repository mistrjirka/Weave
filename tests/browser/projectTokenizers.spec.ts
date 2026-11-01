import { test, expect } from "@playwright/test";
import { openInputTokenizer } from "./helpers";
import { projectSchema } from "../../frontend/src/api";

test("reuse another project's tokenizer without inspection and retain special tokens", async ({
  page,
}) => {
  const info = {
    id: "a".repeat(20),
    repo: "example/shared-tokenizer",
    class: "WordLevel",
    size: 128,
    capacity: 128,
    base_size: 127,
    mask_id: 127,
    special_tokens: { mask_token: { token: "[MASK]", id: 127 } },
  };
  let inspections = 0;
  await page.route("**/api/tokenizers/inspect", async (route) => {
    inspections += 1;
    await route.abort();
  });
  await page.route("**/api/tokenizers/projects", async (route) => {
    await route.fulfill({
      json: [
        {
          project_id: "other-project",
          project_name: "My other model",
          input_path: "nested/tokens",
          tokenizer: info,
        },
      ],
    });
  });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Shapes compatible" }),
  ).toBeVisible();
  await openInputTokenizer(page);
  await page
    .getByRole("button", { name: "Other project", exact: true })
    .click();
  const picker = page.getByRole("combobox", {
    name: "Tokenizer from another project",
  });
  await expect(picker.locator("option")).toContainText([
    "Choose a saved tokenizer…",
    "My other model · example/shared-tokenizer · 128 tokens",
  ]);
  await picker.selectOption({
    label: "My other model · example/shared-tokenizer · 128 tokens",
  });
  await expect(page.locator(".tokenizer-result")).toContainText(
    "example/shared-tokenizer",
  );
  await expect(page.locator(".tokenizer-result")).toContainText("[MASK]");
  await page.getByRole("button", { name: "Close dialog" }).click();
  await expect(page.locator(".tokenizer-chip")).toContainText("128 vocab");
  await expect
    .poll(async () => {
      const project = projectSchema.parse(
        await page.evaluate(() =>
          JSON.parse(localStorage.getItem("weave.project") || "null"),
        ),
      );
      return project.graph.nodes.find((node) => node.id === "ids")?.data
        .tokenizer;
    })
    .toEqual(info);
  expect(inspections).toBe(0);
});
