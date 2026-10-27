import { expect } from "@playwright/test";
import type { Page } from "@playwright/test";

/** Produce a real unsaved metadata edit without changing a network's architecture. */
export async function editProjectDescription(page: Page, value: string) {
  await page.locator(".project-title").click();
  const settings = page.getByRole("dialog", {
    name: "Project settings",
    exact: true,
  });
  const field = settings.getByRole("textbox", {
    name: "Description",
    exact: true,
  });
  await field.fill(value);
  await field.press("Enter");
  await settings
    .getByRole("button", { name: "Close dialog", exact: true })
    .click();
}

/** Create an editable example fixture through the actual project workflow. */
export async function openExample(page: Page, name: string) {
  await page
    .locator(".topbar")
    .getByRole("button", { name: "New project", exact: true })
    .click();
  const form = page.getByRole("dialog", { name: "New project", exact: true });
  await form
    .getByRole("combobox", { name: "Start from", exact: true })
    .selectOption({ label: name === "Blank project" ? "Blank canvas" : name });
  await form
    .getByRole("textbox", { name: "Project name", exact: true })
    .fill(name);
  await form
    .getByRole("button", { name: "Create project", exact: true })
    .click();
  const guard = page.getByRole("dialog", {
    name: "Unsaved changes",
    exact: true,
  });
  if (await guard.isVisible())
    await guard
      .getByRole("button", { name: "Discard & create", exact: true })
      .click();
  await expect(page.locator(".project-title")).toHaveText(name);
  await expect(page.getByRole("dialog")).toHaveCount(0);
}

/** Tokenizers belong to the external token input, opened through its inspector. */
export async function selectTokenInput(page: Page, id = "ids") {
  const backdrop = page.getByRole("button", {
    name: "Close side panel",
    exact: true,
  });
  if (await backdrop.isVisible()) await backdrop.click();
  const node = page.locator(`.flow-wrap .react-flow__node[data-id="${id}"]`);
  if (await node.isVisible()) await node.click();
  else throw new Error("Token input must be visible");
  if ((page.viewportSize()?.width ?? 1600) <= 760) {
    await expect(
      page.getByRole("complementary", { name: "Property inspector" }),
    ).toHaveClass(/is-open/);
  }
}
export async function openInputTokenizer(page: Page, id = "ids") {
  await selectTokenInput(page, id);
  await page.locator(".input-tokenizer-settings .tokenizer-chip").click();
}
