import { test, expect } from "@playwright/test";
import { openInputTokenizer } from "./helpers";

const bundle = (size: number) => ({
  name: "tokenizer.json",
  mimeType: "application/json",
  buffer: Buffer.from(
    JSON.stringify({
      version: "1.0",
      truncation: null,
      padding: null,
      added_tokens: [],
      normalizer: null,
      pre_tokenizer: { type: "Whitespace" },
      post_processor: null,
      decoder: null,
      model: {
        type: "WordLevel",
        vocab: Object.fromEntries(
          ["[UNK]", "hello", "world", "extra"]
            .slice(0, size)
            .map((word, id) => [word, id]),
        ),
        unk_token: "[UNK]",
      },
    }),
  ),
});

for (const width of [1600, 390])
  test(`replace uploaded tokenizer and clear stale preview at ${width}px`, async ({
    page,
  }) => {
    test.setTimeout(60000);
    await page.setViewportSize({ width, height: 1000 });
    await page.goto("/");
    await expect(
      page.getByRole("button", { name: "Shapes compatible" }),
    ).toBeVisible();
    await openInputTokenizer(page);
    const picker = page.getByLabel("Tokenizer files", { exact: true });
    await picker.setInputFiles(bundle(3));
    await expect(page.locator(".tokenizer-status")).toContainText(
      "Applied to",
      { timeout: 45000 },
    );
    await expect(page.locator(".result-heading")).toContainText(
      "tokenizer.json",
    );
    await page.screenshot({
      path: `test-results/tokenizer-summary-${width}.png`,
      fullPage: true,
    });
    await page
      .getByText("Details & tokenization preview", { exact: true })
      .click();
    await page
      .getByRole("textbox", { name: "Tokenization preview" })
      .fill("hello world");
    await page.getByRole("button", { name: "Tokenize sample" }).click();
    await expect(page.locator(".token-pills")).toContainText("hello");
    await picker.setInputFiles(bundle(4));
    await expect(page.locator(".tokenizer-status")).toContainText("3 → 4");
    await expect(page.locator(".token-pills")).toHaveCount(0);
    // The exact same filename and contents can be selected again.
    const response = page.waitForResponse("**/api/tokenizers/upload");
    await picker.setInputFiles(bundle(4));
    expect((await response).ok()).toBeTruthy();
    await expect(page.locator(".tokenizer-status")).toContainText("4 → 4");
    await picker.setInputFiles({
      name: "bad.json",
      mimeType: "application/json",
      buffer: Buffer.from("not json"),
    });
    await expect(page.getByRole("alert")).toContainText("has not changed");
    await expect(page.locator(".token-stats")).toContainText("4");
    const dialog = page.getByRole("dialog");
    expect(
      await dialog.evaluate((el) => el.scrollWidth <= el.clientWidth + 1),
    ).toBeTruthy();
    await page.screenshot({
      path: `test-results/tokenizer-${width}.png`,
      fullPage: true,
    });
    await page.getByRole("button", { name: "Close dialog" }).click();
    await expect(page.locator(".tokenizer-chip")).toContainText("4 vocab");
    await page.reload();
    await expect(
      page.locator('.flow-wrap .react-flow__node[data-id="ids"]'),
    ).toBeVisible();
    // Selection and the input inspector are restored on reload.
    await page.locator(".input-tokenizer-settings .tokenizer-chip").click();
    await expect(page.locator(".result-heading")).toContainText(
      "tokenizer.json",
    );
    await expect(page.locator(".token-stats")).toContainText("4");
  });

test("upload a tokenizer folder with its companion configuration", async ({
  page,
}) => {
  const { mkdtemp, writeFile, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const directory = await mkdtemp(join(tmpdir(), "weave-tokenizer-bundle-"));
  try {
    await writeFile(join(directory, "tokenizer.json"), bundle(4).buffer);
    await writeFile(
      join(directory, "tokenizer_config.json"),
      JSON.stringify({
        tokenizer_class: "PreTrainedTokenizerFast",
        unk_token: "[UNK]",
        eos_token: "extra",
      }),
    );
    await writeFile(
      join(directory, "README.md"),
      "Ignored companion documentation",
    );
    await page.goto("/");
    await expect(
      page.locator('.flow-wrap .react-flow__node[data-id="ids"]'),
    ).toBeVisible();
    await openInputTokenizer(page);
    await page
      .getByLabel("Tokenizer folder", { exact: true })
      .setInputFiles(directory);
    await expect(page.locator(".tokenizer-status")).toContainText(
      "Applied to",
      { timeout: 45000 },
    );
    await page
      .getByText("Details & tokenization preview", { exact: true })
      .click();
    await expect(page.locator(".special-tokens")).toContainText("eos");
    await expect(page.locator(".special-tokens")).toContainText("extra · 3");
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Save & close persists the tokenizer and keeps the dialog open on failure", async ({
  page,
}) => {
  test.setTimeout(60000);
  let savedId: string | undefined;
  await page.goto("/");
  await expect(
    page.locator('.flow-wrap .react-flow__node[data-id="ids"]'),
  ).toBeVisible();
  await openInputTokenizer(page);
  await page
    .getByLabel("Tokenizer files", { exact: true })
    .setInputFiles(bundle(4));
  await expect(page.locator(".tokenizer-status")).toContainText(
    "Not saved yet",
    { timeout: 45000 },
  );
  const pattern = "**/api/projects/project-*";
  await page.route(pattern, (route) =>
    route.request().method() === "PUT"
      ? route.fulfill({
          status: 503,
          json: { detail: "Save temporarily unavailable" },
        })
      : route.continue(),
  );
  await page.getByRole("button", { name: "Save & close", exact: true }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByRole("alert")).toContainText("could not be saved");
  await page.unroute(pattern);
  try {
    const saved = page.waitForResponse(
      (response) =>
        response.request().method() === "PUT" &&
        response.url().includes("/api/projects/"),
    );
    await page
      .getByRole("button", { name: "Save & close", exact: true })
      .click();
    const response = await saved;
    savedId = response.url().split("/").at(-1);
    expect(response.ok()).toBeTruthy();
    await expect(page.getByRole("dialog")).toHaveCount(0);
    await expect(page.getByRole("status")).toContainText(
      "Project saved on server",
    );
    const project = await (
      await page.request.get(`/api/projects/${savedId}`)
    ).json();
    expect(
      project.graph.nodes.find((node: { id: string }) => node.id === "ids").data
        .tokenizer.capacity,
    ).toBe(4);
  } finally {
    if (savedId) await page.request.delete(`/api/projects/${savedId}`);
  }
});

test("black tokenizer dialog keeps confirmation visible on a phone", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.addInitScript(() => localStorage.setItem("weave.theme", "black"));
  await page.goto("/");
  await expect(
    page.locator('.flow-wrap .react-flow__node[data-id="ids"]'),
  ).toBeVisible();
  await openInputTokenizer(page);
  await expect(
    page.getByRole("button", { name: "Save & close", exact: true }),
  ).toBeInViewport();
  await page
    .getByLabel("Tokenizer files", { exact: true })
    .setInputFiles(bundle(4));
  await expect(page.locator(".tokenizer-status")).toContainText(
    "Not saved yet",
    { timeout: 45000 },
  );
  await page
    .getByText("Details & tokenization preview", { exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Save & close", exact: true }),
  ).toBeInViewport();
  const { default: AxeBuilder } = await import("@axe-core/playwright");
  const results = await new AxeBuilder({ page })
    .include('[role="dialog"]')
    .analyze();
  expect(results.violations).toEqual([]);
  await page.screenshot({
    path: "test-results/tokenizer-black-phone.png",
    fullPage: true,
  });
});
