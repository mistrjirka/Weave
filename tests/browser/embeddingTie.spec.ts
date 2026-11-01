import { test, expect, type Page } from "@playwright/test";
import { projectSchema } from "../../frontend/src/api";

async function transformer(page: Page) {
  const templates = await (await page.request.get("/api/templates")).json();
  const project = projectSchema.parse(
    templates.find((item: { id: string }) => item.id === "transformer"),
  );
  project.id = "project-embedding-group-test";
  const head = project.graph.nodes.find((node) => node.data.kind === "head")!;
  const embedding = project.graph.nodes.find(
    (node) => node.data.kind === "embedding",
  )!;
  return { project, head, embedding };
}

async function stored(page: Page) {
  return projectSchema.parse(
    await page.evaluate(() =>
      JSON.parse(localStorage.getItem("weave.project")!),
    ),
  );
}

test("a head joins an embedding weight group and detaches through one picker", async ({
  page,
}) => {
  const { project, head, embedding } = await transformer(page);
  delete head.data.params.tie;
  delete head.data.share;
  embedding.data.share = "token_matrix";
  await page.addInitScript((value) => {
    if (!localStorage.getItem("weave.project"))
      localStorage.setItem("weave.project", JSON.stringify(value));
  }, project);
  await page.goto("/");
  await page.locator(`.react-flow__node[data-id="${head.id}"]`).click();
  const picker = page.getByRole("combobox", { name: "Weight group" });
  await expect(picker).toHaveValue("");
  await expect(picker.locator('option[value="token_matrix"]')).toBeEnabled();
  await picker.selectOption("token_matrix");
  await expect(picker).toHaveValue("token_matrix");
  await expect(
    page.getByRole("combobox", { name: "Output weights" }),
  ).toHaveCount(0);
  await expect
    .poll(
      async () =>
        (await stored(page)).graph.nodes.find((n) => n.id === head.id)?.data
          .share,
    )
    .toBe("token_matrix");
  const joined = await stored(page);
  expect(joined.graph.nodes.find((n) => n.id === embedding.id)!.data.kind).toBe(
    "embedding",
  );
  expect(joined.graph.nodes.find((n) => n.id === head.id)!.data.kind).toBe(
    "head",
  );
  expect(
    joined.graph.nodes.find((n) => n.id === head.id)!.data.params.out,
  ).toBe(head.data.params.out);
  await page.reload();
  await page.locator(`.react-flow__node[data-id="${head.id}"]`).click();
  await expect(picker).toHaveValue("token_matrix");
  await picker.selectOption("");
  await expect
    .poll(async () => {
      const data = (await stored(page)).graph.nodes.find(
        (n) => n.id === head.id,
      )!.data;
      return [data.share ?? null, data.params.tie ?? null];
    })
    .toEqual([null, null]);
});

test("a saved single-head embedding tie migrates to a visible weight group", async ({
  page,
}) => {
  const { project, head, embedding } = await transformer(page);
  delete head.data.share;
  delete embedding.data.share;
  head.data.params.tie = embedding.id;
  await page.addInitScript((value) => {
    if (!localStorage.getItem("weave.project"))
      localStorage.setItem("weave.project", JSON.stringify(value));
  }, project);
  await page.goto("/");
  await page.locator(`.react-flow__node[data-id="${head.id}"]`).click();
  const picker = page.getByRole("combobox", { name: "Weight group" });
  const migrated = await stored(page);
  const group = migrated.graph.nodes.find((n) => n.id === embedding.id)!.data
    .share;
  expect(group).toBeTruthy();
  expect(migrated.graph.nodes.find((n) => n.id === head.id)!.data.share).toBe(
    group,
  );
  await expect(picker).toHaveValue(group!);
});
