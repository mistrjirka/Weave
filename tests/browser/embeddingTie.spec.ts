import { test, expect } from "@playwright/test";
import { projectSchema } from "../../frontend/src/api";

test("output weights can be tied to a named embedding and detached", async ({
  page,
}) => {
  const templates = await (await page.request.get("/api/templates")).json();
  const project = projectSchema.parse(
    templates.find((item: { id: string }) => item.id === "transformer"),
  );
  project.id = "project-embedding-picker-test";
  const head = project.graph.nodes.find((node) => node.data.kind === "head")!;
  const embedding = project.graph.nodes.find(
    (node) => node.data.kind === "embedding",
  )!;
  delete head.data.params.tie;
  await page.addInitScript((value) => {
    if (!localStorage.getItem("weave.project"))
      localStorage.setItem("weave.project", JSON.stringify(value));
  }, project);
  await page.goto("/");
  await page.locator(`.react-flow__node[data-id="${head.id}"]`).click();
  const picker = page.getByRole("combobox", {
    name: "Output weights",
    exact: true,
  });
  await expect(picker).toHaveValue("");
  await picker.selectOption(embedding.id);
  await expect(picker).toHaveValue(embedding.id);
  await expect(picker.locator("option:checked")).toHaveText(
    `Share with ${embedding.data.label}`,
  );
  await page.reload();
  await page.locator(`.react-flow__node[data-id="${head.id}"]`).click();
  await expect(picker).toHaveValue(embedding.id);
  await picker.selectOption("");
  await expect
    .poll(async () =>
      page.evaluate(() => {
        const stored = JSON.parse(localStorage.getItem("weave.project")!);
        return (
          stored.graph.nodes.find(
            (node: { data: { kind: string } }) => node.data.kind === "head",
          ).data.params.tie ?? null
        );
      }),
    )
    .toBeNull();
});
