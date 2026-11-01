import { expect, test } from "@playwright/test";
import { projectSchema } from "../../frontend/src/api";
import {
  exposeTransformerSettings,
  wrapLoopTransformer,
} from "../../frontend/src/transformerControls";

test("wrapper switches edit the inner attention and survive a local reload", async ({
  page,
}) => {
  const response = await page.request.get("/api/templates");
  const templates = await response.json();
  const project = projectSchema.parse(
    templates.find((item: { id: string }) => item.id === "transformer"),
  );
  project.id = "project-attention-booleans-browser-test";
  const loop = project.graph.nodes.find((node) => node.data.kind === "loop")!;
  loop.data.role = "transformer";
  exposeTransformerSettings(loop.data);
  wrapLoopTransformer(loop.data);
  const child = loop.data.body!.nodes.find(
    (node) => node.data.role === "transformer",
  )!;
  const attention = child.data.body!.nodes.find(
    (node) => node.data.kind === "attention",
  )!;
  attention.data.params.qk_norm_eps = 0.000001;
  child.data.params.qk_norm = true;
  delete attention.data.params.qk_norm;
  await page.addInitScript((value) => {
    if (!localStorage.getItem("weave.project"))
      localStorage.setItem("weave.project", JSON.stringify(value));
  }, project);
  const codeRequests: unknown[] = [];
  await page.route("**/api/code", async (route) => {
    codeRequests.push(route.request().postDataJSON());
    await route.continue();
  });
  await page.goto("/");
  await page.locator(`.react-flow__node[data-id="${loop.id}"]`).click();
  const toggle = (name: string) =>
    page.getByRole("switch", { name, exact: true });
  await expect(toggle("qk norm")).toHaveAttribute("aria-checked", "false");
  await toggle("qk norm").click();
  await expect(toggle("qk norm")).toHaveAttribute("aria-checked", "true");
  await toggle("qk norm").click();
  for (const name of [
    "causal",
    "rope",
    "Attention output gate",
    "qkv bias",
    "output bias",
  ])
    await toggle(name).click();
  await expect(toggle("causal")).toHaveAttribute("aria-checked", "false");
  await expect(toggle("rope")).toHaveAttribute("aria-checked", "false");
  await expect(page.getByRole("textbox", { name: "theta" })).toBeDisabled();
  await expect(
    page.getByRole("textbox", { name: "qk norm eps" }),
  ).toBeDisabled();

  const stored = () =>
    page.evaluate(() =>
      JSON.parse(localStorage.getItem("weave.project") || "null"),
    );
  await expect
    .poll(async () => {
      const saved = projectSchema.parse(await stored());
      const wrapper = saved.graph.nodes.find((node) => node.id === loop.id)!;
      const inner = wrapper.data.body!.nodes.find(
        (node) => node.data.role === "transformer",
      )!;
      return inner.data.params.qk_norm;
    })
    .toBe(false);
  const saved = projectSchema.parse(await stored());
  const savedChild = saved.graph.nodes
    .find((node) => node.id === loop.id)!
    .data.body!.nodes.find((node) => node.data.role === "transformer")!;
  const savedAttention = savedChild.data.body!.nodes.find(
    (node) => node.data.kind === "attention",
  )!;
  for (const [key, value] of Object.entries({
    qk_norm: false,
    causal: false,
    rope: false,
    gated: true,
    qkv_bias: true,
    output_bias: true,
  })) {
    expect(savedChild.data.params[key]).toBe(value);
    expect(savedAttention.data.params[key]).toBe(`$${key}`);
  }
  await expect
    .poll(() =>
      codeRequests.some((request) => {
        const serialized = JSON.stringify(request);
        return (
          serialized.includes('"qk_norm":false') &&
          serialized.includes('"causal":false') &&
          serialized.includes('"rope":false')
        );
      }),
    )
    .toBe(true);
  await page.reload();
  await page.locator(`.react-flow__node[data-id="${loop.id}"]`).click();
  await expect(toggle("qk norm")).toHaveAttribute("aria-checked", "false");
  await expect(toggle("causal")).toHaveAttribute("aria-checked", "false");
  await expect(toggle("rope")).toHaveAttribute("aria-checked", "false");
  await page
    .locator(`.react-flow__node[data-id="${loop.id}"]`)
    .getByRole("button", { name: "Open loop body" })
    .click();
  await page.locator(`.react-flow__node[data-id="${child.id}"]`).click();
  await expect(toggle("qk norm")).toHaveAttribute("aria-checked", "false");
  await page.getByRole("button", { name: "Open internal graph" }).click();
  await page.locator(`.react-flow__node[data-id="${attention.id}"]`).click();
  await expect(page.locator(".inspector")).toContainText("$qk_norm");
  await expect(page.locator(".inspector")).toContainText("false");
  await page
    .locator(".property-field")
    .filter({ has: page.getByText("qk norm", { exact: true }) })
    .getByRole("button", { name: "Set independently" })
    .click();
  await expect(toggle("qk norm")).toHaveAttribute("aria-checked", "false");
  await toggle("qk norm").click();
  await expect(toggle("qk norm")).toHaveAttribute("aria-checked", "true");
  const independent = projectSchema
    .parse(await stored())
    .graph.nodes.find((node) => node.id === loop.id)!
    .data.body!.nodes.find((node) => node.id === child.id)!
    .data.body!.nodes.find((node) => node.id === attention.id)!;
  expect(independent.data.params.qk_norm).toBe(true);
});
