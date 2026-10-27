import type { Page } from "@playwright/test";

/** Check card content in diagram coordinates, independent of canvas zoom. */
export function cardStyleViolations(page: Page) {
  return page.locator(".arch-node").evaluateAll((cards) => {
    const issues: string[] = [];
    for (const card of cards) {
      const name = card.getAttribute("aria-label") || "Card";
      const box = card.getBoundingClientRect();
      const scale = box.width / (card as HTMLElement).offsetWidth;
      const body = card.querySelector<HTMLElement>(".node-body");
      const preview = body?.firstElementChild;
      const footer = card.querySelector(".node-footer");
      if (!body || !preview || !footer) {
        issues.push(`${name}: missing card content section`);
        continue;
      }
      const bounds = preview.getBoundingClientRect();
      if (
        (bounds.left - box.left) / scale < 12 ||
        (box.right - bounds.right) / scale < 12
      )
        issues.push(`${name}: diagram touches the card edge`);
      const bodyBox = body.getBoundingClientRect();
      if (
        (bounds.top - bodyBox.top) / scale < 10 ||
        (footer.getBoundingClientRect().top - bounds.bottom) / scale < 10
      )
        issues.push(`${name}: diagram lacks vertical spacing`);
      for (const section of card.querySelectorAll<HTMLElement>(
        ".node-top, .node-title, .open-block",
      )) {
        const style = getComputedStyle(section);
        if (
          parseFloat(style.paddingLeft) < 12 ||
          parseFloat(style.paddingRight) < 12
        )
          issues.push(`${name}: ${section.className} lacks an inset`);
      }
      const badge = card.querySelector(".weight-badge");
      if (badge) {
        const badgeBox = badge.getBoundingClientRect();
        if (
          (badgeBox.left - box.left) / scale < 12 ||
          (box.right - badgeBox.right) / scale < 12
        )
          issues.push(`${name}: weight badge touches the card edge`);
      }
      for (const element of body.querySelectorAll<HTMLElement | SVGSVGElement>(
        "div, span, small, b, svg",
      )) {
        const rect = element.getBoundingClientRect();
        if (!rect.width || !rect.height) continue;
        if (
          (rect.left - box.left) / scale < 8 ||
          (box.right - rect.right) / scale < 8
        )
          issues.push(
            `${name}: ${element.getAttribute("class") || element.tagName} overflows the card gutter`,
          );
        if (
          element instanceof HTMLElement &&
          element.clientWidth &&
          element.scrollWidth > element.clientWidth + 1
        )
          issues.push(
            `${name}: ${element.className || element.tagName} clips its content`,
          );
      }
    }
    return [...new Set(issues)];
  });
}

export async function collectCardSamples(
  page: Page,
  samples: Map<string, string>,
) {
  const cards = await page.locator(".arch-node").evaluateAll((nodes) =>
    nodes.map((card) => ({
      key: [
        card.querySelector(".node-kind")?.textContent,
        card.getAttribute("data-visual-role"),
        card.querySelector(".node-body > :first-child")?.className,
        card.querySelector(".pattern-preview b")?.textContent,
      ].join(" / "),
      html: card.outerHTML,
    })),
  );
  for (const { key, html } of cards)
    if (!samples.has(key)) samples.set(key, html);
}

/** Review actual rendered cards together, including variants in nested graphs. */
export async function saveCardGallery(
  page: Page,
  samples: Map<string, string>,
  name: string,
) {
  const cards = [...samples.values()];
  for (let offset = 0; offset < cards.length; offset += 12) {
    await page.evaluate(
      (html) => {
        document.getElementById("card-review-gallery")?.remove();
        const gallery = document.createElement("div");
        gallery.id = "card-review-gallery";
        Object.assign(gallery.style, {
          position: "absolute",
          top: "0",
          left: "0",
          width: "1380px",
          zIndex: "9999",
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          alignItems: "start",
          justifyItems: "center",
          gap: "24px",
          padding: "24px",
          background: "var(--theme-base, #f7f9f5)",
        });
        gallery.innerHTML = html.join("");
        document.body.append(gallery);
      },
      cards.slice(offset, offset + 12),
    );
    await page
      .locator("#card-review-gallery")
      .screenshot({ path: `data/card-gallery-${name}-${offset / 12 + 1}.png` });
  }
}
