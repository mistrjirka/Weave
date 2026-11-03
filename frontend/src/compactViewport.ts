// Keep canvas interaction in step with the drawer layout in style.css.
// Short, touch-operated landscape screens still need the phone navigation.
export const SHORT_TOUCH_VIEWPORT_QUERY =
  "(max-width: 1000px) and (max-height: 500px) and (pointer: coarse)";

export function isCompactViewport() {
  return window.matchMedia(`(max-width: 760px), ${SHORT_TOUCH_VIEWPORT_QUERY}`)
    .matches;
}
