export const motion = (duration: number) =>
  window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : duration;
export const format = (n: number) =>
  n >= 1e9
    ? (n / 1e9).toFixed(2) + "B"
    : n >= 1e6
      ? (n / 1e6).toFixed(2) + "M"
      : n >= 1e3
        ? (n / 1e3).toFixed(1) + "K"
        : String(n || 0);
