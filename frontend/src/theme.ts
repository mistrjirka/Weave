export type ThemePreference = "system" | "light" | "dark" | "black";
export type ResolvedTheme = Exclude<ThemePreference, "system">;
export const themeStorageKey = "weave.theme";
export const systemThemeQuery = "(prefers-color-scheme: dark)";

export function themePreference(value: unknown): ThemePreference {
  return value === "light" || value === "dark" || value === "black"
    ? value
    : "system";
}

export function resolveTheme(
  preference: ThemePreference,
  systemDark: boolean,
): ResolvedTheme {
  return preference === "system" ? (systemDark ? "dark" : "light") : preference;
}

export function applyTheme(preference: ThemePreference, systemDark: boolean) {
  const theme = resolveTheme(preference, systemDark);
  document.documentElement.dataset.theme = theme;
  document.documentElement.dataset.themePreference = preference;
  document.documentElement.style.colorScheme =
    theme === "light" ? "light" : "dark";
  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute(
      "content",
      { light: "#f7f9f5", dark: "#101a15", black: "#111113" }[theme],
    );
}
