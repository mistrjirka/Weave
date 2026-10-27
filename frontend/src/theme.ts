export type ThemePreference = "system" | "light" | "dark";
export type ResolvedTheme = Exclude<ThemePreference, "system">;
export const themeStorageKey = "weave.theme";
export const systemThemeQuery = "(prefers-color-scheme: dark)";

export function themePreference(value: unknown): ThemePreference {
  return value === "light" || value === "dark" ? value : "system";
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
  document.documentElement.style.colorScheme = theme;
  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute("content", theme === "dark" ? "#101b15" : "#f7f9f5");
}
