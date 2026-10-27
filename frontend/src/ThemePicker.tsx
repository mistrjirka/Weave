import { useEffect, useState } from "react";
import { Monitor, Moon, Sun } from "lucide-react";
import {
  applyTheme,
  systemThemeQuery,
  themePreference,
  themeStorageKey,
} from "./theme";
import type { ThemePreference } from "./theme";

const icons = { system: Monitor, light: Sun, dark: Moon, black: Moon };

export function ThemePicker() {
  const [preference, setPreference] = useState<ThemePreference>(() =>
    themePreference(document.documentElement.dataset.themePreference),
  );
  useEffect(() => {
    const media = window.matchMedia(systemThemeQuery);
    const update = () => applyTheme(preference, media.matches);
    const storage = (event: StorageEvent) => {
      if (event.key === themeStorageKey || event.key === null)
        setPreference(themePreference(event.newValue));
    };
    update();
    media.addEventListener("change", update);
    window.addEventListener("storage", storage);
    return () => {
      media.removeEventListener("change", update);
      window.removeEventListener("storage", storage);
    };
  }, [preference]);

  function choose(value: string) {
    const next = themePreference(value);
    applyTheme(next, window.matchMedia(systemThemeQuery).matches);
    setPreference(next);
    try {
      localStorage.setItem(themeStorageKey, next);
    } catch {
      /* Keep the session choice when storage is unavailable. */
    }
  }

  const Icon = icons[preference];
  return (
    <label className="theme-picker" title={`Color theme: ${preference}`}>
      <Icon size={15} aria-hidden="true" />
      <select
        aria-label="Color theme"
        name="color-theme"
        value={preference}
        onChange={(e) => choose(e.target.value)}
      >
        <option value="system">System</option>
        <option value="light">Light</option>
        <option value="dark">Forest</option>
        <option value="black">Black</option>
      </select>
    </label>
  );
}
