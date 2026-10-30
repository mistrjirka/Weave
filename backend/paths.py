"""User workspace location, independent of application databases and run history."""

import json
import os
from pathlib import Path

INSTALL_SETTINGS = Path(__file__).resolve().parents[1] / ".weave-config.json"


def projects_directory():
    override = os.environ.get("WEAVE_PROJECTS_DIR")
    if override:
        return Path(override).expanduser().absolute()
    settings = (
        json.loads(INSTALL_SETTINGS.read_text()) if INSTALL_SETTINGS.exists() else {}
    )
    location = settings.get("projects_directory") or "~/shared/weave-projects"
    return Path(location).expanduser().absolute()
