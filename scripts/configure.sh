#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - "$@" <<'PY'
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Choose where Weave keeps editable projects")
parser.add_argument("--projects-dir", default="~/shared/weave-projects")
args = parser.parse_args()
location = Path(args.projects_dir).expanduser().absolute()
location.mkdir(parents=True, exist_ok=True)
settings = Path('.weave-config.json')
value = json.loads(settings.read_text()) if settings.exists() else {}
value['projects_directory'] = str(location)
settings.write_text(json.dumps(value, indent=2) + '\n')
print('Weave projects: ' + str(location))
print('Saved in .weave-config.json. Restart Weave to apply; WEAVE_PROJECTS_DIR overrides this setting.')
PY
