"""Documentation is consumed by MCP, offline training exports and GitHub links."""

import asyncio
from pathlib import Path
import re

from backend.documentation import GUIDE_PATHS, guide_path
from backend.mcp_bridge import WeaveBridge, create_mcp_server
from backend.training.layout import write_helpers

ROOT = Path(__file__).resolve().parents[1]


def test_documentation_registry_has_real_files_and_rejects_traversal():
    for identifier, location in GUIDE_PATHS.items():
        assert (ROOT / location).is_file(), identifier
        assert guide_path(identifier) == location
        assert guide_path(location) == location
    assert guide_path("../pyproject.toml") is None
    assert guide_path("docs/../pyproject.toml") is None


def test_mcp_accepts_historical_and_installed_guide_paths():
    import json

    server = create_mcp_server(WeaveBridge())

    def read(identifier):
        result = asyncio.run(server.call_tool("read_weave_doc", {"guide": identifier}))
        return json.loads(result.content[0].text)

    old = read("TRAINING_GUIDE.md")
    current = read("docs/training-guide.md")
    assert old["text"] == current["text"]
    assert "Writing train.py" in current["text"]


def test_exported_authoring_guide_is_self_contained(tmp_path):
    write_helpers(tmp_path)
    guide = (tmp_path / "TRAINING_GUIDE.md").read_text()
    assert "(example_cached_jsonl.py)" in guide
    assert "(https://github.com/mistrjirka/Weave/blob/main/docs/training.md)" in guide
    assert "docs/training.md#host-wide-gpu-performance-tuning" in guide
    assert "(training.md" not in guide
    assert (tmp_path / "example_cached_jsonl.py").is_file()


def test_local_documentation_links_resolve():
    pages = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *ROOT.glob("docs/**/*.md")]
    for page in pages:
        for target in re.findall(r"\]\(([^)]+)\)", page.read_text()):
            if "://" in target or target.startswith(("#", "/", "mailto:")):
                continue
            filename = target.split("#", 1)[0]
            assert (page.parent / filename).exists(), (page.relative_to(ROOT), target)
