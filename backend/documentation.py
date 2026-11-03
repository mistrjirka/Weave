"""Installed guide locations; stable IDs preserve existing MCP client requests."""

GUIDE_PATHS = {
    "README.md": "README.md",
    "INSTALL.md": "docs/install.md",
    "TRAINING.md": "docs/training.md",
    "TRAINING_GUIDE.md": "docs/training-guide.md",
    "MCP.md": "docs/mcp.md",
    "DESIGN_ADVICE.md": "docs/design-advice.md",
    "ARCHITECTURE_AUDIT.md": "docs/reference-models.md",
    "USER_GUIDE.md": "docs/user-guide.md",
    "DOCKER.md": "docs/docker.md",
    "CONTRIBUTING.md": "CONTRIBUTING.md",
    "CHECKPOINT_CHAT_DECODERS.md": "docs/checkpoint-chat-decoders.md",
    "docs/README.md": "docs/README.md",
    "docs/structure.md": "docs/structure.md",
}


def guide_path(identifier):
    """Accept a stable ID or exact installed path, never arbitrary disk paths."""
    return GUIDE_PATHS.get(identifier) or (
        identifier if identifier in GUIDE_PATHS.values() else None
    )
