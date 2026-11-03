# Weave documentation for agents

Weave exposes searchable, installed documentation through its MCP server at
`http://127.0.0.1:8173/mcp/`. Replace the host if the agent runs on another machine.
Connection examples are in https://github.com/mistrjirka/Weave/blob/main/docs/mcp.md.

Use `search_weave_docs(query=...)` to find guidance and
`read_weave_doc(guide="TRAINING_GUIDE.md")` for training-script authoring or
`read_weave_doc(guide="CUSTOM_TRAINING.md")` for custom loops and RL. Long guides
support `start_line` and `max_lines` for pagination. Discover API operations with
`list_weave_operations` and inspect their schemas before calling them.

Offline stage exports include `TRAINING_GUIDE.md` and `custom-training.md` beside
this file. Training scripts and local helpers belong outside `weave_support/`;
that folder contains generated files refreshed by Weave.
