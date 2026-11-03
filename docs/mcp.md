# Weave MCP server

Weave exposes its JSON API and installed guides to MCP clients. Start Weave first with `bash scripts/start.sh`, then configure a local MCP client to launch:

```bash
/absolute/path/to/Weave/.venv/bin/python -m backend.mcp_bridge
```

Set the client's working directory to the Weave checkout, or set `PYTHONPATH` to that checkout. The bridge connects to `http://127.0.0.1:8173` by default. Set `WEAVE_MCP_URL` to another Weave HTTP(S) origin if needed. It must contain no path, query, or embedded credentials. For example, a client configuration can use:

```json
{
  "mcpServers": {
    "weave": {
      "command": "/absolute/path/to/Weave/.venv/bin/python",
      "args": ["-m", "backend.mcp_bridge"],
      "cwd": "/absolute/path/to/Weave"
    }
  }
}
```

When Weave is running, its Streamable HTTP MCP endpoint is `http://127.0.0.1:8173/mcp/` or `http://YOUR_SERVER_IP:8173/mcp/` on the LAN. The MCP SDK checks Host and Origin against the server's hostname and network addresses. Restart Weave if its IP address changes. For Docker or a reverse proxy, set `WEAVE_MCP_PUBLIC_ORIGINS` to the exact public HTTP(S) origin; see [Docker network settings](docker.md#address-settings-and-lifecycle).

Use `list_weave_operations` to find an operation, `describe_weave_operation` for its parameters and referenced JSON schemas, then `read_weave_operation` for GET or `execute_weave_operation` for POST, PUT, PATCH, and DELETE. The bridge exposes JSON operations from Weave's OpenAPI document, including project editing, graph validation, training configuration, and run controls. Binary downloads/uploads, event streams, session login, and reporting ingestion are omitted; use the HTTP API or UI for those. Results are limited to 1 MB.

`search_weave_docs` and `read_weave_doc` access only the installed guides listed in the tool's response. Reading supports line ranges for longer guides.

Training operations keep Weave's normal access control. An HTTP MCP caller may provide its `weave_training` cookie or `x-weave-training-token` header. For a local stdio client, set `WEAVE_TRAINING_TOKEN` in that client's environment if the Weave server requires an access code. The MCP bridge never reads the server's configured access code. Protect the MCP connection and the training credential as you would the Weave UI.

Guides now live under `docs/`. Both historical guide IDs (for example, `TRAINING_GUIDE.md`) and exact installed paths (`docs/training-guide.md`) are accepted. The documentation index is `docs/README.md`.
