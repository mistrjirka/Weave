# Connect Weave through MCP

Weave's MCP server lets an agent edit saved graphs and training scripts, validate models, launch/control training, inspect progress, and test checkpoints. It also provides searchable documentation and small file transfers. Start Weave first; the stdio bridge connects to the running application rather than starting another server.

## Codex: connect over HTTP

On the machine running Codex:

```bash
codex mcp add weave --url http://127.0.0.1:8173/mcp/
codex mcp get weave
```

Use your server's reachable IP/hostname if Codex runs elsewhere. The `/mcp/` suffix is required. For Docker, use the published host port, not an internal container address.

Alternatively, add this to `~/.codex/config.toml` or a trusted project's `.codex/config.toml`:

```toml
[mcp_servers.weave]
url = "http://127.0.0.1:8173/mcp/"
startup_timeout_sec = 20
tool_timeout_sec = 120
```

[Copyable TOML file](examples/codex-http.toml). These settings were checked against the local Codex CLI and [official MCP configuration documentation](https://developers.openai.com/codex/mcp). Codex uses TOML, not the `mcpServers` JSON format.

If the Weave operator enabled a training access code, export it in the environment that launches Codex, then add this field under the same server table:

```toml
env_http_headers = { "x-weave-training-token" = "WEAVE_TRAINING_TOKEN" }
```

Weave expects its custom header, not a bearer token. No access code is needed by default. `codex mcp login` is for OAuth servers; Weave does not provide an OAuth login flow. The access code protects training endpoints, not the whole application.

## JSON configuration for stdio clients

Clients supporting the common `mcpServers` configuration can use:

```json
{
  "mcpServers": {
    "weave": {
      "command": "/absolute/path/to/Weave/.venv/bin/python",
      "args": ["-m", "backend.mcp_bridge"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/Weave",
        "WEAVE_MCP_URL": "http://127.0.0.1:8173"
      }
    }
  }
}
```

[Copyable JSON file](examples/mcp-client.json). Replace both checkout paths. `PYTHONPATH` makes the module discoverable even if the client has no working-directory setting. Use the web interpreter in `.venv`, not the separate training interpreter. Client-specific settings locations differ.

For Codex stdio, use the [stdio TOML example](examples/codex-stdio.toml), or:

```bash
codex mcp add weave --env PYTHONPATH=/absolute/path/to/Weave \
  --env WEAVE_MCP_URL=http://127.0.0.1:8173 \
  -- /absolute/path/to/Weave/.venv/bin/python -m backend.mcp_bridge
```

Stdio may explicitly inherit `WEAVE_TRAINING_TOKEN` from its client environment. It never reads a stored server credential. Do not commit secrets to shared client configuration files.

## Available operations

| Workflow | MCP access |
| --- | --- |
| Discover and inspect API operations | `list_weave_operations`, `describe_weave_operation` |
| Read projects, examples, stages, hardware and checkpoints | `read_weave_operation` |
| Validate shapes, generate code, expand attention or unroll a loop without saving | `read_weave_operation` with the discovered pure-POST operation and `body` |
| Save/delete graphs, edit scripts, launch/stop/resume runs, change validation | `execute_weave_operation` |
| Probe a Python environment | `execute_weave_operation`, because probing launches the selected executable |
| Chat, completion, catalog/decoder settings and regression tests | Discover `/chat` operations; changes use the execution tool, polling uses the read tool |
| Read transcript export | The JSON read adapter parses NDJSON into `data.records`; file download also supports transcripts |
| Monitor a long run | `/runs/{run_id}/status` for a small summary; `/event-records` for chronological pages |
| Upload tokenizer bundles, small datasets or snapshot archives | `upload_weave_files` with caller-supplied base64 bytes |
| Export model/training ZIPs, transcripts or named text/code run artifacts | `download_weave_file`, returning an embedded resource |
| Search/read guides | `search_weave_docs`, `read_weave_doc`; documentation resources also support MCP `resources/read` |
| Start a guided design task | MCP prompt `design_weave_model` |

A read-only flag is returned in operation discovery. The bridge enforces it: a GET that executes an interpreter cannot be called through the read tool. The generic execution tool is conservatively annotated as potentially destructive because it includes delete/stop operations. API validation/permission failures set MCP `isError: true` and retain their HTTP status and details in structured content.

The API remains the authority for authorization, graph validation, file containment and upload formats. The HTTP bridge forwards only the calling client's training header/cookie, never the server's environment credential.

## Example workflow

1. Discover `/templates` and `/projects`; inspect the relevant operation schema.
2. Read a suitable template or saved graph and edit its JSON document.
3. Discover `/validate` and call it with `read_weave_operation(operation_id=..., body=project)`.
4. Generate PyTorch with `/code`. Save through the discovered project PUT operation when requested.
5. Inspect/create training stages, supply a real data loader, and launch explicitly. Launch returns a run ID; tools do not block until training finishes.
6. Poll the run's `/status`, then read `/event-records?after=&limit=100`. Pass `next_after` on subsequent calls; an empty page means there are no more records yet. `kind=metrics` filters events without changing cursor semantics.
7. Use checkpoint chat operations to load/test a checkpoint. Token-by-token SSE is not needed for MCP polling.

## File-transfer bounds

`upload_weave_files` accepts `kind="tokenizer"`, `"dataset"`, or `"snapshot"`, and `files=[{"filename":"tokenizer.json","content_base64":"..."}]`. Tokenizer bundles accept up to 32 unique basenames; dataset/snapshot accepts one file. API format checks still apply. There are no server-file-path arguments.

`download_weave_file` accepts:

- `kind="model", project=...`
- `kind="training_package", run_id=...`
- `kind="transcript", run_id=...`
- `kind="artifact", run_id=..., path="train.py"` (text/code extensions only, confined to that run)

Transfers are capped at **4 MiB** per call. Downloads return filename, byte count, SHA256 and an embedded text/blob resource. Its URI identifies the returned content; it is not a persistent HTTP download URL or a server filesystem path. Large datasets, checkpoint weights and large archives still use the UI/HTTP API. For an existing server-local tokenizer directory, `/tokenizers/inspect` remains available without embedding its files in an MCP request.

JSON operation responses are capped at 1 MB. Event pages use a smaller byte budget and at most 500 records; follow the cursor even when a page is shorter than the requested count. A single unusually large event reports an error and can be read from its run artifact.

Session-login endpoints, scoped training-report ingestion and live SSE transport are intentionally outside the MCP operation bridge. Configure the caller credential, use the training reporting API for scripts, and use the JSON polling endpoints for agents. Viewport pan/zoom and browser-only undo are UI state; the saved graph document is editable through MCP.

## Documentation and connection troubleshooting

The index is the resource `weave://docs/index`; individual guides use `weave://guides/TRAINING_GUIDE.md`. `read_weave_doc` accepts both historical guide IDs and exact installed paths such as `docs/training-guide.md`, with line ranges for long guides.

For user-owned training and RL loops, read `CUSTOM_TRAINING.md` (or
`docs/custom-training.md`). It covers launch settings, checkpoints/resume,
task metrics, artifacts, evaluation and distributed helpers. Stage exports also
include these guides offline and an `AGENTS.md` pointing agents to the MCP docs.

- Connection refused: start Weave, check the published port, and use an address reachable from the MCP client.
- Module not found: use absolute `.venv/bin/python` and checkout `PYTHONPATH`/`cwd` in a stdio configuration.
- Training 401: provide the caller's training access code as shown above.
- Host/Origin rejected behind Docker or a proxy: configure exact `WEAVE_MCP_PUBLIC_ORIGINS`; see [Docker networking](docker.md#address-settings-and-lifecycle). Wildcards are not accepted.
- Tool schema changed after an update: reconnect the MCP client so it refreshes the catalog.

The design follows the [MCP server concepts](https://modelcontextprotocol.io/docs/learn/server-concepts) and [tool result/error specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools). Capability coverage and deliberate limits are recorded in the [MCP review](reviews/mcp-2026-11-03.md).
