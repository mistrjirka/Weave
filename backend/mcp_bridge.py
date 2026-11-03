"""MCP access to Weave's documented JSON API and local guides."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx
import psutil
from mcp.server import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from .documentation import GUIDE_PATHS, guide_path

ROOT = Path(__file__).resolve().parent.parent
GUIDES = GUIDE_PATHS
MAX_RESPONSE = 1_000_000
MAX_GUIDE = 250_000
EXCLUDED_PATHS = {
    "/api/export", "/api/training/session", "/api/training/reports/{run_id}",
    "/api/training/runs/{run_id}/artifact", "/api/training/runs/{run_id}/package",
    "/api/training/runs/{run_id}/events", "/api/training/runs/{run_id}/reporting",
}


def mcp_transport_security() -> TransportSecuritySettings:
    """Allow this machine's network names while rejecting unrelated Host/Origin headers."""
    names = {"localhost", "127.0.0.1", "[::1]", socket.gethostname().lower()}
    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            if address.family in (socket.AF_INET, socket.AF_INET6):
                host = address.address.split("%", 1)[0]
                names.add(f"[{host}]" if address.family == socket.AF_INET6 else host)
    names.discard("")
    hosts = {f"{name}:*" for name in names}
    origins = {f"{scheme}://{name}:*" for scheme in ("http", "https") for name in names}
    # A container sees its own interfaces, not the host's LAN address or proxy
    # hostname. Permit only explicitly configured public origins in addition.
    for value in os.environ.get("WEAVE_MCP_PUBLIC_ORIGINS", "").split(","):
        value = value.strip().rstrip("/")
        if not value:
            continue
        origin = urlsplit(value)
        if (
            origin.scheme not in {"http", "https"} or not origin.hostname
            or origin.username or origin.password or origin.path
            or origin.query or origin.fragment or "*" in value
        ):
            raise ValueError(
                "WEAVE_MCP_PUBLIC_ORIGINS must contain comma-separated "
                "HTTP(S) origins without wildcards"
            )
        origin.port  # Validate malformed/out-of-range ports before serving.
        hosts.add(origin.netloc)
        origins.add(value)
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=sorted(hosts),
        allowed_origins=sorted(origins),
    )


class WeaveBridge:
    def __init__(self, base_url: str = "http://127.0.0.1:8173", app=None):
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.path not in {"", "/"} or parts.query or parts.fragment:
            raise ValueError("WEAVE_MCP_URL must be an HTTP(S) origin without credentials, path, or query")
        self.base_url = base_url.rstrip("/")
        self.app = app

    def client(self) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=self.app) if self.app is not None else None
        return httpx.AsyncClient(base_url=self.base_url, transport=transport, timeout=30.0, follow_redirects=False)

    async def openapi(self) -> dict:
        async with self.client() as client:
            response = await client.get("/openapi.json")
            response.raise_for_status()
            return response.json()

    async def operations(self) -> dict[str, dict]:
        document = await self.openapi()
        result = {}
        for path, methods in document.get("paths", {}).items():
            if not path.startswith("/api/") or path in EXCLUDED_PATHS:
                continue
            for method, operation in methods.items():
                if method.upper() not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                    continue
                body = operation.get("requestBody", {})
                content = body.get("content", {})
                if content and "application/json" not in content:
                    continue
                responses = operation.get("responses", {})
                successful = [v for k, v in responses.items() if str(k).startswith("2")]
                if any(any(t in v.get("content", {}) for t in ("application/zip", "application/octet-stream", "text/event-stream")) for v in successful):
                    continue
                name = operation.get("operationId") or f"{method}_{path}"
                result[name] = {"method": method.upper(), "path": path, "operation": operation}
        return result

    def credentials(self, ctx: Context | None) -> dict[str, str]:
        headers = ctx.headers if ctx is not None else None
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        forwarded = {}
        if headers.get("x-weave-training-token"):
            forwarded["x-weave-training-token"] = headers["x-weave-training-token"]
        elif self.app is None and os.environ.get("WEAVE_TRAINING_TOKEN"):
            forwarded["x-weave-training-token"] = os.environ["WEAVE_TRAINING_TOKEN"]
        cookie = headers.get("cookie", "")
        # Pass through only Weave's training cookie, never unrelated browser cookies.
        from http.cookies import SimpleCookie
        parsed = SimpleCookie()
        try:
            parsed.load(cookie)
        except Exception:
            pass
        if "weave_training" in parsed:
            forwarded["cookie"] = f"weave_training={parsed['weave_training'].coded_value}"
        return forwarded

    async def execute(self, operation_id: str, path_params: dict | None, query: dict | None, body: object | None, ctx: Context | None, *, read_only: bool) -> dict:
        operations = await self.operations()
        if operation_id not in operations:
            return {"error": "Unknown or unsupported operation", "operation_id": operation_id}
        entry = operations[operation_id]
        if (entry["method"] == "GET") != read_only:
            return {"error": "Use read_weave_operation for GET and execute_weave_operation for changes"}
        path = entry["path"]
        params = path_params or {}
        for parameter in entry["operation"].get("parameters", []):
            if parameter.get("in") == "path":
                key = parameter["name"]
                if key not in params:
                    return {"error": f"Missing path parameter: {key}"}
                value = str(params[key])
                if value in {".", ".."} or "/" in value or "\\" in value:
                    return {"error": f"Invalid path parameter: {key}"}
                path = path.replace("{" + key + "}", quote(value, safe=""))
        if "{" in path or "}" in path:
            return {"error": "Unresolved path parameter"}
        if not isinstance(query or {}, dict):
            return {"error": "Query must be an object"}
        kwargs = {"params": query or {}, "headers": self.credentials(ctx)}
        if body is not None:
            kwargs["json"] = body
        try:
            async with self.client() as client:
                async with client.stream(entry["method"], path, **kwargs) as response:
                    payload = bytearray()
                    async for chunk in response.aiter_bytes():
                        payload.extend(chunk)
                        if len(payload) > MAX_RESPONSE:
                            return {"status": response.status_code, "error": "Response exceeds 1 MB; use the Weave UI or HTTP API for large results"}
                    content_type = response.headers.get("content-type", "")
                    if "json" not in content_type:
                        return {"status": response.status_code, "error": "Operation returned non-JSON content"}
                    return {"status": response.status_code, "data": json.loads(payload)}
        except (httpx.HTTPError, ValueError) as exc:
            return {"error": str(exc)}


def create_mcp_server(bridge: WeaveBridge) -> MCPServer:
    server = MCPServer(name="Weave Neural Graph Studio", instructions="Discover operations, inspect their OpenAPI contract, then call JSON endpoints. Training operations require the caller's training credential.")

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def list_weave_operations(search: str = "") -> list[dict]:
        """List available Weave JSON API operations, optionally filtered by text."""
        operations = await bridge.operations()
        term = search.casefold()
        return [
            {"operation_id": key, "method": item["method"], "path": item["path"], "summary": item["operation"].get("summary", "")}
            for key, item in operations.items()
            if term in f"{key} {item['path']} {item['operation'].get('summary', '')}".casefold()
        ]

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def describe_weave_operation(operation_id: str) -> dict:
        """Get an operation's parameters, JSON request body, response schemas, and description."""
        operations = await bridge.operations()
        item = operations.get(operation_id)
        if item is None:
            return {"error": "Unknown or unsupported operation"}
        document = await bridge.openapi()
        schemas = document.get("components", {}).get("schemas", {})
        needed = {}
        pending = [item["operation"]]
        while pending:
            current = pending.pop()
            if isinstance(current, dict):
                ref = current.get("$ref", "")
                prefix = "#/components/schemas/"
                if ref.startswith(prefix):
                    name = ref[len(prefix):]
                    if name in schemas and name not in needed:
                        needed[name] = schemas[name]
                        pending.append(schemas[name])
                pending.extend(current.values())
            elif isinstance(current, list):
                pending.extend(current)
        return {"operation_id": operation_id, **item, "schemas": needed}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def read_weave_operation(operation_id: str, path_params: dict | None = None, query: dict | None = None, ctx: Context | None = None) -> dict:
        """Call a GET Weave JSON API operation. Supply path_params and query as needed."""
        return await bridge.execute(operation_id, path_params, query, None, ctx, read_only=True)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False))
    async def execute_weave_operation(operation_id: str, path_params: dict | None = None, query: dict | None = None, body: object | None = None, ctx: Context | None = None) -> dict:
        """Call a POST, PUT, PATCH, or DELETE Weave JSON API operation."""
        return await bridge.execute(operation_id, path_params, query, body, ctx, read_only=False)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def search_weave_docs(query: str) -> list[dict]:
        """Search the installed Weave guides and return matching lines with context."""
        term = query.strip().casefold()
        if not term or len(term) > 200:
            return [{"error": "Supply a search term of 1 to 200 characters"}]
        matches = []
        for name, location in GUIDES.items():
            lines = (ROOT / location).read_text(encoding="utf-8").splitlines()
            for number, line in enumerate(lines, 1):
                if term in line.casefold():
                    matches.append({"guide": name, "line": number, "text": line[:500]})
                    if len(matches) >= 60:
                        return matches
        return matches

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def read_weave_doc(guide: str, start_line: int = 1, max_lines: int = 160) -> dict:
        """Read an allowlisted Weave guide, optionally by line range."""
        location = guide_path(guide)
        if location is None:
            return {"error": "Unknown guide", "available": list(GUIDES)}
        if start_line < 1 or not 1 <= max_lines <= 300:
            return {"error": "start_line must be positive and max_lines must be 1 to 300"}
        source = ROOT / location
        if source.stat().st_size > MAX_GUIDE:
            return {"error": "Guide exceeds size limit"}
        lines = source.read_text(encoding="utf-8").splitlines()
        return {"guide": guide, "start_line": start_line, "total_lines": len(lines), "text": "\n".join(lines[start_line - 1:start_line - 1 + max_lines])}

    return server


def main() -> None:
    import asyncio
    bridge = WeaveBridge(os.environ.get("WEAVE_MCP_URL", "http://127.0.0.1:8173"))
    asyncio.run(create_mcp_server(bridge).run_stdio_async())


if __name__ == "__main__":
    main()
