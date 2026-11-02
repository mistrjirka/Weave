"""Chat data validation and XML inspection, independent of model/runtime code."""

import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def clean_assistant(text):
    """Templates own turn delimiters. Never feed a generated delimiter back twice."""
    text = text.rstrip()
    while match := re.search(r"(?:<\|end\|>|<\|im_end\|>|<\|eot_id\|>|</s>)$", text):
        text = text[: match.start()].rstrip()
    return text


def inspect_calls(text, tools):
    from jsonschema import Draft202012Validator

    text = clean_assistant(text)
    result = {"kind": "text", "calls": [], "errors": [], "text": text}
    if "<call" not in text and "<no_call" not in text:
        return result
    result["kind"] = "invalid"
    if "<!" in text or "<?" in text:
        result["errors"].append("XML declarations and entities are not supported.")
        return result
    try:
        root = ET.fromstring("<response>" + text + "</response>")
    except ET.ParseError as exc:
        result["errors"].append(f"Incomplete or malformed XML: {exc}")
        return result
    catalog = {tool["function"]["name"]: tool["function"] for tool in tools or []}
    plain = [root.text or ""]
    no_call = False
    for element in root:
        plain.append(element.tail or "")
        if (
            element.tag == "no_call"
            and not element.attrib
            and not list(element)
            and not (element.text or "").strip()
        ):
            no_call = True
            continue
        if element.tag != "call" or set(element.attrib) != {"name"}:
            result["errors"].append(
                "Only <call name=...> and <no_call/> are supported."
            )
            continue
        name = element.get("name")
        function = catalog.get(name)
        errors = []
        if function is None:
            errors.append(f"Undeclared tool: {name}")
        schema = (function or {}).get("parameters") or {
            "type": "object",
            "properties": {},
        }
        properties = schema.get("properties", {})
        arguments = {}
        if (element.text or "").strip():
            errors.append("Call body must contain argument elements only.")
        for arg in element:
            if arg.tag != "arg" or set(arg.attrib) != {"name"} or list(arg):
                errors.append("Arguments must be <arg name=...>escaped value</arg>.")
                continue
            key = arg.get("name")
            if key in arguments:
                errors.append(f"Duplicate argument: {key}")
            if key not in properties:
                errors.append(f"Undeclared argument: {key}")
            value = arg.text or ""
            kind = properties.get(key, {}).get("type", "string")
            if kind != "string":
                try:
                    value = json.loads(value)
                except ValueError:
                    errors.append(f"Argument {key} must be valid JSON for type {kind}.")
            arguments[key] = value
            if (arg.tail or "").strip():
                errors.append("Unexpected text after argument.")
        try:
            errors.extend(
                error.message
                for error in Draft202012Validator(schema).iter_errors(arguments)
            )
        except Exception as exc:
            errors.append(f"Invalid tool schema: {exc}")
        call = {
            "name": name,
            "arguments": arguments,
            "valid": not errors,
            "errors": errors,
        }
        result["calls"].append(call)
        result["errors"].extend(errors)
    if no_call and (result["calls"] or "".join(plain).strip()):
        result["errors"].append("<no_call/> cannot be mixed with calls or other text.")
    if not result["errors"]:
        result["kind"] = "no_call" if no_call else "calls"
    result["text"] = "".join(plain).strip()
    return result


def context_messages(messages):
    """Keep only turns explicitly accepted into context, retaining structured calls."""
    return [
        {
            key: message[key]
            for key in ("role", "content", "tool_calls", "tool_call_id", "name")
            if key in message
        }
        for message in messages
        if message.get("included_in_context", True)
    ]


def catalog_issues(tools, profile="generic"):
    errors, warnings = [], []
    if len(tools or []) > 10:
        warnings.append(
            "More than 10 tools: small models trained on short catalogs may select less reliably. Consider retrieval/top-k tool selection."
        )
    if profile != "xml-flat-v1":
        return {"errors": errors, "warnings": warnings}
    for tool in tools or []:
        fn = tool["function"]
        name = fn["name"]
        if not all(32 <= ord(c) < 127 for c in name):
            errors.append(
                f"{name}: this decoder supports ASCII tool names only; it would silently skip this tool."
            )
        schema = fn.get("parameters") or {"type": "object", "properties": {}}
        allowed_root = {
            "type",
            "properties",
            "required",
            "additionalProperties",
            "description",
            "title",
            "$schema",
        }
        for key in schema.keys() - allowed_root:
            errors.append(f"{name}: decoder does not enforce schema keyword {key}.")
        if schema.get("type", "object") != "object" or not isinstance(
            schema.get("properties", {}), dict
        ):
            errors.append(f"{name}: parameters must be a flat object schema.")
            continue
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(required, list) or any(
            not isinstance(k, str) or k not in props for k in required
        ):
            errors.append(
                f"{name}: every required argument must be declared in properties."
            )
        for key, prop in props.items():
            if not key or not all(32 <= ord(c) < 127 for c in key):
                errors.append(
                    f"{name}.{key}: this decoder supports nonempty ASCII argument names only."
                )
            if not isinstance(prop, dict):
                errors.append(f"{name}.{key}: argument schema must be an object.")
                continue
            if prop.get("type", "string") not in (
                "string",
                "integer",
                "number",
                "boolean",
            ):
                errors.append(
                    f"{name}.{key}: decoder supports primitive string/integer/number/boolean arguments only; nested and nullable schemas are unsupported."
                )
            for keyword in prop.keys() - {
                "type",
                "enum",
                "description",
                "title",
                "default",
                "examples",
            }:
                errors.append(f"{name}.{key}: decoder does not enforce {keyword}.")
            if "enum" in prop and (
                not isinstance(prop["enum"], list)
                or not prop["enum"]
                or any(
                    isinstance(v, (dict, list))
                    or (isinstance(v, str) and not v.isascii())
                    for v in prop["enum"]
                )
            ):
                errors.append(
                    f"{name}.{key}: enum must be a nonempty list of ASCII scalar values."
                )
    return {"errors": errors, "warnings": warnings}


def presets():
    mobile = json.loads(
        Path(__file__)
        .with_name("chat_presets")
        .joinpath("mobile-actions.json")
        .read_text()
    )
    generic = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    return [
        {
            "id": "mobile-actions",
            "name": "Mobile Actions (7 tools)",
            "tools": mobile,
            "system_template": "Current date and time given in YYYY-MM-DDTHH:MM:SS format: {datetime}\nDay of week is {weekday}\nYou are a model that can do function calling with the following functions\n",
        },
        {
            "id": "generic-demo",
            "name": "Generic demo",
            "tools": generic,
            "system_template": "You are a helpful assistant with access to tools.",
        },
    ]
