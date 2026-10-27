"""Deterministic compiler: explicit forward statements, reusable generated classes."""

from __future__ import annotations
import ast
import io
import json
from pathlib import Path
import zipfile
from functools import lru_cache
import black
from .core import params, validate_project
from .codegen import SourceGenerator


@lru_cache(maxsize=8)
def support_source(filename, modified_ns):
    """Format shared support code once; omit the legacy graph-interpreter API."""
    source = Path(__file__).with_name(filename).read_text()
    if filename == "runtime.py":
        lines = source.splitlines(keepends=True)
        removed = set()
        for statement in ast.parse(source).body:
            if isinstance(
                statement, (ast.ClassDef, ast.FunctionDef)
            ) and statement.name in {"Loop", "Functional", "make_module"}:
                removed.update(range(statement.lineno - 1, statement.end_lineno))
        source = "".join(
            line for index, line in enumerate(lines) if index not in removed
        )
    return black.format_str(source, mode=black.Mode(line_length=88))


def compile_project(project):
    report = validate_project(project)
    if not report["valid"]:
        raise ValueError(report)
    cfg = project["config"]
    source, files = SourceGenerator(project, report).generate()
    source = black.format_str(source, mode=black.Mode(line_length=88))
    ast.parse(source)
    files["model.py"] = source
    for filename in ("runtime.py", "export_helpers.py", "sparse_runtime.py"):
        files[filename] = support_source(
            filename, Path(__file__).with_name(filename).stat().st_mtime_ns
        )
    files["project.json"] = json.dumps(project, indent=2)
    files["requirements.txt"] = (
        "torch>=2.5\ntransformers>=4.48,<5\nsentencepiece>=0.2\n"
    )
    input_nodes = {
        params(n, cfg).get("index", 0): n
        for n in project["graph"]["nodes"]
        if n["data"]["kind"] == "input"
    }
    last_required = max(
        (
            index
            for index, node in input_nodes.items()
            if not params(node, cfg).get("optional")
        ),
        default=-1,
    )
    # Read back the public signature so the example uses the same safe names.
    classes = [
        statement
        for statement in ast.parse(source).body
        if isinstance(statement, ast.ClassDef)
    ]
    forward = next(
        statement
        for statement in classes[-1].body
        if isinstance(statement, ast.FunctionDef) and statement.name == "forward"
    )
    input_names = [argument.arg for argument in forward.args.args[1:]]
    declarations = []
    arguments = []
    for index in range(last_required + 1):
        node = input_nodes.get(index)
        settings = params(node, cfg) if node else {}
        name = input_names[index]
        if node is None or settings.get("optional"):
            declarations.append(f"{name} = None")
        else:
            dims = [
                value if isinstance(value, int) else 1 if value == "B" else 8
                for value in settings.get("shape", ["B", "T"])
            ]
            declarations.append(
                f"{name} = torch.zeros({tuple(dims)!r}, dtype=torch.{settings.get('dtype', 'int64')})"
            )
        arguments.append(name)
    files["example.py"] = black.format_str(
        "import torch\nfrom model import Model\n\n"
        + "\n".join(declarations)
        + "\n\nmodel = Model().eval()\nwith torch.no_grad():\n    output = model("
        + ", ".join(arguments)
        + ')\nprint(output.shape if hasattr(output, "shape") else [v.shape for v in output])\n',
        mode=black.Mode(line_length=88),
    )
    tokenizer = project.get("tokenizer", {}) or {}
    files["load_tokenizer.py"] = (
        'from transformers import AutoTokenizer\nfrom pathlib import Path\n\ndef load_tokenizer():\n    local = Path(__file__).parent / "tokenizer"\n    return AutoTokenizer.from_pretrained(str(local) if local.exists() else '
        + repr(tokenizer.get("repo") or tokenizer.get("source") or "TOKENIZER_PATH")
        + ", trust_remote_code=False)\n"
    )
    files["README.md"] = (
        "# "
        + project.get("name", "Exported graph")
        + "\n\nRandomly initialized PyTorch architecture. Install requirements, then run `python example.py`.\n\n`model.py` contains named layers, direct tensor operations, and explicit loops. Specialized PyTorch layers live in `runtime.py`; named tensor helpers live in `export_helpers.py`, and attention-mask support lives in `sparse_runtime.py`. Keep these files together.\n\nDifferentiable blocks support standard PyTorch autograd. The example uses `torch.no_grad()` for inference. Training requires your own data, loss and optimizer; sampling/token-selection outputs are discrete.\n\nTokenizer files, when selected, are included in `tokenizer/`. A declared custom Python shape contract is not a runtime proof.\n\n## Source references\n"
        + json.dumps(project.get("references", []), indent=2)
        + "\n"
    )
    root = Path(__file__).parent.parent
    files["THIRD_PARTY_NOTICES.md"] = (root / "THIRD_PARTY_NOTICES.md").read_text()
    for filename in ("needle_LICENSE", "thimble_LICENSE"):
        files["licenses/" + filename] = (root / "references" / filename).read_text()
    return files


def export_zip(project, tokenizer_dir=None):
    files = compile_project(project)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            archive.writestr(path, content)
        if tokenizer_dir and Path(tokenizer_dir).is_dir():
            for path in Path(tokenizer_dir).rglob("*"):
                if path.is_file():
                    archive.write(
                        path, "tokenizer/" + str(path.relative_to(tokenizer_dir))
                    )
    return buffer.getvalue()
