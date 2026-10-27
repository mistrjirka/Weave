"""Deterministic compiler: explicit forward statements, reusable generated classes."""

from __future__ import annotations
import ast
import io
import json
from pathlib import Path
import zipfile
from functools import lru_cache
import black
from .core import params, validate_project, ordered
from .tokenizer_bindings import bind_input_tokenizers, input_tokenizers
from .codegen import SourceGenerator, has_moe


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
    document = project
    project, _, _ = bind_input_tokenizers(project, ordered)
    project["config"] = {**project["config"], **project["graph"].get("variables", {})}
    cfg = project["config"]
    source, files = SourceGenerator(project, report).generate()
    source = black.format_str(source, mode=black.Mode(line_length=88))
    ast.parse(source)
    files["model.py"] = source
    for filename in ("runtime.py", "export_helpers.py", "sparse_runtime.py"):
        files[filename] = support_source(
            filename, Path(__file__).with_name(filename).stat().st_mtime_ns
        )
    files["project.json"] = json.dumps(document, indent=2)
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
    if (
        has_moe(project["graph"])
        and len(report["outputs"]) == 1
        and report["outputs"][0]
        and report["outputs"][0]["dtype"] == "float32"
    ):
        files["train_example.py"] = black.format_str(
            "import torch\nfrom model import Model\nfrom runtime import moe_auxiliary_loss\n\n"
            + "\n".join(declarations)
            + "\n\nmodel = Model().train()\noptimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)\noptimizer.zero_grad(set_to_none=True)\noutput = model("
            + ", ".join(arguments)
            + ")\n# Replace this placeholder with your task loss and real training data.\ntask_loss = output.float().square().mean()\nloss = task_loss + moe_auxiliary_loss(model)\nloss.backward()\noptimizer.step()\nprint('loss:', loss.item())\n",
            mode=black.Mode(line_length=88),
        )
    configured = input_tokenizers(document)
    if not configured and document.get("tokenizer"):
        configured = {
            node["id"]: document["tokenizer"]
            for node in document["graph"]["nodes"]
            if node["data"]["kind"] == "input"
            and node["data"].get("params", {}).get("dtype", "int64") == "int64"
        }
    files["tokenizers.json"] = json.dumps(configured, indent=2)
    files["load_tokenizer.py"] = black.format_str(
        '''from transformers import AutoTokenizer
from pathlib import Path
import json


def load_tokenizer(input_node=None):
    """Load the tokenizer for an input node path, e.g. source or target."""
    root = Path(__file__).parent
    configured = json.loads((root / "tokenizers.json").read_text())
    if input_node is None:
        if not configured:
            raise ValueError("No tokenizer is configured on a token input.")
        if len({info["id"] for info in configured.values()}) > 1:
            raise ValueError("Choose an input node: " + ", ".join(configured))
        input_node = next(iter(configured))
    info = configured[input_node]
    local = root / "tokenizers" / info["id"]
    legacy = root / "tokenizer"
    source = local if local.is_dir() else legacy if legacy.is_dir() else info.get("repo")
    if not source:
        raise ValueError("Tokenizer files are unavailable for input " + input_node)
    return AutoTokenizer.from_pretrained(str(source), trust_remote_code=False)
''',
        mode=black.Mode(line_length=88),
    )
    files["README.md"] = (
        "# "
        + project.get("name", "Exported graph")
        + "\n\nRandomly initialized PyTorch architecture. Install requirements, then run `python example.py`.\n\n`model.py` contains named layers, direct tensor operations, and explicit loops. Specialized PyTorch layers live in `runtime.py`; named tensor helpers live in `export_helpers.py`, and attention-mask support lives in `sparse_runtime.py`. Keep these files together.\n\nDifferentiable blocks support standard PyTorch autograd. The example uses `torch.no_grad()` for inference. Training requires your own data, loss and optimizer; sampling/token-selection outputs are discrete.\n\nTokenizer files selected on input nodes are included in `tokenizers/<id>/`; `tokenizers.json` maps input node paths to their tokenizer. Use `load_tokenizer(input_node='source')` when inputs use different tokenizers. A declared custom Python shape contract is not a runtime proof.\n\n## Source references\n"
        + json.dumps(project.get("references", []), indent=2)
        + "\n"
    )
    if has_moe(project["graph"]):
        files[
            "README.md"
        ] += "\n## Mixture of experts\n\nWhen present, `train_example.py` demonstrates adding `moe_auxiliary_loss(model)` to a placeholder task loss. Set the balancing weight to zero to disable it. Experts receive selected tokens packed into a single sequence; token-mixing operations inside an expert therefore do not see the original sequences. Routing has no capacity limit or distributed optimization.\n"
    root = Path(__file__).parent.parent
    files["THIRD_PARTY_NOTICES.md"] = (root / "THIRD_PARTY_NOTICES.md").read_text()
    for filename in ("needle_LICENSE", "thimble_LICENSE"):
        files["licenses/" + filename] = (root / "references" / filename).read_text()
    return files


def export_zip(project, tokenizer_dir=None, tokenizer_dirs=None):
    files = compile_project(project)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            archive.writestr(path, content)
        for token_id, directory in (tokenizer_dirs or {}).items():
            for path in Path(directory).rglob("*"):
                if path.is_file():
                    archive.write(
                        path,
                        "tokenizers/"
                        + token_id
                        + "/"
                        + str(path.relative_to(directory)),
                    )
        if tokenizer_dir and Path(tokenizer_dir).is_dir():
            for path in Path(tokenizer_dir).rglob("*"):
                if path.is_file():
                    archive.write(
                        path, "tokenizer/" + str(path.relative_to(tokenizer_dir))
                    )
    return buffer.getvalue()
