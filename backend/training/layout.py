"""Portable, namespaced support code kept separate from user training files."""

from __future__ import annotations

import ast
import io
import hashlib
from pathlib import Path
import re
import zipfile

PACKAGE = "weave_support"
HERE = Path(__file__).parent
MANAGED_MODULES = {
    "model",
    "runtime",
    "export_helpers",
    "sparse_runtime",
    "load_tokenizer",
    "weave_training",
    "text_training",
    "trainer",
    "data_stream",
    "cache_identity",
    "token_cache",
}


def rewrite_imports(source, prefix, modules=None):
    """Qualify only local generated modules; preserve user comments and formatting."""
    modules = MANAGED_MODULES | set(modules or ())
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source  # Keep incomplete SSH edits editable; launch validates syntax.
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line.encode()))
    edits = []
    content = source.encode()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module in modules
        ):
            start = starts[node.lineno - 1] + node.col_offset
            end = starts[node.end_lineno - 1] + node.end_col_offset
            old = content[start:end].decode()
            new = re.sub(
                r"\bfrom\s+" + re.escape(node.module) + r"\b",
                "from " + prefix + node.module,
                old,
                count=1,
            )
            edits.append((start, end, new.encode()))
        elif isinstance(node, ast.Import) and any(
            alias.name in modules for alias in node.names
        ):
            changed = ast.Import(
                names=[
                    (
                        ast.alias(
                            name=prefix + alias.name, asname=alias.asname or alias.name
                        )
                        if alias.name in modules
                        else alias
                    )
                    for alias in node.names
                ]
            )
            if prefix == ".":
                local = [alias for alias in node.names if alias.name in modules]
                external = [alias for alias in node.names if alias.name not in modules]
                nodes = ([ast.Import(names=external)] if external else []) + [
                    ast.ImportFrom(module=None, names=local, level=1)
                ]
                indent = content[
                    starts[node.lineno - 1] : starts[node.lineno - 1] + node.col_offset
                ].decode()
                replacement = ("\n" + indent).join(ast.unparse(item) for item in nodes)
                edits.append(
                    (
                        starts[node.lineno - 1] + node.col_offset,
                        starts[node.end_lineno - 1] + node.end_col_offset,
                        replacement.encode(),
                    )
                )
            else:
                edits.append(
                    (
                        starts[node.lineno - 1] + node.col_offset,
                        starts[node.end_lineno - 1] + node.end_col_offset,
                        ast.unparse(changed).encode(),
                    )
                )
    for start, end, replacement in sorted(edits, reverse=True):
        content = content[:start] + replacement + content[end:]
    return content.decode()


LEGACY_RECIPE_HASHES = {
    "17baba3054bea765bc2bebdb7421be61f76a488fae526057e353c48214acbc47"
}


def normalize_script(source, recipe):
    # Only exact unedited shipped recipes are replaced. Edited scripts retain their logic.
    if (
        recipe in ("causal", "sft")
        and hashlib.sha256(source.encode()).hexdigest() in LEGACY_RECIPE_HASHES
    ):
        return (HERE / "template.py").read_text()
    return rewrite_imports(source, PACKAGE + ".")


def write_support(directory, bundle):
    """Graph exports remain ordinary Python; training uses an explicit package."""
    from .service import inside

    package = Path(directory) / PACKAGE
    package.mkdir(parents=True, exist_ok=True)
    generated = []
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        modules = {
            Path(name).stem
            for name in archive.namelist()
            if name.endswith(".py") and "/" not in name
        }
        for name in archive.namelist():
            target = inside(package, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(name)
            if name == "README.md":
                data = (
                    data.decode()
                    .replace("python example.py", "python -m weave_support.example")
                    .encode()
                )
            if name.endswith(".py"):
                data = rewrite_imports(data.decode(), ".", modules).encode()
            target.write_bytes(data)
            generated.append(PACKAGE + "/" + name)
    generated.extend(write_helpers(package))
    requirements = package / "requirements.txt"
    requirements.write_text(
        requirements.read_text() + "jsonschema>=4,<5\npyarrow>=18,<27\nnumpy>=1.26,<3\n"
    )
    return generated


def write_helpers(package):
    package = Path(package)
    package.mkdir(parents=True, exist_ok=True)
    generated = []
    for name in (
        "text_training.py",
        "weave_training.py",
        "trainer.py",
        "data_stream.py",
        "cache_identity.py",
        "token_cache.py",
    ):
        source = rewrite_imports((HERE / name).read_text(), ".")
        target = package / name
        if not target.exists() or target.read_text() != source:
            target.write_text(source)
        generated.append(PACKAGE + "/" + name)
    # Ship the authoring guide and optional complete example with offline exports.
    for source_path, name in (
        (HERE.parent.parent / "TRAINING_GUIDE.md", "TRAINING_GUIDE.md"),
        (
            HERE.parent.parent / "examples" / "train_cached_jsonl.py",
            "example_cached_jsonl.py",
        ),
    ):
        target = package / name
        content = source_path.read_text()
        if name == "TRAINING_GUIDE.md":
            content = content.replace(
                "examples/train_cached_jsonl.py", "example_cached_jsonl.py"
            ).replace(
                "(TRAINING.md)",
                "(https://github.com/mistrjirka/Weave/blob/main/TRAINING.md)",
            )
        if not target.exists() or target.read_text() != content:
            target.write_text(content)
        generated.append(PACKAGE + "/" + name)
    target = package / "__init__.py"
    source = '"""Generated support. Edit train.py and your own helpers in the parent folder."""\nfrom .weave_training import Run, atomic_checkpoint_path\n'
    if not target.exists() or target.read_text() != source:
        target.write_text(source)
    generated.append(PACKAGE + "/__init__.py")
    return generated
