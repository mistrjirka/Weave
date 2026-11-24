"""Advisory inspection of reporting instrumentation."""


def instrumentation_warnings(source):
    """Advisory syntax inspection, not a claim that callbacks actually execute."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [
            {
                "code": "syntax",
                "message": f"Python syntax error at line {exc.lineno}: {exc.msg}",
            }
        ]
    delegated = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "weave_support.text_training"
        for alias in node.names
        if alias.name == "main"
    }
    trainer_classes = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "weave_support.trainer"
        for alias in node.names
        if alias.name == "Trainer"
    }
    if trainer_classes and any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "train"
        for node in ast.walk(tree)
    ):
        return []  # The optional Trainer owns the reporting/checkpoint hooks.
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in delegated
        for node in ast.walk(tree)
    ):
        return (
            []
        )  # The shipped trainer implements metrics/samples/checkpoints/commands.
    methods = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    if "handle_commands" in methods:
        methods.add("poll_commands")
    if (
        any(
            isinstance(node, ast.ImportFrom)
            and node.module in ("weave_support.custom_training", "custom_training")
            and any(alias.name == "Checkpointer" for alias in node.names)
            for node in ast.walk(tree)
        )
        and "save" in methods
    ):
        methods.add("checkpoint")
    expected = {
        "log": "No loss or metric reporting found. Use run.log_loss(loss) with Run(optimizer=...), or run.log().",
        "sample": "No generated-sample reporting found.",
        "checkpoint": "No checkpoint reporting found.",
        "poll_commands": "No cooperative stop/snapshot polling found.",
    }
    return [
        {"code": "missing-" + key, "message": message}
        for key, message in expected.items()
        if key not in methods and not (key == "log" and "log_loss" in methods)
    ]
