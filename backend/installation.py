"""Standard-library installation, persistent configuration and server entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / ".weave-config.json"
ENVIRONMENT = {
    "host": "WEAVE_HOST",
    "port": "WEAVE_PORT",
    "data_directory": "WEAVE_DATA",
    "projects_directory": "WEAVE_PROJECTS_DIR",
    "training_enabled": "WEAVE_TRAINING_ENABLED",
    "training_python": "WEAVE_TRAINING_PYTHON",
}


def read_settings():
    if not SETTINGS.exists():
        return {}
    value = json.loads(SETTINGS.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{SETTINGS} must contain a JSON object")
    return value


def executable(value):
    candidate = shutil.which(str(value))
    if not candidate:
        raise ValueError(f"Python executable not found: {value}")
    # Preserve a venv symlink: resolving it would select the base interpreter.
    return str(Path(candidate).absolute())


def save_settings(values):
    values = {**read_settings(), **values}
    fd, temporary = tempfile.mkstemp(prefix=".weave-config-", dir=ROOT)
    try:
        with os.fdopen(fd, "w") as target:
            json.dump(values, target, indent=2)
            target.write("\n")
        os.replace(temporary, SETTINGS)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return values


def configuration(args):
    values = {}
    for field in ("host", "port", "training_enabled"):
        value = getattr(args, field, None)
        if value is not None:
            values[field] = value
    for field in ("projects_directory", "data_directory"):
        value = getattr(args, field, None)
        if value is not None:
            path = Path(value).expanduser().absolute()
            path.mkdir(parents=True, exist_ok=True)
            values[field] = str(path)
    if getattr(args, "training_python", None):
        values["training_python"] = executable(args.training_python)
    return values


def summary(settings):
    print(f"Configuration: {SETTINGS}")
    print(
        f'Address: http://{settings.get("host", "0.0.0.0")}:{settings.get("port", 8173)}'
    )
    print(f'Projects: {settings.get("projects_directory", "~/shared/weave-projects")}')
    print(f'Internal data: {settings.get("data_directory", ROOT / "data")}')
    enabled = settings.get("training_enabled", False)
    print(
        f'Local training: {"enabled" if enabled else "disabled (editor/export only)"}'
    )
    if enabled:
        print(
            f'Training Python: {settings.get("training_python", ROOT / ".venv/bin/python")}'
        )
    print("Start: bash scripts/start.sh (environment variables are optional overrides)")


def run(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, cwd=ROOT, **kwargs)


def check_tools():
    if not shutil.which("uv"):
        raise ValueError(
            "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
        )
    for tool in ("node", "npm"):
        if not shutil.which(tool):
            raise ValueError(
                f"{tool} is required. Install Node.js 20.19+ and npm first."
            )
    run(
        [
            "node",
            "-e",
            'const [major,minor]=process.versions.node.split(".").map(Number); if(major<20 || (major===20 && minor<19)) { console.error("Node.js 20.19+ is required"); process.exit(1); }',
        ]
    )


def install(args):
    python = args.python
    check_tools()
    # Existing installations keep their selected training environment unless an
    # explicit installation mode was requested. Fresh installs include CPU torch.
    previous = read_settings()
    mode = args.training
    if mode is None:
        mode = (
            "none"
            if previous.get("training_enabled") is False
            else ("existing" if previous.get("training_python") else "cpu")
        )
    training_python = args.training_python or previous.get("training_python")
    if args.training_python and args.training == "none":
        raise ValueError("--training-python cannot be combined with --training none")
    if args.training_python:
        mode = "existing"
    if mode == "existing" and not training_python:
        raise ValueError("--training existing requires --training-python PATH")
    run(["uv", "sync", "--locked", "--no-dev", "--python", python])
    run(["npm", "ci"])
    run(["npm", "run", "build"])
    values = configuration(args)
    values.setdefault(
        "projects_directory",
        previous.get("projects_directory", str(Path.home() / "shared/weave-projects")),
    )
    Path(values["projects_directory"]).mkdir(parents=True, exist_ok=True)
    if mode != "none":
        if mode == "cpu":
            training_python = str(ROOT / ".venv-training/bin/python")
            if not Path(training_python).exists():
                run(
                    [
                        "uv",
                        "venv",
                        "--python",
                        python,
                        Path(training_python).parent.parent,
                    ]
                )
            run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    training_python,
                    "torch",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ]
            )
        training_python = executable(training_python)
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                training_python,
                "-r",
                "training-requirements.txt",
            ]
        )
        run(
            [
                training_python,
                "-c",
                'import torch, transformers, numpy, jsonschema, pyarrow; print("Training ready: PyTorch", torch.__version__, "CUDA", torch.version.cuda)',
            ]
        )
        values.update(training_enabled=True, training_python=training_python)
    else:
        values["training_enabled"] = False
    summary(save_settings(values))


def serve():
    if not (ROOT / "dist/index.html").is_file():
        raise ValueError(
            "Frontend build is missing. Run bash scripts/install.sh first."
        )
    for key, value in read_settings().items():
        if key in ENVIRONMENT:
            if key == "training_enabled":
                if type(value) is not bool:
                    raise ValueError("training_enabled must be true or false")
                value = "1" if value else "0"
            os.environ.setdefault(ENVIRONMENT[key], str(value))
    host = os.environ.get("WEAVE_HOST", "0.0.0.0")
    port = port_number(os.environ.get("WEAVE_PORT", "8173"))
    os.chdir(ROOT)
    if os.environ.get("WEAVE_TRAINING_ENABLED") == "1":
        interpreter = executable(
            os.environ.get("WEAVE_TRAINING_PYTHON", sys.executable)
        )
        os.environ["WEAVE_TRAINING_PYTHON"] = interpreter
        print(f"Local training enabled; Python: {interpreter}", flush=True)
    else:
        print(
            "Local training disabled. Enable with: bash scripts/configure.sh --enable-training --training-python /path/to/python",
            flush=True,
        )
    import uvicorn

    uvicorn.run("backend.app:app", host=host, port=port)


def port_number(value):
    try:
        port = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("Port must be an integer") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
    return port


def options(parser):
    parser.add_argument("--projects-dir", dest="projects_directory")
    parser.add_argument("--data-dir", dest="data_directory")
    parser.add_argument("--host")
    parser.add_argument("--port", type=port_number)
    parser.add_argument(
        "--training-python",
        help="Existing training interpreter (CUDA/ROCm/CPU); its torch build is preserved",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Install and configure Weave without required environment variables"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser(
        "install", help="Install dependencies, build UI, and save configuration"
    )
    options(setup)
    setup.add_argument(
        "--python",
        default="3.12",
        help="Python version/path managed by uv (default: 3.12)",
    )
    setup.add_argument(
        "--training",
        choices=["cpu", "existing", "none"],
        help="Fresh default: cpu; existing installations retain their interpreter",
    )
    configure = commands.add_parser(
        "configure", help="Update only supplied settings; install no packages"
    )
    options(configure)
    toggle = configure.add_mutually_exclusive_group()
    toggle.add_argument(
        "--enable-training", dest="training_enabled", action="store_true", default=None
    )
    toggle.add_argument(
        "--disable-training", dest="training_enabled", action="store_false"
    )
    commands.add_parser(
        "run", help="Start with persisted settings and optional environment overrides"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "install":
            install(args)
        elif args.command == "configure":
            summary(save_settings(configuration(args)))
        else:
            serve()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Weave setup: {exc}\n")


if __name__ == "__main__":
    main()
