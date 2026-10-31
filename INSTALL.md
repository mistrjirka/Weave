# Install Weave

Weave currently targets Linux (including WSL2). The graph editor needs uv, Node.js 20.19+ and npm. The installer obtains Python 3.12 through uv; you can select another Python 3.11+ with `--python`. GPU drivers and system packages remain your responsibility. Native Windows training is not supported; the cache builder uses POSIX locks. macOS has not been verified.

## Default: browser editor and CPU training

```bash
git clone https://github.com/mistrjirka/Weave.git
cd Weave
bash scripts/install.sh
bash scripts/start.sh
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first if missing. The script checks Node/npm, runs `uv sync --locked --no-dev`, `npm ci`, and `npm run build`. Python is downloaded automatically if needed. A fresh default install also creates `.venv-training`, installs CPU PyTorch plus the training helper dependencies, verifies imports, and enables local training. No training starts until you supply a data loader and launch a stage.

The app listens on `0.0.0.0:8173`. Use `http://127.0.0.1:8173` on the server or `http://SERVER_IP:8173` elsewhere. This is a single-user service for a trusted machine/LAN, with server-side Python execution. It is not an authenticated multi-user hosting platform.

For editor/export only:

```bash
bash scripts/install.sh --training none
```

The CPU training environment is omitted and local execution is disabled. Training recipes, export and terminal preparation remain available. Enable local execution later by rerunning the installer with `--training cpu` or `--training-python PATH`.

## GPU training with an existing environment

Choose the PyTorch version/build for your GPU from the [official PyTorch installer](https://pytorch.org/get-started/locally/). In particular, older GPUs may require an older compatible wheel; Weave does not guess a CUDA build from driver presence.

```bash
uv venv --python 3.12 ~/shared/weave-training
# Install your chosen torch build into that environment, for example:
# uv pip install --python ~/shared/weave-training/bin/python torch --index-url URL_FROM_PYTORCH
bash scripts/install.sh --training-python ~/shared/weave-training/bin/python
bash scripts/start.sh
```

The installer adds `training-requirements.txt` to the selected environment, verifies imports, and saves its absolute interpreter path. It does not install or replace torch in an existing training environment. The Training UI shows the interpreter, Python/PyTorch/CUDA versions and available devices. Each stage can choose another interpreter. Default GPU/device and precision choices belong in your `train.py`.

The web environment (`.venv`) is locked by `uv.lock`. Training dependencies are intentionally separate: PyTorch builds, additional data libraries and user recipe requirements depend on hardware and experiments. Run snapshots record the actual installed versions. Updating the web app does not make arbitrary training recipes reproducible automatically.

## Persistent configuration

```bash
bash scripts/configure.sh --projects-dir ~/shared/weave-projects \
  --data-dir /absolute/path/to/app-data --host 0.0.0.0 --port 8173
bash scripts/configure.sh --enable-training --training-python /absolute/path/to/python
bash scripts/start.sh
```

Only supplied options change. Configuration is stored in the git-ignored `.weave-config.json`; unknown keys and existing selections are preserved. Run `bash scripts/configure.sh` without options to print the current saved settings. `--disable-training` disables local process launch without removing environments, scripts or checkpoints. This configuration applies when using `scripts/start.sh`; invoking uvicorn directly bypasses the launcher settings except the existing project-directory setting.

| Saved setting | Default | Optional environment override |
| --- | --- | --- |
| `host` | `0.0.0.0` | `WEAVE_HOST` |
| `port` | `8173` | `WEAVE_PORT` |
| `projects_directory` | `~/shared/weave-projects` | `WEAVE_PROJECTS_DIR` |
| `data_directory` | `<checkout>/data` | `WEAVE_DATA` |
| `training_enabled` | Enabled by default installer; otherwise disabled | `WEAVE_TRAINING_ENABLED=1` or `0` |
| `training_python` | `<checkout>/.venv-training/bin/python` for new CPU installs | `WEAVE_TRAINING_PYTHON` |

Environment overrides take precedence. `.env.example` documents them; copying it to `.env` does not load it. Optional `WEAVE_TRAINING_TOKEN` protects training endpoints with an operator access code; it does not authenticate the rest of the application. No token is required by default. Restart the server after changing settings.

Projects and caches are separate from environments. Back up the projects directory and internal app data for project graphs, run records, snapshots and tokenizer assets. Token caches may live elsewhere as chosen by your recipe. See [training operations](TRAINING.md) and [recipe authoring](TRAINING_GUIDE.md).

## Restart and startup after reboot

Manual restart uses the saved settings:

```bash
cd /absolute/path/to/Weave
bash scripts/start.sh
```

On Linux with systemd, copy [the example user service](scripts/weave.service.example) to `~/.config/systemd/user/weave.service`, replacing both checkout paths with your actual path. Then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now weave.service
journalctl --user -u weave.service -f
```

Stop a manually started instance before enabling the service on the same port. User services normally start at login. For startup before login, an administrator can enable lingering for your user with `loginctl enable-linger USERNAME`. The installer does not alter system services or request sudo.

## Updates and development

```bash
git pull
bash scripts/install.sh
# Restart the process/service after installation.
```

Rerunning the installer retains the selected training interpreter and explicit disabled state unless a training option is supplied. Back up custom code before updates. `--training cpu` explicitly chooses the managed CPU environment; `--training-python` explicitly chooses an external one.

`pyproject.toml` is the web dependency source; `uv.lock` commits the resolution. `requirements.txt` is an exported compatibility file, regenerated with `uv export --locked --no-dev --no-hashes --output-file requirements.txt`. For development, use `uv sync --locked --group dev` and `npm ci`; use `npm run dev` for Vite on port 5173 alongside the backend. Computational tests additionally require an appropriate PyTorch build in the test interpreter.

## Troubleshooting

- Missing uv/Node/npm: install those prerequisites, then rerun. The script does not install system packages.
- Missing frontend build: run the installer; `start.sh` does not fetch packages or build assets on every start.
- Training disabled: check `bash scripts/configure.sh`, then enable/select an installed interpreter. Check for environment overrides from your shell or service.
- Missing torch or data libraries: install them into the interpreter selected in Training, not your shell's unrelated Python.
- Address already in use: stop the other instance or configure another port.
- Using a remote browser: server paths and selected Python refer to the server, not your laptop. SSH forwarding also works: `ssh -L 8173:127.0.0.1:8173 USER@SERVER`.
