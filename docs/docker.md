# Run Weave in Docker

Docker packages the browser UI and Python server. You do not need Node.js, uv or Python on the host. Use Docker Engine with Compose and Buildx, or Docker Desktop with Linux containers.

The image runs as an unprivileged user. By default it contains the editor and PyTorch exporter; CPU or NVIDIA training is an explicit build choice. Images are built from this repository, not downloaded from a published Weave registry.

## Editor and export

```bash
git clone https://github.com/mistrjirka/Weave.git
cd Weave
mkdir -p weave-state weave-projects weave-settings
export WEAVE_UID=$(id -u) WEAVE_GID=$(id -g)
docker compose up --build -d
```

Open **http://127.0.0.1:8173**. On Linux, use the UID/GID of your normal user so files remain editable from your host editor; do not build with UID 0. Docker Desktop users can normally omit the exports and use the default UID/GID of 1000.

The bind mounts are separate:

| Host directory | Container path | Contents |
| --- | --- | --- |
| `weave-projects/` | `/workspace/projects` | Editable project graphs and stage scripts |
| `weave-state/` | `/workspace/state` | Database, tokenizer uploads, run snapshots, checkpoints and logs |
| `weave-settings/` | `/workspace/settings` | Persistent `weave.json` configuration, including optional local decoder registrations |

A named volume holds downloaded model/package/compiler caches under `/home/weave/.cache`. Back up the three bind-mounted directories together. `docker compose down` preserves them; `down -v` also deletes the cache volume. Mounts are git-ignored and excluded from image builds.

The container paths shown by the UI are paths **inside the container**. For example, `/workspace/projects/my-model/stages/.../train.py` is `weave-projects/my-model/stages/.../train.py` on the host. An existing host Python environment cannot be reused inside an image; install training dependencies in the image.

## CPU training

```bash
WEAVE_IMAGE_TARGET=training docker compose up --build -d
```

This builds a separate `/opt/training` environment with CPU PyTorch, the training requirements, and a C/C++ compiler for supported `torch.compile` workloads. Training is enabled, but the starter script still needs your data loader. There is no dataset baked into the image.

## NVIDIA training

Install a compatible host NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Select a PyTorch package and index from the [official installer](https://pytorch.org/get-started/locally/) or [previous versions](https://pytorch.org/get-started/previous-versions/). Newer CUDA wheels do not necessarily support older GPUs.

For example, **if PyTorch 2.6.0 / CUDA 12.6 is suitable for your GPU and driver**:

```bash
WEAVE_IMAGE_TARGET=training \
TORCH_PACKAGE='torch==2.6.0' \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

The GPU override requests all NVIDIA devices using Docker's [GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/). Select training and validation devices in Weave. To expose a subset, edit the override to replace `count: all` with `device_ids: ["0", "1"]`; container device indices may differ from host indices. Do not set both `count` and `device_ids`.

ROCm passthrough and Apple GPU training are not provided by this Compose setup. CPU editor/export remains available on Docker Desktop. Only Linux x86-64 is currently smoke-tested; wheel availability still limits other architectures.

## Data and additional Python packages

Mount datasets explicitly; for example, add this entry under the service's `volumes`:

```yaml
- /absolute/host/corpus:/datasets/corpus:ro
```

Your loader then reads `/datasets/corpus`. Token caches must use a writable mount such as `/workspace/state/token-cache`; read-only corpus mounts cannot store caches. Shared-memory size defaults to 2 GiB and can be increased for larger multiprocessing loaders.

Keep custom packages in a derived image so they survive recreation. After building a training image with `docker compose build`, use its image name from `docker compose images`:

```dockerfile
FROM your-built-training-image
USER root
RUN uv pip install --python /opt/training/bin/python your-data-library
USER weave
```

Optional decoder modules must also be installed or mounted inside the container and registered with **container** paths in `weave-settings/weave.json`. See [checkpoint decoders](checkpoint-chat-decoders.md).

## Address, settings and lifecycle

The default host binding is loopback. For LAN access:

```bash
WEAVE_BIND_ADDRESS=0.0.0.0 docker compose up -d
```

Use `WEAVE_PUBLISHED_PORT=8174` to change the host port. Internally the app stays on port 8173. This is a trusted single-user application that runs Python; expose it only to users you trust.

For remote MCP access, also specify the public origin, because the container cannot discover the host's LAN address:

```bash
WEAVE_BIND_ADDRESS=0.0.0.0 \
WEAVE_MCP_PUBLIC_ORIGINS=http://192.168.1.10:8173 \
docker compose up -d
```

`WEAVE_MCP_PUBLIC_ORIGINS` is a comma-separated list of exact HTTP(S) origins; wildcard hosts and URL paths are rejected. Ordinary browser UI access does not require it. The `/mcp/` endpoint retains host/origin checks.

Compose reads the above substitution variables from your shell or its `.env` file. This differs from the native shell launcher, which does not load `.env`. Repeat your chosen build options on updates, or store them in an untracked `.env`:

```dotenv
WEAVE_IMAGE_TARGET=training
WEAVE_UID=1000
WEAVE_GID=1000
# TORCH_PACKAGE=torch==2.6.0
# TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
```

Use `docker compose logs -f` for server logs. `restart: unless-stopped` starts the app when Docker starts. Rebuild after pulling updates, and use the same GPU override if applicable:

```bash
git pull
docker compose up --build -d
```

**Stopping or recreating the container also stops its training and chat processes.** Checkpoint/pause training before an update; resume from a saved checkpoint afterward. The independent-training behavior of a native systemd server restart does not apply across container destruction.

The Dockerfile has `editor` and `training` targets and uses multi-stage builds to keep Node.js out of the runtime image. PyTorch and custom recipe dependencies are hardware-dependent, not covered by the web `uv.lock`; freeze your selected versions when reproducible training environments are required.

## Troubleshooting

- `docker: unknown command: docker buildx`: install the Buildx plugin supplied by your Docker/platform packages. Compose builds require it; `docker build` may fall back to Docker's deprecated legacy builder.
- Permission denied on bind mounts: create the directories as your normal user and rebuild with matching `WEAVE_UID`/`WEAVE_GID`. Existing directories retain their ownership; the image does not recursively change host files.
- No GPU devices: verify the host driver, NVIDIA Container Toolkit and GPU override before choosing a CUDA device in the UI. Seeing devices does not ensure that a selected PyTorch wheel supports their compute capability.
