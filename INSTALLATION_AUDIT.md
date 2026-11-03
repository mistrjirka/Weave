# Installation audit · 2026-11-03

## Findings

The native installer has no Arch/pacman dependency, machine-specific username, dataset path, GPU index or server address. It uses Bash, uv, Node/npm and Python. `~/shared/weave-projects` and port 8173 are defaults, configurable during installation or afterward. PyTorch is deliberately separate from the web environment and selected for the user's hardware.

The current platform boundary is **POSIX**, rather than Arch. Training uses `fcntl.flock`, process groups and Unix signals. Native Windows is unsupported; WSL2 and Linux containers are the available routes. macOS, Linux ARM and ROCm container passthrough have not been verified by this audit. Package/wheel availability remains a dependency on the target architecture and libc.

Machine-specific installation settings, project data, tokenizers, checkpoints and external decoder modules are intentionally not committed. They are deployment state, not required source files. The checked-in repository contains the code, Python/npm lockfiles, reference assets, examples and instructions needed for a fresh install. Optional local decoders must be supplied separately.

## Corrections

- Native reinstall uses `uv sync --inexact` so additional packages such as user-installed PyTorch are retained. A selected interpreter in the web venv is retained rather than replaced by the default Python version.
- Missing existing Python/PyTorch is reported before dependency installation or UI builds.
- `WEAVE_CONFIG_FILE` places settings outside the checkout. Writes remain atomic in the destination directory. This supports persistent Docker settings and read-only application source.
- Configuration summaries show effective environment overrides, and data-directory paths expand `~` consistently.
- `configure.sh` reports a missing uv prerequisite clearly; unsupported native Windows installation reports the platform requirement.
- Docker source files have explicit ownership so restrictive host checkout permissions do not prevent the unprivileged container user from reading them.
- Images include reference assets, licenses, authoring guides and the example needed by stage generation. The build context excludes projects, local configuration, environments and credentials.
- Docker keeps editable projects, internal state and settings in separate bind mounts. Compiler/package/model caches use a writable named volume. CPU training uses its own Python environment.
- Remote MCP origins can be declared explicitly for container/reverse-proxy deployments. Existing host/origin protection stays enabled.

## Verification

| Check | Result |
| --- | --- |
| Fresh native editor-only install in an isolated checkout | Passed |
| Fresh separate CPU training environment | Passed |
| Native API: project save, graph validation, PyTorch ZIP export | Passed |
| Native training: generated model, backward/optimizer, checkpoint, metrics | Passed; 2 optimizer updates |
| Debian 12 / Python 3.12 Docker editor image build and startup | Passed |
| Container API: save, validation, export, settings on persistent mount | Passed |
| Docker Compose CPU training image | Passed; 2 optimizer updates with metrics |
| Named cache volume writable by container user | Passed |
| NVIDIA container device passthrough | Both RTX 2080 Ti and Tesla V100 detected |
| CPU and GPU Compose configuration validation | Passed |
| Installer and MCP unit/integration tests | 16 passed |

These are smoke tests, not a complete operating-system or GPU-wheel compatibility matrix. The CUDA-wheel training image and non-x86 platforms were not exercised; NVIDIA device visibility alone is not a CUDA training benchmark. Installation tests used isolated storage and did not alter the existing server or training projects.

Docker restart/recreation stops processes inside the container. This differs from the native deployment's independently supervised training processes; checkpoint before replacing a training container.

## References

- [uv sync behavior](https://docs.astral.sh/uv/concepts/projects/sync/): exact syncing removes extra packages; inexact syncing preserves them.
- [uv Docker integration](https://docs.astral.sh/uv/guides/integration/docker/): locked dependencies and virtual environments in images.
- [Docker GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/): optional GPU access with explicit device capabilities.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html): host runtime requirements.
- [PyTorch installation](https://pytorch.org/get-started/locally/): hardware-specific package selection.

The README was shortened from approximately 4,700 to 470 words. It now leads with purpose, an actual screenshot and installation, with the manual moved to [USER_GUIDE.md](USER_GUIDE.md). The structure follows [GitHub's README guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes); wording follows [Google's clear, direct technical style](https://developers.google.com/style/tone). Screenshots show example graphs in an isolated installation, with manually arranged nodes and no personal training data.

The audit host has Docker Engine/Compose and NVIDIA passthrough but lacks the Buildx plugin. Image builds were verified with Docker's legacy builder and then launched through Compose; install Buildx for the normal Compose build command. This is a host prerequisite, not an Arch dependency in the installer.
