#!/usr/bin/env bash
# Install the shared environment for the paper code (Linux, Python 3.11, CUDA 12.8).
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$project_root/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
if ! command -v uv >/dev/null; then
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi
if [ ! -x .venv/bin/python ]; then
    uv venv --python "${LAPLACIAN_PYTHON:-python3.11}" .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11 for the pinned environment"'
uv pip install --python .venv/bin/python --torch-backend cu128 -r setup/requirements.lock
uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
    -e pytorch-image-models -e dinov3
# nanochat is an application; DINO's task-local measurements are also importable.
.venv/bin/python - <<'PYTHON'
from pathlib import Path
import site
root = Path.cwd()
pth = Path(site.getsitepackages()[0]) / "laplacian_sources.pth"
pth.write_text("".join(str(root / name) + "\n" for name in ("nanochat", "dinov3")))
PYTHON
uv pip check --python .venv/bin/python
echo "Ready: source $project_root/.venv/bin/activate"
