#!/usr/bin/env bash
# Install tileops with --no-deps so pip never re-resolves tilelang (which would drift the
# cu129 stack). tilelang must already be present — baked into the runner image, or
# installed by the developer locally — before this runs.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONSTRAINTS="${REPO_ROOT}/constraints.txt"

# Install tilelang-metax and tvm-ffi
pip install -r 3rdparty/tilelang-metax/requirements.txt
pip install -r 3rdparty/tilelang-metax/requirements-dev.txt
python -m build -w 3rdparty/tilelang-metax
pip install --target=/data/cache/ci/site-packages --no-deps 3rdparty/tilelang-metax/dist/tilelang-0.1.9+maca.gitf7451dfc-cp38-abi3-linux_x86_64.whl
pip install --target=/data/cache/ci/site-packages --no-deps "z3-solver>=4.13.0,<4.15.5"
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.3 python -m build -w 3rdparty/tilelang-metax/3rdparty/tvm/3rdparty/tvm-ffi
pip install --target=/data/cache/ci/site-packages --no-deps 3rdparty/tilelang-metax/3rdparty/tvm/3rdparty/tvm-ffi/dist/apache_tvm_ffi-0.1.3-cp312-abi3-linux_x86_64.whl
pip install --target=/data/cache/ci/site-packages --python-version 3.10.0 --no-deps flash-linear-attention==0.4.0 -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com

if ! python3 -c "import tilelang" >/dev/null 2>&1; then
  {
    echo "::error::tilelang is not importable. Provision it first, then re-run:"
    echo "  release:  python3 -m pip install --no-deps -c \"${CONSTRAINTS}\" tilelang==<version>"
    echo "  main:     python3 -m pip install --no-deps <prebuilt main-commit tilelang wheel>"
    echo "(the CI runner image bakes tilelang; this script never installs it)"
  } >&2
  exit 1
fi

# tileops only; its runtime deps come from the runner image.
python3 -m pip install -e "${REPO_ROOT}" --no-deps -c "${CONSTRAINTS}"

python3 -c "import tileops, tilelang; print('install_tileops: tileops + tilelang import OK')"
