#!/usr/bin/env bash
set -euo pipefail

# Build the MACA backend from the same TileLang revision as the runner image.
SITE_PACKAGES="${SITE_PACKAGES:-/ci-cache/site-packages}"
MACA_INDEX="https://repos.metax-tech.com/r/maca-pypi/simple"
MACA_HOST="repos.metax-tech.com"
PIP_ARGS=(--index-url "${MACA_INDEX}" --trusted-host "${MACA_HOST}")

export PYTHONPATH="${SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m pip install --upgrade --target="${SITE_PACKAGES}" \
  "maca-python" "${PIP_ARGS[@]}"

unset USE_CUDA USE_ROCM USE_METAL
TILELANG_BUILD_DIR="$(mktemp -d /tmp/tilelang-metax-build.XXXXXX)"
trap 'rm -rf "${TILELANG_BUILD_DIR}"' EXIT
git clone --branch 0.1.15 --recursive \
  https://github.com/tile-ai/tilelang-metax.git "${TILELANG_BUILD_DIR}"
python3 -m pip install --upgrade --target="${SITE_PACKAGES}" \
  --no-deps --no-cache-dir --force-reinstall \
  "${TILELANG_BUILD_DIR}/3rdparty/tvm/3rdparty/tvm-ffi"
USE_MACA=ON python3 -m pip install --upgrade --target="${SITE_PACKAGES}" \
  --no-build-isolation --no-deps --no-cache-dir --force-reinstall \
  "${TILELANG_BUILD_DIR}"

python3 -m pip install --quiet --no-cache-dir --target="${SITE_PACKAGES}" \
  "transformers==4.45.2" "${PIP_ARGS[@]}"
python3 - <<'PY'
import transformers
assert transformers.__version__ == "4.45.2", transformers.__version__
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
assert mamba_chunk_scan_combined is not None
print("mamba_ssm baseline import OK")
PY

# Benchmark-only baselines. Install without dependencies so these packages cannot
# replace the runner's MetaX torch/runtime stack.
python3 -m pip install --target="${SITE_PACKAGES}" --no-deps \
  "sqlalchemy==2.0.52" "flag_gems==5.0.2"

python3 -m pip install --target="${SITE_PACKAGES}" --python-version 3.10.0 --no-deps \
  "flash-linear-attention==0.4.0" "${PIP_ARGS[@]}"

PYTHONPATH="${SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY'
import importlib.metadata as metadata
import tilelang

installed = metadata.version("tilelang")
assert installed.startswith("0.1.15+maca"), installed
print(f"tilelang {installed} from {tilelang.__file__}")
print("maca-python installed")
PY
