#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUNNER_TEMP:-}" ]]; then
  echo "::error::RUNNER_TEMP is not set; cannot create a run-local nightly venv"
  exit 1
fi

if [[ -z "${GITHUB_RUN_ID:-}" || -z "${GITHUB_RUN_ATTEMPT:-}" || -z "${GITHUB_JOB:-}" ]]; then
  echo "::error::GitHub run metadata is incomplete; cannot create a scoped nightly venv"
  exit 1
fi

# Install system dependencies
sudo apt-get update && sudo apt-get install -y git

RUNTIME_ROOT="${RUNNER_TEMP}/nightly-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${GITHUB_JOB}"
VENV_PATH="${RUNTIME_ROOT}/venv"

rm -rf "${RUNTIME_ROOT}"
mkdir -p "${RUNTIME_ROOT}"

# Add conda to PATH for python access
export PATH="/opt/conda/bin:/usr/bin:${PATH}"

# Set MACA environment variables
export USE_MACA=ON

# Clear CXX/CC to prevent cu-bridge compiler from interfering with TileLang
# kernel compilation. Torch cpp extension may set these to cu-bridge paths,
# which causes "__macro_mxcc.h not found" errors when TVM tries to link .so
unset CXX CC

echo "Creating run-local nightly venv at ${VENV_PATH}"
python -m venv --system-site-packages "${VENV_PATH}"

# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel --no-user

# Install pytest-xdist for parallel test execution (required by warmup_kernel_cache.py)
pip install --no-cache-dir pytest-xdist

# Install TileLang (MACA version)
pip install --no-cache-dir git+https://github.com/tile-ai/tilelang-metax.git@f7451dfce010234f540472d442b3f68d4f94232e

{
  echo "RUNTIME_ROOT=${RUNTIME_ROOT}"
  echo "VENV_PATH=${VENV_PATH}"
} >> "${GITHUB_ENV}"

echo "Nightly venv ready:"
echo "RUNTIME_ROOT=${RUNTIME_ROOT}"
echo "VENV_PATH=${VENV_PATH}"
python --version
python -m pip --version