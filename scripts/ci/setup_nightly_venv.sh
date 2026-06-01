#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Setup nightly venv with cache reuse
# =============================================================================
# This script creates a nightly venv that can reuse:
#   1. Cached venv from /data/cache/venv (based on pyproject.toml hash)
#   2. TileLang/Triton kernel compilation cache from /data/cache/tilelang
#   3. Pip wheel cache from /data/cache/pip, /data/cache/wheels
#
# The cache key is computed from:
#   - Python version
#   - TileLang git commit hash
#   - pyproject.toml dependency sections
# =============================================================================

# Install system dependencies
sudo apt-get update && sudo apt-get install -y git

# Add conda to PATH for python access
export PATH="/opt/conda/bin:/usr/bin:${PATH}"

# Set MACA environment variables
export USE_MACA=ON

# Clear CXX/CC to prevent cu-bridge compiler from interfering with TileLang
# kernel compilation. Torch cpp extension may set these to cu-bridge paths,
# which causes "__macro_mxcc.h not found" errors when TVM tries to link .so
unset CXX CC

# =============================================================================
# Resolve cache paths and compute venv cache key
# =============================================================================

TRUSTED_CACHE_ROOT="/data/cache"
TILELANG_COMMIT="f7451dfce010234f540472d442b3f68d4f94232e"

# Compute venv cache key from pyproject.toml (same logic as gpu-smoke.yml)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYPROJECT_PATH="${PROJECT_ROOT}/pyproject.toml"

if [[ ! -f "$PYPROJECT_PATH" ]]; then
  echo "::error::pyproject.toml not found at ${PYPROJECT_PATH}"
  exit 1
fi

# Compute hash using ci_venv_hash.py (same as gpu-smoke.yml benchmark job)
PYPROJECT_HASH=$(python "${SCRIPT_DIR}/ci_venv_hash.py" "$PYPROJECT_PATH")

HASH_INPUT="$(
  {
    echo "python=3.12"
    echo "tilelang=${TILELANG_COMMIT}"
    echo "install=PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev]'"
    echo "pyproject=${PYPROJECT_HASH}"
  } | sha256sum | cut -c1-16
)"

VENV_PREFIX="tileops_ci_venv"
VENV_DIR="${VENV_PREFIX}_${HASH_INPUT}"
TRUSTED_VENV_PATH="${TRUSTED_CACHE_ROOT}/venv/${VENV_DIR}"

# =============================================================================
# Setup cache directories
# =============================================================================

TILELANG_CACHE_DIR="${TRUSTED_CACHE_ROOT}/tilelang"
TILELANG_TMP_DIR="${TILELANG_CACHE_DIR}/tmp"
TRITON_CACHE_DIR="${TRUSTED_CACHE_ROOT}/triton"
PIP_CACHE_DIR="${TRUSTED_CACHE_ROOT}/pip"
WHEEL_DIR="${TRUSTED_CACHE_ROOT}/wheels"

mkdir -p "${TILELANG_CACHE_DIR}" "${TILELANG_TMP_DIR}" "${TRITON_CACHE_DIR}" "${PIP_CACHE_DIR}" "${WHEEL_DIR}"

# =============================================================================
# Check if cached venv exists and is valid
# =============================================================================

VENV_REUSED="false"
if [[ -x "${TRUSTED_VENV_PATH}/bin/python" ]]; then
  # Verify the cached venv is functional
  echo "Found cached venv at ${TRUSTED_VENV_PATH}, validating..."
  if "${TRUSTED_VENV_PATH}/bin/python" -c "import tileops, torch; print('Cache venv validation passed')" 2>/dev/null; then
    VENV_REUSED="true"
    echo "Cached venv is valid and will be reused"
  else
    echo "Cached venv exists but failed validation, will recreate"
  fi
fi

# =============================================================================
# Create or reuse venv
# =============================================================================

if [[ "$VENV_REUSED" == "true" ]]; then
  VENV_PATH="${TRUSTED_VENV_PATH}"
  echo "Using cached venv at ${VENV_PATH}"
else
  # Create a new venv in the trusted cache location
  echo "Creating fresh venv at ${TRUSTED_VENV_PATH}"
  python -m venv --system-site-packages "${TRUSTED_VENV_PATH}"

  # shellcheck source=/dev/null
  source "${TRUSTED_VENV_PATH}/bin/activate"

  python -m pip install --upgrade pip setuptools wheel --no-user

  # Install pytest-xdist for parallel test execution
  pip install --no-cache-dir pytest-xdist

  # Install TileLang (MACA version)
  pip install --no-cache-dir git+https://github.com/tile-ai/tilelang-metax.git@${TILELANG_COMMIT}

  # Install tileops in dev mode
  PIP_NO_BUILD_ISOLATION=1 PIP_NO_CACHE_DIR=1 pip install --no-deps -e '.[dev]'

  VENV_PATH="${TRUSTED_VENV_PATH}"
fi

# =============================================================================
# Export environment variables for subsequent steps
# =============================================================================

{
  echo "VENV_PATH=${VENV_PATH}"
  echo "VENV_REUSED=${VENV_REUSED}"
  echo "TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR}"
  echo "TILELANG_TMP_DIR=${TILELANG_TMP_DIR}"
  echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
  echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}"
  echo "WHEEL_DIR=${WHEEL_DIR}"
  echo "USE_MACA=ON"
} >> "${GITHUB_ENV:-/dev/null}"

# =============================================================================
# Print summary
# =============================================================================

echo ""
echo "=== Nightly venv setup complete ==="
echo "VENV_PATH=${VENV_PATH}"
echo "VENV_REUSED=${VENV_REUSED}"
echo "TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR}"
echo "TILELANG_TMP_DIR=${TILELANG_TMP_DIR}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo ""

if [[ "${VENV_REUSED}" == "true" ]]; then
  echo "CACHE_SIGNAL: nightly venv=hit"
else
  echo "CACHE_SIGNAL: nightly venv=miss (fresh venv created)"
fi

# Print cache stats
echo ""
echo "=== Cache stats ==="
for cache_dir in "${TILELANG_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${WHEEL_DIR}"; do
  if [[ -d "${cache_dir}" ]]; then
    file_count=$(find "${cache_dir}" -type f 2>/dev/null | wc -l || echo "0")
    cache_size=$(du -sh "${cache_dir}" 2>/dev/null | cut -f1 || echo "unknown")
    echo "${cache_dir}: ${file_count} files, ${cache_size}"
  fi
done
