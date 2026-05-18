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

# Set MACA SDK paths for compilation (fixes __macro_mxcc.h not found)
if [[ -d "/opt/maca" ]]; then
  export MACA_PATH=/opt/maca
  export C_INCLUDE_PATH="${MACA_PATH}/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
  export CPLUS_INCLUDE_PATH="${MACA_PATH}/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
  export LIBRARY_PATH="${MACA_PATH}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${MACA_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "MACA SDK paths configured: MACA_PATH=${MACA_PATH}"
fi

echo "Creating run-local nightly venv at ${VENV_PATH}"
python -m venv --system-site-packages "${VENV_PATH}"

# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel --no-user

# Install TileLang (MACA version)
pip install --no-cache-dir git+https://github.com/tile-ai/tilelang-metax.git@1efc3ba8a7c48bc97047e562223538d2dcb2222d

{
  echo "RUNTIME_ROOT=${RUNTIME_ROOT}"
  echo "VENV_PATH=${VENV_PATH}"
} >> "${GITHUB_ENV}"

echo "Nightly venv ready:"
echo "RUNTIME_ROOT=${RUNTIME_ROOT}"
echo "VENV_PATH=${VENV_PATH}"
python --version
python -m pip --version
python -c "import torch; print(f'torch={torch.__version__}')"
