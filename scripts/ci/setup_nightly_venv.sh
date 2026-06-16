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

RUNTIME_ROOT="${RUNNER_TEMP}/nightly-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${GITHUB_JOB}"
VENV_PATH="${RUNTIME_ROOT}/venv"

rm -rf "${RUNTIME_ROOT}"
mkdir -p "${RUNTIME_ROOT}"

echo "Creating run-local nightly venv at ${VENV_PATH}"
python -m venv --system-site-packages "${VENV_PATH}"

# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel --no-user

# Install tilelang-metax and tvm-ffi
pip install -r 3rdparty/tilelang-metax/requirements.txt
pip install build
python -m build -w 3rdparty/tilelang-metax
pip install --target=/data/cache/site-packages --no-deps --upgrade 3rdparty/tilelang-metax/dist/tilelang-0.1.9+maca.gitf7451dfc-cp38-abi3-linux_x86_64.whl
python -m build -w 3rdparty/tilelang-metax/3rdparty/tvm/3rdparty/tvm-ffi
pip install --target=/data/cache/site-packages --no-deps --upgrade 3rdparty/tilelang-metax/3rdparty/tvm/3rdparty/tvm-ffi/dist/apache_tvm_ffi-0.1.3.dev11+gae346ec92.d20260616-cp312-abi3-linux_x86_64.whl
pip install --target=/data/cache/site-packages --python-version 3.10.0 --no-deps flash-linear-attention==0.4.0 -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com

{
  echo "RUNTIME_ROOT=${RUNTIME_ROOT}"
  echo "VENV_PATH=${VENV_PATH}"
} >> "${GITHUB_ENV}"

echo "Nightly venv ready:"
echo "RUNTIME_ROOT=${RUNTIME_ROOT}"
echo "VENV_PATH=${VENV_PATH}"
python --version
python -m pip --version
