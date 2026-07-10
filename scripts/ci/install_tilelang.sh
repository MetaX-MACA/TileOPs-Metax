#!/usr/bin/env bash
set -euo pipefail

pip install -r 3rdparty/tilelang-metax/requirements.txt
pip install -r 3rdparty/tilelang-metax/requirements-dev.txt
python -m build -w 3rdparty/tilelang-metax
pip install --target=/ci-cache/site-packages --no-deps 3rdparty/tilelang-metax/dist/tilelang-0.1.9+maca.gitf7451dfc-cp38-abi3-linux_x86_64.whl
pip install --target=/ci-cache/site-packages --no-deps "z3-solver>=4.13.0,<4.15.5"
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.3 python -m build -w 3rdparty/tilelang-metax/3rdparty/tvm/3rdparty/tvm-ffi
pip install --target=/ci-cache/site-packages --no-deps 3rdparty/tilelang-metax/3rdparty/tvm/3rdparty/tvm-ffi/dist/apache_tvm_ffi-0.1.3-cp312-abi3-linux_x86_64.whl
pip install --target=/ci-cache/site-packages --python-version 3.10.0 --no-deps flash-linear-attention==0.4.0 -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com
