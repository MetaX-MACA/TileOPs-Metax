#!/usr/bin/env bash
set -euo pipefail

SITE_PACKAGES="${SITE_PACKAGES:-/ci-cache/site-packages}"
TVM_FFI_SRC="3rdparty/tilelang-metax/3rdparty/tvm/3rdparty/tvm-ffi"

# Build apache-tvm-ffi from source below; skip the PyPI pin in requirements.txt
# so it cannot overwrite the SITE_PACKAGES install (plain 0.1.11 has no +g<sha>).
grep -vE '^apache-tvm-ffi' "3rdparty/tilelang-metax/requirements.txt" | pip install -r /dev/stdin
pip install -r "3rdparty/tilelang-metax/requirements-dev.txt"

# --- tilelang ---
# Installed version looks like: 0.1.11+maca.git5675cade
desired_tilelang_git="$(git -C "3rdparty/tilelang-metax" rev-parse --short=8 HEAD)"
installed_tilelang="$(PYTHONPATH="${SITE_PACKAGES}" python -c "from importlib.metadata import version; print(version('tilelang'))" 2>/dev/null || true)"

if [[ -n "${installed_tilelang}" && "${installed_tilelang}" == *"git${desired_tilelang_git}"* ]]; then
  echo "tilelang already at ${installed_tilelang} (matches git${desired_tilelang_git}); skipping build/install"
else
  echo "tilelang installed='${installed_tilelang:-<missing>}' desired=git${desired_tilelang_git}; building and force-installing"
  # Drop stale top-level tvm that can shadow tilelang's vendored copy.
  rm -rf "${SITE_PACKAGES}/tvm" "${SITE_PACKAGES}"/tvm-*.dist-info
  rm -rf "3rdparty/tilelang-metax/dist"
  python -m build -w "3rdparty/tilelang-metax"
  shopt -s nullglob
  tilelang_whls=("3rdparty/tilelang-metax/dist"/tilelang-*git"${desired_tilelang_git}"*.whl)
  shopt -u nullglob
  if [[ "${#tilelang_whls[@]}" -ne 1 ]]; then
    echo "error: expected exactly 1 tilelang wheel for git${desired_tilelang_git}, found ${#tilelang_whls[@]}:" >&2
    printf '  %s\n' "${tilelang_whls[@]:-}" >&2
    ls -la "3rdparty/tilelang-metax/dist" >&2 || true
    exit 1
  fi
  rm -rf "${SITE_PACKAGES}/tilelang" "${SITE_PACKAGES}"/tilelang-*.dist-info
  pip install --upgrade --force-reinstall --target="${SITE_PACKAGES}" --no-deps "${tilelang_whls[0]}"
fi

pip install --target="${SITE_PACKAGES}" --no-deps "z3-solver>=4.13.0,<4.15.5"

# --- apache-tvm-ffi ---
# setuptools_scm local version looks like: 0.1.12.dev0+g3c35034fd.d20260714
# Use --short=7 so the prefix still matches when scm lengthens the node for uniqueness.
desired_tvm_ffi_git="$(git -C "${TVM_FFI_SRC}" rev-parse --short=7 HEAD)"
installed_tvm_ffi="$(PYTHONPATH="${SITE_PACKAGES}" python -c "from importlib.metadata import version; print(version('apache-tvm-ffi'))" 2>/dev/null || true)"

if [[ -n "${installed_tvm_ffi}" && "${installed_tvm_ffi}" == *"g${desired_tvm_ffi_git}"* ]]; then
  echo "apache-tvm-ffi already at ${installed_tvm_ffi} (matches g${desired_tvm_ffi_git}); skipping build/install"
else
  echo "apache-tvm-ffi installed='${installed_tvm_ffi:-<missing>}' desired=g${desired_tvm_ffi_git}; building and force-installing"
  rm -rf "${TVM_FFI_SRC}/dist"
  python -m build -w "${TVM_FFI_SRC}"
  shopt -s nullglob
  tvm_ffi_whls=("${TVM_FFI_SRC}/dist"/apache_tvm_ffi-*g"${desired_tvm_ffi_git}"*.whl)
  shopt -u nullglob
  if [[ "${#tvm_ffi_whls[@]}" -ne 1 ]]; then
    echo "error: expected exactly 1 apache_tvm_ffi wheel for g${desired_tvm_ffi_git}, found ${#tvm_ffi_whls[@]}:" >&2
    printf '  %s\n' "${tvm_ffi_whls[@]:-}" >&2
    ls -la "${TVM_FFI_SRC}/dist" >&2 || true
    exit 1
  fi
  rm -rf "${SITE_PACKAGES}/tvm_ffi" "${SITE_PACKAGES}"/apache_tvm_ffi-*.dist-info
  pip install --upgrade --force-reinstall --target="${SITE_PACKAGES}" --no-deps "${tvm_ffi_whls[0]}"
fi

pip install --target="${SITE_PACKAGES}" --python-version 3.10.0 --no-deps flash-linear-attention==0.4.0 \
  -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com
