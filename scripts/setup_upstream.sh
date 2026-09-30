#!/bin/bash
# Clone the pinned upstream repositories, apply the StructRL patches, and install the RoboCasa365
# Hydra configs into RLinf.
#
#   bash scripts/setup_upstream.sh [--openpi] [--libero]
#
# Default: RLinf, robocasa, robosuite and Isaac-GR00T in $PROJECT_ROOT (RoboCasa365 with GR00T-N1.5).
#   --openpi  also the RoboCasa openpi fork in $PROJECT_ROOT/openpi (pi0.5 SFT and evaluation)
#   --libero  also a second patched RLinf and the RLinf LIBERO fork in $LIBERO_RL_ROOT
#             (default $PROJECT_ROOT/libero_rl), and LIBERO for the evaluation client in
#             $PROJECT_ROOT/LIBERO
# PROJECT_ROOT defaults to the repository root. Re-running is safe: existing checkouts are reused and
# each patch is applied once.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"
LIBERO_RL_ROOT="${LIBERO_RL_ROOT:-$PROJECT_ROOT/libero_rl}"
GIT_BASE="${GIT_BASE:-https://github.com}"

WITH_OPENPI=0
WITH_LIBERO=0
for arg in "$@"; do
  case "$arg" in
    --openpi) WITH_OPENPI=1 ;;
    --libero) WITH_LIBERO=1 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

clone() {  # <owner/repo> <commit> <dir>
  local dir="$3"
  [ -d "$dir/.git" ] || git clone -q "$GIT_BASE/$1" "$dir"
  git -C "$dir" checkout -q "$2"
  echo "$dir @ $(git -C "$dir" rev-parse --short HEAD)"
}

apply_patch() {  # <dir> <patch file in patches/>
  local dir="$1" name="$2" mark="$1/.structrl_$2.applied"
  if [ -f "$mark" ]; then
    echo "  $name already applied"
  else
    git -C "$dir" apply "$REPO/patches/$name"
    touch "$mark"
    echo "  applied $name"
  fi
}

RLINF_COMMIT=499363965432843cdbae607a561d59b68f785d34

clone RLinf/RLinf "$RLINF_COMMIT" "$PROJECT_ROOT/RLinf"
apply_patch "$PROJECT_ROOT/RLinf" rlinf_core.patch
cp -r "$REPO/configs/." "$PROJECT_ROOT/RLinf/examples/embodiment/config/"
echo "  installed configs/ into RLinf/examples/embodiment/config/"

clone robocasa/robocasa 8f3c96ec8d1bfcd8126cad2bca887da98d30e997 "$PROJECT_ROOT/robocasa"
apply_patch "$PROJECT_ROOT/robocasa" robocasa_gymutils.patch
clone ARISE-Initiative/robosuite 232ce7d4a6ed89c949a9aba024a05c8c32fdd08b "$PROJECT_ROOT/robosuite"
clone NVIDIA/Isaac-GR00T 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96 "$PROJECT_ROOT/Isaac-GR00T"
apply_patch "$PROJECT_ROOT/Isaac-GR00T" isaacgroot_shim.patch

if [ "$WITH_OPENPI" = 1 ]; then
  clone robocasa-benchmark/openpi 5a6beda9ff99da30b4e1b59320f6a32971d7c397 "$PROJECT_ROOT/openpi"
  apply_patch "$PROJECT_ROOT/openpi" openpi_rc365.patch
fi

if [ "$WITH_LIBERO" = 1 ]; then
  mkdir -p "$LIBERO_RL_ROOT"
  clone RLinf/RLinf "$RLINF_COMMIT" "$LIBERO_RL_ROOT/RLinf"
  apply_patch "$LIBERO_RL_ROOT/RLinf" rlinf_core.patch
  apply_patch "$LIBERO_RL_ROOT/RLinf" libero_addon.patch
  clone RLinf/LIBERO 0c5e40cc4ae63e09c14e7df6f74481e9ee8585f7 "$LIBERO_RL_ROOT/LIBERO"
  clone Lifelong-Robot-Learning/LIBERO 8f1084e3132a39270c3a13ebe37270a43ece2a01 "$PROJECT_ROOT/LIBERO"
fi

echo "done. export PROJECT_ROOT=$PROJECT_ROOT"
[ "$WITH_LIBERO" = 1 ] && echo "      export LIBERO_RL_ROOT=$LIBERO_RL_ROOT"
exit 0
