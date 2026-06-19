#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# OctaSoma — one-click installer for Debian / Ubuntu systems
# ---------------------------------------------------------------------------
#
# This script:
#   1. Checks system dependencies (cargo, python3-dev, python3-venv).
#   2. Creates a Python virtual environment (`.venv`).
#   3. Installs `maturin` inside the venv.
#   4. Compiles the Rust crate as a native Python wheel via `maturin develop --release`.
#
# Usage:
#   chmod +x install.sh && ./install.sh
# ---------------------------------------------------------------------------

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[-]${NC} $*"; exit 1; }

# ---- system dep check ------------------------------------------------------
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        warn "missing: $1 — attempting apt install ..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq "$2" || err "failed to install $2"
    fi
    log "found: $1 ($(command -v "$1"))"
}

log "OctaSoma installer — checking system dependencies ..."
check_cmd cargo       cargo
check_cmd rustc       cargo
check_cmd python3     python3
check_cmd python3-config python3-dev
check_cmd pip3        python3-pip

# Ensure python3-venv is present.
if ! python3 -m venv --help &>/dev/null; then
    warn "python3-venv missing — installing ..."
    sudo apt-get install -y -qq python3-venv || err "cannot install python3-venv"
fi

# ---- venv ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

if [ ! -d "${VENV_DIR}" ]; then
    log "creating virtual environment at ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
else
    log "virtual environment already exists at ${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
log "activated venv ($(python3 --version))"

# ---- maturin + build -------------------------------------------------------
log "upgrading pip and installing maturin ..."
pip install --upgrade pip -q
pip install maturin -q

log "building OctaSoma native wheel (maturin develop --release) ..."
cd "${SCRIPT_DIR}"
maturin develop --release

log "verifying import ..."
python3 -c "from octasoma import OctaSomaCore; print('OctaSomaCore imported successfully')"

echo ""
log "============================================"
log " OctaSoma installation complete!"
log ""
log " Activate the environment:"
log "   source ${VENV_DIR}/bin/activate"
log ""
log " Quick test:"
log "   python3 -c \"from octasoma import OctaSomaCore; m = OctaSomaCore(8, 42); m.insert([0.1]*8, b'hello'); print(m.query([0.1]*8))\""
log "============================================"
