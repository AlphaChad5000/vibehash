#!/bin/sh
# Install Vibe Hash on macOS or Linux, then start its guided setup. In a terminal:
#
#     curl -fsSL https://vibehash.org/install.sh | sh
#
# Finds Python 3.10 or newer, installs or updates Vibe Hash from PyPI, then runs `vibehash setup`, which asks
# the rest. A Python that doesn't take packages directly (Homebrew's, Debian's and Ubuntu's) gets Vibe Hash in
# its own environment, ~/.vibehash/venv. Safe to run again. To join a network other than the public one, give
# its node: curl -fsSL https://vibehash.org/install.sh | VIBEHASH_NODE=192.168.1.20 sh

# Everything is in functions that run on the last line, so a truncated download runs nothing.

find_python() {
    for py in python3 python; do
        if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
            echo "$py"
            return 0
        fi
    done
    return 1
}

no_python() {
    echo "Vibe Hash needs Python 3.10 or newer, and this computer doesn't have it yet."
    case "$(uname -s)" in
        Darwin) echo "Install it from https://www.python.org/downloads/ (or: brew install python), then run this again." ;;
        *) echo "Install it with your package manager (on Ubuntu or Debian: sudo apt install python3 python3-venv)," \
                "then run this again." ;;
    esac
    exit 1
}

# The mining dashboard's lettering, in its gold-to-violet gradient: one colour for each quarter of the width.
banner() {
    if [ "$(locale charmap 2>/dev/null)" != UTF-8 ]; then
        printf '\n VIBE HASH   useful-work miner\n\n'
        return
    fi
    if [ ! -t 1 ] || [ -n "${NO_COLOR:-}" ]; then
        set -- '' '' '' '' ''
    elif [ "${COLORTERM:-}" = truecolor ] || [ "${COLORTERM:-}" = 24bit ]; then
        set -- '\033[1;38;2;255;205;60m' '\033[1;38;2;255;150;70m' '\033[1;38;2;255;95;150m' '\033[1;38;2;180;110;255m' \
               '\033[0;38;2;125;135;148m'
    else  # 256 colours, for terminals without true colour (macOS Terminal)
        set -- '\033[1;38;5;221m' '\033[1;38;5;209m' '\033[1;38;5;205m' '\033[1;38;5;141m' '\033[0;38;5;245m'
    fi
    reset=''
    [ -n "$1" ] && reset='\033[0m'
    # shellcheck disable=SC2059  # the colours are printf escapes, so they belong in the format
    printf "\n $1%s$2%s$3%s$4%s$5   %s$reset\n" '█ █ █ █▄' '▄ █▀▀  ' ' █ █ ▄▀█' ' █▀ █ █' 'useful-work miner'
    # shellcheck disable=SC2059
    printf " $1%s$2%s$3%s$4%s$5   %s$reset\n\n" '▀▄▀ █ █▄' '█ ██▄  ' ' █▀█ █▀█' ' ▄█ █▀█' 'installer · vibehash.org'
}

# True if pip can install into this Python: it has pip, and it isn't marked externally managed (PEP 668).
takes_packages() {
    "$1" -m pip --version >/dev/null 2>&1 && "$1" -c '
import os, sys, sysconfig
marked = os.path.exists(os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED"))
sys.exit(marked and sys.prefix == sys.base_prefix)' 2>/dev/null
}

# Make Vibe Hash's own environment in $2 from Python $1. Ubuntu and Debian leave out the part of Python that puts
# pip in new environments (python3-venv), so without it the environment starts bare and gets pip from PyPA.
make_venv() {
    if [ -f "$2/pyvenv.cfg" ]; then
        rm -rf "$2"  # an environment without pip: what an earlier, interrupted attempt left
    fi
    "$1" -m venv "$2" >/dev/null 2>&1 && return 0
    rm -rf "$2"
    "$1" -m venv --without-pip "$2" && curl -fsSL https://bootstrap.pypa.io/get-pip.py | "$2/bin/python" - --quiet
}

main() {
    banner
    py=$(find_python) || no_python
    venv="${VIBEHASH_HOME:-$HOME/.vibehash}/venv"
    if ! takes_packages "$venv/bin/python" && ! takes_packages "$py"; then
        echo "This Python doesn't take packages directly, so Vibe Hash gets its own environment in $venv"
        if ! make_venv "$py" "$venv"; then
            echo "Couldn't make it; the message above says why. On Ubuntu or Debian, sudo apt install python3-venv"
            echo "fixes most causes. Then run this again."
            exit 1
        fi
    fi
    if takes_packages "$venv/bin/python"; then
        py="$venv/bin/python"  # its own environment, made by this script now or before
    fi

    echo "Installing Vibe Hash with $py..."
    if ! "$py" -m pip install --upgrade --quiet --disable-pip-version-check vibehash; then
        echo
        echo "Installing didn't work; pip's message above says why. Check you're online, then run this again."
        exit 1
    fi
    "$py" -c "from importlib.metadata import version; print('Vibe Hash', version('vibehash'), 'is installed.')"

    if [ "$py" = "$venv/bin/python" ]; then
        mkdir -p "$HOME/.local/bin" && ln -sf "$venv/bin/vibehash" "$HOME/.local/bin/vibehash"
        echo "It's in its own environment, so type vibehash where the guide says python3 -m vibehash."
        case ":$PATH:" in
            *":$HOME/.local/bin:"*) ;;
            *) echo "New terminals find vibehash once ~/.local/bin is on your PATH; until then, type ~/.local/bin/vibehash." ;;
        esac
        set -- "$venv/bin/vibehash" setup
    else
        set -- "$py" -m vibehash setup
    fi
    echo

    # curl | sh gives this script the download as its input, so setup's questions read the keyboard instead
    if (: </dev/tty) 2>/dev/null; then
        "$@" </dev/tty
    else
        "$@"
    fi
}

main
