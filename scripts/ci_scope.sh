#!/usr/bin/env bash
# Which tests a pull request needs, from the files it changes (tests.yml runs this). Prints two lines for
# $GITHUB_OUTPUT: `scope=full|light|none` and `modules=...`.
#
#   full   anything near consensus, the chain, training or verification changed: the whole fast suite runs.
#   light  only edge files changed (the app, the launcher, the dashboards, docs): just their own tests run,
#          with no PyTorch install. Pushes to main always run full, so nothing goes untested for long.
#   none   docs only.
#
# Usage: scripts/ci_scope.sh BASE_REF   (e.g. origin/main); with no argument it says full.
set -euo pipefail

if [ $# -eq 0 ] || [ -z "$1" ]; then
  echo "scope=full"; echo "modules="; exit 0
fi

# Modules whose own tests are enough when only they change: nothing in them decides what the chain accepts, and
# their tests pass without PyTorch (checked with torch, transformers and safetensors blocked). cli.py is not here:
# over a dozen test modules import it, so a change to it runs everything. Nor is selftest: its tests need PyTorch.
EDGE="app watch minerui trainui termui advice hardware netstats records settings unattended emblems"
# Test modules that may run in the light set: the edge modules' own, plus these (also PyTorch-free).
LIGHT_TESTS=" tests.test_launcher tests.test_cli_docs tests.test_setup tests.test_menu "

mods="" full=0
add() { case " $mods " in *" $1 "*) ;; *) mods="$mods $1" ;; esac; }
light_test() {  # a test module, or full when it isn't one the light set can run
  local t="$1" m="${1#tests.test_}"
  if [[ " $EDGE " == *" $m "* || "$LIGHT_TESTS" == *" $t "* ]]; then add "$t"; else full=1; fi
}

while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in
    docs/CLI.md) add tests.test_cli_docs ;;
    *.md|docs/*|LICENSE|.gitignore|floorplan/*|.github/workflows/launcher.yml|.github/workflows/publish.yml) ;;
    launcher/*|app.ps1) add tests.test_launcher ;;
    scripts/ci_scope.sh|.github/workflows/tests.yml) add tests.test_launcher ;;
    tests/test_*.py) light_test "tests.$(basename "$f" .py)" ;;
    vibehash/*.py)
      m=$(basename "$f" .py)
      if [[ " $EDGE " == *" $m "* ]]; then
        [ -f "tests/test_$m.py" ] && add "tests.test_$m"
      else
        full=1
      fi ;;
    *) full=1 ;;
  esac
done < <(git diff --name-only "$1...HEAD")

if [ $full -eq 1 ]; then echo "scope=full"; echo "modules="
elif [ -z "$mods" ]; then echo "scope=none"; echo "modules="
else echo "scope=light"; echo "modules=${mods# }"
fi
