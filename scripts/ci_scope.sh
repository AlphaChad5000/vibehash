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

# Modules whose own tests are enough when only they change: nothing in them decides what the chain accepts.
EDGE="app watch cli minerui trainui termui advice selftest hardware netstats records settings unattended emblems menu"

mods="" full=0
add() { case " $mods " in *" $1 "*) ;; *) mods="$mods $1" ;; esac; }

while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in
    docs/CLI.md) add tests.test_cli_docs ;;
    *.md|docs/*|LICENSE|.gitignore|floorplan/*|.github/workflows/launcher.yml|.github/workflows/publish.yml) ;;
    launcher/*|install/*) add tests.test_launcher ;;
    scripts/ci_scope.sh|.github/workflows/tests.yml) add tests.test_launcher ;;
    tests/test_*.py) add "tests.$(basename "$f" .py)" ;;
    vibehash/*.py)
      m=$(basename "$f" .py)
      if [[ " $EDGE " == *" $m "* ]]; then
        [ -f "tests/test_$m.py" ] && add "tests.test_$m"
        [ "$m" = cli ] && { add tests.test_cli_docs; add tests.test_setup; add tests.test_menu; add tests.test_app; }
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
