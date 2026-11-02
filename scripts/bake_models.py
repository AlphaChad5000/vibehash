"""Bake the model catalog into the guide's command builder.

The guide is a static page and can't ask Hugging Face anything, so the facts its Train tab shows for each model in
vibehash/catalog.py (size, licence, gated or not, the GPUs a team needs) are written into it. Run this after changing
the catalog, or now and then to refresh the numbers:

    python scripts/bake_models.py [--cached]

It asks Hugging Face afresh (--cached: what it said in the last day will do) and rewrites the block between
`// models:begin` and `// models:end` in docs/guide.html.
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibehash import catalog  # noqa: E402

GUIDE = Path(__file__).resolve().parent.parent / "docs" / "guide.html"
BLOCK = re.compile(r"(// models:begin[^\n]*\n)(.*?)(\n *// models:end)", re.S)


def baked(m: catalog.Model, f: catalog.Facts) -> dict:
    """One model as the page's script reads it: everything already in words, so the page does no maths."""
    return {"name": m.name, "title": m.title, "repo": m.repo, "params": catalog.params_text(f.parameters),
            "experts": catalog.experts_text(f) if f.experts else "", "weights": catalog.size_text(f.weight_bytes),
            "estimated": f.estimated, "licence": f.licence or "not stated", "gated": f.gated,
            "team": [catalog.team(f, m)[gb] for gb in catalog.GPU_GB], "held": catalog.size_text(catalog.held_bytes(f, m)),
            "left_out": [f"{what} ({catalog.size_text(b)}): {why}" for what, why, b in m.left_out]}


def script_json(value) -> str:
    """JSON safe inside the page's inline <script>. What Hugging Face says (a licence name, say) is anyone's text, and a
    `</script>` in it would end the script and start whatever followed. JSON's unicode escapes keep every <, > and &
    (and anything not ASCII) out of the page's markup, while meaning the same to the script."""
    return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def block(facts: dict[str, catalog.Facts]) -> list[str]:
    """The lines between the markers: the catalog's models with their facts, and when Hugging Face was asked."""
    checked = datetime.fromtimestamp(min(f.checked for f in facts.values()), timezone.utc).strftime("%Y-%m-%d")
    lines = ["  var MODELS = ["] + [f"    {script_json(baked(m, facts[m.repo]))}," for m in catalog.CATALOG]
    lines[-1] = lines[-1].rstrip(",")
    return lines + ["  ];", f"  var MODELS_GPU_GB = {script_json(list(catalog.GPU_GB))}, MODELS_CHECKED = {script_json(checked)};"]


def main() -> None:
    facts, errors = catalog.lookup([m.repo for m in catalog.CATALOG], refresh="--cached" not in sys.argv[1:])
    missing = [m.repo for m in catalog.CATALOG if m.repo not in facts or m.repo in errors]
    if missing:
        sys.exit("Hugging Face didn't answer for " + ", ".join(f"{r} ({errors.get(r, 'never looked up')})" for r in missing))
    lines = block(facts)
    checked = json.loads(lines[-1].rsplit("= ", 1)[1].rstrip(";"))
    text = GUIDE.read_text(encoding="utf-8")
    if not BLOCK.search(text):
        sys.exit(f"{GUIDE} has no `// models:begin` ... `// models:end` block")
    GUIDE.write_text(BLOCK.sub(lambda mt: mt.group(1) + "\n".join(lines) + mt.group(3), text, count=1),
                     encoding="utf-8", newline="\n")
    print(f"baked {len(catalog.CATALOG)} models into {GUIDE}, facts from Hugging Face as of {checked}")


if __name__ == "__main__":
    main()
