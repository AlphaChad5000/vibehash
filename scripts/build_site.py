"""Build the vibehash.org website from the guide.

docs/guide.html is written for the claude.ai artifact viewer, which wraps a page in its own <html> and <head>.
A standalone site needs that wrapper itself, plus what search engines and link previews read.

    python scripts/build_site.py SITE_DIR

writes index.html, CNAME and .nojekyll into SITE_DIR, a checkout of github.com/AlphaChad5000/vibehash.org, which
GitHub Pages serves at https://vibehash.org. Commit and push SITE_DIR to publish.
"""
import html
import re
import sys
from pathlib import Path

DOMAIN = "vibehash.org"
TITLE = "Vibe Hash: mining that trains real AI models"
DESCRIPTION = ("Vibe Hash is a proof-of-useful-work test network: computers earn VIBE by training small, checkable "
               "pieces of AI models instead of hashing. Install it, mine, train a model on your own data, or run "
               "a network of validators.")
# The brand mark (a pickaxe of three blocks) in the dashboard's gold.
ICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 3 4'%3E%3Cpath fill='%23f1b23c' "
        "d='M0 0h1v3H0zM2 0h1v3H2zM1 3h1v1H1z'/%3E%3C/svg%3E")
# What the artifact viewer's wrapper adds that the guide relies on.
RESET = ":root { color-scheme: light; } img { max-width: 100%; } [hidden] { display: none !important; }"


def build(guide: Path, out: Path) -> None:
    text = guide.read_text(encoding="utf-8")
    cut = text.index("</style>") + len("</style>")  # the guide's title, fonts and styles; the page follows
    head = text[:cut].replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
    head = re.sub(r"<title>.*?</title>", f"<title>{html.escape(TITLE)}</title>", head, count=1)  # a home page's name
    desc = html.escape(DESCRIPTION, quote=True)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{desc}">
<link rel="canonical" href="https://{DOMAIN}/">
<meta property="og:type" content="website">
<meta property="og:url" content="https://{DOMAIN}/">
<meta property="og:title" content="{html.escape(TITLE, quote=True)}">
<meta property="og:description" content="{desc}">
<link rel="icon" href="{ICON}">
<style>{RESET}</style>
{head}
</head>
<body>
{text[cut:].strip()}
</body>
</html>
"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    (out / "CNAME").write_text(DOMAIN + "\n", encoding="utf-8", newline="\n")  # GitHub Pages' custom domain
    (out / ".nojekyll").write_text("", encoding="utf-8")  # serve the files as they are
    print(f"wrote {out / 'index.html'} ({len(page.encode()):,} bytes), CNAME and .nojekyll")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    build(Path(__file__).resolve().parent.parent / "docs" / "guide.html", Path(sys.argv[1]))
