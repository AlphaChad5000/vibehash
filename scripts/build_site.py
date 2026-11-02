"""Build the vibehash.org website: the front page and the guide.

docs/site/index.html is the front page, a whole page as it is, which reads the network live from the node's /network.
docs/guide.html is written for the claude.ai artifact viewer, which wraps a page in its own <html> and <head>; on the
site it's /guide/, so it needs that wrapper itself, plus what search engines and link previews read.

    python scripts/build_site.py SITE_DIR

writes index.html (and og.png, its link preview), guide/index.html, the one-line installers (install.ps1 and install.sh),
the whitepaper (whitepaper.pdf), CNAME and .nojekyll into SITE_DIR, a checkout of github.com/AlphaChad5000/vibehash.org,
which GitHub Pages serves at https://vibehash.org. Commit and push SITE_DIR to publish.
"""
import html
import re
import sys
from pathlib import Path

DOMAIN = "vibehash.org"
TITLE = "Vibe Hash guide: mine, train a model, run a network"
DESCRIPTION = ("How Vibe Hash works and how to use it: install it and mine, train a model on your own data, or run "
               "a network of validators. Rounds, checks, payment, penalties and the job file.")
# The brand mark (a pickaxe of three blocks) in the dashboard's gold.
ICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 3 4'%3E%3Cpath fill='%23f1b23c' "
        "d='M0 0h1v3H0zM2 0h1v3H2zM1 3h1v1H1z'/%3E%3C/svg%3E")
# What the artifact viewer's wrapper adds that the guide relies on.
RESET = ":root { color-scheme: light; } img { max-width: 100%; } [hidden] { display: none !important; }"


def build(guide: Path, front: Path, out: Path) -> None:
    text = guide.read_text(encoding="utf-8")
    # The guide's brand mark leads home, to the front page.
    text = text.replace('<a class="brand" href="#overview"', '<a class="brand" href="/"', 1)
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
<link rel="canonical" href="https://{DOMAIN}/guide/">
<meta property="og:type" content="website">
<meta property="og:url" content="https://{DOMAIN}/guide/">
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
    (out / "guide").mkdir(parents=True, exist_ok=True)
    (out / "guide" / "index.html").write_text(page, encoding="utf-8", newline="\n")
    for f in front.parent.iterdir():  # the front page and what it links: og.png, the link preview (docs/brand/og.html)
        if f.is_file():
            (out / f.name).write_bytes(f.read_bytes())
    (out / "CNAME").write_text(DOMAIN + "\n", encoding="utf-8", newline="\n")  # GitHub Pages' custom domain
    (out / ".nojekyll").write_text("", encoding="utf-8")  # serve the files as they are
    for script in ("install.ps1", "install.sh"):  # irm https://vibehash.org/install.ps1 | iex, and curl … | sh
        (out / script).write_bytes((guide.parent.parent / script).read_bytes())
    (out / "whitepaper.pdf").write_bytes((guide.parent / "whitepaper.pdf").read_bytes())  # vibehash.org/whitepaper.pdf
    print(f"wrote index.html, guide/index.html ({len(page.encode()):,} bytes), install.ps1, install.sh, whitepaper.pdf, "
          "CNAME and .nojekyll")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    docs = Path(__file__).resolve().parent.parent / "docs"
    build(docs / "guide.html", docs / "site" / "index.html", Path(sys.argv[1]))
