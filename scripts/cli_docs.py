"""Write docs/CLI.md, the reference for every `vibehash` command, from the command line's own parser.

    python scripts/cli_docs.py            # rewrite docs/CLI.md
    python scripts/cli_docs.py --check    # exit 1 if docs/CLI.md is out of date (tests/test_cli_docs.py does this)

Everything in the file comes from cli.parser(): each command's help and description, its arguments and flags with
their defaults and choices, so the reference can't say something `--help` doesn't. The one thing kept here by hand is
EXAMPLES, a command line or two per command, most of them lifted from the README. A positional argument with choices
(like `cache`'s show/clear/sweep) is written out one choice a line when its help names each choice as "choice: what
it does", so it reads like the command groups (wallet, job, ...) do.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vibehash import cli  # noqa: E402

OUT = ROOT / "docs" / "CLI.md"
PROG = "vibehash"

# A command line or two per command, by its full name. Anything not here just gets its usage line.
EXAMPLES: dict[str, list[str]] = {
    "setup": ["vibehash setup", "vibehash setup --node 192.168.1.20",
              "VIBEHASH_PASSWORD='choose-one' nohup vibehash setup --yes --deepseek no > setup.log 2>&1"],
    "install-gpu": ["vibehash install-gpu"],
    "testnet": ["vibehash testnet", "vibehash testnet --validators 4", "vibehash testnet --job none --workers 3 --device cuda",
                "vibehash testnet --evo2"],
    "wallet create": ["vibehash wallet create", "vibehash wallet create --out gpu1.json"],
    "wallet restore": ["vibehash wallet restore"],
    "wallet show": ["vibehash wallet show", "vibehash wallet show --wallet gpu1.json"],
    "genesis create": ["vibehash genesis create --chain-id my-testnet-1 --out genesis.json "
                       "--bootstrap NODE_ADDR=10 --validator NODE_ADDR=http://192.168.1.20:8545 --alloc REQUESTER_ADDR=500"],
    "node": ["vibehash node --genesis genesis.json --wallet node.json --data-dir node-data --host 0.0.0.0",
             "vibehash node --genesis genesis.json --wallet bob.json --data-dir bob-data --host 0.0.0.0 "
             "--announce http://203.0.113.7:8545 --peer http://192.168.1.20:8545",
             "VIBEHASH_ADMIN_TOKEN=SECRET vibehash node --genesis genesis.json --wallet node.json --data-dir node-data --host 0.0.0.0"],
    "tls cert": ["vibehash tls cert --host 192.168.1.20"],
    "admin": ["vibehash admin", "vibehash admin --node https://node.vibehash.org --token SECRET --node http://10.0.0.5:8545",
              "vibehash admin --browser", "vibehash admin --install-shortcut --node https://node.vibehash.org --token SECRET",
              "vibehash admin --new-token", "vibehash admin --new-mfa"],
    "tls fingerprint": ["vibehash tls fingerprint node-cert.pem"],
    "worker": ["vibehash worker", "vibehash worker --device cuda:1 --wallet gpu1.json", "vibehash worker --job fruit-demo",
               "VIBEHASH_PASSWORD='choose-one' nohup vibehash worker --plain > worker.log 2>&1 &"],
    "cache": ["vibehash cache", "vibehash cache clear", "vibehash cache clear fruit-demo", "vibehash cache sweep"],
    "devices": ["vibehash devices"],
    "selftest": ["vibehash selftest", "vibehash selftest --model qwen3-0.6b"],
    "balance": ["vibehash balance", "vibehash balance vibe1..."],
    "send": ["vibehash send --to vibe1... --amount 2.5"],
    "stake": ["vibehash stake --amount 10"],
    "unstake": ["vibehash unstake --amount 10"],
    "post-job": ["vibehash post-job --spec job.json --budget 150 --dataset examples/demo-dataset.npz"],
    "cancel-job": ["vibehash cancel-job", "vibehash cancel-job my-first-job"],
    "validator join": ["vibehash validator join --url http://203.0.113.7:8545 --wallet bob.json"],
    "validator leave": ["vibehash validator leave"],
    "validator list": ["vibehash validator list"],
    "example": ["vibehash example"],
    "train": ["vibehash train fruit.csv --budget 100", "vibehash train my-data.csv --label price --hidden 256,256",
              "vibehash train chats.jsonl --base qwen3-0.6b"],
    "models": ["vibehash models"],
    "predict": ["vibehash predict new-rows.csv", "vibehash predict fruit-demo.npz new-rows.csv --out predictions.csv"],
    "job status": ["vibehash job status", "vibehash job status fruit-demo"],
    "job watch": ["vibehash job watch"],
    "job download": ["vibehash job download", "vibehash job download fruit-demo --out fruit.npz"],
    "job history": ["vibehash job history fruit-demo --csv fruit-demo.csv"],
    "job template": ["vibehash job template --out job.json", "vibehash job template --gpu"],
    "chain verify": ["vibehash chain verify", "vibehash chain verify --genesis genesis.json"],
}


def subcommands(p: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    return next((a for a in p._actions if isinstance(a, argparse._SubParsersAction)), None)


def helps(sp: argparse._SubParsersAction) -> dict[str, str]:
    """Each subcommand's one-line help, as `--help` lists them."""
    return {a.dest: a.help or "" for a in sp._choices_actions}


def leaves(p: argparse.ArgumentParser, path: list[str] | None = None):
    """Every command that does something, with its full name, in the parser's order."""
    sp = subcommands(p)
    if sp is None:
        yield " ".join(path or []), p
        return
    for name, child in sp.choices.items():
        yield from leaves(child, (path or []) + [name])


def usage(p: argparse.ArgumentParser) -> str:
    return " ".join(p.format_usage().split()).removeprefix("usage: ")


def choices_text(choices) -> str:
    if isinstance(choices, range):
        return f"{choices.start}-{choices.stop - 1}"
    return ", ".join(str(c) for c in choices)


def each_choice(action: argparse.Action) -> list[tuple[str, str]] | None:
    """A positional's choices with what each does, when its help is written "a: ...; b: ...; c: ..."."""
    if not action.choices or isinstance(action.choices, range) or not action.help:
        return None
    parts = re.split(r";\s+(?=\w[\w-]*:\s)", action.help)
    found = dict(part.split(":", 1) for part in parts if ":" in part)
    if set(found) != {str(c) for c in action.choices}:
        return None
    return [(str(c), found[str(c)].strip()) for c in action.choices]


def flag_name(action: argparse.Action) -> str:
    names = ", ".join(action.option_strings)
    if action.nargs == 0:
        return names
    metavar = action.metavar or ("{" + ",".join(str(c) for c in action.choices) + "}" if action.choices else action.dest.upper())
    if isinstance(metavar, tuple):
        metavar = " ".join(metavar)
    return f"{names} {metavar}"


def default_text(action: argparse.Action) -> str:
    """The default, unless the help already says it or there is nothing worth saying."""
    d = action.default
    if d in (None, argparse.SUPPRESS, False, "") or action.nargs == 0 or "default" in (action.help or "").lower():
        return ""
    return f"Default: `{d}`."


def sentences(*parts: str) -> str:
    """The parts that say anything, each ending in a full stop, as one line."""
    said = [p.strip() for p in parts if p and p.strip()]
    return " ".join(p if p.endswith(".") else p + "." for p in said)


def portable(text: str) -> str:
    """Defaults that name this computer's folders, written as any computer's."""
    from vibehash import settings
    return text.replace(str(settings.home()), "~/.vibehash").replace("~/.vibehash\\", "~/.vibehash/")


def argument_lines(p: argparse.ArgumentParser) -> list[str]:
    lines = []
    positionals = [a for a in p._actions if not a.option_strings and not isinstance(a, argparse._SubParsersAction)]
    options = [a for a in p._actions if a.option_strings and a.help != argparse.SUPPRESS
               and not isinstance(a, argparse._HelpAction)]
    if positionals:
        lines.append("Arguments:")
        lines.append("")
        for a in positionals:
            optional = " (optional)" if a.nargs in ("?", "*") else ""
            per_choice = each_choice(a)
            if per_choice:
                lines.append(f"- `{a.dest}`{optional}: one of")
                for choice, what in per_choice:
                    lines.append(f"  - `{choice}`: {what}")
            else:
                one_of = f"one of {choices_text(a.choices)}" if a.choices else ""
                lines.append(f"- `{a.dest}`{optional}: {sentences(a.help or '', one_of, default_text(a))}".rstrip(": "))
        lines.append("")
    if options:
        lines.append("Flags:")
        lines.append("")
        for a in options:
            required = " (required)" if a.required else ""
            help_ = (a.help or "").replace("%(default)s", str(a.default))
            lines.append(f"- `{flag_name(a)}`{required}: {sentences(help_, default_text(a))}".rstrip(": "))
        lines.append("")
    return lines


def command_section(name: str, p: argparse.ArgumentParser, help_: str, level: int) -> list[str]:
    lines = [f"{'#' * level} {PROG} {name}", ""]
    if help_:
        lines += [help_[0].upper() + help_[1:] + ("" if help_.endswith(".") else "."), ""]
    if p.description and p.description != help_:
        lines += [" ".join(p.description.split()), ""]
    lines += ["```", usage(p), "```", ""]
    lines += argument_lines(p)
    if name in EXAMPLES:
        lines += ["Examples:", "", "```bash", *EXAMPLES[name], "```", ""]
    return lines


def render(ap: argparse.ArgumentParser | None = None) -> str:
    ap = ap or cli.parser()
    top = subcommands(ap)
    top_help = helps(top)
    lines = [
        f"# The `{PROG}` command line",
        "",
        "Every command, from the command line's own parser. Generated by `python scripts/cli_docs.py`; edit the help "
        "text in `vibehash/cli.py` (or `selftest.py`) and run it again, since `tests/test_cli_docs.py` fails when "
        "this file is out of date.",
        "",
        f"`{PROG}` and `python -m vibehash` are the same program. With no arguments at a terminal it opens a menu. "
        "`--node` and `--wallet` default to what `setup` saved. Wallet passwords are read from the terminal, or from "
        "`$VIBEHASH_PASSWORD` on machines nobody is watching. For the ideas behind the commands, see the "
        "[README](../README.md) and the [guide](https://vibehash.org/guide/).",
        "",
        "## Global flags",
        "",
    ]
    for a in ap._actions:
        if a.option_strings and not isinstance(a, argparse._HelpAction):
            lines.append(f"- `{flag_name(a)}`: {a.help or ('print the version' if a.dest == 'version' else '')}")
    lines += ["", "## Commands", ""]
    for name, child in top.choices.items():
        sp = subcommands(child)
        if sp is None:
            lines.append(f"- [`{PROG} {name}`](#{anchor(name)}): {top_help[name]}")
        else:
            lines.append(f"- `{PROG} {name}`: {top_help[name]}")
            for sub_name, sub_help in helps(sp).items():
                lines.append(f"  - [`{PROG} {name} {sub_name}`](#{anchor(f'{name} {sub_name}')}): {sub_help}")
    lines.append("")
    for name, child in top.choices.items():
        sp = subcommands(child)
        if sp is None:
            lines += command_section(name, child, top_help[name], 2)
        else:
            lines += [f"## {PROG} {name}", "", top_help[name][0].upper() + top_help[name][1:] + ".", ""]
            for sub_name, grandchild in sp.choices.items():
                lines += command_section(f"{name} {sub_name}", grandchild, helps(sp)[sub_name], 3)
    return portable("\n".join(lines).rstrip() + "\n")


def anchor(name: str) -> str:
    """GitHub's anchor for a heading "vibehash NAME"."""
    return f"{PROG}-{name}".replace(" ", "-")


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        current = OUT.read_text(encoding="utf-8").replace("\r\n", "\n") if OUT.exists() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is out of date: run python scripts/cli_docs.py")
            return 1
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {sum(1 for _ in leaves(cli.parser()))} commands")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
