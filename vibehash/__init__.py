"""Vibe Hash: a proof-of-useful-work cryptocurrency prototype that pays for verified AI training."""


def _version() -> str:
    """This code's release. A source checkout's pyproject.toml first: the package metadata a virtual environment holds
    can be an older install's (a venv with 0.5.7 installed ran a 0.5.15 checkout). Then the installed package's
    metadata, else "unknown"."""
    from pathlib import Path

    try:
        import re
        text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
        if re.search(r'^name\s*=\s*"vibehash"', text, re.M):
            return re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)
    except (OSError, AttributeError):
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("vibehash")
        except PackageNotFoundError:
            pass
    except ImportError:
        pass
    return "unknown"


__version__ = _version()
