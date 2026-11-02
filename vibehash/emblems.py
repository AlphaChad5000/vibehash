"""Small pixel-art emblems of the model makers in the catalog, for the training dashboard's forge (trainui.py).

When a job fine-tunes an open-weight model, the forge works that model's emblem instead of a blade, so it's plain at
a glance which model is being trained. Each one is a simple rendering of its maker's logo, at most 18 by 10 pixels,
in their colours, with no wordmarks or slogans, used only to say whose model it is. They go by model family (an
organisation and the start of a model's name), so only that maker's own models of that family get one; anything else
gets a generic gem, and jobs trained from a CSV file (the only kind today) keep the blade. The owner chose to keep
them as drawn, logo-like ones included (2026-11-01).

Disclaimer: All product names, logos, and brands are property of their respective owners and are used purely for
identification. Vibe Hash isn't affiliated with or endorsed by them.

The pixels are drawn two to a character cell with half blocks: the top pixel as the foreground of "▀", the bottom one
as its background. Without colour, or in ASCII, the shapes still read in "█▀▄" or "#~_".
"""
from __future__ import annotations

from dataclasses import dataclass

WIDTH, HEIGHT = 18, 10  # the most room the forge has for an emblem, in pixels: five rows of half blocks


@dataclass(frozen=True)
class Emblem:
    name: str
    rows: tuple[str, ...]  # HEIGHT strings of pixels: a palette letter, or "." for none
    palette: dict

    @property
    def width(self) -> int:
        return len(self.rows[0])

    def pixel(self, x: int, y: int) -> tuple | None:
        """The colour at (x, y), or None where there's nothing."""
        ch = self.rows[y][x] if 0 <= y < len(self.rows) and 0 <= x < self.width else "."
        return None if ch == "." else self.palette[ch]


def _emblem(name: str, palette: dict, *rows: str) -> Emblem:
    assert len(rows) == HEIGHT and len({len(r) for r in rows}) == 1 and len(rows[0]) <= WIDTH, name
    assert set("".join(rows)) <= set(palette) | {"."}, name
    return Emblem(name, rows, palette)


EMBLEMS = {e.name: e for e in (
    # DeepSeek: a blue whale, facing left, its tail raised.
    _emblem("deepseek", {"b": (77, 107, 254), "d": (52, 78, 214), "l": (160, 178, 255), "w": (235, 240, 255)},
            "...............b.b",
            "..............bbbb",
            "....bbbbbbb....bb.",
            "..bbbbbbbbbbbb.bb.",
            ".bbbbbbbbbbbbbbb..",
            "bbwbbbbbbbbbbbd...",
            "bbbbbbbbbbbbbd....",
            ".llllllbbbbdd.....",
            "..llllllll........",
            "....llll.........."),
    # Meta: the infinity loop, light blue to deep blue.
    _emblem("meta", {"l": (0, 140, 255), "d": (0, 100, 224)},
            "...lll......ddd...",
            "..llllll..dddddd..",
            ".lllllllldddddddd.",
            "lll...lllddd...ddd",
            "ll.....lldd.....dd",
            "ll.....lldd.....dd",
            "lll...lllddd...ddd",
            ".lllllllldddddddd.",
            "..llllll..dddddd..",
            "...lll......ddd..."),
    # Qwen: a violet hexagon ring, lit from above.
    _emblem("qwen", {"p": (98, 84, 232), "l": (150, 135, 255)},
            "...lppppppl...",
            "..lllpppplll..",
            ".llll....llll.",
            ".lll......lll.",
            ".lll......lll.",
            ".ppp......ppp.",
            ".ppp......ppp.",
            ".pppp....pppp.",
            "..pppllllppp..",
            "...pllllllp..."),
    # Gemma (Google): a four-pointed star in blues, brightest at its heart.
    _emblem("gemma", {"b": (66, 133, 244), "l": (138, 180, 248), "w": (225, 236, 254)},
            ".......bb.......",
            ".......bb.......",
            "......bllb......",
            "....bbllllbb....",
            "bbbbbllwwllbbbbb",
            "bbbbbllwwllbbbbb",
            "....bbllllbb....",
            "......bllb......",
            ".......bb.......",
            ".......bb......."),
    # OpenAI: the knot, as a light rosette round a hexagonal hole.
    _emblem("openai", {"w": (236, 236, 236), "g": (150, 150, 150)},
            "............",
            "............",
            "...ggwwwg...",
            "...wgwwgw...",
            "..wwwwwwww..",
            ".gwww..wwwg.",
            ".ggww..wwgg.",
            "..wwwwwwww..",
            "...wgwwgw...",
            "...ggwwwg..."),
    # Microsoft (Phi): four squares.
    _emblem("microsoft", {"r": (242, 80, 34), "g": (127, 186, 0), "b": (0, 164, 239), "y": (255, 185, 0)},
            ".........",
            "rrrr.gggg",
            "rrrr.gggg",
            "rrrr.gggg",
            "rrrr.gggg",
            ".........",
            "bbbb.yyyy",
            "bbbb.yyyy",
            "bbbb.yyyy",
            "bbbb.yyyy"),
    # Mistral: a block M, banded yellow to red.
    _emblem("mistral", {"1": (255, 216, 0), "2": (255, 175, 0), "3": (255, 130, 5), "4": (250, 80, 15),
                        "5": (225, 5, 0)},
            "111.........111",
            "111.........111",
            "222222...222222",
            "222222...222222",
            "333333333333333",
            "333333333333333",
            "444...444...444",
            "444...444...444",
            "555.........555",
            "555.........555"),
    # Hugging Face (SmolLM): the hugging face, smiling, hands up front.
    _emblem("huggingface", {"y": (255, 210, 30), "k": (58, 59, 69), "h": (255, 157, 11)},
            "......yyyy......",
            ".....yyyyyy.....",
            "....yyyyyyyy....",
            "...yykkyykkyy...",
            "...yyyyyyyyyy...",
            "...yykyyyykyy...",
            "..hyyykkkkyyyh..",
            ".hhhyyyyyyyyhhh.",
            ".hhhhyyyyyyhhhh.",
            ".hhh..yyyy..hhh."),
    # Anyone else's model: a cut gem, in the dashboard's violet.
    _emblem("generic", {"v": (170, 110, 240), "l": (215, 175, 255), "d": (120, 70, 190)},
            "...llllllll...",
            "..lvvvvvvvvl..",
            ".lvvvvvvvvvvl.",
            "vvvvvvvvvvvvvv",
            ".vvvvvvvvvvvd.",
            "..vvvvvvvvdd..",
            "...vvvvvvdd...",
            "....vvvvdd....",
            ".....vvdd.....",
            "......dd......"),
)}

# Whose emblem a model gets: its family, as (organisation, the start of its name), in lower case. Only these: Google's
# T5 or ViT aren't Gemma, and Meta's Prompt-Guard isn't Llama, so they get the generic gem, as do others' repackagings.
FAMILIES = (("deepseek-ai", ("deepseek",), "deepseek"), ("meta-llama", ("llama", "meta-llama"), "meta"),
            ("qwen", ("qwen",), "qwen"), ("google", ("gemma",), "gemma"), ("openai", ("gpt-oss",), "openai"),
            ("microsoft", ("phi",), "microsoft"),
            ("mistralai", ("mistral", "mixtral", "ministral", "magistral", "devstral", "codestral"), "mistral"),
            ("huggingfacetb", ("smollm",), "huggingface"))


def for_base(base: str | None) -> Emblem | None:
    """The emblem for a job's base model, as `org/name`, `hf:org/name@commit` or a catalog name; None for no base
    model (a model trained from scratch on a CSV file, which keeps the blade)."""
    if not base:
        return None
    from . import catalog  # catalog names, such as deepseek-v4.1-flash

    model = catalog.find(base)
    repo = model.repo if model else base.strip()
    repo = (repo[3:] if repo.lower().startswith("hf:") else repo).split("@")[0].lower()
    org, _, name = repo.partition("/")
    family = next((e for o, starts, e in FAMILIES if org == o and name.startswith(starts)), "generic")
    return EMBLEMS[family]


def cells(e: Emblem) -> list[list[tuple[tuple | None, tuple | None]]]:
    """The emblem as HEIGHT // 2 rows of character cells, each its (top, bottom) pixel colours."""
    return [[(e.pixel(x, 2 * r), e.pixel(x, 2 * r + 1)) for x in range(e.width)] for r in range(HEIGHT // 2)]
