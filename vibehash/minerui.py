"""The live mining dashboard that `vibehash worker` and `vibehash testnet` show in a terminal.

A drill bores through rock while the worker trains, slows down while it checks other
workers' results, fills its ore cart with every unit it commits, and throws gold sparks
when VIBE lands. Below it: the round and phase, what the worker is doing, which jobs it
mines (a selector: auto, or one of the jobs with budget left), its speed, the work mined
this session, earnings and stake, and the latest events.

Plain ANSI escape codes, nothing to install. Only used on a real terminal; `--plain` (or
piping the output) keeps the classic log, and `worker --job` picks the job there. NO_COLOR
turns the colours off. What it shares with the training dashboard (trainui.py), from the
palette and the rock to the screen and keys, is in termui.py.

An Evo job (es.py) gets a scene of its own in place of the drill (esui.py): a chain of blocks, the open one
incubating the round's population, and below it the generation, population, puzzles, difficulty, mean reward, the
bytes each generation puts on the chain, and whether the model was rebuilt from the chain or fetched.
"""
from __future__ import annotations

import collections
import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass

from . import es, esui, termui
from .client import JobChoice, NodeClient, NodeError, WorkerClient
from .economics import REP_MAX, REP_MIN
from .termui import (DIM, GOLD, GREEN, LAMP, PHASES, RED, SPARKS, TEXT, TIMBER, WHITE, YELLOW, _cells, _fit, _line,
                     _n, _noise, _si, _vibe, rgb)
from .termui import supported  # noqa: F401  (the commands ask minerui whether to show it)

RAIL, TIE, CART = rgb(150, 152, 165), rgb(110, 86, 62), rgb(176, 116, 60)
DRILL, DRILL_DIM, TIP = rgb(80, 200, 255, True), rgb(60, 110, 140), rgb(225, 245, 255, True)

GLYPHS = {**termui.GLYPHS, "rail": "═", "tie": "╪", "stripe": "▓", "cart": "▐▌"}
ASCII = {**termui.ASCII, "rail": "=", "tie": "+", "stripe": "=", "cart": "[]"}
BAND, TUNNEL, RAILS = 7, (2, 3, 4), 4  # rows in the mine picture; the drill bores rows 2-4
JOBS_EVERY = 30.0  # seconds between asking the node which jobs have budget left, for the selector
HINTS = ("  tab or 0-9 to pick", "  tab/0-9")  # the selector's keys, at the end of its line when there's room


def selector(g: dict, options: list[str], sel: int, width: int, keys: bool = True) -> list:
    """The job selector's line, at most `width` wide: "job:", then each option ("auto" first, then the jobs) with
    its number key, the chosen one (`options[sel]`) in brackets, and the keys at the end if they fit (`keys`: only
    when they work). When it doesn't all fit, the names not chosen get shorter (to 16, then 12 characters), the keys
    go, the names go down to 8, and then the last options give way to "+N more", auto too. The chosen one always
    shows, and is cut only when it alone is too wide for the line."""
    more, n = g["more"], len(options)
    hints = HINTS if keys else ()

    def line(hint: str, cap: int | None, keep: int, own: int | None = None) -> list:
        segs = [("job:", DIM)]
        shown = [i for i in range(n) if i < keep or i == sel]
        for i in shown:
            on, name, limit = i == sel, options[i], own if i == sel else cap
            if limit is not None and len(name) > limit:
                name = name[:max(1, limit - len(more))] + more
            segs += [(" [" if on else "  ", GOLD), (f"{i} " if i < 10 else "", DIM), (name, GOLD if on else TEXT),
                     ("]" if on else " ", GOLD)]
        if len(shown) < n:
            segs.append((f" +{n - len(shown)} more", DIM))
        return segs + [(hint, DIM)]

    tries = ([(h, None, n) for h in hints] + [(h, c, n) for c in (16, 12) for h in hints[1:]]
             + [("", c, n) for c in (None, 16, 12)] + [(h, 8, n) for h in (*hints[1:], "")]
             + [(h, 8, k) for k in range(n - 1, -1, -1) for h in (*hints[1:], "")])
    for t in tries:
        segs = line(*t)
        if sum(len(s) for s, _ in segs) <= width:
            return segs
    over = sum(len(s) for s, _ in line("", 8, 0)) - width  # the chosen one alone is too wide: cut it to fit
    return line("", 8, 0, max(1, len(options[sel]) - over))


@dataclass
class Rig:
    name: str
    client: WorkerClient
    start: int | None = None  # balance + stake when the dashboard started
    total: int | None = None
    balance: int = 0
    stake: int = 0
    reputation: float = 0.0
    nonce: int | None = None  # transactions the wallet has sent: a drop with no new one is a slash
    sent: int = 0  # what the wallet's own transactions took out this session, which isn't lost earnings

    @property
    def earned(self) -> int:
        return 0 if self.start is None or self.total is None else self.total - self.start + self.sent


class MinerDashboard:
    """One or more workers' live view. `run()` draws until `stop` is set; the workers run on their own threads."""

    def __init__(self, rigs: list[tuple[str, WorkerClient]], node_url: str, footer: str = "", evo: bool = False):
        """`evo`: show the Evo scene until a job is known (an Evo testnet's); after that the job decides."""
        self.rigs = [Rig(name, c) for name, c in rigs]
        self.api, self.node_url, self.footer = NodeClient(node_url), node_url, footer
        self.feed: collections.deque = collections.deque(maxlen=40)
        self.g = termui.glyphs_for(GLYPHS, ASCII)
        self.scroll, self.frame_no = 0.0, 0
        self.popup: tuple[float, int] | None = None  # (time, syn) of the last payout
        self.phase_seen: dict[tuple, float] = {}
        self.chain_id = ""
        self.started = time.time()
        # One selector for every rig: they share the first one's choice (a worker's --job, or auto).
        self.choice = getattr(self.rigs[0].client, "choice", None) or JobChoice()
        for r in self.rigs:
            r.client.choice = self.choice
        self.jobs: list[str] = []  # jobs with budget left, as the node lists them (_poll_jobs)
        self.keyed = False  # keys are being read (run), so the selector shows them
        self.evo = evo
        self.lineages: dict[str, esui.Lineage] = {}  # each Evo job's generations, followed from the node (_poll_lineage)
        self.eg: dict | None = None  # the Evo scene's glyphs (_eg)

    def note(self, text: str, level: int = logging.INFO) -> None:
        self.feed.append((time.time(), level, text))

    # --- the job selector ------------------------------------------------------------------------------------

    def options(self) -> list[str | None]:
        """What the selector offers: auto (None), then the jobs with budget left, and the chosen one if the node no
        longer lists it (the worker puts itself back on auto once it notices)."""
        chosen = self.choice.job_id
        return [None, *self.jobs, *([chosen] if chosen and chosen not in self.jobs else [])]

    def pick(self, job_id: str | None) -> None:
        """Mine `job_id` only (None: every job), on every rig, and say so in the feed if that's a change."""
        if self.choice.set(job_id):
            self.note(f"mining {job_id} only" if job_id else "mining every job (auto)")

    def press(self, key: str) -> None:
        """One key, as termui.keypresses names it: Tab, j or the right arrow picks the next option, k or the left
        arrow the one before (Shift+Tab too, where the terminal tells it from Tab: a Windows console doesn't); a or 0
        auto, and 1-9 the job with that number."""
        opts = self.options()
        at = opts.index(self.choice.job_id) if self.choice.job_id in opts else 0
        if key in ("tab", "j", "right"):
            self.pick(opts[(at + 1) % len(opts)])
        elif key in ("shift-tab", "k", "left"):
            self.pick(opts[(at - 1) % len(opts)])
        elif key in ("a", "A", "0"):
            self.pick(None)
        elif len(key) == 1 and key in "123456789" and int(key) < len(opts):
            self.pick(opts[int(key)])

    def _selector(self, width: int):
        opts = self.options()
        sel = opts.index(self.choice.job_id) if self.choice.job_id in opts else 0
        evo = self._evo_ids()
        return selector(self.g, ["auto", *(f"{j} (Evo)" if j in evo else j for j in opts[1:])], sel, width, self.keyed)

    def _evo_ids(self) -> set[str]:
        """The jobs the rigs have loaded that are Evo jobs, which the selector marks (no requests for it)."""
        return {j for r in self.rigs for j, job in list(getattr(r.client, "jobs", {}).items()) if getattr(job, "es", False)}

    def _poll_jobs(self, stop: threading.Event) -> None:
        """The selector's jobs, from the node's /network, every JOBS_EVERY seconds: off the drawing thread."""
        while not stop.is_set():
            try:
                self.jobs = self.api.open_jobs()
            except NodeError:
                pass  # keep the last list: the node is busy, or from before /network
            stop.wait(JOBS_EVERY)

    # --- background polling: balances arrive over HTTP, which mustn't stall the animation ------------

    def _poll(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.chain_id = self.chain_id or self.api.status()["chain_id"]
            except NodeError:
                pass
            paid = 0
            for rig in self.rigs:
                try:
                    acct = self.api.account(rig.client.keys.address)
                except NodeError:
                    continue
                total, nonce = acct["balance"] + acct["stake"], acct.get("nonce")
                if rig.start is None:
                    rig.start = total
                elif rig.total is not None and total != rig.total:
                    if total > rig.total:
                        paid += total - rig.total
                    elif nonce is not None and rig.nonce is not None and nonce > rig.nonce:
                        # The wallet sent something itself (a transfer, a job's budget, unstaking): a slash sends nothing
                        rig.sent += rig.total - total
                        self.note(f"{rig.name} sent {_vibe(rig.total - total)} in its own transaction (not a slash)")
                    else:
                        self.note(f"{rig.name} lost {_vibe(rig.total - total)}: slashed", logging.ERROR)
                rig.total, rig.balance, rig.stake, rig.reputation = total, acct["balance"], acct["stake"], acct["reputation"]
                rig.nonce = nonce
            if paid:
                self.popup = (time.time(), paid)
                height = self.rigs[0].client.stats.height
                self.note(f"paid {_vibe(paid, sign=True)} in block {height}" if height else f"paid {_vibe(paid, sign=True)}",
                          logging.INFO + 1)
            stop.wait(1.5)

    def _poll_lineage(self, stop: threading.Event) -> None:
        """An Evo job's generations, followed from the node's blocks and records while it's the one on screen (esui.
        Lineage): off the drawing thread, and only for Evo jobs. Those sealed while watched go in the feed."""
        while not stop.is_set():
            lead = self.rigs[0].client.stats
            job = lead.job
            if job is not None and getattr(job, "es", False):
                lin = self.lineages.setdefault(job.job_id, esui.Lineage(job.job_id))
                was = lin.caught_up
                try:
                    for gen in lin.poll(self.api, lead.height):
                        for text, level in lin.news(gen, self._eg()) if was else ():
                            self.note(text, level)
                except Exception:  # noqa: BLE001 - the node is busy, or said something unreadable: try again later
                    pass
            stop.wait(1.0)

    # --- drawing -----------------------------------------------------------------------------------------

    def _band(self, w: int) -> list[str]:
        """The mine: layered rock, and a timbered tunnel the drill has bored, with rails and an ore cart."""
        g, f, off = self.g, self.frame_no, int(self.scroll)
        stats = [r.client.stats for r in self.rigs]
        busy = any(s.busy for s in stats)
        dx = max(16, int(w * 0.45))  # the drill tip's column; everything left of it is already mined
        rows = [[(" ", None)] * w for _ in range(BAND)]
        for r in range(BAND):
            for c in range(w):
                x, mined = off + c, c <= dx
                lamp = termui.lamp_distance(x)
                if mined and r in TUNNEL:
                    if r == RAILS:
                        rows[r][c] = (g["tie"], TIE) if x % 4 == 0 else (g["rail"], RAIL)
                    elif x % 12 == 0:
                        rows[r][c] = (g["post"], TIMBER)  # pit props holding the roof up
                    elif r == TUNNEL[0] and lamp == 0:
                        rows[r][c] = (g["lamp"], LAMP)  # lanterns hung from the roof
                    continue
                if mined and r == TUNNEL[0] - 1 and x % 12 in (11, 0, 1):
                    rows[r][c] = (g["beam"], TIMBER)
                    continue
                # Ore is only left in rock the drill hasn't reached, and tunnel rock glows near the lanterns.
                rows[r][c] = termui.rock(x, r, g, termui.lit_by(lamp) if mined else 1, 0 if mined else 4 if r in TUNNEL else 1)

        def put(r, c, text, style):
            termui.put(rows, r, c, text, style)

        top, mid, low = TUNNEL
        # Ore cart: one nugget per unit committed this round.
        committed = sum(s.committed for s in stats)
        load = (g["ore"] * min(committed, 4)).ljust(4, "·" if g is GLYPHS else ".")
        cx = dx - 13
        put(mid, cx, g["cart"][0], CART)
        put(mid, cx + 1, load[:4], GOLD if committed else DIM)
        put(mid, cx + 5, g["cart"][1], CART)
        put(mid, cx + 6, "───" if g is GLYPHS else "---", RAIL)
        wheel = "oO"[(int(self.scroll) // 2) % 2] if busy else "o"
        put(low, cx + 1, wheel, TEXT)
        put(low, cx + 4, wheel, TEXT)
        # The drill: a body whose stripes turn while it works, and a spinning tip.
        style = DRILL if busy or any(s.round for s in stats) else DRILL_DIM
        put(top, dx - 4, g["top"] * 4, style)
        put(low, dx - 4, g["bottom"] * 4, style)
        body = [g["full"] if (i + (f if busy else 0)) % 3 else g["stripe"] for i in range(4)]
        put(mid, dx - 4, "".join(body), style)
        put(mid, dx, g["tips"][f % 3] if busy else g["tips"][0], TIP if busy else style)
        # Sparks off the rock face, gold coins after a payout.
        paid = self.popup and time.time() - self.popup[0] < 2.5
        if busy or paid:
            for i in range(7 if busy else 4):
                n = _noise(f, i)
                r, c = top - 1 + n % 5, dx + 1 + (n >> 4) % 4
                if paid and i % 2:
                    put(r, c, g["coin"], GOLD)
                else:
                    put(r, c, g["sparks"][(n >> 8) % 5], SPARKS[(n >> 12) % 4])
        if paid:
            label = f" {_vibe(self.popup[1], sign=True)} "
            put(0, max(0, dx - len(label) // 2), label, GOLD)
        return [_cells(row) for row in rows]

    # --- an Evo job's scene and lines (esui.py) ------------------------------------------------------------

    def _eg(self) -> dict:
        """The Evo scene's glyphs: ASCII with the dashboard's ASCII, else esui's own if this terminal shows them."""
        if self.g is ASCII:
            return esui.ASCII
        if self.eg is None:
            self.eg = esui.glyphs()
        return self.eg

    def _evo_state(self, lead):
        """(the Evo job on screen or None, its generations so far, the open round's generation number or None, the
        phase its block shows). A round's block is sealed once the chain reaches its height (rounds are numbered by
        the block that seals them); the open block is then a new, empty one, once its generation is in the lineage."""
        job = lead.job if lead.job is not None and getattr(lead.job, "es", False) else None
        lin = self.lineages.get(job.job_id) if job is not None else None
        gens = list(lin.gens) if lin is not None else []
        if job is None or lead.round is None:
            return job, gens, None, lead.phase
        if lead.height >= lead.round:
            return job, gens, len(gens) + 1, "next" if gens and gens[-1].height >= lead.round else "idle"
        return job, gens, len(gens) + 1, lead.phase

    def _chain(self, width: int, lead) -> list[str]:
        job, gens, number, phase = self._evo_state(lead)
        lin = self.lineages.get(job.job_id) if job is not None else None
        now = time.time()
        first = self.phase_seen.get((lead.round, lead.phase), now)
        frac = 1 - max(0.0, lead.phase_ends - now) / max(lead.phase_ends - first, 0.1)
        return esui.scene(self._eg(), width, gens, now, self.frame_no, number=number, phase=phase,
                          progress=lead.progress, frac=frac, pairs=job.spec.pairs if job is not None else 8,
                          landed=lin.landed if lin is not None else None,
                          busy=any(r.client.stats.busy for r in self.rigs),
                          waiting="waiting for a round" if job is not None else "waiting for the Evo job")

    def _evo_lines(self, add, width: int, lead, stats) -> None:
        """Generation and population, difficulty, mean reward, what the chain holds, how the model was had."""
        eg, pick = self._eg(), esui.pick
        job, gens, number, _ = self._evo_state(lead)
        if job is None:
            add(("evo       ", DIM), ("Evo mode: training by evolution, the chain is the model", TEXT))
            add(("job       ", DIM), ("waiting for the Evo job to reach a block", DIM))
            return
        spec = job.spec
        lin = self.lineages.get(job.job_id)
        mine = sum(s.assigned for s in stats) * spec.members
        gen = [("generation ", DIM), (f"{number or len(gens) + 1}", WHITE)]
        pop = [("   population ", DIM), (f"{2 * spec.pairs}", WHITE)]
        add(*pick(width, [*gen, *pop, (f" ({spec.pairs} twin pairs, {mine} yours)", DIM), ("   puzzles ", DIM),
                          (f"{spec.problems}", WHITE), (" a round", DIM)],
                  [*gen, *pop, (f" ({mine} yours)", DIM), ("   puzzles ", DIM), (f"{spec.problems}", WHITE)],
                  [*gen, *pop, ("   puzzles ", DIM), (f"{spec.problems}", WHITE)], [*gen, *pop]))
        if getattr(spec, "rule", 1) == 2:  # Evo v2 (es2.py): no level; its task, and which cycle's average it's from
            m = re.match(r"^evo2-c(\d+)-", job.job_id)
            src = (f"the mean of {len(spec.start)} earlier jobs" if len(spec.start) > 1 else
                   f"{spec.start[0][0]}" if spec.start else "the base model")
            tl = [("task      ", DIM), (spec.task, WHITE), *([(f" · cycle {m.group(1)}", TEXT)] if m else [])]
            add(*pick(width, [*tl, (f", starting from {src}", DIM)], tl))
        else:
            level = None
            for r in self.rigs:  # the round's level, from the state a worker has for it
                got = getattr(getattr(r.client, "es", None), "states", {}).get(job.job_id)
                if got is not None:
                    level = es.level_of(got[1])
                    break
            last_level = gens[-1].level if gens else None
            level = level if level is not None else last_level if last_level is not None else spec.level
            moved = ([(f" (up from {last_level})", GREEN)] if last_level is not None and level > last_level else
                     [(f" (down from {last_level})", YELLOW)] if last_level is not None and level < last_level else [])
            k, target, _ = spec.retarget
            rule = f" · moves every {k} rounds toward {target:.0%} solved" if k else " · fixed"
            lv = [("level     ", DIM), (f"{level}", WHITE), (": " + esui.level_words(level, eg), TEXT), *moved]
            add(*pick(width, [*lv, (rule, DIM)], lv))
        scored = [x for x in gens if x.mean is not None]
        if scored:
            last = scored[-1]
            delta = last.mean - scored[-2].mean if len(scored) > 1 else None
            change = ([(f" ({delta * 100:+.1f} pts)", GREEN if delta > 0 else RED if delta < 0 else DIM)]
                      if delta is not None else [])
            rew = [("reward    ", DIM), (f"{last.mean:.1%}", GOLD), *change]
            best = lin.best() if lin is not None else None
            add(*pick(width, [*rew, (f", the mean of generation {last.number}", DIM), (" · best ", DIM), (f"{best:.1%}", TEXT)],
                      [*rew, (f", generation {last.number}", DIM)], rew))
        else:
            add(*pick(width, [("reward    ", DIM), ("-", DIM), ("   the first generation's lands when its block is sealed", DIM)],
                      [("reward    ", DIM), ("-", DIM)]))
        if gens:
            total = sum(x.bytes for x in gens)
            here = [("chain     ", DIM), (esui.size(gens[-1].bytes), WHITE), (f" for generation {gens[-1].number}", DIM)]
            alls = [(" · ", DIM), (esui.size(total), WHITE), (f" for all {len(gens)}", DIM)]
            add(*pick(width, [*here, *alls, (f", which rebuild {spec.n:,} numbers", DIM)], [*here, *alls], here))
        else:
            add(("chain     ", DIM), ("seeds and scores, a few hundred bytes a generation", DIM))
        rebuilt = sum(getattr(getattr(r.client, "es", None), "rebuilt", 0) for r in self.rigs)
        fetched = sum(getattr(getattr(r.client, "es", None), "fetched", 0) for r in self.rigs)
        how = ([("rebuilt from the chain ", TEXT), (f"{rebuilt}x", WHITE), (" · fetched ", DIM), (f"{fetched}x", WHITE)]
               if rebuilt or fetched else [("not needed yet", DIM)])
        where = (f" · on {self.rigs[0].client.backend.describe()}" if len(self.rigs) == 1 else f" · {len(self.rigs)} workers")
        units = f" · {_n(sum(s.units for s in stats), 'unit')}, {_n(sum(s.checks for s in stats), 'check')}"
        add(*pick(width, [("model     ", DIM), *how, (where + units, DIM)], [("model     ", DIM), *how, (where, DIM)],
                  [("model     ", DIM), *how]))

    def _phase_bar(self, s, width: int):
        return termui.phase_bar(self.g, self.phase_seen, s.round, s.phase, s.phase_ends, width, [(f"    block {s.height}", DIM)],
                               s.progress)

    def frame(self) -> str:
        width, height = shutil.get_terminal_size((100, 32))
        w = max(40, width - 1)
        stats = [r.client.stats for r in self.rigs]
        lead = stats[0]
        evo = bool(lead.job.es) if lead.job is not None else self.evo  # an Evo job's scene (esui.py), not the drill
        busy_training = any(s.busy and s.activity.startswith(("training", "uploading")) for s in stats)
        self.scroll += 1.0 if busy_training else 0.35 if any(s.busy for s in stats) else 0.0
        self.frame_no += 1
        lines: list[str] = []
        add = lambda *segs: lines.append(_line([(" ", None), *segs], w))  # noqa: E731
        room = w - 1 - len(termui.BANNER[0]) - 3
        addr = self.rigs[0].client.keys.address
        tails = [_fit(room, f"Evo miner: training by evolution  ·  {self.chain_id}", f"Evo miner  ·  {self.chain_id}",
                      self.chain_id or "Evo miner") if evo else
                 _fit(room, f"useful-work miner  ·  {self.chain_id}", self.chain_id or "useful-work miner"),
                 _fit(room, addr, f"{addr[:10]}{'…' if self.g is GLYPHS else '..'}{addr[-4:]}") if len(self.rigs) == 1
                 else f"{len(self.rigs)} workers"]
        termui.header(add, self.g, ASCII, tails)
        lines.append("")
        lines += [" " + row for row in (self._chain(w - 2, lead) if evo else self._band(w - 2))]
        lines.append("")
        add(*self._phase_bar(lead, w))
        mark = (self.g["tips"][self.frame_no % 3] if lead.busy else self.g["tips"][0]) + " "
        add((mark, PHASES.get(lead.phase, DIM)), (lead.activity if len(self.rigs) == 1 else
             f"{sum(s.busy for s in stats)} of {len(stats)} workers busy · {lead.activity}", TEXT))
        add(*self._selector(w - 1))
        lines.append("")
        job = lead.job
        what = f"{job.job_id} ({job.spec.n_params:,} {job.spec.precision} weights)" if job else "no job yet"
        if evo:
            self._evo_lines(add, w - 1, lead, stats)
        elif len(self.rigs) == 1:
            c = self.rigs[0].client
            add(("rig       ", DIM), (c.backend.describe(), WHITE))
            add(("speed     ", DIM), (_si(lead.flops_per_sec, "FLOP/s") if lead.flops_per_sec else "measuring...", WHITE),
                (f"  on {what}", DIM))
            add(("round     ", DIM), (f"{lead.assigned}", WHITE), (" assigned · ", DIM), (f"{lead.trained}", WHITE),
                (" trained · ", DIM), (f"{lead.committed}", WHITE), (" committed · ", DIM), (f"{lead.checked}", WHITE),
                (" checked", DIM))
            add(("session   ", DIM), (_n(lead.units, "unit"), WHITE), (" · ", DIM), (_n(lead.checks, "check"), WHITE),
                (" · ", DIM), (_n(lead.rounds, "round"), WHITE), (" · ", DIM),
                (_si(lead.training_flops, "FLOP"), WHITE), (" trained", DIM))
        else:
            dev = max(8, min(34, w - 2 - 11 - 7 - 8 - 15))  # the device column gives way on narrow terminals
            add((f"{'worker':<11}{'device':<{dev}}{'units':>7}{'checks':>8}{'earned':>15}", DIM))
            for rig in self.rigs:
                s = rig.client.stats
                add((f"{rig.name:<11}", WHITE), (f"{rig.client.backend.describe()[:dev - 2]:<{dev}}", TEXT),
                    (f"{s.units:>7}{s.checks:>8}", WHITE), (f"{_vibe(rig.earned, sign=True):>15}", GREEN if rig.earned else DIM))
            add(("job       ", DIM), (what, TEXT), ("   speed ", DIM), (_si(sum(s.flops_per_sec for s in stats), "FLOP/s"), WHITE))
        lines.append("")
        earned, rig = sum(r.earned for r in self.rigs), self.rigs[0]
        rep = max(0.0, min(1.0, (rig.reputation - REP_MIN) / (REP_MAX - REP_MIN))) if rig.total is not None else 0.0
        if rig.total is None:
            add(("wallet    ", DIM), ("checking balance...", DIM))
        else:
            add(("wallet    ", DIM), (_vibe(sum(r.balance for r in self.rigs)), GOLD), ("   ", None),
                (f"{_vibe(earned, sign=True)} this session", GREEN if earned > 0 else DIM))
            add(("stake     ", DIM), (_vibe(sum(r.stake for r in self.rigs)), WHITE), ("   reputation ", DIM),
                (self.g["bar"] * round(rep * 10), GREEN), (self.g["empty"] * (10 - round(rep * 10)), DIM),
                (f" {rig.reputation:.2f}", TEXT))
        lines.append("")
        termui.feed_lines(add, self.feed, max(0, height - len(lines) - 3))
        while len(lines) < height - 2:
            lines.append("")
        add(("Ctrl+C to stop", DIM), (f"   ·   node {self.node_url}", DIM), (f"   ·   {self.footer}" if self.footer else "", DIM))
        return termui.finish(lines, height, self.g is ASCII)

    def run(self, stop: threading.Event, tick=None, on_start=None) -> None:
        """Draw until `stop` is set or Ctrl+C, calling `tick()` every frame. Worker and node log lines go to the
        event feed meanwhile; start the workers in `on_start()` so their first lines land there too. Keys pressed
        move the job selector (press), read between frames, never waiting (termui.keypresses)."""
        loggers = [logging.getLogger(n) for n in ("vibehash.worker", "vibehash.node")]
        handler, saved = termui.Feed(self), [(lg, lg.propagate) for lg in loggers]
        for lg in loggers:
            lg.addHandler(handler)
            lg.propagate = False
        threading.Thread(target=self._poll, args=(stop,), daemon=True, name="dashboard-poll").start()
        threading.Thread(target=self._poll_jobs, args=(stop,), daemon=True, name="dashboard-jobs").start()
        threading.Thread(target=self._poll_lineage, args=(stop,), daemon=True, name="dashboard-evo").start()
        try:
            with termui.alternate_screen() as out, termui.keypresses() as read:
                self.keyed = read is not None

                def every_frame():
                    for key in read() if read else ():
                        self.press(key)
                    if tick:
                        tick()
                if on_start:
                    on_start()
                termui.animate(out, self.frame, lambda: not stop.is_set(), every_frame)
        finally:
            for lg, propagate in saved:
                lg.removeHandler(handler)
                lg.propagate = propagate

    def summary(self) -> str:
        units, checks = sum(r.client.stats.units for r in self.rigs), sum(r.client.stats.checks for r in self.rigs)
        rounds = max(r.client.stats.rounds for r in self.rigs)
        mins = (time.time() - self.started) / 60
        return (f"Mined {units} units and checked {checks} results over {rounds} rounds in {mins:.0f} min; "
                f"earned {_vibe(sum(r.earned for r in self.rigs), sign=True)} this session.")
