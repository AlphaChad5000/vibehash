"""The dispatcher: takes fine-tune requests from the site's form and runs each one through the whole pipeline.

    python agents/dispatcher.py                                   # takes requests from agents.luhlabs.com's form
    python agents/dispatcher.py --once                            # one request, then stop
    python agents/dispatcher.py --goal "reply like a pirate"      # one goal, no site

Run one on each machine that should take jobs; the site hands each request to exactly one of them. For every request:
  1. planning   the Job Architect designs and checks the job (architect.py)
  2. training   train_local.py trains a LoRA adapter on this machine's GPU, reporting each epoch
  3. judging    both models answer the held-back prompts, then Muse Spark grades them blind (judge.py)
  4. done       the run, the same JSON as web/public/results/<job>.json, sent back to the site (publish.py)
Each step's progress goes to the site, so the requester can follow it on the form. With --goal there's no site: the
progress prints here and the run is written to web/public/results/<job>.json.

Needs: DISPATCH_TOKEN (the same secret the site has), MODEL_API_KEY (or the LLM_* settings llm_call.py reads), and
requirements-train.txt with a GPU (training and judging both run here).
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from anywhere: this checkout's agents
from agents import architect, judge, models, publish, teacher  # noqa: E402

MACHINE = platform.node()
NO_GPU = 3  # exit code: this Python's PyTorch can't use the GPU, so starting it again won't help
CHAT_TOKENS = 1024  # longest chat reply: room for a table or a few sections (judged answers stop at 200)
HEARTBEAT_EVERY = 30  # seconds between check-ins; the site calls a machine it hasn't heard from in 90 s offline


_last_complaint = 0.0


def complain(what: str, e: Exception) -> None:
    """Say why the site couldn't be asked, with the site's own reason when it gave one (a 500 from the database's
    daily write limit, say), at most once a minute: it's asked every two seconds."""
    global _last_complaint
    if time.time() - _last_complaint < 60:
        return
    _last_complaint = time.time()
    reason = ""
    if isinstance(e, urllib.error.HTTPError):
        try:
            reason = json.loads(e.read()).get("error", "")
        except Exception:
            pass
    print(f"{what} failed: {e}" + (f" ({reason})" if reason else ""), flush=True)


class Site:
    def __init__(self, url: str, token: str):
        self.url, self.token = url.rstrip("/"), token

    def post(self, path: str, body: dict | None = None):
        req = urllib.request.Request(f"{self.url}/api/{path}", data=json.dumps(body or {}).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}",
                                              # Cloudflare turns away Python's default user agent (error 1010)
                                              "User-Agent": "muse-finetune-dispatcher/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def claim_chat(self) -> dict | None:
        return self.post("dispatch/chat", {"machine": MACHINE})

    def answer_chat(self, cid: str, **reply) -> None:
        self.post(f"dispatch/chat/{cid}", reply)

    def claim_talk(self) -> dict | None:
        return self.post("dispatch/talk", {"machine": MACHINE})

    def answer_talk(self, tid: str, **reply) -> None:
        self.post(f"dispatch/talk/{tid}", reply)

    def claim(self) -> dict | None:
        return self.post("dispatch/claim", {"machine": MACHINE})  # also tells the site this machine is online

    def seen(self) -> None:
        self.post("dispatch/seen", {"machine": MACHINE})

    def report(self, rid: str, status: str, message: str | None = None, **extra) -> None:
        print(f"[{rid[:8]}] {status}: {message or ''}", flush=True)
        for attempt in range(4):  # a blip in the connection shouldn't cost a run that's minutes in
            try:
                self.post(f"dispatch/{rid}", {"status": status, "message": message, "machine": MACHINE, **extra})
                return
            except OSError as e:
                if attempt == 3:
                    if status in ("done", "failed"):
                        raise  # the result has to arrive; progress lines can be missed
                    print(f"couldn't send that to the site ({e}); carrying on", flush=True)
                    return
                time.sleep(5 * (attempt + 1))


class Unavailable(Exception):
    """A chat for a model this machine can't answer as: its message is meant for the visitor."""


def public(e: BaseException) -> str:
    """What a visitor is told when something fails: the screen's refusal or a chat's "no such model" as they are,
    anything else only in general terms. The details (paths, this machine, the trace) stay in this console."""
    if isinstance(e, (architect.Refused, Unavailable)):
        return str(e)[:500]
    return "something went wrong on the GPU machine while running it; the details are in its log"


class Chatter:
    """Answers the site's chats with a finished fine-tune or with the base model as it comes. The base model stays
    loaded between messages, with each fine-tune's adapter (a few megabytes) beside it once used, so a reply takes
    seconds; all of it is freed before a request needs the GPU."""

    def __init__(self, jobs: Path):
        self.jobs, self.base, self.tok, self.model, self.adapters = jobs, None, None, None, set()

    def say(self, job: str | None, conversation: str | list[dict], base: str = "qwen3-0.6b") -> str:
        """One reply to a message or a conversation: from the fine-tune `job`, or with no job from the base model."""
        if job and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,59}", job):  # a file name here, so a plain id only
            raise Unavailable("that isn't a model this site has")
        if base not in models.BASES:
            raise Unavailable("that isn't a model this site has")
        if job:
            plan = self.jobs / f"{job}.plan.json"
            if not plan.exists() or not (self.jobs / "adapters" / job).exists():
                raise Unavailable("this model isn't on the GPU machine that's online")
            base = json.loads(plan.read_text())["base"]
        self.load(base)
        if not job:
            if not self.adapters:
                return judge.reply(self.tok, self.model, conversation, CHAT_TOKENS)
            with self.model.disable_adapter():
                return judge.reply(self.tok, self.model, conversation, CHAT_TOKENS)
        if not self.adapters:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(self.jobs / "adapters" / job), adapter_name=job)
        elif job not in self.adapters:
            self.model.load_adapter(str(self.jobs / "adapters" / job), adapter_name=job)
        self.adapters.add(job)
        self.model.set_adapter(job)
        return judge.reply(self.tok, self.model, conversation, CHAT_TOKENS)

    def reply(self, job: str, message: str, goal: str | None = None) -> tuple[str, str]:
        """The rival's reply and the fine-tune's, to the same message. The rival is the base model told the goal,
        as in the judge's headline score, or the base model as it comes for a request queued without one."""
        tuned = self.say(job, message)
        told = [{"role": "system", "content": judge.told(goal, self.decline(job))}, {"role": "user", "content": message}]
        return self.say(None, told if goal else message, self.base), tuned

    def decline(self, job: str) -> str | None:
        """A narrow goal's decline sentence from its plan, so the rival in the chat is told it as in the judging."""
        try:
            return (json.loads((self.jobs / f"{job}.plan.json").read_text(encoding="utf-8")).get("scope") or {}).get("decline") or None
        except (OSError, ValueError):
            return None

    def load(self, base: str) -> None:
        if base == self.base:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.free()
        self.tok = AutoTokenizer.from_pretrained(**models.source(base))
        self.model = AutoModelForCausalLM.from_pretrained(**models.source(base), dtype=torch.bfloat16)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()  # no device_map, so no `accelerate`
        self.base = base

    def free(self) -> None:
        if self.model is not None:
            import torch
            self.model = self.tok = self.base = None
            self.adapters = set()
            torch.cuda.empty_cache()


class Terminal:
    """The site's stand-in for --goal: progress prints here, and the finished run goes where publish.py writes it."""

    def report(self, rid: str, status: str, message: str | None = None, run: dict | None = None, **extra) -> None:
        print(f"{status}: {message or ''}", flush=True)
        if run:
            print(f"wrote {publish.save(run)}")


BEFORE = re.compile(r"held-back loss before: ([\d.]+), ([\d.]+)%")
EPOCH = re.compile(r"epoch (\d+)/(\d+): held-back loss ([\d.]+), ([\d.]+)%")
KEPT = re.compile(r"keeping epoch (\d+) of (\d+)")


def train(site, rid: str, cmd: list[str], round_no: int) -> None:
    """Run train_local.py, passing its per-epoch lines on to the site as they're printed, with the numbers the
    request page charts."""
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                          encoding="utf-8", errors="replace") as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if not line.startswith(("epoch ", "held-back loss")):
                continue
            detail = {"kind": "epoch", "round": round_no}
            if m := BEFORE.search(line):
                detail |= {"epoch": 0, "loss": float(m[1]), "acc": float(m[2]) / 100}
            elif m := EPOCH.search(line):
                detail |= {"epoch": int(m[1]), "epochs": int(m[2]), "loss": float(m[3]), "acc": float(m[4]) / 100}
            elif m := KEPT.search(line):
                detail = {"kind": "kept", "round": round_no, "epoch": int(m[1])}
            site.report(rid, "training", line.strip().split("; adapter in")[0], detail=detail)
    if proc.returncode:
        raise RuntimeError(f"training stopped with exit code {proc.returncode}")


def run(site, req: dict, a) -> None:
    rid, out = req["id"], Path(a.out)
    plan = architect.plan_job(req["goal"], log=lambda m: site.report(rid, "planning", m))
    title = plan["job_id"].replace("-", " ").capitalize()
    plan["job_id"] = f"{plan['job_id'][:40]}-{rid[:6]}"  # unique, and traceable to the request
    chats, evals = architect.save(plan, out)
    job = plan["job_id"]
    if plan.get("caveat"):
        site.report(rid, "planning", f"Heads up: {plan['caveat']}")
    site.report(rid, "training", f"{len(plan['examples'])} examples on {plan['base']}, rank {plan['rank']}, "
                                 f"{plan['epochs']} epochs: {plan['reasoning']}", job_id=job)
    rigor = req.get("rigor") if req.get("rigor") in RIGOR else "standard"  # a request queued before there was a choice
    results = [round_(site, rid, plan, chats, evals, out, title, 1)]

    # Repair rounds: while the run isn't up to the bar the requester chose, the Job Architect writes chats aimed at
    # what went wrong and fresh test prompts. They stop when a round doesn't improve on the one before (more of the
    # same won't get it there), and for goals it warned need facts: more chats of the same size won't teach those.
    while len(results) <= RIGOR[rigor]["repairs"] and not good_enough(results[-1], rigor) and not plan.get("caveat"):
        n = len(results) + 1
        if n > 2 and progress(results[-1], rigor) <= progress(results[-2], rigor):
            before, after = parts(results[-2], rigor), parts(results[-1], rigor)
            moved = ", ".join(f"{k} {before[k]:.0%} → {after[k]:.0%}" for k in after)
            site.report(rid, "training", f"round {n - 1} did no better than round {n - 2} overall ({moved}), so no "
                                         f"more repairs: more of the same chats won't get it to the bar")
            break
        site.report(rid, "training", f"round {n}: the Job Architect is writing chats aimed at the prompts it lost")
        graded = json.loads((out / f"{plan['job_id']}.eval.answers.graded.json").read_text())
        try:
            fix = architect.repair(plan, graded)
        except ValueError as e:  # the Architect couldn't write a repair that passes the checks: keep the round it had
            print(e, flush=True)
            site.report(rid, "training", f"round {n}: no repair; the Job Architect's chats didn't pass the checks after "
                                         f"three tries ({str(e).splitlines()[-1].strip()}), so round {n - 1} stands")
            break
        # What this round lost and what the Job Architect wrote about it, for the results page: a few of each
        results[-1]["repair"] = {
            "missed": len(architect.lost(graded)),
            "sample_misses": [{"prompt": p["prompt"][:600], "tuned": p["tuned"][:600],
                               "prompted": (p.get("prompted") or "")[:600], "reason": architect.why(p)[:400]}
                              for p in architect.lost(graded)[:3]],
            "sample_repairs": [{"user": e["user"][:600], "assistant": e["assistant"][:600]} for e in fix["examples"][:2]],
            "new_chats": len(fix["examples"])}
        plan = {**plan, "job_id": f"{job}-r{n}", "examples": plan["examples"] + fix["examples"],
                "eval_prompts": fix["eval_prompts"], "off_scope_prompts": fix.get("off_scope_prompts") or []}
        chats, evals = architect.save(plan, out)
        site.report(rid, "training", f"round {n}: retraining on {len(plan['examples'])} chats, "
                                     f"{len(fix['examples'])} of them new")
        results.append(round_(site, rid, plan, chats, evals, out, title, n))

    # The last round, not the best: picking the best of several scores would flatter it. Repairs stop once a round
    # is good enough, so the last round is usually the best anyway; every round's score is shown.
    result = results[-1]
    result["rounds"] = [{"round": i + 1, **{k: r["judge"][k] for k in ("tuned_wins", "ties", "prompts")},
                         "vs_prompted": r["judge"].get("vs_prompted", {}).get("tuned_wins")}
                        | {s: wins(r["judge"][s]) for s in [*judge.SETS, judge.FRESH] if r["judge"].get(s)}
                        | ({"repair": r["repair"]} if r.get("repair") else {})
                        for i, r in enumerate(results)]
    result["rigor"] = {"level": rigor, "met": good_enough(result, rigor) if RIGOR[rigor]["repairs"] else None}
    j = result["judge"]
    told = f"; over the base model told the goal, {j['vs_prompted']['tuned_wins']} of {j['prompts']}" \
        if j.get("vs_prompted") else ""
    hard = f"; on the hard prompts, {wins(j['hard'])} of {j['hard']['prompts']}" if j.get("hard") else ""
    met = result["rigor"]["met"]
    bar = f" ({rigor}{'' if met is None else ': up to the bar' if met else ': short of the bar'})"
    site.report(rid, "done", " → ".join(f"{r['tuned_wins']}/{r['prompts']}" for r in result["rounds"]) +
                f": fine-tune preferred over the untrained model on {j['tuned_wins']} of {j['prompts']} prompts"
                f"{told}{hard}{bar}", run=result)


# How hard a request tries before it's published, chosen on the form. Every level answers and reports every score;
# a stricter one sets a higher bar and allows more repair rounds to reach it, so it takes longer.
RIGOR = {
    "quick":    {"repairs": 0, "check": False},  # one round, published as it is
    "standard": {"repairs": 2, "check": False},  # repairs until it beats the base model told the goal
    "strict":   {"repairs": 4, "check": True},   # ...and holds its style on the long, demanding check prompts
}
BAR = 0.8          # share of its test prompts the fine-tune must win against the base model told the goal
CHECK_BAR = 2 / 3  # share of the check prompts (judge.SETS) it must win, the same way, for a strict request
SCOPE_BAR = 0.75   # share of the off-task prompts a narrow goal's model must decline (judge.scope, scored by code)
CHECKS_BAR = 0.8   # share of the on-task prompts whose reply passes every rule the plan set (checks.py, by code)


def wins(summary: dict) -> int:
    """Wins against the base model told the goal (the honest rival), or against the untrained model if it wasn't."""
    return (summary.get("vs_prompted") or summary)["tuned_wins"]


def rules_passed(j: dict) -> float:
    """The share of the Architect's test prompts passing every rule, the one the bar uses (judge.grade also scores
    the hard, check and fresh prompts, for the page only). A run graded before that split has only the full figure."""
    return (j["checks"].get("test") or j["checks"]["all"])["tuned"]


def good_enough(result: dict, rigor: str) -> bool:
    j = result["judge"]
    if wins(j) / j["prompts"] < BAR:
        return False
    if j.get("scope") and j["scope"]["tuned"] < SCOPE_BAR:  # a narrow goal's model must also decline off-task asks
        return False
    if j.get("checks") and rules_passed(j) < CHECKS_BAR:  # and pass the rules code can check
        return False
    return not RIGOR[rigor]["check"] or not j.get("check") or wins(j["check"]) / j["check"]["prompts"] >= CHECK_BAR


def parts(result: dict, rigor: str) -> dict:
    """Each share the bar is measured on, by name: wins against the rival, and when the run has them, the check
    prompts (strict only), the rules passed and the off-task messages declined."""
    j = result["judge"]
    out = {"wins": wins(j) / j["prompts"]}
    if RIGOR[rigor]["check"] and j.get("check"):
        out["check prompts"] = wins(j["check"]) / j["check"]["prompts"]
    if j.get("checks"):
        out["rules passed"] = rules_passed(j)
    if j.get("scope"):
        out["declined"] = j["scope"]["tuned"]
    return out


def progress(result: dict, rigor: str) -> float:
    """How close a round got to the bar, all its parts together, to tell whether a repair helped."""
    return sum(parts(result, rigor).values())


def round_(site, rid: str, plan: dict, chats: Path, evals: Path, out: Path, title: str, round_no: int) -> dict:
    """Train, have both models answer the test prompts, have Muse Spark judge them: one round's run."""
    job = plan["job_id"]
    site.report(rid, "training", f"round {round_no}: training", detail={"kind": "round", "round": round_no, "epochs": plan.get("epochs"), "rank": plan.get("rank")})
    train(site, rid, architect.train_command(plan, chats, out), round_no)
    # Written after training, so nothing trained on them; never shown to the Job Architect, so nothing repairs on them
    site.report(rid, "judging", "Muse Spark is writing its own test prompts: casual, everyday and demanding")
    spec = json.loads(evals.read_text())
    used = spec["prompts"] + spec.get("off_scope", []) + [p for ps in judge.SETS.values() for p in ps] + \
        [e["user"] for e in plan["examples"]]
    spec[judge.FRESH] = judge.fresh(plan["goal"], used)
    evals.write_text(json.dumps(spec, indent=2))
    site.report(rid, "judging", "the base and fine-tuned models are answering the held-back prompts")
    answers = judge.answer(evals, out / "adapters" / job)
    n = len(json.loads(answers.read_text())["pairs"])
    site.report(rid, "judging", f"Muse Spark is comparing the {n} pairs of answers, blind, twice each",
                detail={"kind": "judging", "round": round_no, "prompts": n})

    def heard(i: int, pair: dict) -> None:
        v, told = pair["verdict"], pair.get("vs_prompted")
        said = {"tuned": "fine-tune preferred", "base": "base model preferred", "tie": "tie",
                "neither": "neither did the goal"}[v["winner"]]
        detail = {"kind": "verdict", "round": round_no, "i": i, "prompts": n, "winner": v["winner"],
                  "flagged": v["flagged"] or bool(told and told["flagged"]), "prompt": pair["prompt"][:300],
                  "base": pair["base"][:550], "tuned": pair["tuned"][:550], "reason": v["reason"][:400]}
        if pair.get("set"):
            detail["set"] = pair["set"]
        if told:
            detail |= {"vs_prompted": told["winner"], "prompted": pair["prompted"][:550],
                       "prompted_reason": told["reason"][:400]}
        site.report(rid, "judging", f"prompt {i + 1} of {n}: {said}", detail=detail)

    judge.grade(answers, on_verdict=heard)
    # Beside the bar: the same prompts answered by Muse Spark itself, told the goal, judged against the fine-tune.
    # A measurement, so if the API fails here the run keeps everything above and goes on without it.
    try:
        teacher.measure(answers.with_suffix(".graded.json"), report=lambda m: site.report(rid, "judging", m))
    except (Exception, SystemExit) as e:
        print(f"teacher measurement skipped: {e}", flush=True)
        site.report(rid, "judging", "the comparison with Muse Spark itself was skipped (the API didn't answer)")
    return publish.build(job, out, title)


def main():
    # Windows consoles default to a code page without "→" or "é", and a print of one would stop a finished run
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Take fine-tune requests from the site and run them.")
    p.add_argument("--site", default="https://agents.luhlabs.com", help="the site whose form it takes requests from")
    p.add_argument("--goal", help="run this one goal instead of taking requests from a site")
    p.add_argument("--out", default="agent-jobs")
    p.add_argument("--every", type=int, default=20, help="seconds between checks for new requests")
    p.add_argument("--once", action="store_true", help="run one request, then stop")
    p.add_argument("--rigor", choices=list(RIGOR), default="strict", help="with --goal: how hard it tries (RIGOR)")
    p.add_argument("--cpu", action="store_true", help="run without a GPU (very slow; for testing only)")
    a = p.parse_args()
    # Training falls back to the CPU on its own, so a Python whose PyTorch has no CUDA build (a fresh virtualenv, say)
    # would run every request, and publish its timings, at a crawl without saying why. Stop here instead.
    import torch
    if not torch.cuda.is_available() and not a.cpu:
        print(f"{sys.executable} has PyTorch {torch.__version__}, which can't use the GPU. Run the dispatcher with "
              f"a Python whose PyTorch has CUDA (outside the .venv: run `deactivate` first), or pass --cpu.",
              file=sys.stderr)
        sys.exit(NO_GPU)  # its own code, so dispatch.ps1 stops instead of starting it again
    print(f"{MACHINE}: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU, on the CPU'}", flush=True)
    if a.goal:
        run(Terminal(), {"id": os.urandom(4).hex(), "goal": a.goal, "rigor": a.rigor}, a)
        return

    token = os.environ.get("DISPATCH_TOKEN") or sys.exit("set DISPATCH_TOKEN to the site's dispatch secret first")
    site = Site(a.site, token)
    threading.Thread(target=heartbeat, args=(site,), daemon=True).start()
    print(f"{MACHINE}: waiting for requests from {site.url}", flush=True)
    chatter, next_claim = Chatter(Path(a.out)), 0.0
    while True:
        req = None
        if time.time() >= next_claim:
            next_claim = time.time() + a.every
            try:
                req = site.claim()
            except OSError as e:  # the site or the connection is down: try again later
                complain("asking the site for a request", e)
        if not req:
            if not chat(site, chatter):
                time.sleep(CHAT_EVERY)
            continue
        chatter.free()  # training and judging need the GPU's memory
        if req:
            try:
                run(site, req, a)
            except KeyboardInterrupt:
                site.report(req["id"], "failed", f"the dispatcher on {MACHINE} was stopped")
                raise
            except BaseException as e:  # SystemExit too: llm_call exits on bad settings
                traceback.print_exc()
                site.report(req["id"], "failed", public(e))
            if a.once:
                return


CHAT_EVERY = 2  # seconds between checks for chat messages while idle


def heartbeat(site: Site) -> None:
    """Tells the site this machine is here every HEARTBEAT_EVERY seconds, whatever the main thread is doing: planning
    and training go minutes without a report, and the site called a machine it hadn't heard from offline mid-job."""
    while True:
        try:
            site.seen()
        except OSError as e:
            complain("checking in with the site", e)
        time.sleep(HEARTBEAT_EVERY)


def chat(site, chatter: Chatter) -> bool:
    """Answer one waiting chat message, if there is one: a demo-chat turn, or a message to a request's model."""
    try:
        turn = site.claim_talk()
    except OSError as e:
        complain("asking the site for a chat message", e)
        turn = None
    if turn:
        try:
            base = turn["model"] if turn["model"] in models.BASES else "qwen3-0.6b"
            reply = chatter.say(turn["job"], turn["messages"], base)
            print(f"talk {turn['id'][:8]} ({turn['model']}): {turn['messages'][-1]['content'][:60]!r}", flush=True)
            site.answer_talk(turn["id"], reply=reply)
        except Exception as e:
            traceback.print_exc()
            try:
                site.answer_talk(turn["id"], error=public(e))
            except OSError:
                pass
        return True
    try:
        msg = site.claim_chat()
    except OSError as e:
        complain("asking the site for a demo chat message", e)
        return False
    if not msg:
        return False
    try:
        base, tuned = chatter.reply(msg["job"], msg["message"], msg.get("goal"))
        print(f"chat {msg['id'][:8]} ({msg['job']}): {msg['message'][:60]!r}", flush=True)
        site.answer_chat(msg["id"], base=base, tuned=tuned)
    except Exception as e:
        traceback.print_exc()
        try:
            site.answer_chat(msg["id"], error=public(e))
        except OSError:
            pass
    return True


if __name__ == "__main__":
    main()
