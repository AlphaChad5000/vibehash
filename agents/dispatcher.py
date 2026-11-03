"""The dispatcher: takes fine-tune requests from the site's form and runs each one through the whole pipeline.

    python agents/dispatcher.py                                   # takes requests from agents.luhlabs.com's form
    python agents/dispatcher.py --once                            # one request, then stop
    python agents/dispatcher.py --goal "reply like a pirate"      # one goal, no site

Run one on each machine that should take jobs; the site hands each request to exactly one of them. For every request:
  1. planning   the Job Architect designs and checks the job (architect.py)
  2. training   train_local.py trains a LoRA adapter on this machine's GPU, reporting each epoch
  3. judging    both models answer the held-back prompts, then Muse Spark grades them blind (judge.py)
  4. done       the run, the same JSON as web/public/results/run.json, sent back to the site (publish.py)
Each step's progress goes to the site, so the requester can follow it on the form. With --goal there's no site: the
progress prints here and the run is written to web/public/results/run.json.

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
import time
import traceback
import urllib.request
from pathlib import Path

import architect
import judge
import models
import publish

MACHINE = platform.node()


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

    def claim(self) -> dict | None:
        return self.post("dispatch/claim", {"machine": MACHINE})  # also tells the site this machine is online

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


class Chatter:
    """Answers the site's chat messages: the base model and a finished fine-tune, one reply each. The base model
    stays loaded between messages (a reply then takes seconds) and is freed before a request needs the GPU."""

    def __init__(self, jobs: Path):
        self.jobs, self.base, self.tok, self.model, self.adapters = jobs, None, None, None, set()

    def reply(self, job: str, message: str) -> tuple[str, str]:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        plan = self.jobs / f"{job}.plan.json"
        adapter = self.jobs / "adapters" / job
        if not plan.exists() or not adapter.exists():
            raise FileNotFoundError(f"this machine doesn't have the model {job} (it was trained elsewhere)")
        base = json.loads(plan.read_text())["base"]
        if base != self.base:
            self.free()
            repo = models.BASES[base]
            self.tok = AutoTokenizer.from_pretrained(repo)
            loaded = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")
            self.model, self.base = PeftModel.from_pretrained(loaded, str(adapter), adapter_name=job), base
            self.adapters = {job}
        elif job not in self.adapters:
            self.model.load_adapter(str(adapter), adapter_name=job)  # a few megabytes next to the same base
            self.adapters.add(job)
        self.model.set_adapter(job)
        tuned = judge.reply(self.tok, self.model, message)
        with self.model.disable_adapter():
            base_reply = judge.reply(self.tok, self.model, message)
        return base_reply, tuned

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
            publish.OUT.write_text(json.dumps(run, indent=2), encoding="utf-8")
            print(f"wrote {publish.OUT}")


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
    results = [round_(site, rid, plan, chats, evals, out, title, 1)]

    # Repair rounds: while the judge isn't convinced, the Job Architect writes chats aimed at what went wrong and
    # fresh test prompts. Not for goals it warned need facts: more chats of the same size won't teach those.
    while len(results) <= REPAIRS and not usable(results[-1]) and not plan.get("caveat"):
        n = len(results) + 1
        site.report(rid, "training", f"round {n}: the Job Architect is writing chats aimed at the prompts it lost")
        graded = json.loads((out / f"{plan['job_id']}.eval.answers.graded.json").read_text())
        fix = architect.repair(plan, graded)
        plan = {**plan, "job_id": f"{job}-r{n}", "examples": plan["examples"] + fix["examples"],
                "eval_prompts": fix["eval_prompts"]}
        chats, evals = architect.save(plan, out)
        site.report(rid, "training", f"round {n}: retraining on {len(plan['examples'])} chats, "
                                     f"{len(fix['examples'])} of them new")
        results.append(round_(site, rid, plan, chats, evals, out, title, n))

    result = max(results, key=score)  # each round was judged on its own new prompts
    result["rounds"] = [{"round": i + 1, **{k: r["judge"][k] for k in ("tuned_wins", "ties", "prompts")}}
                        for i, r in enumerate(results)]
    j = result["judge"]
    site.report(rid, "done", " → ".join(f"{r['tuned_wins']}/{r['prompts']}" for r in result["rounds"]) +
                f": fine-tune preferred on {j['tuned_wins']} of {j['prompts']} prompts", run=result)


REPAIRS = 2     # at most this many repair rounds after the first, so a request finishes in minutes
USABLE = 0.8    # the judge preferring the fine-tune on this share of prompts ends the repairs


def score(result: dict) -> float:
    return result["judge"]["tuned_wins"] / result["judge"]["prompts"]


def usable(result: dict) -> bool:
    return score(result) >= USABLE


def round_(site, rid: str, plan: dict, chats: Path, evals: Path, out: Path, title: str, round_no: int) -> dict:
    """Train, have both models answer the test prompts, have Muse Spark judge them: one round's run."""
    job = plan["job_id"]
    site.report(rid, "training", f"round {round_no}: training", detail={"kind": "round", "round": round_no})
    train(site, rid, architect.train_command(plan, chats, out), round_no)
    site.report(rid, "judging", "the base and fine-tuned models are answering the held-back prompts")
    answers = judge.answer(evals, out / "adapters" / job)
    n = len(json.loads(answers.read_text())["pairs"])
    site.report(rid, "judging", f"Muse Spark is comparing the {n} pairs of answers, blind, twice each",
                detail={"kind": "judging", "round": round_no, "prompts": n})

    def heard(i: int, pair: dict) -> None:
        v = pair["verdict"]
        said = {"tuned": "fine-tune preferred", "base": "base model preferred", "tie": "tie"}[v["winner"]]
        site.report(rid, "judging", f"prompt {i + 1} of {n}: {said}",
                    detail={"kind": "verdict", "round": round_no, "i": i, "prompts": n, "winner": v["winner"],
                            "prompt": pair["prompt"][:300], "base": pair["base"][:700], "tuned": pair["tuned"][:700],
                            "reason": v["reason"][:500]})

    judge.grade(answers, on_verdict=heard)
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
    a = p.parse_args()
    if a.goal:
        run(Terminal(), {"id": os.urandom(4).hex(), "goal": a.goal}, a)
        return

    token = os.environ.get("DISPATCH_TOKEN") or sys.exit("set DISPATCH_TOKEN to the site's dispatch secret first")
    site = Site(a.site, token)
    print(f"{MACHINE}: waiting for requests from {site.url}", flush=True)
    chatter, next_claim = Chatter(Path(a.out)), 0.0
    while True:
        req = None
        if time.time() >= next_claim:
            next_claim = time.time() + a.every
            try:
                req = site.claim()
            except OSError as e:  # the site or the connection is down: try again later
                print(f"couldn't reach the site ({e})", flush=True)
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
                site.report(req["id"], "failed", f"{type(e).__name__}: {e}"[:500])
            if a.once:
                return


CHAT_EVERY = 2  # seconds between checks for chat messages while idle


def chat(site, chatter: Chatter) -> bool:
    """Answer one waiting chat message, if there is one."""
    try:
        msg = site.claim_chat()
    except OSError:
        return False
    if not msg:
        return False
    try:
        base, tuned = chatter.reply(msg["job"], msg["message"])
        print(f"chat {msg['id'][:8]} ({msg['job']}): {msg['message'][:60]!r}", flush=True)
        site.answer_chat(msg["id"], base=base, tuned=tuned)
    except Exception as e:
        traceback.print_exc()
        try:
            site.answer_chat(msg["id"], error=f"{type(e).__name__}: {e}"[:300])
        except OSError:
            pass
    return True


if __name__ == "__main__":
    main()
