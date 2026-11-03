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
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import architect
import judge
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

    def claim(self) -> dict | None:
        return self.post("dispatch/claim")

    def report(self, rid: str, status: str, message: str | None = None, **extra) -> None:
        print(f"[{rid[:8]}] {status}: {message or ''}", flush=True)
        self.post(f"dispatch/{rid}", {"status": status, "message": message, "machine": MACHINE, **extra})


class Terminal:
    """The site's stand-in for --goal: progress prints here, and the finished run goes where publish.py writes it."""

    def report(self, rid: str, status: str, message: str | None = None, run: dict | None = None, **extra) -> None:
        print(f"{status}: {message or ''}", flush=True)
        if run:
            publish.OUT.write_text(json.dumps(run, indent=2), encoding="utf-8")
            print(f"wrote {publish.OUT}")


def train(site, rid: str, cmd: list[str]) -> None:
    """Run train_local.py, passing its per-epoch lines on to the site as they're printed."""
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if line.startswith(("epoch ", "held-back loss")):
                site.report(rid, "training", line.strip())
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
    train(site, rid, architect.train_command(plan, chats, out))

    site.report(rid, "judging", "the base and fine-tuned models are answering the held-back prompts")
    answers = judge.answer(evals, out / "adapters" / job)
    site.report(rid, "judging", "Muse Spark is comparing the answers, blind")
    judge.grade(answers)
    result = publish.build(job, out, title)
    j = result["judge"]
    site.report(rid, "done", f"fine-tune preferred on {j['tuned_wins']} of {j['prompts']} prompts", run=result)


def main():
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
    while True:
        try:
            req = site.claim()
        except OSError as e:  # the site or the connection is down: try again later
            print(f"couldn't reach the site ({e})", flush=True)
            req = None
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
        else:
            time.sleep(a.every)


if __name__ == "__main__":
    main()
