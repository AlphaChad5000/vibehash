"""Command line: setup, wallets, a node, a worker, transactions, and chain auditing.

    python -m vibehash setup                 # node address, wallet, GPU support and stake, step by step
    python -m vibehash worker                # then start earning
    python -m vibehash testnet               # or run a whole network on this computer

    python -m vibehash wallet create --out me.json
    python -m vibehash genesis create --chain-id my-testnet --alloc vibe1...=1000 --stake vibe1...=10 \
        --validator vibe1...=http://10.0.0.5:8545 --out genesis.json
    python -m vibehash node --genesis genesis.json --wallet node.json --data-dir node-data [--peer URL]
    python -m vibehash validator join --url http://my-public-address:8545
    python -m vibehash stake --amount 10
    python -m vibehash job template --out job.json
    python -m vibehash post-job --spec job.json --budget 300 [--dataset data.npz]
    python -m vibehash chain verify --genesis genesis.json

--node and --wallet default to what `setup` saved (see settings.py). Wallet passwords are read from
the terminal, or from $VIBEHASH_PASSWORD for unattended nodes.
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import re
import secrets
import shutil
import socket
import sys
import threading
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlsplit

from . import crypto, hardware, mnemonic, settings, tls
from . import economics as econ
from . import fees
from . import wallet as wallets
from .client import NodeClient, NodeError, WorkerClient
from .crypto import KeyPair
from .ledger import Block, Chain, canonical
from .model import ModelSpec
from .tasks import SYN_PER_VIBE, InvalidJob, TrainingJob


def to_syn(text: str) -> int:
    try:
        v = Decimal(text) * SYN_PER_VIBE
    except InvalidOperation:
        raise argparse.ArgumentTypeError(f"not an amount: {text}")
    if v < 0 or v != v.to_integral_value():
        raise argparse.ArgumentTypeError("amount must be non-negative with at most 8 decimals")
    return int(v)


def fmt(syn: int) -> str:
    """12.5 VIBE, not 12.50000000 VIBE; exact to the last syn."""
    return f"{(Decimal(syn) / SYN_PER_VIBE).normalize():,f} VIBE"


def password(prompt: str, confirm: bool = False) -> str | None:
    if "VIBEHASH_PASSWORD" in os.environ:
        return os.environ["VIBEHASH_PASSWORD"]
    if not sys.stdin.isatty():
        return None
    pw = getpass.getpass(prompt)
    if confirm and pw != getpass.getpass("Repeat password: "):
        sys.exit("passwords don't match")
    return pw


def load_wallet(path: str) -> KeyPair:
    pw = password(f"Password for {path}: ") if wallets.is_encrypted(path) else None
    return wallets.load(path, pw)


def prog() -> str:
    """How we were started, so printed next steps can be pasted as they are."""
    if Path(sys.argv[0]).stem.lower() == "vibehash":
        return "vibehash"
    return ("python" if os.name == "nt" else "python3") + " -m vibehash"


def ask(question: str, assume_yes: bool = False) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        return False
    return input(f"{question} [Y/n] ").strip().lower() in ("", "y", "yes")


def node_of(a) -> str:
    """--node, the node setup saved, or the public network's."""
    return settings.node_url(a.node)


def wallet_of(a) -> str:
    path = settings.wallet_path(a.wallet)
    if not path:
        sys.exit(f"no wallet: run `{prog()} setup`, or add --wallet FILE")
    return str(path)


def normalise_url(text: str, https: bool = False) -> str:
    """'192.168.1.20' -> 'http://192.168.1.20:8545' (https:// with `https`). An address typed with a scheme is kept."""
    text = text.strip().rstrip("/")
    if "://" in text:
        return text
    if text == urlsplit(settings.PUBLIC_NODE).hostname:  # the public node serves HTTPS on the usual port
        return settings.PUBLIC_NODE
    scheme = "https" if https else "http"
    return f"{scheme}://{text}" if urlsplit(f"http://{text}").port else f"{scheme}://{text}:8545"


def _kv(items: list[str]) -> dict[str, int]:
    out = {}
    for item in items or []:
        addr, _, amount = item.partition("=")
        if not addr.startswith("vibe1") or not amount:
            sys.exit(f"expected ADDRESS=AMOUNT, got {item!r}")
        out[addr] = to_syn(amount)
    return out


def _validators(items: list[str]) -> dict[str, str]:
    from .ledger import valid_endpoint

    out = {}
    for item in items or []:
        addr, _, url = item.partition("=")
        if not addr.startswith("vibe1") or not valid_endpoint(normalise_url(url) if url else ""):
            sys.exit(f"expected ADDRESS=URL (the address other nodes reach that validator at), got {item!r}")
        out[addr] = normalise_url(url)
    return out


# --- commands ----------------------------------------------------------------------------

def show_words(words: str, indent: str = "") -> None:
    """Recovery words, numbered down the columns the way people copy them, with what they're for."""
    w = words.split()
    rows = (len(w) + 3) // 4
    print(f"\n{indent}Your {len(w)} recovery words. Write them down in order and keep them offline: they restore this\n"
          f"{indent}wallet if the file or its password is ever lost ({prog()} wallet restore), and anyone who sees\n"
          f"{indent}them can take its coins. They are not saved anywhere and won't be shown again.\n")
    for r in range(rows):
        print(indent + "  " + "".join(f"{i + 1:>3}. {w[i]:<10}" for i in range(r, len(w), rows)).rstrip())
    print()


def _read_words() -> str:
    if "VIBEHASH_MNEMONIC" in os.environ:
        return os.environ["VIBEHASH_MNEMONIC"]
    if sys.stdin.isatty():
        return getpass.getpass("Recovery words, separated by spaces (they aren't shown as you type): ")
    return sys.stdin.readline()


def _passphrase(a, confirm: bool) -> str:
    """The optional BIP39 passphrase: an extra secret on top of the words. Different passphrase, different wallet."""
    if not getattr(a, "passphrase", False):
        return ""
    if "VIBEHASH_MNEMONIC_PASSPHRASE" in os.environ:
        return os.environ["VIBEHASH_MNEMONIC_PASSPHRASE"]
    if not sys.stdin.isatty():
        sys.exit("--passphrase needs a terminal, or set $VIBEHASH_MNEMONIC_PASSPHRASE")
    phrase = getpass.getpass("Recovery passphrase: ")
    if confirm and phrase != getpass.getpass("Repeat passphrase: "):
        sys.exit("passphrases don't match")
    return phrase


def _new_wallet_password(a) -> str | None:
    pw = None if a.no_password else password("New wallet password: ", confirm=True)
    if not a.no_password and not pw:
        sys.exit("a password is required (or pass --no-password for a throwaway test key)")
    return pw


def cmd_wallet_create(a):
    out = Path(a.out or settings.default_wallet())
    if out.exists():
        sys.exit(f"{out} already exists; refusing to overwrite a wallet")
    words = mnemonic.generate()
    keys = mnemonic.keypair(words, _passphrase(a, confirm=True))
    pw = _new_wallet_password(a)
    wallets.save(keys, out, pw, mnemonic.PATH)
    print(f"created {out}\naddress: {keys.address}")
    show_words(words)
    if not pw:
        print("warning: stored unencrypted", file=sys.stderr)


def cmd_wallet_restore(a):
    """Rebuild a wallet file from its recovery words (and passphrase, if it had one)."""
    out = Path(a.out or settings.default_wallet())
    if out.exists():
        sys.exit(f"{out} already exists; refusing to overwrite a wallet (restore with --out NEW-FILE)")
    keys = mnemonic.keypair(mnemonic.check(_read_words()), _passphrase(a, confirm=False), a.path)
    pw = _new_wallet_password(a)
    wallets.save(keys, out, pw, a.path)
    print(f"restored {out}\naddress: {keys.address}")
    if not pw:
        print("warning: stored unencrypted", file=sys.stderr)


def cmd_wallet_show(a):
    path = wallet_of(a)
    doc = json.loads(Path(path).read_text())
    hd = wallets.hd_path(path)
    print(f"address: {doc['address']}\nencrypted: {'crypto' in doc}\n"
          f"recovery words: {f'yes (path {hd})' if hd else 'no (made before recovery words; back up the file)'}")


def cmd_genesis_create(a):
    g = {"chain_id": a.chain_id, "timestamp": int(time.time()), "allocations": _kv(a.alloc), "stakes": _kv(a.stake)}
    if a.bootstrap:
        g["bootstrap"] = _kv(a.bootstrap)
    if a.validator:
        g["validators"] = _validators(a.validator)
        stake = lambda v: g["stakes"].get(v, 0) + g.get("bootstrap", {}).get(v, 0)  # noqa: E731
        unstaked = [v for v in g["validators"] if stake(v) < econ.MIN_STAKE]
        if unstaked:
            sys.exit(f"validators need a stake of at least {fmt(econ.MIN_STAKE)} in the genesis (--bootstrap or --stake): "
                     f"{', '.join(unstaked)}")
    if a.earn_in:
        g["earn_in"] = True
    g["economics"] = econ.SCHEDULES[a.economics].to_dict()
    hash_ = Chain.from_genesis(g).genesis_hash  # also checks the allocations fit under the supply cap
    Path(a.out).write_text(json.dumps(g, indent=2))
    print(f"wrote {a.out}  genesis hash {hash_}\n{_describe_economics(econ.SCHEDULES[a.economics])}")
    if a.earn_in:
        print(f"workers can start with no stake: their first {fmt(econ.MIN_STAKE)} of pay becomes their stake")


def _describe_economics(e: econ.Economics) -> str:
    s = e.halving_interval * econ.MAINNET_BLOCK_S
    span = f"{s / (365.25 * 86400):.3g} years" if s > 86400 * 365 else f"{s / 86400:.3g} days" if s > 86400 else f"{s / 3600:.3g} hours"
    cap = f", never more than {fmt(e.max_supply)} in all" if e.max_supply else ""
    return (f"issuance ({e.name}): {fmt(e.initial_subsidy)} per round, halving every {e.halving_interval:,} blocks "
            f"(~{span} of {econ.MAINNET_BLOCK_S} s rounds); {fmt(e.total_issuance())} issued in total{cap}")


def cmd_node(a):
    from .coordinator import CoordinatorConfig
    from .node import Node, PhaseTimes  # only nodes need the server
    from .ratelimit import Limits

    if crypto.BACKEND != "openssl":
        print("warning: `cryptography` not installed; using slow pure-Python signatures", file=sys.stderr)
    if _in_onedrive(Path(a.data_dir)):
        print(f"warning: {a.data_dir} is inside OneDrive, whose syncing can corrupt a running node's database", file=sys.stderr)
    genesis = json.loads(Path(a.genesis).read_text())
    times = PhaseTimes(a.register_s, a.compute_s, a.reveal_s, a.verify_s)
    limits = Limits.parse(a.rate_limit, local=a.rate_limit_local, max_connections=a.max_connections)
    if bool(a.tls_cert) != bool(a.tls_key):
        sys.exit("--tls-cert and --tls-key go together")
    context = tls.server_context(a.tls_cert, a.tls_key) if a.tls_cert else None
    node = Node(a.data_dir, genesis, load_wallet(a.wallet), a.host, a.port, times, config=CoordinatorConfig(min_fee=a.min_fee),
                keep_rounds=a.keep_rounds, limits=limits, tls=context, peers=[normalise_url(p) for p in a.peer or []],
                announce=normalise_url(a.announce) if a.announce else None).start()
    if context:
        fp = tls.cert_fingerprint(a.tls_cert)
        print(f"serving HTTPS; certificate sha256 fingerprint {fp}\n"
              f"if it's self-signed, workers trust it with: {prog()} setup --node https://ADDRESS:{a.port} --tls-fingerprint {fp}")
    if a.dataset:
        for path in a.dataset:
            print(f"dataset {path} -> {node.store.add_dataset(path)}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.stop()


def cmd_tls_cert(a):
    """A self-signed certificate for a testnet node, and how workers pin it."""
    hosts = a.host or tls.default_hosts()
    if not tls.can_make_certs():
        sys.exit(f"making a certificate here needs the `cryptography` package. With openssl instead:\n"
                 f"  {tls.openssl_command(a.cert, a.key, hosts, a.days)}\n"
                 f"then show its fingerprint with: {prog()} tls fingerprint {_q(a.cert)}")
    fp = tls.make_self_signed(a.cert, a.key, hosts, a.days)
    print(f"wrote {a.cert}, and its private key {a.key} (keep that one private)\n"
          f"valid {a.days} days for {', '.join(hosts)}\nsha256 fingerprint {fp}\n\n"
          f"Serve with:    {prog()} node ... --tls-cert {_q(a.cert)} --tls-key {_q(a.key)}\n"
          f"Workers join:  {prog()} setup --node https://ADDRESS:8545 --tls-fingerprint {fp}")


def cmd_tls_fingerprint(a):
    print(tls.cert_fingerprint(a.cert))


def cmd_worker(a):
    from . import minerui

    url = node_of(a)
    try:
        w = WorkerClient(url, load_wallet(wallet_of(a)), a.memory_mb, a.flops, device=a.device, cache_dir=a.cache_dir)
    except RuntimeError as e:
        sys.exit(f"error: {e}")
    status = w.node.status()
    acct, need = w.node.account(w.keys.address), status["min_stake"]
    low = earning = None
    if acct["stake"] < need and status.get("earn_in"):
        earning = (f"earning your stake: your next {fmt(need - acct['stake'])} of pay becomes stake. Until then you get "
                   "small batches of work and don't check other workers'")
    elif acct["stake"] < need:
        low = f"stake is {fmt(acct['stake'])}; stake at least {fmt(need)} to get work"
    if not minerui.supported(a.plain):
        print(f"worker {w.keys.address} training on {w.backend.describe()} ({hardware.describe()})")
        if earning:
            print(earning)
        if low:
            print(f"warning: {low}", file=sys.stderr)
        try:
            w.run()
        except KeyboardInterrupt:
            pass
        return
    dash = minerui.MinerDashboard([("you", w)], url)
    if earning:
        dash.note(earning, logging.INFO + 1)
    if low:
        dash.note(f"{low}: {prog()} stake --amount {need // SYN_PER_VIBE}", logging.WARNING)
    stop = threading.Event()
    try:
        dash.run(stop, on_start=lambda: threading.Thread(target=w.run, args=(stop,), daemon=True, name="worker").start())
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    print(dash.summary())


def cmd_devices(a):
    from .compute import TorchBackend, list_devices, self_test, torch_available

    found = [(d, None if d.id == "cpu" else self_test(TorchBackend(d.id))) for d in list_devices()]
    if a.json:
        print(json.dumps([{"id": d.id, "label": d.label, "ready": p is None, "problem": p} for d, p in found]))
        return
    print("Devices this computer can train on (best first):")
    for d, problem in found:
        state = "ready" if problem is None else f"NOT USABLE: {problem[:120]}. Update the graphics driver."
        print(f"  {d.id:<8} {d.label}  [{state}]")
    if not torch_available():
        print(f"\nNo GPU support installed. `{prog()} install-gpu` finds your graphics card and installs support for it.")


def _setup_gpu(assume_yes: bool) -> None:
    """Find the GPU, install the PyTorch build for it (after asking), and self-test it."""
    from . import gpusetup

    listed = [d for d in gpusetup.devices() if d["id"] != "cpu"]
    if any(d["ready"] for d in listed):
        for d in listed:
            print(f"   {d['label']}: {'ready' if d['ready'] else 'not usable: ' + str(d['problem'])[:120]}")
        return
    if listed:  # PyTorch sees the card but can't run on it: the driver, not the build, is the problem
        for d in listed:
            print(f"   {d['label']} isn't usable ({str(d['problem'])[:120]}). Update its graphics driver, then run setup again.")
        return
    gpus = gpusetup.detect()
    if not gpus:
        print("   no graphics card PyTorch supports; training will use the CPU, which works fine")
        return
    for g in gpus:
        print(f"   found {g.describe()}")
    print("   checking which PyTorch builds are published for it...")
    gpu, build, why = gpusetup.choose(gpus)
    if build is None:
        print(f"   can't train on {gpu.name} yet: {why}. Training will use the CPU.")
        return
    print(f"   it needs PyTorch built for {build.label}, a large download (up to 3 GB) into\n   {sys.executable}")
    if not ask("   Install it now?", assume_yes):
        print(f"   skipped. Install it later with: {prog()} install-gpu")
        return
    if code := gpusetup.install(build):
        print(f"   the install failed (pip exit code {code}). Training will use the CPU until it's fixed.")
        return
    for d in gpusetup.devices():
        print(f"   {d['id']:<8} {d['label']}  [{'ready' if d['ready'] else 'NOT USABLE: ' + str(d['problem'])[:120]}]")


def cmd_install_gpu(a):
    print("Graphics card")
    _setup_gpu(a.yes)


def cmd_setup(a):
    """Everything a new worker needs, in order. Safe to re-run: finished steps are kept."""
    print(f"Vibe Hash setup. Your settings and wallet are kept in {settings.home()}\n\n1. Node")
    url = settings.chosen_node(a.node)
    if not url and sys.stdin.isatty():
        url = input(f"   Press Enter to join the public network ({urlsplit(settings.PUBLIC_NODE).hostname}),\n"
                    "   or type the address of another network's node (like 192.168.1.20): ").strip()
    url = url or settings.PUBLIC_NODE
    status = None
    if url:
        url = normalise_url(url, https=bool(a.tls_fingerprint or a.tls_ca))
        settings.save(node=url)
        if a.tls_fingerprint or a.tls_ca:
            settings.save_tls_trust(url, a.tls_ca and str(Path(a.tls_ca).resolve()),
                                    a.tls_fingerprint and tls.normalize_fingerprint(a.tls_fingerprint))
            print(f"   trusting the node's certificate by its {'fingerprint' if a.tls_fingerprint else 'CA'}")
        try:
            status = NodeClient(url).status()
            print(f"   connected to {url}: network {status['chain_id']}, block {status['height']}")
        except NodeError as e:
            print(f"   saved {url}, but couldn't reach it ({e}). Check the address, or run setup again later.")
    else:
        print(f"   skipped. Add one later with: {prog()} setup --node ADDRESS")

    print("\n2. Wallet")
    path = settings.wallet_path(a.wallet) or settings.default_wallet()
    keys = None
    if path.exists():
        address = json.loads(path.read_text())["address"]
        print(f"   using {path}\n   address {address}")
    else:
        pw = password("   Choose a wallet password: ", confirm=True)
        if not pw:
            sys.exit("a wallet password is required (set VIBEHASH_PASSWORD for unattended setups)")
        words = mnemonic.generate()
        keys = mnemonic.keypair(words)
        wallets.save(keys, path, pw, mnemonic.PATH)
        address = keys.address
        print(f"   created {path}\n   address {address}")
        show_words(words, "   ")
    settings.save(wallet=str(path))

    print("\n3. Graphics card")
    if a.gpu == "skip":
        print("   skipped")
    else:
        _setup_gpu(a.yes)

    print("\n4. Stake")
    staked = False
    if status is None:
        print("   skipped (no node yet)")
    else:
        node, need = NodeClient(url), status["min_stake"]
        fee = status.get("min_fee", 0)
        acct = node.account(address)
        short = max(need - acct["stake"], 0)
        if not short:
            staked = True
            print(f"   staked {fmt(acct['stake'])}: ready to take work")
        elif status.get("earn_in") and acct["balance"] < short + fee:
            staked = True  # nothing to do: the work itself pays the stake in
            print(f"   no stake needed to start: this network lets you earn it. Your first {fmt(short)} of pay\n"
                  "   becomes your stake; until then you get small batches of work")
        elif acct["balance"] < short + fee:
            print(f"   balance {fmt(acct['balance'])}; staking needs at least {fmt(need)}. Ask the node's operator to send\n"
                  f"   VIBE to {address}, then run setup again (or: {prog()} stake --amount {need // SYN_PER_VIBE})")
        else:
            amount = a.stake or min(acct["balance"] - fee, max(short, 10 * SYN_PER_VIBE))
            if (a.stake or sys.stdin.isatty()) and ask(f"   Stake {fmt(amount)} now? Only cheating can cost you it.", bool(a.stake)):
                node.send_tx(keys or load_wallet(str(path)), "stake", "system", amount)
                staked = True
                print("   submitted; it takes effect in the next block")
            else:
                print(f"   not staked yet. When you're ready: {prog()} stake --amount {amount // SYN_PER_VIBE}")
    if staked:
        print(f"\nAll set. Start earning with:\n   {prog()} worker")
    else:
        print(f"\nAlmost there: once you've staked, start earning with:\n   {prog()} worker")


def cmd_balance(a):
    addr = a.address or json.loads(Path(wallet_of(a)).read_text())["address"]
    node = NodeClient(node_of(a))
    acct = node.account(addr)
    print(f"address:    {addr}\nbalance:    {fmt(acct['balance'])}\nstake:      {fmt(acct['stake'])}")
    if acct.get("bootstrap"):
        print(f"            of which {fmt(acct['bootstrap'])} is bootstrap stake: it votes, but can't be withdrawn")
    for amount, height in acct["unbonding"]:
        print(f"unbonding:  {fmt(amount)} (spendable at block {height})")
    print(f"reputation: {acct['reputation']:.2f}")
    print(f"tx fee:     {fmt(node.min_fee())} per transaction on this node, paid from the balance")


def _send(a, kind, recipient, amount, memo=""):
    keys = load_wallet(wallet_of(a))
    node = NodeClient(node_of(a))
    fee = node.min_fee() if a.fee is None else a.fee
    txid = node.send_tx(keys, kind, recipient, amount, memo, fee)
    print(f"submitted {kind} {txid} with a {fmt(fee)} fee; it lands in the next block")


def _fee_flag(p) -> None:
    p.add_argument("--fee", type=to_syn, help="transaction fee in VIBE, paid to the block's proposer (default: the node's minimum)")


def cmd_send(a):
    _send(a, "transfer", a.to, a.amount)


def cmd_stake(a):
    _send(a, "stake", "system", a.amount)


def cmd_unstake(a):
    _send(a, "unstake", "system", a.amount)


def cmd_post_job(a):
    keys = load_wallet(wallet_of(a))
    node = NodeClient(node_of(a))
    spec = json.loads(Path(a.spec).read_text())
    if a.dataset:
        spec["data"] = {"kind": "npz", "sha256": node.upload_dataset(keys, a.dataset)}
        print(f"uploaded dataset {spec['data']['sha256']}")
    try:
        job = TrainingJob.from_spec_json(json.dumps(spec), keys.address, a.budget)
    except InvalidJob as e:
        sys.exit(f"invalid job spec: {e}")
    fee = node.min_fee() if a.fee is None else a.fee
    txid = node.send_tx(keys, "post_job", "system", a.budget, job.spec_json(), fee)
    print(f"posted job {job.job_id} with budget {fmt(a.budget)} and a {fmt(fee)} fee ({txid})")


def template_job(job_id: str, gpu: bool, requester: str = "", budget: int = 0) -> TrainingJob:
    if gpu:
        # Big enough that GPUs pay off, small enough that a laptop CPU can still finish a unit.
        return TrainingJob(job_id, requester, budget=budget, fee_per_round=20 * SYN_PER_VIBE, units_per_round=16,
                           spec=ModelSpec(8, (512, 512), 4, "float32"), lr=0.05, batch_size=256,
                           tolerance=1e-4, memory_mb=2048)
    return TrainingJob(job_id, requester, budget=budget, fee_per_round=10 * SYN_PER_VIBE, units_per_round=12)


def cmd_job_template(a):
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} already exists; refusing to overwrite it")
    try:
        job = template_job(a.id, a.gpu)
    except InvalidJob as e:
        sys.exit(f"invalid job id: {e}")
    out.write_text(json.dumps(job.to_spec(), indent=2) + "\n")
    print(f"wrote {out}; edit it, then post it with post-job --spec {out}")


def cmd_cancel_job(a):
    _send(a, "cancel_job", "system", 0, _job_of(a))


def cmd_validator_join(a):
    """Announce this wallet as a validator, reachable at --url: from the next block it votes, and leads rounds."""
    from .ledger import valid_endpoint

    url = normalise_url(a.url)
    if not valid_endpoint(url):
        sys.exit(f"{a.url} isn't an address other nodes could reach (like http://203.0.113.7:8545)")
    node = NodeClient(node_of(a))
    if "consensus" not in node.status():
        sys.exit("this node's chain is run by a single node, not validators; its genesis would need to name some")
    _send(a, "validator", "system", 0, url)
    print(f"Run your node with --announce {url} and the same wallet, so the other validators can reach it.")


def cmd_validator_leave(a):
    _send(a, "validator", "system", 0, "")


def cmd_validator_list(a):
    node = NodeClient(node_of(a))
    st, vals = node.status(), node.get("/validators")
    if "consensus" not in st:
        print("this chain is run by a single node: it has no validators")
        return
    total = sum(v["stake"] for v in vals.values() if v["votes"]) or 1
    for addr, v in sorted(vals.items(), key=lambda kv: -kv[1]["stake"]):
        share = f"{v['stake'] / total:6.1%}" if v["votes"] else "no vote (stake below the minimum)"
        print(f"  {addr}  {fmt(v['stake']):>14}  {share}  {v['url']}")
    c = st["consensus"]
    print(f"\nvoting on block {c['height']} (consensus round {c['round']}, {c['step']}); "
          f"leading it: {c['proposer']} at {st.get('leader')}")


# --- training a model on your own data ---------------------------------------------------

def _job_home(job_id: str) -> Path:
    """Where `train` keeps what the network doesn't: column and class names, and the input scaling."""
    return settings.home() / "jobs" / job_id


def _job_of(a) -> str:
    job_id = a.job_id or settings.load().get("last_job")
    if not job_id:
        sys.exit(f"which job? Give its name, e.g. `{prog()} job status fruit-3fa2`")
    return job_id


def _job_node(a, job_id: str) -> NodeClient:
    """The node a job lives on: --node, the saved one, or the one `train` posted it to."""
    receipt = _job_home(job_id) / "job.json"
    if not settings.chosen_node(a.node) and receipt.exists():
        return NodeClient(json.loads(receipt.read_text())["node"])
    return NodeClient(node_of(a))


def cmd_train(a):
    """A CSV file in, a training job on the network out, with sensible model settings chosen for it. On a terminal
    it then follows the job on the live training dashboard."""
    from . import tabular, trainui
    from .blobs import file_sha256

    src = Path(a.data)
    try:
        hidden = tuple(int(v) for v in a.hidden.split(","))
    except ValueError:
        sys.exit("--hidden takes layer widths separated by commas, like 128,128")
    try:
        if src.suffix.lower() == ".npz":  # already prepared: inputs scaled and labels numbered
            from .model import Dataset
            data = Dataset.from_npz(src)
            n_out = int(max(data.y_train.max(), data.y_val.max())) + 1
            table = tabular.Table([f"x{i}" for i in range(data.x_train.shape[1])], None, data.x_train, None)
            classes, mean, std = [str(i) for i in range(max(2, n_out))], [0.0] * len(table.features), [1.0] * len(table.features)
        else:
            table = tabular.read_csv(src, a.label)
            data, classes, mean, std = tabular.to_dataset(table)
            mean, std = mean.tolist(), std.tolist()
    except (tabular.TableError, OSError, KeyError, ValueError) as e:
        sys.exit(f"error: {e}")
    job_id = a.id or f"{re.sub(r'[^a-z0-9._-]+', '-', src.stem.lower()).strip('-._')[:40] or 'model'}-{secrets.token_hex(2)}"
    home = _job_home(job_id)
    if home.exists():
        sys.exit(f"you already have a job called {job_id}; pick another name with --id")
    npz = tabular.save_dataset(data, home / "dataset.npz")
    try:
        job = TrainingJob(job_id, "", budget=a.budget, fee_per_round=a.fee, units_per_round=12,
                          spec=ModelSpec(len(table.features), hidden, len(classes), "float32"),
                          data_source={"kind": "npz", "sha256": file_sha256(npz)}, lr=0.05,
                          batch_size=min(128, len(data.y_train)), tolerance=1e-4)
    except InvalidJob as e:
        shutil.rmtree(home, ignore_errors=True)
        sys.exit(f"error: {e}")
    url = node_of(a)
    (home / "job.json").write_text(json.dumps({
        "job_id": job_id, "node": url, "source": str(src.resolve()), "label": table.label, "features": table.features,
        "classes": classes, "mean": mean, "std": std, "train_rows": len(data.y_train), "val_rows": len(data.y_val),
        "budget": a.budget, "fee": a.fee}, indent=2))
    keys, node = load_wallet(wallet_of(a)), NodeClient(url)
    try:
        node.upload_dataset(keys, npz)
        node.send_tx(keys, "post_job", "system", a.budget, job.spec_json())
    except NodeError:
        shutil.rmtree(home, ignore_errors=True)
        raise
    settings.save(last_job=job_id)
    rounds = a.budget // a.fee if a.fee else 0
    widths = " → ".join(str(w) for w in job.spec.widths)
    shown = ", ".join(table.features[:6]) + (f" and {len(table.features) - 6} more" if len(table.features) > 6 else "")
    print(f"Posted job {job_id}\n"
          f"  data      {src.name}: {len(data.y_train):,} rows to train on, {len(data.y_val):,} held back to score the model\n"
          f"  inputs    {len(table.features)} columns: {shown}\n"
          f"  predicts  {table.label or 'class'}: {', '.join(classes[:8])}{' …' if len(classes) > 8 else ''}\n"
          f"  model     {widths}, float32 ({job.spec.n_params:,} weights)\n"
          f"  budget    {fmt(a.budget)}: up to {rounds} rounds at {fmt(a.fee)} each, paid only for work that passes its checks")
    if trainui.supported(a.plain):  # follow it live, as the network trains it
        _follow(job_id, url, budget=a.budget, posted=True)
    else:
        print(f"\nWatch it train:     {prog()} job watch\n"
              f"When it's trained:  {prog()} job download")


def cmd_example(a):
    """Write the sample data, so training can be tried before you have data of your own."""
    from importlib import resources

    Path(a.dir).mkdir(parents=True, exist_ok=True)
    for name, what in (("fruit.csv", "320 measured fruit with their names, to train on"),
                       ("fruit-new.csv", "12 more without names, to predict")):
        out = Path(a.dir) / name
        if out.exists():
            print(f"{out} is already there")
            continue
        out.write_bytes(resources.files("vibehash").joinpath("data", name).read_bytes())
        print(f"wrote {out}: {what}")
    print(f"\nTrain on it:  {prog()} train {_q(Path(a.dir) / 'fruit.csv')}")


def _sparkline(values: list[float]) -> str:
    bars = "▁▂▃▄▅▆▇█"
    try:
        bars.encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        bars = ".:-=+*#@"
    lo, hi = min(values), max(values)
    return "".join(bars[0 if hi == lo else round((v - lo) / (hi - lo) * 7)] for v in values[-40:])


def cmd_job_status(a):
    job_id = _job_of(a)
    node = _job_node(a, job_id)
    info = node.get(f"/job/{job_id}")
    rounds, escrow, fee = info["rounds"], info["escrow"], info["spec"]["fee_per_round"]
    training = escrow > 0
    now = node.status()
    print(f"job {job_id}: {'training' if training else 'finished'}"
          + (f", in round {now['round']} right now" if now.get("job_id") == job_id else ""))
    if rounds:
        acc = [r["val_acc"] for r in rounds]
        print(f"  rounds     {len(rounds)} done, {sum(r['accepted_units'] for r in rounds):,} units of training accepted")
        print(f"  accuracy   {acc[-1]:.1%} on held-back rows (started at {acc[0]:.1%}; best {max(acc):.1%})")
        if len(acc) > 1:
            print(f"             {_sparkline(acc)}")
        print(f"  loss       {rounds[-1]['val_loss']:.4f}")
    else:
        print("  rounds     none yet: it starts when the next round opens" if training else "  rounds     none")
    left = f", about {escrow // fee} more rounds" if training and fee else ""
    print(f"  budget     {fmt(escrow)} left{left}")
    print(f"\n{'Download the model so far' if training else 'Download the model'}: {prog()} job download {job_id}")
    if training:
        print(f"Watch it train live: {prog()} job watch {job_id}")


def _follow(job_id: str, url: str, budget: int | None = None, posted: bool = False, plain: bool = False) -> None:
    """Follow a training job until its budget is spent, or Ctrl+C: the live dashboard on a terminal, else a log line
    per event. Then say where it stands, and what's next."""
    from . import trainui

    watch = trainui.JobWatch(job_id, url, budget=budget, posted=posted)
    try:
        if trainui.supported(plain):
            trainui.TrainDashboard(watch).run()
        else:
            trainui.follow_plain(watch)
    except KeyboardInterrupt:
        pass
    p, job = prog(), watch.job
    print(f"\n{watch.summary()}")
    if job is not None and job.rounds:
        print(f"{'Download the model' if watch.finished_at is not None else 'Download the model so far'}:  "
              f"{p} job download {job_id}")
    if watch.finished_at is None:
        print(f"Watch it again:  {p} job watch {job_id}")


def cmd_job_watch(a):
    """A training job, live: accuracy and loss climbing round by round, the budget left, whose turn it is."""
    from . import trainui

    job_id = _job_of(a)
    url = _job_node(a, job_id).url
    receipt = _job_home(job_id) / "job.json"
    posted = json.loads(receipt.read_text()) if receipt.exists() else {}
    if not posted:  # not posted from this computer: check the node has heard of it, rather than wait on a typo
        try:
            trainui.Reader(url).get(f"/job/{job_id}")
        except NodeError as e:
            if e.status == 404:
                sys.exit(f"error: {url} has no job called {job_id}")
        except ValueError:
            pass  # an answer that isn't JSON: the dashboard shows what's wrong, and keeps asking
    budget = posted.get("budget")
    _follow(job_id, url, budget=budget if isinstance(budget, int) else None, posted=bool(posted), plain=a.plain)


def cmd_job_download(a):
    import numpy as np

    from . import tabular

    job_id = _job_of(a)
    info, raw = _job_node(a, job_id).download_model(job_id)
    job = TrainingJob.from_spec_json(json.dumps(info["spec"]), info["owner"], 0)
    weights = np.frombuffer(raw, dtype=job.wire_dtype).astype(np.float64)
    receipt_path = _job_home(job_id) / "job.json"
    n_in, n_out = job.spec.n_in, job.spec.n_out
    if receipt_path.exists():
        r = json.loads(receipt_path.read_text())
        features, classes, mean, std, label = r["features"], r["classes"], r["mean"], r["std"], r.get("label")
    else:
        print("note: this job wasn't posted from this computer, so the model has no column or class names, and the\n"
              "      rows you predict on must already be scaled the way its dataset was", file=sys.stderr)
        features, classes, mean, std, label = [f"x{i}" for i in range(n_in)], [str(i) for i in range(n_out)], [0.0] * n_in, [1.0] * n_in, None
    rounds = info["rounds"]
    acc = rounds[-1]["val_acc"] if rounds else None
    source = json.loads(receipt_path.read_text()).get("source") if receipt_path.exists() else None
    out = Path(a.out or (f"{Path(source).stem}-model.npz" if source else f"{job_id}.npz"))  # my-data.csv -> my-data-model.npz
    tabular.save_model(out, job.spec, weights, features, classes, mean, std,
                       {"job_id": job_id, "label": label, "height": (info.get("model") or {}).get("height"), "rounds": len(rounds),
                        "val_acc": acc})
    settings.save(last_model=str(out.resolve()))
    score = f", {acc:.1%} right on held-back rows" if acc is not None else " (not trained yet: these are its starting weights)"
    print(f"saved {out}: {n_in} inputs → {label or 'class'} ({', '.join(classes[:6])}{' …' if len(classes) > 6 else ''}), "
          f"trained for {len(rounds)} rounds{score}\n"
          f"Use it on new rows:  {prog()} predict new-rows.csv")


def cmd_predict(a):
    from . import tabular

    path = a.model or settings.load().get("last_model")
    if not path:
        sys.exit(f"which model? Download one with `{prog()} job download`, or name the .npz file before the CSV")
    try:
        model = tabular.load_model(path)
        table = tabular.read_csv(a.data, label=model.info.get("label"), features=model.features)
    except (tabular.TableError, OSError) as e:
        sys.exit(f"error: {e}")
    probs = model.predict(table.x)
    names, sure = [model.classes[i] for i in probs.argmax(axis=1)], probs.max(axis=1)
    if a.out:
        tabular.write_predictions(a.data, a.out, names, sure)
        print(f"wrote {a.out}: {len(names):,} rows with a predicted and a confidence column")
    else:
        show = min(len(names), a.rows)
        width = max(9, *(len(n) for n in names[:show]))
        try:
            wrong = "  ✗" if "✗".encode(sys.stdout.encoding or "ascii") else ""
        except (UnicodeEncodeError, LookupError):
            wrong = "  x"
        print(f"{'row':>5}  {'predicted':<{width}}  confidence" + ("  actual" if table.y else ""))
        for i in range(show):
            actual = f"  {table.y[i]}{'' if table.y[i] == names[i] else wrong}" if table.y else ""
            print(f"{i + 1:>5}  {names[i]:<{width}}  {sure[i]:>9.1%}{actual}")
        if show < len(names):
            print(f"  … {len(names) - show:,} more rows. Add --out predictions.csv to save them all.")
    if table.y:
        right = sum(p == t for p, t in zip(names, table.y))
        print(f"\n{right / len(names):.1%} right ({right:,} of {len(names):,} rows have the same answer in the {table.label!r} column)")


class _Downloaded:
    """A round's checkpoints and dataset, fetched from the node into a scratch folder to recheck the round."""

    def __init__(self, node: NodeClient, root: Path):
        from .blobs import BlobStore

        self.node, self.blobs, self.datasets, self.arrays = node, BlobStore(root / "blobs"), BlobStore(root / "datasets"), {}

    def array(self, blob_id: str, job):
        import numpy as np

        if blob_id not in self.arrays:
            self.node.download("blob", blob_id, self.blobs)  # hash-checked on arrival
            self.arrays[blob_id] = np.array(self.blobs.load(blob_id, job.wire_dtype, job.spec.n_params), dtype=np.float64)
        return self.arrays[blob_id]

    def attach(self, job) -> None:
        from .model import Dataset

        sha = job.data_source["sha256"]
        job.attach_data(Dataset.from_npz(self.node.download("dataset", sha, self.datasets)), sha)


def cmd_chain_verify(a):
    """Download every block and replay it locally. Trust nothing the node says about balances."""
    node = NodeClient(node_of(a))
    genesis = json.loads(Path(a.genesis).read_text()) if a.genesis else node.get("/genesis")
    chain = Chain.from_genesis(genesis)
    status = node.status()
    if status["genesis_hash"] != chain.genesis_hash:
        sys.exit("node is on a different network than your genesis file")
    import tempfile

    from . import receipts
    from .consensus import check_certificate
    from .crypto import address_of
    from .replay import replay_block

    height, replayed, unrecorded, problems, records, certified = 1, 0, 0, [], {}, 0
    # Rounds whose checkpoints the node still keeps get their weight-based results re-run too.
    window = status.get("keeps_checkpoints_for", 0) if a.recheck else 0
    first_kept, recheck_stats, rechecked_rounds = status["height"] - window + 1, {}, 0
    scratch = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # Windows holds memory-mapped files a moment

    def record_of(h: int) -> dict | None:
        if h not in records:
            try:
                records[h] = node.get(f"/round/{h}/record")
            except NodeError:
                records[h] = None
        return records[h]

    while height <= status["height"]:
        for d in node.get(f"/blocks?from={height}&to={height + 99}"):
            block = Block.from_dict(d)
            leader = chain.state.validators.get(address_of(block.proposer_pubkey))
            if chain.state.validators:  # a chain validators run: each block needs their signatures to be final
                problem = check_certificate(d.get("certificate"), block.height, block.hash.hex(),
                                            chain.state.validator_set(), chain.chain_id)
                if problem:
                    problems.append(f"block {block.height}: {problem}")
                else:
                    certified += 1
            if block.summary is not None and a.replay:
                if not block.summary.record_hash:
                    unrecorded += 1  # sealed before blocks carried round records
                elif record_of(block.height) is None:
                    problems.append(f"block {block.height}: its round record isn't available")
                else:
                    # The round's checkpoints are with the validator that led it.
                    source = NodeClient(leader) if leader and leader.rstrip("/") != node.url else node
                    kept = _Downloaded(source, Path(scratch.name)) if window and block.height >= first_kept else None
                    found = replay_block(chain, block, record_of(block.height), kept, recheck_stats if kept else None)
                    problems += [f"block {block.height}: {p}" for p in found]
                    replayed += 1
                    rechecked_rounds += kept is not None
            chain.append(block)
            height += 1

    # Receipts: the node's signed promises that messages it accepted would be recorded (receipts.py).
    files = [Path(p) for p in a.receipts] if a.receipts else sorted((settings.home() / "cache").glob("*/receipts.jsonl"))
    held = []
    for f in files:
        try:
            held += [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError) as e:
            problems.append(f"can't read receipts from {f}: {e}")
    tally: dict[str, int] = {}
    for rcpt in (r for r in held if r.get("chain_id") == chain.chain_id):
        n = rcpt.get("round")
        block = chain.blocks[n] if isinstance(n, int) and 0 < n <= chain.tip.height else None
        status_, detail = receipts.audit(rcpt, block, record_of(n) if block is not None and block.summary else None)
        tally[status_] = tally.get(status_, 0) + 1
        if status_ in ("censored", "invalid"):
            problems.append(detail)
    ok = chain.validate()
    st = chain.state
    print(f"verified {len(chain.blocks)} blocks: {'VALID' if ok else 'INVALID'}")
    print(f"supply {fmt(st.total_supply())} = genesis {fmt(st.genesis)} + minted {fmt(st.minted)} - burned {fmt(st.burned)}")
    if certified or chain.state.validators:
        print(f"{certified} blocks final: each signed by validators holding more than two thirds of the stake "
              f"({len(chain.state.validator_set())} validators now)")
    if a.replay:
        replay_problems = [p for p in problems if p.startswith("block ")]
        print(f"replayed {replayed} rounds from their records: "
              + ("every decision follows from the signed messages" if not replay_problems else f"{len(replay_problems)} problems")
              + (f" ({unrecorded} older rounds have no record)" if unrecorded else ""))
        if rechecked_rounds:
            redone, gone = recheck_stats.get("rechecked", 0), recheck_stats.get("unchecked", 0)
            disputed = sum(" check for " in p for p in replay_problems)
            print(f"re-ran {redone} weight-based checks in the last {rechecked_rounds} rounds from their checkpoints: "
                  + ("every one matches its record" if not disputed else f"{disputed} don't match")
                  + (f"; {gone} couldn't be redone (checkpoints gone)" if gone else "")
                  + (f". Earlier rounds' checkpoints are pruned (the node keeps {window}), so their results come from the records"
                     if replayed > rechecked_rounds else ""))
    scratch.cleanup()
    if tally:
        print(f"checked {sum(tally.values())} receipts: " + ", ".join(f"{v} {k}" for k, v in sorted(tally.items()))
              + (". The node dropped messages it had accepted." if tally.get("censored") else ""))
    for p in problems:
        print(f"  {p}")
    sys.exit(0 if ok and not problems else 1)


TESTNET_WORKERS = 8  # wallets staked in a local testnet's genesis; --workers picks how many of them run


def _lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # a UDP connect sends nothing; it just picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return "THIS-COMPUTER"


def _q(path) -> str:
    return f'"{path}"' if " " in str(path) else str(path)


def _free_ports(n: int) -> list[int]:
    socks = [socket.socket() for _ in range(n)]
    try:
        for s in socks:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def _in_onedrive(path: Path) -> bool:
    roots = [os.environ.get(v) for v in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")]
    return any(r and path.resolve().is_relative_to(Path(r).resolve()) for r in roots)


def cmd_testnet(a):
    """A whole network on this computer: node, workers and a training job, from one command.
    Its wallets are unencrypted throwaway keys, so never send real coins to them."""
    from . import minerui
    from .node import Node, PhaseTimes

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with the node's log, even when piped
    d = Path(a.dir) if a.dir else settings.home() / "testnet"
    if a.reset and d.exists():
        if not (d / "genesis.json").exists():
            sys.exit(f"{d} doesn't look like a testnet folder; not deleting it")
        shutil.rmtree(d)
    if _in_onedrive(d):
        print(f"warning: {d} is inside OneDrive, whose syncing can corrupt a running node's database; "
              "use --dir somewhere else", file=sys.stderr)
    existing = json.loads((d / "genesis.json").read_text()) if (d / "genesis.json").exists() else None
    count = (len(existing.get("validators", {})) or 1) if existing else (a.validators or 1)
    if existing and a.validators and a.validators != count:
        sys.exit(f"this testnet has {count} validator{'s' * (count > 1)}; add --reset to start one with {a.validators}")
    validators = ["node", *(f"validator-{i}" for i in range(2, count + 1))]
    worker_names = [f"worker-{i}" for i in range(1, TESTNET_WORKERS + 1)]
    keys = {}
    for name in ["requester", *validators, *worker_names]:
        if not (d / f"{name}.json").exists():
            wallets.save(KeyPair.generate(), d / f"{name}.json", None)
        keys[name] = wallets.load(d / f"{name}.json", None)
    if existing is None:
        # Validators run the chain, even just one: each listens on the address the genesis gives it. They vote with
        # bootstrap stake, which can't be spent; the workers start with nothing and earn their stake. Only the
        # requester is handed coins, to pay for the demo jobs.
        public = _lan_ip() if a.host == "0.0.0.0" else "127.0.0.1" if a.host in ("127.0.0.1", "localhost") else a.host
        ports = _free_ports(count) if a.port == 0 else [a.port + i for i in range(count)]
        genesis = {"chain_id": f"vibehash-local-{secrets.token_hex(3)}", "timestamp": int(time.time()),
                   "allocations": {keys["requester"].address: 10_000 * SYN_PER_VIBE}, "stakes": {},
                   "bootstrap": {keys[v].address: 10 * SYN_PER_VIBE for v in validators}, "earn_in": True,
                   "validators": {keys[v].address: f"http://{public}:{port}" for v, port in zip(validators, ports)}}
        (d / "genesis.json").write_text(json.dumps(genesis, indent=2))
    genesis = json.loads((d / "genesis.json").read_text())
    named = genesis.get("validators", {})
    ports = [urlsplit(named[keys[v].address]).port for v in validators] if named else [a.port]
    if a.host == "0.0.0.0" and count > 1 and any(urlsplit(u).hostname == "127.0.0.1" for u in named.values()):
        print("warning: this testnet's validators were given this computer's local-only address, so other computers "
              "can't reach validators 2 and up; --reset rebuilds it with its network address", file=sys.stderr)
    workers = worker_names[:a.workers]
    if not a.verbose:
        logging.getLogger("vibehash.worker").setLevel(logging.WARNING)  # the node's log tells the story

    times = PhaseTimes(a.register_s, a.compute_s, a.reveal_s, a.verify_s, idle_s=1.0)
    nodes = []
    try:
        for v, port in zip(validators, ports):
            nodes.append(Node(d / ("node-data" if v == "node" else f"{v}-data"), genesis, keys[v], a.host, port, times))
    except OSError as e:
        for n in nodes:
            n.server.server_close()
            n.store.close()
        sys.exit(f"couldn't start a node on port {port} ({e}). Is one already running? Try --port.")
    node = nodes[0]
    for n in nodes:
        n.coord.config.min_fee = a.min_fee
    try:  # before the node starts rounds: a GPU's self-test takes seconds, and the first round shouldn't go unworked
        clients = [(n, WorkerClient(node.url, keys[n], device=a.device, cache_dir=d / "cache" / n)) for n in workers]
    except RuntimeError as e:
        for n in nodes:
            n.server.server_close()
            n.store.close()
        sys.exit(f"error: {e}")
    for n in nodes:
        n.start()
    api, s = NodeClient(node.url), node.chain.state
    start = {n: s.balance(keys[n].address) + s.stake(keys[n].address) for n in workers}
    stop = threading.Event()

    port = urlsplit(node.url).port
    url = f"http://{_lan_ip() if a.host == '0.0.0.0' else urlsplit(node.url).hostname}:{port}"
    p = prog()
    dash = minerui.MinerDashboard(clients, node.url, footer=f"kept in {d}") if minerui.supported(a.plain) else None
    say = dash.note if dash else (lambda text: print("  " + text))
    posted = None

    def ensure_job():
        nonlocal posted
        s = node.chain.state
        if any(v > 0 for v in s.escrow.values()) or (posted and posted not in s.jobs):
            return  # a job is running, or the one just posted hasn't reached a block yet
        job = template_job(f"demo-{len(s.jobs) + 1}", a.job == "gpu", keys["requester"].address, a.budget)
        if s.balance(keys["requester"].address) >= job.budget + node.coord.config.min_fee:
            api.send_tx(keys["requester"], "post_job", "system", job.budget, job.spec_json())
            posted = job.job_id
            say(f"posted job {job.job_id}: {job.spec.n_params:,} {job.spec.precision} weights, "
                f"budget {job.budget / SYN_PER_VIBE:,.0f} VIBE")

    hints = (f"Try these in another terminal:\n"
             f"  {p} balance --node {url} --wallet {_q(d / 'worker-1.json')}\n"
             f"  {p} chain verify --node {url} --genesis {_q(d / 'genesis.json')}")
    if a.host == "0.0.0.0":
        hints += (f"\nOther computers join with `{p} setup --node {url}`. Send each one VIBE to stake:\n"
                  f"  {p} send --node {url} --wallet {_q(d / 'requester.json')} --to THEIR-ADDRESS --amount 20")
    if not dash:
        print(f"\nLocal testnet {genesis['chain_id']}, kept in {d}\n"
              f"  node     {url}\n"
              + (f"  validators {len(nodes)}, on ports {', '.join(str(p) for p in ports)}; they take turns leading rounds "
                 f"and vote on every block\n" if len(nodes) > 1 else "")
              + f"  workers  {len(clients)}, training on {clients[0][1].backend.describe()}\n\n{hints}\n"
              "Stop with Ctrl+C. Running it again carries on where it stopped; --reset starts over.\n")
    elif a.host == "0.0.0.0":
        dash.note(f"open to other computers: they join with `{p} setup --node {url}`", logging.INFO + 1)

    threads = [threading.Thread(target=c.run, args=(stop,), daemon=True, name=n) for n, c in clients]
    rounds, height = 0, node.chain.tip.height

    def begin():
        for t in threads:
            t.start()
        ensure_job()

    def tick():
        nonlocal rounds, height
        tip = node.chain.tip
        if tip.height == height:
            return
        height = tip.height
        ensure_job()
        if tip.summary is None:
            return
        rounds += 1
        if not dash:
            s = node.chain.state
            earned = ", ".join(f"{n} {(s.balance(keys[n].address) + s.stake(keys[n].address) - start[n]) / SYN_PER_VIBE:.2f}"
                               for n in workers)
            print(f"  block {tip.height}: earned this session {earned} VIBE; "
                  f"{sum(s.escrow.values()) / SYN_PER_VIBE:,.0f} VIBE of job budget left")
        if a.rounds is not None and rounds >= a.rounds:
            stop.set()

    try:
        if dash:
            dash.run(stop, tick, on_start=begin)
        else:
            begin()
            while not stop.is_set():
                time.sleep(0.5)
                tick()
    except KeyboardInterrupt:
        if not dash:
            print("\nstopping...")
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=10)
        for n in nodes:
            n.stop()
    if dash:
        print(f"{dash.summary()}\nThe chain and wallets are kept in {d}; run it again to carry on.\n\n{hints}")


def _min_fee_flag(p) -> None:
    p.add_argument("--min-fee", type=to_syn, default=str(Decimal(fees.DEFAULT_MIN_FEE) / SYN_PER_VIBE),
                   help=f"smallest fee in VIBE this node relays a transaction for (default {fmt(fees.DEFAULT_MIN_FEE)}; 0 relays free ones)")


def _rate_limit_flags(p) -> None:
    p.add_argument("--rate-limit", default="default", metavar="SPEC",
                   help="per-client request limits: default, off, or overrides like tx=1/10,read=100 (requests a second"
                        "/burst; buckets: read, worker, tx, upload, other). Clients on this computer are never limited")
    p.add_argument("--rate-limit-local", action="store_true",
                   help="limit clients on this computer (127.0.0.1) too, e.g. behind a reverse proxy, where everyone looks local")
    p.add_argument("--max-connections", type=int, default=256, help="connections served at once (default 256)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vibehash", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    NODE, WALLET = "node address (default: the one `setup` saved)", "wallet file (default: the one `setup` saved)"

    p = sub.add_parser("setup", help="get this computer ready to work: node, wallet, GPU support, stake")
    p.add_argument("--node", help="node address, e.g. 192.168.1.20 or http://node.example:8545")
    p.add_argument("--wallet", help=f"wallet file to use or create (default {settings.default_wallet()})")
    p.add_argument("--gpu", choices=["auto", "skip"], default="auto", help="find the GPU and install PyTorch for it")
    p.add_argument("--stake", type=to_syn, help="stake this much VIBE without asking")
    p.add_argument("--yes", action="store_true", help="install GPU support without asking")
    p.set_defaults(fn=cmd_setup)
    p.add_argument("--tls-fingerprint", metavar="SHA256", help="trust the https node's certificate with this fingerprint "
                   "(how workers trust a self-signed testnet node; ask its operator)")
    p.add_argument("--tls-ca", metavar="FILE", help="verify the https node's certificate with this CA file")

    p = sub.add_parser("install-gpu", help="find your graphics card and install the PyTorch build for it")
    p.add_argument("--yes", action="store_true", help="don't ask before downloading"); p.set_defaults(fn=cmd_install_gpu)

    p = sub.add_parser("testnet", help="run a whole network on this computer: node, workers and a job")
    p.add_argument("--workers", type=int, default=2, choices=range(1, TESTNET_WORKERS + 1), metavar=f"1-{TESTNET_WORKERS}")
    p.add_argument("--validators", type=int, choices=range(1, 8), metavar="1-7",
                   help="validator nodes, on --port and the ports after it; they take turns leading rounds and vote on "
                        "every block (default 1; fixed when the testnet is first made)")
    p.add_argument("--dir", help=f"where its wallets and chain live (default {settings.home() / 'testnet'})")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 lets other computers join"); p.add_argument("--port", type=int, default=8545)
    p.add_argument("--job", choices=["gpu", "small"], default="gpu", help="the job template to train (see `job template`)")
    p.add_argument("--budget", type=to_syn, default="500", help="VIBE per job; a new job is posted when one runs out")
    p.add_argument("--device", default="auto", help="what the workers train on: auto, cpu, cuda, xpu, mps")
    p.add_argument("--rounds", type=int, help="stop after this many rounds (default: run until Ctrl+C)")
    p.add_argument("--reset", action="store_true", help="delete this testnet's chain and wallets and start over")
    p.add_argument("--plain", action="store_true", help="print log lines instead of the live mining dashboard")
    for name, default in (("register-s", 3.0), ("compute-s", 8.0), ("reveal-s", 3.0), ("verify-s", 4.0)):
        p.add_argument(f"--{name}", type=float, default=default)
    p.set_defaults(fn=cmd_testnet)
    _min_fee_flag(p)

    w = sub.add_parser("wallet").add_subparsers(dest="wcmd", required=True)
    p = w.add_parser("create", help="make a wallet and show its 24 recovery words once")
    p.add_argument("--out", help=f"default {settings.default_wallet()}")
    p.add_argument("--no-password", action="store_true")
    p.add_argument("--passphrase", action="store_true", help="add a BIP39 passphrase to the words (asked for; optional)")
    p.set_defaults(fn=cmd_wallet_create)
    p = w.add_parser("restore", help="rebuild a wallet from its recovery words (read from the terminal or $VIBEHASH_MNEMONIC)")
    p.add_argument("--out", help=f"default {settings.default_wallet()}")
    p.add_argument("--no-password", action="store_true")
    p.add_argument("--passphrase", action="store_true", help="the words had a BIP39 passphrase (asked for)")
    p.add_argument("--path", default=mnemonic.PATH, help=f"derivation path (default {mnemonic.PATH})")
    p.set_defaults(fn=cmd_wallet_restore)
    p = w.add_parser("show"); p.add_argument("--wallet", help=WALLET); p.set_defaults(fn=cmd_wallet_show)

    g = sub.add_parser("genesis").add_subparsers(dest="gcmd", required=True)
    p = g.add_parser("create"); p.add_argument("--chain-id", required=True); p.add_argument("--alloc", action="append", metavar="ADDR=VIBE")
    p.add_argument("--stake", action="append", metavar="ADDR=VIBE")
    p.add_argument("--validator", action="append", metavar="ADDR=URL",
                   help="a validator and the address other nodes reach it at (it needs a --stake too). Without any, "
                        "one node runs the chain alone")
    p.add_argument("--out", required=True); p.set_defaults(fn=cmd_genesis_create)
    p.add_argument("--economics", choices=sorted(econ.SCHEDULES), default="testnet",
                   help="issuance schedule: testnet (fast halvings, for trying things) or mainnet (Bitcoin-like, ~4-year halvings)")
    p.add_argument("--bootstrap", action="append", metavar="ADDR=VIBE",
                   help="stake for a first validator that votes but can never be withdrawn or spent; burned if it leaves")
    p.add_argument("--no-earn-in", dest="earn_in", action="store_false",
                   help="require workers to be given stake; by default they can start with none and earn it")

    p = sub.add_parser("node", help="run a node")
    p.add_argument("--genesis", required=True); p.add_argument("--wallet", required=True); p.add_argument("--data-dir", required=True)
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8545)
    p.add_argument("--peer", action="append", metavar="URL",
                   help="another node to sync from and gossip to (validators find each other from the chain anyway)")
    p.add_argument("--announce", metavar="URL", help="the address other nodes reach this one at, if not --host:--port")
    p.add_argument("--dataset", action="append", help="add an .npz dataset to this node's store")
    p.add_argument("--keep-rounds", type=int, default=10,
                   help="rounds whose opened checkpoints stay downloadable, so anyone can re-run their checks (default 10)")
    for name, default in (("register-s", 3.0), ("compute-s", 20.0), ("reveal-s", 5.0), ("verify-s", 10.0)):
        p.add_argument(f"--{name}", type=float, default=default)
    p.set_defaults(fn=cmd_node)
    _min_fee_flag(p)
    _rate_limit_flags(p)
    p.add_argument("--tls-cert", metavar="FILE", help="serve HTTPS with this PEM certificate (see `tls cert`)")
    p.add_argument("--tls-key", metavar="FILE", help="the certificate's private key")

    t = sub.add_parser("tls", help="HTTPS certificates for nodes").add_subparsers(dest="tcmd", required=True)
    p = t.add_parser("cert", help="make a self-signed certificate for a testnet node")
    p.add_argument("--cert", default="node-cert.pem", help="certificate file to write (default node-cert.pem)")
    p.add_argument("--key", default="node-key.pem", help="private key file to write (default node-key.pem)")
    p.add_argument("--host", action="append", help="a name or IP address workers reach the node at "
                   "(repeat for several; default: localhost and this computer's name and address)")
    p.add_argument("--days", type=int, default=365); p.set_defaults(fn=cmd_tls_cert)
    p = t.add_parser("fingerprint", help="print a certificate's SHA-256 fingerprint, for setup --tls-fingerprint")
    p.add_argument("cert"); p.set_defaults(fn=cmd_tls_fingerprint)

    p = sub.add_parser("worker", help="contribute compute and earn VIBE")
    p.add_argument("--node", help=NODE); p.add_argument("--wallet", help=WALLET)
    p.add_argument("--device", default="auto", help="auto (first GPU found, else CPU), cpu, cuda, cuda:1, xpu, mps")
    p.add_argument("--memory-mb", type=int, help="override detected memory"); p.add_argument("--flops", type=float, help="override benchmark")
    p.add_argument("--cache-dir", help="where checkpoints and datasets are kept (default ~/.vibehash/cache)")
    p.add_argument("--plain", action="store_true", help="print log lines instead of the live mining dashboard")
    p.set_defaults(fn=cmd_worker)

    p = sub.add_parser("devices", help="list the CPUs and GPUs this computer can train on")
    p.add_argument("--json", action="store_true", help="machine-readable output"); p.set_defaults(fn=cmd_devices)

    p = sub.add_parser("balance"); p.add_argument("--node", help=NODE); p.add_argument("--wallet", help=WALLET)
    p.add_argument("address", nargs="?"); p.set_defaults(fn=cmd_balance)
    p = sub.add_parser("send"); p.add_argument("--node", help=NODE); p.add_argument("--wallet", help=WALLET)
    p.add_argument("--to", required=True); p.add_argument("--amount", type=to_syn, required=True); p.set_defaults(fn=cmd_send)
    _fee_flag(p)
    for name, fn in (("stake", cmd_stake), ("unstake", cmd_unstake)):
        p = sub.add_parser(name); p.add_argument("--node", help=NODE); p.add_argument("--wallet", help=WALLET)
        p.add_argument("--amount", type=to_syn, required=True); p.set_defaults(fn=fn)
        _fee_flag(p)
    p = sub.add_parser("post-job"); p.add_argument("--node", help=NODE); p.add_argument("--wallet", help=WALLET)
    p.add_argument("--spec", required=True); p.add_argument("--budget", type=to_syn, required=True); p.add_argument("--dataset")
    p.set_defaults(fn=cmd_post_job)
    _fee_flag(p)
    p = sub.add_parser("cancel-job"); p.add_argument("--node", help=NODE); p.add_argument("--wallet", help=WALLET)
    p.add_argument("job_id", nargs="?", help="default: the last job you trained"); p.set_defaults(fn=cmd_cancel_job)
    _fee_flag(p)

    v = sub.add_parser("validator", help="validators vote on blocks and take turns leading rounds").add_subparsers(dest="vcmd", required=True)
    p = v.add_parser("join", help="announce your (staked) wallet as a validator")
    p.add_argument("--url", required=True, help="the address other nodes reach your node at, e.g. http://203.0.113.7:8545")
    p.add_argument("--node", help=NODE); p.add_argument("--wallet", help=WALLET); p.set_defaults(fn=cmd_validator_join)
    p = v.add_parser("leave", help="stop validating"); p.add_argument("--node", help=NODE); p.add_argument("--wallet", help=WALLET)
    p.set_defaults(fn=cmd_validator_leave)
    p = v.add_parser("list", help="the validators, their stake and voting share")
    p.add_argument("--node", help=NODE); p.set_defaults(fn=cmd_validator_list)

    p = sub.add_parser("example", help="write sample data (fruit.csv) to try training with")
    p.add_argument("--dir", default=".", help="folder to write it to (default: this one)"); p.set_defaults(fn=cmd_example)

    p = sub.add_parser("train", help="train a model on a CSV file: posts it as a job for the network to train")
    p.add_argument("data", help="CSV with a header row: numeric input columns, plus the column to predict")
    p.add_argument("--label", help="the column to predict (default: the last one)")
    p.add_argument("--budget", type=to_syn, default="100", help="VIBE to spend, paid only for work that passes its checks (default 100)")
    p.add_argument("--fee", type=to_syn, default="10", help="VIBE per round of training (default 10)")
    p.add_argument("--hidden", default="128,128", help="hidden layer widths (default 128,128)")
    p.add_argument("--id", help="job name (default: from the file name)")
    p.add_argument("--node", help=NODE); p.add_argument("--wallet", help=WALLET)
    p.add_argument("--plain", action="store_true", help="print the plan and return, instead of following the job on the "
                   "live training dashboard (piped output does this anyway)")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("predict", help="use a downloaded model on new rows of a CSV file")
    p.add_argument("model", nargs="?", help="the .npz file `job download` saved (default: the last one)")
    p.add_argument("data", help="CSV with the same input columns")
    p.add_argument("--out", help="write every row with its prediction to this CSV")
    p.add_argument("--rows", type=int, default=20, help="rows to show when not writing a file (default 20)")
    p.set_defaults(fn=cmd_predict)

    j = sub.add_parser("job").add_subparsers(dest="jcmd", required=True)
    p = j.add_parser("status", help="how a training job is going: rounds, accuracy, budget left")
    p.add_argument("job_id", nargs="?", help="default: the last job you trained"); p.add_argument("--node", help=NODE)
    p.set_defaults(fn=cmd_job_status)
    p = j.add_parser("watch", help="follow a training job live, until its budget is spent: accuracy climbing round by "
                     "round, budget left, events")
    p.add_argument("job_id", nargs="?", help="default: the last job you trained"); p.add_argument("--node", help=NODE)
    p.add_argument("--plain", action="store_true", help="print a line per event instead of the live dashboard")
    p.set_defaults(fn=cmd_job_watch)
    p = j.add_parser("download", help="save a job's trained model to use with `predict`")
    p.add_argument("job_id", nargs="?", help="default: the last job you trained"); p.add_argument("--out", help="file to save (default JOB.npz)")
    p.add_argument("--node", help=NODE); p.set_defaults(fn=cmd_job_download)
    p = j.add_parser("template", help="write a starter job file, for full control with post-job")
    p.add_argument("--out", default="job.json"); p.add_argument("--id", default="my-first-job")
    p.add_argument("--gpu", action="store_true", help="a larger float32 model sized for GPUs")
    p.set_defaults(fn=cmd_job_template)

    c = sub.add_parser("chain").add_subparsers(dest="ccmd", required=True)
    p = c.add_parser("verify", help="download every block, re-check it from genesis, and replay every round")
    p.add_argument("--node", help=NODE); p.add_argument("--genesis")
    p.add_argument("--no-replay", dest="replay", action="store_false", help="skip replaying rounds from their records")
    p.add_argument("--no-recheck", dest="recheck", action="store_false",
                   help="don't download recent rounds' checkpoints to re-run their weight-based checks")
    p.add_argument("--receipts", action="append", metavar="FILE",
                   help="a worker's receipts.jsonl to check (default: every worker's on this computer)")
    p.set_defaults(fn=cmd_chain_verify)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    try:
        a.fn(a)
    except FileNotFoundError as e:
        sys.exit(f"error: {e.filename} doesn't exist")
    except (NodeError, wallets.WalletError, ValueError, OSError) as e:
        sys.exit(f"error: {e}")
