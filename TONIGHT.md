# Test night: four home PCs and a few RunPods on the public network

The plan for 2026-11-02: the owner's laptop and desktop, a friend's laptop and
desktop, and a few rented RunPod GPUs, all mining on https://node.vibehash.org.
They earn on fruit-demo, train a real fine-tune of Qwen3-0.6B together, try the
Muse agents, and maybe take DeepSeek pilot work. Every machine keeps a record of
what it did, to read the next day.

## 0. Before anyone starts (owner, about 45 minutes)

Tonight's fixes only reach the machines through PyPI (the one-line installers
install from there), and the faster rounds only happen once the public node runs
them. So, in order:

1. Merge branch `tonight` into `main` (a PR; CI runs `test (3.10)` and `test (3.12)`).
2. Release 0.5.7: bump `version` in `pyproject.toml`, commit, push, then tag
   `v0.5.7` and push the tag (README, "Publishing a release"). PyPI has it a few
   minutes later. Check with `pip index versions vibehash`.
3. Upgrade the node, in a terminal where you can type your SSH key's passphrase:

   ```powershell
   scp .\deploy\setup-node.sh root@165.227.37.0: ; ssh -t root@165.227.37.0 bash setup-node.sh --domain node.vibehash.org
   ```

   For a few minutes after the restart the node drops some connections, so
   workers may lose a round or two. Then check the block times:
   `curl -s "https://node.vibehash.org/blocks?from=<height-10>"`. The
   think-shorter-pilot's empty rounds should be about 39 s apart, not 1,522 s.

Without step 3, a newcomer gets a paid fruit-demo round only every ~26 minutes.
Without step 2, machines installing tonight get 0.5.6: no `selftest`, no log
files, and on Windows a `setup` with nobody at the keyboard waits forever.

## 1. Every Windows PC

In PowerShell:

```powershell
irm https://vibehash.org/install.ps1 | iex
```

It installs Vibe Hash, offers Python if you need it, and starts `setup`. Setup
asks for a wallet password and shows the wallet's 24 recovery words once. Write
them down. Answer the GPU question yes, so it installs the right PyTorch.

Then check the machine:

```powershell
vibehash selftest
```

Every line should say `ok`. It trains a real fruit-demo unit and replays it the
way the network's checkers will, checks the clock (more than 2 minutes off and
the node refuses your messages; fix it under Settings > Time > Sync now), and
ends with what this machine can take and the next command. Note the wallet
address it prints (`vibehash wallet show` prints it too).

## 2. Stake every wallet (owner)

A worker needs 5 VIBE staked before it can check others' work or take fine-tune
units. Earning it on fruit-demo takes a few rounds. Sending it is instant. The
owner's wallet has about 4,150 VIBE. From the owner's PC, for each new address:

```powershell
vibehash send --to vibe1... --amount 10
```

Then on that machine:

```powershell
vibehash stake --amount 5
```

## 3. Start mining

```powershell
vibehash worker
```

The dashboard shows the mine. Leave it running. Its `job:` line, under the
round, picks what this PC mines: auto (every job it can do, the default) or one
job, whose rounds alone it then takes. Tab or a number key moves it, `0` is auto.
The worker prints where its
records go when it starts: `~/.vibehash/logs/worker-DATE.log` and
`~/.vibehash/logs/rounds.jsonl` (section 8). A PC with two GPUs runs one worker
per GPU, each with its own wallet (`vibehash wallet create --out gpu1.json`,
then `vibehash worker --device cuda:1 --wallet gpu1.json`). Stake that wallet too.

## 4. RunPod pods

Use a template with PyTorch (the official RunPod PyTorch one), a 50 GB+ volume
(more for DeepSeek, section 7), and open its web terminal or SSH. Each pod needs
outbound internet only, so no ports need exposing.

```bash
curl -fsSL https://vibehash.org/install.sh | sh
```

On a pod, the wallet and settings go to `/workspace/.vibehash`, the part that
survives a restart. The pod's PyTorch is kept as it is. After a restart, run the
installer again: the wallet is still there.

To leave a worker running after you close the terminal:

```bash
export VIBEHASH_PASSWORD='the-wallet-password'
nohup vibehash worker --plain > /workspace/worker.log 2>&1 &
```

To have a pod mine one job only (the fine-tune in section 5, say), add `--job`
and the job's id, as the dashboard's `job:` line lists it. The pod then sits the
other jobs' rounds out. A job the node doesn't list stops the worker at once,
naming the ones it does.

```bash
nohup vibehash worker --plain --job fruit-demo > /workspace/worker.log 2>&1 &
```

A pod with several GPUs: one worker per GPU, each with its own wallet, as in
section 3 (`--device cuda:N`). `vibehash selftest` works on pods too.

## 5. Train a real fine-tune together

Once at least two staked GPU workers are mining, post a fine-tune of Qwen3-0.6B
on the 2,379 everyday conversations already on the owner's PC. It needs the
tokenizer extra on the posting PC:

```powershell
python -m pip install "vibehash[finetune]"
```

```powershell
vibehash train $env:USERPROFILE\.vibehash\cache\datasets-src\everyday.jsonl --base qwen3-0.6b --budget 100
```

It shows the pre-flight (model, data, rounds, cost) and asks before posting.
Each worker fetches the 1.5 GB of weights once, so the first round may train
nothing. Jobs take turns, so its rounds come every third round, between
fruit-demo's and the pilot's. A round is at most ~4–6 minutes and ends sooner
once its work is in. Follow it:

```powershell
vibehash job watch
```

Afterwards, the trained adapter, its history and a short report:

```powershell
vibehash job download
```

That writes the PEFT adapter folder with `training-history.csv` and
`TRAINING.md` in it. `vibehash job history <job> --csv file.csv` fetches the
history from any computer.

## 6. The Muse agents (optional)

`agents/README.md` has the details. With a Meta API key (dev.meta.ai, API keys
tab; US only while it's a preview), from the repository folder on a PC with the
GPU venv:

```powershell
$env:MODEL_API_KEY = "paste-the-key"
python agents/architect.py "answer customer-support questions in two friendly sentences" --post
```

The architect has Muse Spark (`muse-spark-1.3`) write 40+ example chats and the
settings, checks them, then runs `vibehash train`, which still asks "Post it?".
After the job trains and `job download`:

```powershell
python agents/judge.py answer agent-jobs/<job>.eval.json <job>-lora
python agents/judge.py grade agent-jobs/<job>.answers.json
```

`answer` needs PyTorch, transformers and peft (the owner's
`~\.vibehash\venv` has them). `grade` costs well under a dollar.

## 7. DeepSeek pilot (optional, pods)

The think-shorter-pilot job waits for two staked workers, each with an 8 GB+
NVIDIA GPU and 16 GB+ RAM. Its prep downloads about 30 GB of DeepSeek weights
per piece, and the worker asks first. On a pod, answer in advance:

```bash
nohup vibehash worker --plain --deepseek yes > /workspace/worker.log 2>&1 &
```

Give such a pod a volume of 100 GB or more. For the pilot, a rejected prep unit
is redone and nobody is slashed. `https://node.vibehash.org/network` shows the
prep's progress (`pieces_done` of 774).

## 8. The records

On every machine, in `~/.vibehash` (on a pod `/workspace/.vibehash`):

| File | What's in it |
|---|---|
| `logs/worker-DATE.log` | everything the dashboard's feed showed, with times |
| `logs/rounds.jsonl` | a line per round: job, units given, committed and accepted (and why not), checks done, seconds, VIBE earned, balance and stake |
| `jobs/JOB/history.csv` | a posted job's rounds: loss, accuracy, units, budget left |

Copy each machine's `rounds.jsonl` somewhere. Every line names its computer and
wallet, so they can be put together. `vibehash worker` also prints a summary when
you stop it with Ctrl+C.

## If something goes wrong

| What you see | What to do |
|---|---|
| `selftest` says the clock is off | sync the clock (Windows: Settings > Time > Sync now) |
| node out of reach | try a phone hotspot (a dorm or campus network may block it); `curl https://node.vibehash.org/status` should answer |
| "waits for staked workers" | stake the wallet (section 2), or keep mining fruit-demo until it has 5 VIBE |
| out of GPU memory | close games, browsers and other GPU programs; Qwen3-0.6B needs about 4 GB |
| "another worker ... already mines with wallet" | each worker needs its own wallet (section 3) |
| a pod restarted | run the installer again, then the worker; the wallet is in `/workspace/.vibehash` |
| setup stops with "VIBEHASH_PASSWORD" | there's no terminal to ask in: set it, as in section 4 |
