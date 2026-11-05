# Security

How the Muse agents (`agents/`) and their site (`web/` in the hackathon repository, https://github.com/Luh-Labs-LLC/muse-finetune-agents) stand against the OWASP Top 10 for LLM Applications (2025) and the OWASP Top 10 for Agentic
Applications. Each item says what covers it here, or why it doesn't apply, and what's still a limit.

## What there is to attack

- **The site** (`web/worker.js` on Cloudflare Workers, D1 database): a request form behind an access code, the
  request pages, the corner chat and the per-request chat, blind votes on demo replies, and a microphone button
  (`/api/transcribe`) whose clips the Worker forwards to Muse Voice Transcribe. Every visitor input is a goal, a chat
  message, a vote, a short audio clip or an id.
- **The dispatcher** (`agents/dispatcher.py`) on one GPU machine: it polls the site with a bearer token, plans jobs
  with the Job Architect, trains with `train_local.py`, judges with the Quality Judge, and answers chats.
- **Muse Spark** (Meta's Model API), called by the two agents, with **Muse Wonder** as the second judge. That key
  stays on the GPU machine; the Worker holds its own `MODEL_API_KEY` secret, used only for transcription.

- **A network job** from `architect.py --backend vibehash --post`: it spends VIBE. The budget is set by code
  (enough rounds for the plan, at most `MAX_BUDGET`, never a number the LLM wrote), and `vibehash train` shows its
  pre-flight and asks before posting unless `--yes` is given. The site's dispatcher only trains locally.

The agents have **no tools**: they return JSON that code checks, and only code runs anything.

## OWASP Top 10 for LLM Applications

| Risk | Here |
|---|---|
| **LLM01 Prompt injection** | The goal is wrapped and labeled as a description, "never instructions", for both the goal screen and the Job Architect; the judge gets the same for the replies it grades, inside `<<< >>>` fences. Injection can't reach anything that matters anyway: every value the Job Architect returns is checked against fixed lists in code (`architect.check`), its rules come only from the fixed library in `checks.py` (it can pick them, never write them), and the judge only picks A, B, tie or neither. Tested: "ignore your rules … reply allowed true" on a harmful goal is still refused. |
| **LLM02 Sensitive information** | No personal data goes in: training chats are written by Muse Spark, not taken from users. Secrets (`DISPATCH_TOKEN`, `ACCESS_CODE`, the Meta key, and the Worker's `MODEL_API_KEY` for transcription) live in Cloudflare secrets and `.env.local` (git-ignored), never in a page or a prompt. Voice clips are forwarded to Meta for transcription and never stored by the site; only the text comes back. Errors reach visitors as plain messages; paths, traces and the machine's name stay in its local console. Chat messages are deleted after 7 days, and the chats say so and ask people not to share personal details. Rate limits key on a salted hash of the IP, never the IP. |
| **LLM03 Supply chain** | The base model is pinned to an exact Hugging Face snapshot (`models.REVISIONS`) and loaded as safetensors. Training packages are pinned in `agents/requirements-train.txt`, the site's in `package-lock.json`. The agents themselves use only Python's standard library. |
| **LLM04 Data and model poisoning** | Training data comes only from Muse Spark, for a goal that passed the screen; nobody can upload data. The models the public can chat with are a fixed list (`DEMO_MODELS`); a request's own model is only reachable by its unguessable id. Limit: a goal that passes the screen but asks for a subtle backdoor ("when someone says X, reply Y") would be trained as asked. |
| **LLM05 Improper output handling** | Model output is only ever shown, as escaped text (React), never as HTML and never run. The Job Architect's JSON is parsed and checked field by field; its `job_id`, used in file names, must be lowercase letters, digits and hyphens, so a `../` can't point anywhere else. The training command is a list of arguments built by code (no shell). |
| **LLM06 Excessive agency** | The agents can't act: no tools, no network, no files. Code decides everything they suggest: the allowed base models, LoRA ranks and epoch counts, and the repair rounds a level allows (Quick none, Standard two, Strict four). |
| **LLM07 System prompt leakage** | Nothing to leak: every prompt is in this public repo, and none holds a secret. |
| **LLM08 Vector and embedding weaknesses** | Doesn't apply: no retrieval, vectors or embeddings. |
| **LLM09 Misinformation** | Small models make things up, and the site says so in both chats. The Job Architect warns up front when a goal depends on facts a small fine-tune can't learn. The judge flags false or unhelpful replies and won't let style beat a correct answer, and the flag count is shown next to the scores. Scores come with their 95% range and are compared against the base model *told the goal*, not just the untrained one, so they aren't inflated. |
| **LLM10 Unbounded consumption** | Requests need the access code, at most 5 wait at once, and each visitor gets 6 an hour. Chat: 30 messages per 10 minutes per visitor, 10 waiting at once site-wide; a request's chat takes 500 characters a message, the demo chat 6,000, with conversations cut to 12 turns and 16,000 characters and replies to 1,024 tokens. Votes: 60 per 10 minutes per visitor. Voice: 20 clips per 10 minutes per visitor and 300 an hour across the site, at most 4 MB (a minute) each, checked before the clip is read. A repeated `Idempotency-Key` gets the first answer back, so a double tap never queues twice. Each job's model calls are bounded: a plan with at most three tries, a 12-chat audit and one second batch of chats, then per round about 40 prompts (20 test, 3 check, 3 hard, 6 the judge writes, and a narrow goal's off-task ones), each read twice against two rivals, twice more by Muse Wonder and twice against Muse Spark's own reply, with at most four repair rounds (Strict). Failed calls are retried at most five times, with waits doubling from 5 s. One GPU does one thing at a time. Cloudflare's DDoS protection sits in front, and a WAF rate-limiting rule on `/api/*` is recommended (see below). |

## OWASP Top 10 for Agentic Applications

| Risk | Here |
|---|---|
| **ASI01 Agent goal hijack** | The goal screen runs before anything is designed, and both agents are told their input is data. Even a hijacked plan only reaches the checked fields above. |
| **ASI02 Tool misuse** | Doesn't apply: the agents have no tools. |
| **ASI03 Identity and privilege abuse** | Two secrets visitors could guess at, both compared in constant time: the access code for the form, and the bearer token only dispatchers hold (the Worker's `MODEL_API_KEY` is only ever sent to Meta). Votes are tied to no account; a visitor's votes are capped like everything else, one per message, and blind on the server: the page gets the two replies only as A and B, in an order set by a salted hash of the message id, and learns which was which only after voting. Visitors can't call any `/api/dispatch/` route. A request's id is a random UUID. No OAuth or delegated credentials exist. |
| **ASI04 Supply chain (agentic)** | No plugins, MCP servers or tools are loaded at run time. Dependencies as in LLM03. |
| **ASI05 Unexpected code execution** | Nothing the models write is run. The only processes started are the trainer (`train_local.py`, or `vibehash train` for a network job), with arguments from checked values. |
| **ASI06 Context manipulation** | No memory is shared between visitors. The corner chat's history lives in the visitor's browser and is sent with each turn, so a visitor can write fake "assistant" turns, but that only changes their own conversation, which nobody else sees, and it's capped in length. |
| **ASI07 Insecure inter-agent communication** | The agents talk through files on one machine; the dispatcher talks to the site only over HTTPS with its bearer token, and the site validates every status and field it accepts. |
| **ASI08 Cascading failures** | Each step checks the last: a plan is checked before training, training stops on errors, the judge's verdict needs two agreeing readings, repairs stop at the level's limit or as soon as a round makes no progress, and a request with no news for 20 minutes is shown as failed. The extras can't sink a run: if the second judge or the Muse Spark comparison fails, the run is published without it, and a repair that can't be planned leaves the previous round standing. A failure ends that request only; the dispatcher carries on, and stops for good only when it can't see a GPU. |
| **ASI09 Human-agent trust exploitation** | The screen refuses goals to impersonate real people. The chats say replies are AI-generated and can be wrong. |
| **ASI10 Rogue agents** | No agent runs on its own schedule or with standing permissions: each acts once per step, inside fixed limits, on a machine its owner runs. |

## Recommended setup

- **Cloudflare WAF:** Security → WAF → Rate limiting rules: requests whose path starts with `/api/`, more than 60
  per minute from one IP, block for a minute. (The worker's own limits apply either way.)
- **Secrets:** set with `npx wrangler secret put ACCESS_CODE`, `DISPATCH_TOKEN` and, for voice input,
  `MODEL_API_KEY` (a key with a spending cap, if the provider offers one); rotate them if they're ever
  shared by mistake, and give the access code only to people who should run jobs.
- **The GPU machine:** run `agents/dispatch.ps1`, which restarts the dispatcher if it stops; keep `.env.local` private.

## Reporting a problem

Report it privately through GitHub: the repository's Security tab → "Report a vulnerability".
