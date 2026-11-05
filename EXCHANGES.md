# Vibe Hash for exchanges, wallets and listing sites

How to follow the chain, spot deposits, send withdrawals and read the supply,
from the node's own API. Everything here is what the code in this repository
does; file and function names point at it.

**Read this first.** Vibe Hash is a **testnet**. VIBE has no value, and
nothing here has been audited. The public network, `vibehash-public-1`, has
one validator, run by the project's owner, who can halt or censor it alone.
There is no mainnet yet. HANDOFF.md, "Before VIBE is a real, listed
currency", lists what stands between this and one. This page is for people
building against the testnet now, and it says what a mainnet will keep.

Units: **1 VIBE = 100,000,000 syn.** Every amount in the API is a whole
number of syn.

## 1. Run your own node

Don't trust someone else's node for deposits. Run your own: it downloads
every block, checks it, and serves the same API to your systems.

1. Get the network's genesis file and check its hash against one you got
   from somewhere else (the project's site, its source, another operator):

   ```bash
   curl -s https://node.vibehash.org/genesis > genesis.json
   curl -s https://node.vibehash.org/status   # "genesis_hash": compare it
   ```

2. Make a wallet for the node. It holds no coins and needs no stake; a node
   needs a key to sign with even when it doesn't vote.

   ```bash
   python -m vibehash wallet create --out follower.json
   ```

3. Start it, pointed at a node of the network. Keep its data folder out of
   OneDrive and other syncing folders (HANDOFF.md says why):

   ```bash
   VIBEHASH_PASSWORD=... python -m vibehash node --genesis genesis.json --wallet follower.json \
       --data-dir /var/lib/vibehash-follower --peer https://node.vibehash.org --port 8545
   ```

A node whose wallet isn't a validator **follows**: it never proposes or
votes, and it takes a block only once it applies to the chain (signatures,
balances, the rewards and slashes every node works out for itself), its
round replays from the record it commits to (`replay.py`), and its commit
certificate checks out (`consensus.check_certificate`). It checks the whole
chain again from genesis every time it starts (`Store.open_chain`), and
refuses a data folder whose blocks or certificates don't. It passes
transactions you send it on to the validators (`p2p.py`).

This works on a chain whose genesis names validators: `vibehash-public-1`
and every network `genesis create --validator` or `vibehash testnet` makes.
A chain whose genesis names none (made before 2026-10-29) is run by a single
node that seals every block; read from that node, and don't start a second
one on it.

`python -m vibehash chain verify --node URL --genesis genesis.json` is the
same check from the outside: it downloads every block, applies and replays
each one, and checks each certificate, trusting nothing the node says.

For people rather than programs, `http://YOUR-NODE:8545/explorer` shows the
chain in a browser: blocks, transactions, addresses, jobs, validators and
the supply, read from the same API.

## 2. Finality

Validators agree on each block with Tendermint's BFT voting
(`consensus.py`, DESIGN.md §7). A block is **final once validators holding
more than two thirds of the stake have signed a precommit for it**. Those
signatures are its **commit certificate**, stored with the block and served
in `/blocks`. There are no forks to wait out and no reorganisations: a block
your node serves is final, since a node adopts a block only with a valid
certificate. **One block is enough.** Two different blocks can be final at
one height only if more than a third of the stake signs both, and that would
leave the signed, conflicting votes behind as evidence (slashing them isn't
built yet: DESIGN.md §9).

A certificate, as `/blocks` serves it:

```json
{
  "block": "82d94a722d6383453479c7f3cba34b06cad0d43b69475127c6462548b454afe0",
  "height": 2,
  "round": 0,
  "votes": [
    {
      "pubkey": "05d8de2a42475ca6b5d8228bb5da97580d7fbc62e7d89eb29dcca6728a66690a",
      "sig": "cdc3e8dd857681e9a77b4a89da8f38e0e158e69134aa1628267f661449da0a16a7259d4a3e7d6f32323e9097c272ecb027f379b69cb5fdbd597b5a1e086b6802"
    }
  ]
}
```

To check one yourself: `block` must be the block's `hash` and `height` its
height. Each vote is an Ed25519 signature by `pubkey` over the canonical
JSON (below) of `[chain_id, "precommit", height, round, block_hash]`. Count
each validator once, by its address (`vibe1` and the first 36 hex digits of
sha256(pubkey)), and only if it was in the validator set the block was voted
by, the set in the state just before it (`/validators` shows today's). The
stake of those that signed must be more than two thirds of that set's whole
stake: `3 × signed > 2 × total`. `consensus.check_certificate` is the
reference.

## 3. Following the chain and spotting deposits

Poll your node's `/status` (every few seconds is plenty: a block seals each
round while a job runs, 38 s on a node's defaults, and a transaction with no
job running gets a block of its own within seconds), and when `height`
moves, fetch the new blocks:

```
GET /blocks?from=N&to=M
```

It answers at most 100 blocks, and fewer when they're big (a reply stops at
about 4 MB): ask again from the last height you got plus one. A block, as
your node serves it (block 2 of a local `vibehash testnet`: a training round that also
carried one transfer, with its job's fee release and three rewards):

```json
{
  "height": 2,
  "prev_hash": "4ef2f9188b28325decfe0cc324a25578e665472a4823264f620eaf23e0a4a941",
  "timestamp": 1793849001,
  "summary": {
    "job_id": "demo-1",
    "accepted_units": 7,
    "worker_flops": {
      "vibe196c8f08ba8683db56c57cf459eff27bc5377": 12902400,
      "vibe1b9e97fa2e1661bcd73bb16be45eb32f9f50a": 12902400
    },
    "verifier_flops": {},
    "slashes": [],
    "accepted": [
      "vibe196c8f08ba8683db56c57cf459eff27bc5377",
      "vibe1b9e97fa2e1661bcd73bb16be45eb32f9f50a"
    ],
    "missed": [],
    "params_hash": "7a054b5487d4f17d27e1ba9911d093b3a8dcfff5d369549b930331cb6876082a",
    "proposer": "vibe1e08bd1ed26ddd4d9f5880492211782e40086",
    "record_hash": "65652a150385320ec3226d1533e0f08d232453255c2718da8da71b07ccbdeaa2"
  },
  "txs": [
    {
      "kind": "transfer",
      "sender": "vibe1b45a2af293ec0b6d993718e94c04aea667bb",
      "recipient": "vibe18969b81df88f982576cecdedef6886224c96",
      "amount": 12500000000,
      "nonce": 1,
      "memo": "deposit 1042",
      "pubkey": "3767d75cece68f7a1cc582260c16eb512b62cc8fd4ae9160b76945a130af3d98",
      "signature": "89200223e58d625114ef84fb24d88b75b27e9fa3d6383b562707ab85e5fcc70d511c2bd0d80a4f68c3ea385a155daa817ec94eaa51ceeb9df58fa6f3f2524d08",
      "chain_id": "vibehash-local-88b599",
      "fee": 10000,
      "txid": "80a69b3c02cf9caad496e7501962ddbaae64d21b0028f3dc70e49cefa6ef2252"
    },
    {
      "kind": "fee_release",
      "sender": "system",
      "recipient": "system",
      "amount": 583333333,
      "nonce": 0,
      "memo": "demo-1",
      "pubkey": "",
      "signature": "",
      "chain_id": "vibehash-local-88b599",
      "txid": "8dbb5d33438f3a0b002f220c05033b7087cad7bad8f6c391032448bf8c48a45f"
    },
    {
      "kind": "reward",
      "sender": "system",
      "recipient": "vibe196c8f08ba8683db56c57cf459eff27bc5377",
      "amount": 2233333333,
      "nonce": 0,
      "memo": "work",
      "pubkey": "",
      "signature": "",
      "chain_id": "vibehash-local-88b599",
      "txid": "af6aa4c3af4901b8346c118bc6acfaaf80b714366e106d50b52bdadf4089de91"
    },
    {
      "kind": "reward",
      "sender": "system",
      "recipient": "vibe1b9e97fa2e1661bcd73bb16be45eb32f9f50a",
      "amount": 2233333333,
      "nonce": 0,
      "memo": "work",
      "pubkey": "",
      "signature": "",
      "chain_id": "vibehash-local-88b599",
      "txid": "1b936e8646abb8250ba0b8279d76d2c78c091b073358c2f50154a4f296f9bc25"
    },
    {
      "kind": "reward",
      "sender": "system",
      "recipient": "vibe1e08bd1ed26ddd4d9f5880492211782e40086",
      "amount": 279166666,
      "nonce": 0,
      "memo": "propose",
      "pubkey": "",
      "signature": "",
      "chain_id": "vibehash-local-88b599",
      "txid": "3cf79fc076759608702539ce6335fdac5ea9a8928d9ae03aa4b2384f14a50f93"
    }
  ],
  "chain_id": "vibehash-local-88b599",
  "proposer_pubkey": "05d8de2a42475ca6b5d8228bb5da97580d7fbc62e7d89eb29dcca6728a66690a",
  "signature": "1411bf7d1cf21ed7d747767dc85edce4f68e41234152c1015b98aafa588aad07e914e0fd06c938943b0933075740f09784e0f8543715488280b8724d33491d06",
  "version": 3,
  "hash": "82d94a722d6383453479c7f3cba34b06cad0d43b69475127c6462548b454afe0",
  "proposer": "vibe1e08bd1ed26ddd4d9f5880492211782e40086",
  "certificate": {
    "block": "82d94a722d6383453479c7f3cba34b06cad0d43b69475127c6462548b454afe0",
    "height": 2,
    "round": 0,
    "votes": [
      {
        "pubkey": "05d8de2a42475ca6b5d8228bb5da97580d7fbc62e7d89eb29dcca6728a66690a",
        "sig": "cdc3e8dd857681e9a77b4a89da8f38e0e158e69134aa1628267f661449da0a16a7259d4a3e7d6f32323e9097c272ecb027f379b69cb5fdbd597b5a1e086b6802"
      }
    ]
  }
}
```

- `hash`, `proposer` and each transaction's `txid` are worked out by the node
  for you. Block and transaction ids are sha256 over length-prefixed parts
  (`crypto.h256`), so they're awkward to compute outside Python; everything
  else in the block is what the proposer signed. `version` is missing from
  version 1 blocks, and `fee` from transactions that pay none.
- `timestamp` is Unix seconds. `summary` is the round the block sealed (the
  job, the work accepted and paid for, the slashes), or `null` for a block
  that only carries transactions.

**A deposit is a `transfer` whose `recipient` is one of your addresses.**
Its `amount` (syn) is what the recipient receives. The sender also paid
`fee` (syn), which goes to the block's proposer. `memo` is the sender's
note, at most 256 bytes; use it as a deposit reference if you give users
one address with a tag rather than an address each. Nothing checks a memo.

Transactions come in three kinds:

| Kind | Who signs | What it does |
|---|---|---|
| `genesis` | nobody (block 0 only) | allocations, stakes, bootstrap stake and the network's rules |
| `transfer`, `stake`, `unstake`, `post_job`, `cancel_job`, `validator` | the `sender`, with `pubkey` and `signature` | user transactions: they need the sender's next `nonce` and the chain id |
| `slash`, `fee_release`, `reward` | nobody (`pubkey` and `signature` are empty, `sender` is `system` for rewards and fee releases) | system transactions: every node works them out from the block's round summary and refuses a block whose list differs (`ledger.apply_block`) |

System transactions move coins too: a `reward` (memo `work`, `verify` or
`propose`) pays `recipient`, a `slash` takes stake from `sender` and gives
it to `recipient` or burns it (`recipient` `burn`), and a `fee_release`
moves a job's budget into the round's pool (`memo` is the job). A block
lists its slashes first, then its user transactions, then its fee release
and rewards.

**Balances also change without a transaction.** A block's transaction fees
are credited to its proposer, unstaked coins come back to the balance 20
blocks after their `unstake` (`ledger.UNBONDING_ROUNDS`), and on a network
with earn-in a reward to an address with less than 5 VIBE staked goes into
its stake first. So credit
deposits from `transfer` transactions, and reconcile against
`GET /account/ADDRESS`, which is the state after the tip:

```json
{
  "address": "vibe18969b81df88f982576cecdedef6886224c96",
  "balance": 12500000000,
  "stake": 0,
  "bootstrap": 0,
  "unbonding": [],
  "nonce": 0,
  "reputation": 0.8
}
```

`balance` is spendable. `stake` can't be spent (`bootstrap` of it never can),
`unbonding` lists `[amount, height]` pairs that become spendable after that
height, and `nonce` is how many transactions the address has sent. Keep
deposit addresses for deposits only: don't stake or mine with them.

## 4. Addresses

An address is `vibe1` followed by 36 hex digits: the first 18 bytes of
sha256(the Ed25519 public key), in small letters (`crypto.address_of`).
That **canonical form** is what blocks, the API and the ledger hold. Keys are
derived from 24 BIP39 recovery words along `m/44'/1'/0'/0'/0'` (SLIP-0010
Ed25519; coin type 1, "testnet, all coins"; a mainnet will register its
own), in `mnemonic.py`.

People see a **checksummed form**, with some letters capitalised: EIP-55's
scheme with sha256. Take h = sha256 of the canonical address (with
`vibe1`); each letter a-f among the 36 digits is a capital when the hex digit
of h at the same position is 8 or more. Lowercased, it's the canonical
address again.

To validate what a user types (`crypto.parse_address` does exactly this):

1. Drop spaces around it. It must start `vibe1` (in any case) and have 36
   more characters, each 0-9 or a-f in either case.
2. All small letters: take it as it is (unchecked: that's how addresses were
   shown before checksums).
3. Any capital letter: the capitals must be exactly the checksummed form's,
   or refuse it; all capitals is checked too.
4. Store and send the lowercase form.

**Send to the canonical form only.** The ledger credits any recipient that
starts `vibe1`, and coins sent to any other spelling, a checksummed one
included, can never be spent. Nodes refuse to relay such a transfer
(`ledger._relayable`), and `NodeClient.send_tx` lowercases a checksummed
recipient before signing, but a signer of your own must do the same.
`GET /account/ADDRESS` reads a checksummed address as the canonical one.

## 5. Sending withdrawals

The simplest way is the Python client, from a hot wallet file:

```python
from vibehash import wallet
from vibehash.client import NodeClient

keys = wallet.load("hot-wallet.json", password)
node = NodeClient("http://127.0.0.1:8545")
txid = node.send_tx(keys, "transfer", "vibe1...", 12_500_000_000, memo="withdrawal 1042")  # 125 VIBE
```

`send_tx` checks the recipient (section 4), reads the sender's nonce from
`/account`, pays the node's minimum fee (`min_fee` in `/status`; 0.0001 VIBE
on a node's defaults) unless you pass `fee=`, signs, and posts. The command
line does the same without a memo:
`python -m vibehash send --to ADDRESS --amount 125 --wallet hot-wallet.json`.

**Nonces.** Each user transaction carries its sender's next nonce: the
`nonce` `/account` shows, plus one for each of that sender's transactions
still waiting. `send_tx` reads it from `/account`, so it can't send a second
transaction before the first is in a block (the node refuses it, "bad
nonce"). To send several at once, sign them yourself with consecutive
nonces (`Tx.signed(keys, "transfer", recipient, amount, nonce, memo, chain_id,
fee)` and `POST /tx` with `tx.to_dict()`); the node takes a transaction that
applies after those it already holds. There's no replace-by-fee: a waiting
transaction's nonce can't be reused, and the mempool lives in the node's
memory, so a restart empties it. If a withdrawal isn't in a block after a
few rounds, check `/account`'s nonce before signing it again.

**What gets signed.** `POST /tx` takes:

```json
{
  "kind": "transfer",
  "sender": "vibe1b45a2af293ec0b6d993718e94c04aea667bb",
  "recipient": "vibe18969b81df88f982576cecdedef6886224c96",
  "amount": 12500000000,
  "nonce": 1,
  "memo": "deposit 1042",
  "pubkey": "3767d75cece68f7a1cc582260c16eb512b62cc8fd4ae9160b76945a130af3d98",
  "signature": "89200223e58d625114ef84fb24d88b75b27e9fa3d6383b562707ab85e5fcc70d511c2bd0d80a4f68c3ea385a155daa817ec94eaa51ceeb9df58fa6f3f2524d08",
  "chain_id": "vibehash-local-88b599",
  "fee": 10000
}
```

The signature is Ed25519, by the key whose address is `sender`, over the
canonical JSON of `[chain_id, kind, sender, recipient, amount, nonce, memo]`,
with `fee` appended when it isn't 0 (`Tx.payload`). Canonical JSON is
Python's `json.dumps(obj, sort_keys=True, separators=(",", ":"))`: no
spaces, and characters past ASCII written as `\uXXXX`. The node answers
`{"ok": true, "txid": "..."}`, or a 400 saying why not (a fee below its
minimum, a bad nonce, too little balance, a recipient that isn't canonical, a
memo over 256 bytes).

**Seeing it included.** Look for the `txid` in the blocks after it, as for
deposits; once it's in a block your node serves, it's final.

## 6. Supply

`GET /supply` says how much VIBE exists and where it is, from the state
every node derives from the blocks (`supply.py`), so any honest node answers
the same at the same height:

```json
{
  "height": 2,
  "unit": "syn",
  "syn_per_vibe": 100000000,
  "total": 1005162499999,
  "circulating": 953745833332,
  "unspendable": 0,
  "staked": 2000000000,
  "bootstrap": 1000000000,
  "unbonding": 0,
  "escrow": 49416666667,
  "genesis": 1001000000000,
  "minted": 5000000000,
  "burned": 837500001,
  "max_supply": null,
  "schedule_issuance": 2099999997690,
  "subsidy": 5000000000,
  "vibe": {
    "total": "10051.62499999",
    "circulating": "9537.45833332",
    "unspendable": "0",
    "staked": "20",
    "bootstrap": "10",
    "unbonding": "0",
    "escrow": "494.16666667",
    "genesis": "10010",
    "minted": "50",
    "burned": "8.37500001",
    "schedule_issuance": "20999.9999769",
    "subsidy": "50"
  },
  "economics": {
    "name": "testnet",
    "initial_subsidy": 5000000000,
    "halving_interval": 210,
    "max_supply": 0,
    "count": "paid_rounds"
  },
  "paid_rounds": 1,
  "halving": {
    "counts": "paid_rounds",
    "interval": 210,
    "halvings": 0,
    "next_at": 210,
    "to_go": 209,
    "subsidy_after": 2500000000
  }
}
```

- Where the coins are: `circulating` (balances at real addresses: what keys
  can spend), `unspendable` (balances at recipients no key has, from
  transfers to a mistyped address before nodes refused them), `staked`
  (`bootstrap` of it is the genesis's validator stake, which votes but can
  never be withdrawn or spent, and is burned when its validator leaves),
  `unbonding` and `escrow` (job budgets not yet paid out). They add up to
  `total`.
- How it came to be: `total` = `genesis` + `minted` − `burned`. Burned coins
  are the burned half of slashes, each round's pool nobody earned, and
  bootstrap stake whose validator left. On a network whose schedule counts
  blocks (`economics` without `"count": "paid_rounds"`, as on
  `vibehash-public-1`), every round mints its subsidy and burns what nobody
  earned, so empty rounds show up in both `minted` and `burned`.
- The schedule: `max_supply` is the cap on all coins ever (`null`: none, as
  on the testnet schedule), `schedule_issuance` every subsidy the schedule
  will ever pay before the cap, `subsidy` what the next round that pays for
  work mints, `paid_rounds` how many have (on schedules that count them; else
  `null`), and `halving` where the next halving is, on the clock the schedule
  counts (`counts`: `paid_rounds` or `blocks`).
- `vibe` gives each amount as an exact decimal VIBE string.

`GET /supply/total` and `GET /supply/circulating` answer one plain-text
number in VIBE, such as `10524.0001`, with `Content-Type: text/plain`: the
format CoinGecko and CoinMarketCap ask for. "Circulating" here is spendable
balances only; a listing site that counts staked coins as circulating can
add `staked` from `/supply`.

## 7. Versions and upgrades

`/status` says which software a node runs and which rules its chain is on:

```json
{
  "chain_id": "vibehash-local-88b599",
  "genesis_hash": "6c618b17570654b7bb5830db68cfb1404e77fbe2d19706d9199de46a6517225d",
  "height": 2,
  "tip": "82d94a722d6383453479c7f3cba34b06cad0d43b69475127c6462548b454afe0",
  "min_fee": 10000,
  "subsidy": 5000000000,
  "version": "0.5.17",
  "block_version": 3,
  "tip_version": 3,
  "min_client": "0.5.14",
  "paid_rounds": 1
}
```

- `version`: the node's release. `block_version`: the version of the blocks
  it makes (`ledger.VERSION`). `tip_version`: the version of the chain's
  newest block (the genesis counts as 1). A chain's block versions never go
  down, and a node refuses a block of a version it doesn't know (DESIGN.md
  §10, "Block versions"). When `tip_version` is newer than your node's
  `block_version`, your node has stopped following: upgrade it.
- `min_client`: the oldest release whose workers the node takes (`vibehash
  node --min-client`, default 0.5.14). Clients send `User-Agent:
  vibehash/VERSION`, and a worker registering from an older release is told
  to upgrade. Clients that send no version are taken as before.
- `paid_rounds`: rounds that paid for work so far, on networks whose schedule
  counts them.
- Nodes released before these fields (0.5.16 and earlier) have none of
  them, and Vibe Hash's commands work with them as before.

Vibe Hash commands warn in one line when a node is on newer rules than they
know, and `worker` won't start. There's no upgrade process validators agree
on yet: today an upgrade is every validator installing the new release
(`deploy/setup-node.sh` does it for the public node) before a block of a new
version appears.

## 8. What's not there yet

- **No audit, and no mainnet.** Nobody outside the project has reviewed
  ledger, consensus or verification code. VIBE is a testnet coin with no
  value.
- **One validator** on the public network: its operator can stop or censor
  it. Several independent validators come first.
- **No double-sign slashing**: conflicting votes are only kept as evidence.
- **No replace-by-fee or fee estimation**, and the mempool isn't kept across
  a node restart.
- **No transaction lookup by id**, no address index and no websocket: scan
  `/blocks`, as above.
- Transaction signatures name the chain id but not the genesis hash, so a
  transaction could be replayed on a new network that reuses a chain id
  (DESIGN.md §10, "Still open").
- **No multisig or hardware wallet support.** A wallet file is one Ed25519
  key (scrypt and AES-256-GCM, `wallet.py`).
- The ledger takes transfers to non-canonical recipients in blocks from
  nodes without today's relay policy; making it a block rule waits for a
  block version.
