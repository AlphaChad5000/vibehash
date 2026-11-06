#!/usr/bin/env bash
# Set up a public Vibe Hash node on a fresh Ubuntu 24.04 server. Run it as root (or with sudo once root can't log in):
#
#   sudo bash setup-node.sh --domain node.vibehash.org [--fund vibe1...=10000] [--email you@example.com]
#
# It installs Vibe Hash from PyPI, makes the node's wallet and the network's genesis, gets a Let's Encrypt
# certificate, and runs the node as a service that serves HTTPS on port 443 and restarts itself after a crash or
# reboot. The genesis makes this node the network's first validator, with bootstrap stake (it votes, but can
# never be spent), and lets workers start with nothing and earn their stake. --fund gives one wallet VIBE to pay
# for the first training jobs, since nobody else starts with any.
#
# Running it again upgrades Vibe Hash and restarts the node; the wallet, genesis and chain are kept.
#
# A second network on the same server (an Evo Hash testnet beside the main one, say) is a named node with its own
# port, folders and service, sharing the server's Vibe Hash install:
#
#   sudo bash setup-node.sh --name evo --domain evo.vibehash.org --port 8443 --chain-id vibehash-evo-1 --es \
#     --stake vibe1...=10 --fund vibe1...=10000
#
# --name NAME keeps it in /var/lib/vibehash-NAME and /etc/vibehash-NAME, run by the service vibehash-node-NAME.
# --es makes a network that takes Evo Hash jobs, and --stake gives a wallet stake from the start (an Evo job's work
# is checked only by staked workers, so newcomers can earn theirs only once someone is staked). Both, like --fund,
# only count when the genesis is made, the first time. Upgrading one node upgrades the shared install, so restart
# the others too (systemctl restart 'vibehash-node*') to run the same release.
# The domain's A record must already point at this server. Getting the certificate accepts Let's Encrypt's terms
# (https://letsencrypt.org/repository/). The email is optional: certificates renew by themselves, and Let's Encrypt
# only uses it for account notices.
set -euo pipefail

DOMAIN="" EMAIL="" FUND="" CHAIN_ID="vibehash-public-1" ECONOMICS="mainnet" NAME="" PORT=443 ES="" STAKES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --email) EMAIL="$2"; shift 2 ;;
    --fund) FUND="$2"; shift 2 ;;
    --chain-id) CHAIN_ID="$2"; shift 2 ;;
    --economics) ECONOMICS="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --es) ES=1; shift ;;
    --stake) STAKES+=("$2"); shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done
if [ -z "$DOMAIN" ]; then
  echo "usage: bash setup-node.sh --domain node.example.org [--fund vibe1...=VIBE] [--email you@example.org]"
  echo "       [--name NAME --port PORT --chain-id ID] [--es] [--stake vibe1...=VIBE]"
  exit 1
fi
[[ -z "$NAME" || "$NAME" =~ ^[a-z0-9][a-z0-9-]{0,30}$ ]] || { echo "--name takes lowercase letters, digits and dashes"; exit 1; }
[[ "$PORT" =~ ^[0-9]+$ ]] && [ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || { echo "--port takes a port number"; exit 1; }
if [ -n "$NAME" ] && [ "$CHAIN_ID" = "vibehash-public-1" ]; then
  echo "a named node runs its own network: give it its own --chain-id (vibehash-public-1 is the main node's)"; exit 1
fi
[ "$(id -u)" = 0 ] || { echo "run it as root"; exit 1; }

SUFFIX=${NAME:+-$NAME}  # "" for the main node, whose folders and service keep their names
DATA=/var/lib/vibehash$SUFFIX ETC=/etc/vibehash$SUFFIX VENV=/opt/vibehash SERVICE=vibehash-node$SUFFIX
VH="$VENV/bin/vibehash" WALLET="$DATA/node-wallet.json" GENESIS="$DATA/genesis.json"
if [ "$PORT" = 443 ]; then URL="https://$DOMAIN"; else URL="https://$DOMAIN:$PORT"; fi
say() { printf '\n== %s\n' "$*"; }
as_node() { (cd "$DATA" && sudo -u vibehash --preserve-env=VIBEHASH_PASSWORD,VIBEHASH_HOME "$@"); }

say "Checking $DOMAIN points at this server"
POINTS_AT=$(getent ahostsv4 "$DOMAIN" | awk 'NR == 1 { print $1 }' || true)
if ! hostname -I | tr ' ' '\n' | grep -qxF "${POINTS_AT:-none}"; then
  echo "$DOMAIN points at ${POINTS_AT:-nothing}, not this server ($(hostname -I | awk '{ print $1 }'))."
  echo "Add an A record for it at your DNS provider, wait a few minutes, and run this again."
  exit 1
fi

say "Installing packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q python3-venv certbot ufw

say "Installing Vibe Hash"
id vibehash >/dev/null 2>&1 || useradd --system --home-dir /var/lib/vibehash --create-home --shell /usr/sbin/nologin vibehash
[ "$DATA" = /var/lib/vibehash ] || install -d -m 750 -o vibehash -g vibehash "$DATA"
[ -x "$VENV/bin/pip" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q --upgrade --no-cache-dir vibehash
"$VENV/bin/pip" show vibehash | grep '^Version'

# The wallet's password, made once and readable only by root and the node.
install -d -m 750 -g vibehash "$ETC"
if [ ! -f "$ETC/env" ]; then
  (umask 077; printf 'VIBEHASH_PASSWORD=%s\nVIBEHASH_HOME=%s\n' "$(head -c 33 /dev/urandom | base64 | tr -d '+/=')" "$DATA" > "$ETC/env")
  chgrp vibehash "$ETC/env"
  chmod 640 "$ETC/env"
fi
set -a; . "$ETC/env"; set +a

if [ ! -f "$WALLET" ]; then
  say "Creating the node's wallet. Write down its recovery words: they're shown only now"
  as_node "$VH" wallet create --out "$WALLET"
fi
ADDRESS=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['address'])" "$WALLET")

if [ ! -f "$GENESIS" ]; then
  say "Creating the network's genesis"
  args=(genesis create --chain-id "$CHAIN_ID" --economics "$ECONOMICS" --out "$GENESIS"
        --bootstrap "$ADDRESS=10" --validator "$ADDRESS=$URL")
  [ -n "$FUND" ] && args+=(--alloc "$FUND")
  [ -n "$ES" ] && args+=(--es)
  for s in "${STAKES[@]}"; do args+=(--stake "$s"); done
  as_node "$VH" "${args[@]}"
fi

say "Opening the firewall for HTTP (certificate renewals) and HTTPS"
# SSH is opened to everyone only the first time, while the firewall is still off, so this run can't lock you out.
# Once it's on, its SSH rules are left as they are: an upgrade mustn't undo a narrower rule made afterwards (SSH
# allowed only from your own addresses, or only over Tailscale; README, "Lock down the server").
firewall=$(ufw status || true)  # read in full first: with pipefail, grep -q quitting early could fail ufw's write
if ! grep -q '^Status: active' <<<"$firewall"; then
  ufw allow OpenSSH >/dev/null
fi
ufw allow 80/tcp >/dev/null
ufw allow 443/tcp >/dev/null
[ "$PORT" = 443 ] || ufw allow "$PORT/tcp" >/dev/null
ufw --force enable >/dev/null

say "Getting an HTTPS certificate for $DOMAIN"
# The node reads its own copy of the certificate; renewals (certbot's timer) refresh it and restart the node.
install -d -m 755 /etc/letsencrypt/renewal-hooks/deploy
HOOK=/etc/letsencrypt/renewal-hooks/deploy/$SERVICE.sh
cat > "$HOOK" <<EOF
#!/bin/sh
# certbot runs every deploy hook after any renewal: act only on this node's domain (RENEWED_DOMAINS is unset when
# this script runs it by hand, below).
case " \${RENEWED_DOMAINS:-$DOMAIN} " in *" $DOMAIN "*) ;; *) exit 0 ;; esac
install -d -m 750 -g vibehash $ETC/tls
install -m 640 -g vibehash /etc/letsencrypt/live/$DOMAIN/fullchain.pem $ETC/tls/cert.pem
install -m 640 -g vibehash /etc/letsencrypt/live/$DOMAIN/privkey.pem $ETC/tls/key.pem
systemctl try-restart $SERVICE 2>/dev/null || true  # (on the first install the service doesn't exist yet)
EOF
chmod 755 "$HOOK"
if [ -n "$EMAIL" ]; then contact=(-m "$EMAIL"); else contact=(--register-unsafely-without-email); fi
certbot certonly --standalone --non-interactive --agree-tos "${contact[@]}" -d "$DOMAIN" --keep-until-expiring
sh "$HOOK"

say "Running the node as a service"
cat > "/etc/systemd/system/$SERVICE.service" <<EOF
[Unit]
Description=Vibe Hash node ($URL)
After=network-online.target
Wants=network-online.target

[Service]
User=vibehash
Group=vibehash
EnvironmentFile=$ETC/env
WorkingDirectory=$DATA
ExecStart=$VH node --genesis $GENESIS --wallet $WALLET --data-dir $DATA/data --host 0.0.0.0 --port $PORT --announce $URL --tls-cert $ETC/tls/cert.pem --tls-key $ETC/tls/key.pem
Restart=always
RestartSec=5
AmbientCapabilities=CAP_NET_BIND_SERVICE
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=$DATA

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable "$SERVICE" >/dev/null
systemctl restart "$SERVICE"

say "Checking it answers"
for _ in $(seq 1 30); do
  curl -fsS "$URL/status" >/dev/null 2>&1 && break
  sleep 1
done
if status=$(curl -fsS "$URL/status" 2>/dev/null); then
  echo "$status" | python3 -c "import json, sys; s = json.load(sys.stdin); print(f\"network {s['chain_id']}, block {s['height']}\")"
  printf '\nDone. The node address is %s\n  validator %s\n  logs:     journalctl -u %s -f\n' "$URL" "$ADDRESS" "$SERVICE"
else
  echo "The node didn't answer. Its log says:"
  journalctl -u "$SERVICE" -n 30 --no-pager
  exit 1
fi
