#!/usr/bin/env bash
# Start the "You vs Optimal" server and expose it with a public URL that
# auto-reconnects if the free tunnel drops.
#
#   ./serve_public.sh
#
# Keep this running while your friend plays; press Ctrl-C to stop everything.
# The free localhost.run URL changes whenever the tunnel reconnects -- for a
# stable, always-on address, deploy instead (see DEPLOY.md).

set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"

cleanup() {
  echo
  echo "shutting down..."
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
  [[ -n "${SSH_PID:-}" ]] && kill "$SSH_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

# 1) start the local server if it isn't already answering
if ! curl -s -m 3 -o /dev/null "http://localhost:$PORT/"; then
  echo "starting server on :$PORT ..."
  PORT="$PORT" python3 -m webapp.server >/tmp/yahtzee_server.log 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 30); do
    curl -s -m 2 -o /dev/null "http://localhost:$PORT/" && break || sleep 1
  done
fi
echo "server ready at http://localhost:$PORT"

# 2) keep a public tunnel alive, printing the URL each time it (re)connects
echo "opening public tunnel (auto-reconnects if it drops)..."
while true; do
  ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=20 \
      -o ExitOnForwardFailure=yes -R 80:localhost:"$PORT" localhost.run 2>&1 \
    | while IFS= read -r line; do
        url="$(printf '%s' "$line" | grep -oE 'https://[a-z0-9]+\.lhr\.life' || true)"
        [[ -n "$url" ]] && { echo; echo "  ==> PUBLIC URL:  $url"; echo "      (send this to your friend)"; echo; }
      done
  echo "tunnel dropped; reconnecting in 3s..."
  sleep 3
done
