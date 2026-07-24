# Deploying the server so your friend can play

The app is friend-vs-optimal-AI: your friend opens a URL and plays the
algorithm. Nothing about it is machine-specific — it binds `0.0.0.0:$PORT`, holds
no server-side session state, and loads the 12 MB `ev_table.pkl` at startup
(~230 MB resident, comfortably inside free-tier limits).

There are two ways to get a link to your friend.

## Option 1 — Instant link, tied to your machine (ngrok)

Fastest way to play *right now*. Keep the local server running and expose it:

```bash
python3 -m webapp.server          # terminal 1  (http://localhost:8000)
ngrok http 8000                   # terminal 2  -> prints a public https URL
```

Send your friend the `https://….ngrok-free.app` URL. Caveats: it only works
while your computer and the `ngrok` command are running, and the free URL
changes each time you restart ngrok. Great for a quick game, not for "always on."

## Option 2 — Always-on hosted server (recommended for a friend abroad)

A permanent URL that works whether or not your laptop is on. The repo is ready
to deploy as-is.

### Render (free, easiest)

1. Push this folder to a GitHub repo.
2. On https://render.com → **New → Blueprint**, pick the repo. It reads
   `render.yaml` and deploys automatically.
   (Or **New → Web Service**, Build: `pip install -r requirements.txt`,
   Start: `python -m webapp.server`.)
3. You get a permanent `https://yahtzee-vs-optimal.onrender.com` URL.

Note: the free plan sleeps after ~15 min idle; the next visit waits a few seconds
while it reloads the table. Fine for casual play.

### Any container host (Fly.io, Railway, Cloud Run, a VPS…)

A `Dockerfile` is included:

```bash
docker build -t yahtzee .
docker run -p 8000:8000 yahtzee      # test locally
```

Then push the image to your host of choice. `fly launch` (Fly.io) auto-detects
the Dockerfile; Railway deploys from the repo directly.

## What your friend needs

Just the URL and a browser — desktop or phone. No install, no account. The game
is keyboard- and mouse-friendly (press `?` in-game for shortcuts).
