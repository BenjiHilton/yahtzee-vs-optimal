# Play on the web — You vs Optimal

A browser version of the human-vs-algorithm game. Zero extra dependencies
(standard-library HTTP server + the NumPy the engine already uses).

## Run it locally

```bash
cd /Users/benjaminhilton/Desktop/yatzy
python3 -m webapp.server
# open http://localhost:8000
```

Pick who goes first, press **Roll**, tap dice to **keep** them, **Reroll** once,
then click a highlighted box in the **You** column to score. Optimal replies
automatically (a short "thinking…" spinner appears during its endgame moves).
The **Hint** button shows what the algorithm would do on your turn.

### Keyboard-first (no mouse needed)

Every input has a shortcut, shown on the page (key chips on the dice, buttons,
and each scorecard box) and in a full legend you open with **?**:

* **N** new game &middot; **F** flip who goes first &middot; **?** / **Esc** help
* **R** / **Space** / **Enter** roll &middot; **1–5** keep/un-keep a die &middot;
  **R** / **Space** reroll the rest &middot; **H** hint &middot; **Enter** score the hinted box
* Score in a box by its letter: **Q W E A S D** = Ones…Sixes,
  **Z X C** = 3-Kind / 4-Kind / Full House, **V B Y G** = Sm / Lg Straight / Yahtzee / Chance
  (each box's key lights up gold when you can score there right now).

## Let a friend in another country play

The game logic runs on the server, so your friend just needs to reach it over
the internet. Two options:

### A) Quick & temporary — a tunnel (no deployment)

Keep `python3 -m webapp.server` running locally, then expose it:

```bash
# with cloudflared (no account needed):
cloudflared tunnel --url http://localhost:8000
# or with ngrok:
ngrok http 8000
```

Either prints a public `https://…` URL — send that to your friend and they can
play against the algorithm from anywhere. The tunnel lasts as long as your
machine and the command stay running.

### B) Permanent — deploy to a free host

The repo is deploy-ready (`requirements.txt` + `Procfile`, and the server binds
`0.0.0.0:$PORT`). On Render / Railway / Fly.io:

1. Push this folder to a Git repo.
2. Create a new **Web Service** from it.
3. Build command: `pip install -r requirements.txt`
4. Start command: `python -m webapp.server`

That's it — you'll get a permanent URL. The 12 MB `ev_table.pkl` is loaded once
at startup (~1s); the first request may wait a moment while the container warms.

## Notes

* One shared solver serves everyone with warm caches; a lock serialises the
  heavy win-probability computations, which is fine for a handful of players.
* Each browser holds its own game state and posts it back per action, so the
  server is stateless — restarts don't drop in-progress games (the browser keeps
  the board), and it scales to several simultaneous games trivially.
* This build is **you/your friend vs the optimal AI**. Live human-vs-human in a
  shared room (via a room code + WebSockets) is a natural next step if you want
  it.
