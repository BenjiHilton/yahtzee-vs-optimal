"""Zero-dependency web server for 'You vs Optimal' Yahtzee.

Runs on the standard library only (plus NumPy, which the engine already uses):

    python -m webapp.server            # then open http://localhost:8000
    PORT=8000 python -m webapp.server

The browser holds the game state and posts it back with each action; the server
is authoritative for dice rolls, rule checks, and the AI's moves.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# allow running as `python webapp/server.py` too
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yahtzee import game  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class Handler(BaseHTTPRequestHandler):
    server_version = "YahtzeeVs/1.0"

    # -- helpers ------------------------------------------------------------
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    # -- routes -------------------------------------------------------------
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
        if self.path == "/app.js":
            return self._file(os.path.join(STATIC_DIR, "app.js"), "application/javascript")
        if self.path == "/styles.css":
            return self._file(os.path.join(STATIC_DIR, "styles.css"), "text/css")
        self.send_error(404)

    def do_POST(self):
        try:
            data = self._read_json()
            if self.path == "/api/new":
                return self._json(game.new_game(first=data.get("first", "you")))
            if self.path == "/api/roll":
                return self._json(game.api_roll(data["state"]))
            if self.path == "/api/reroll":
                return self._json(game.api_reroll(data["state"], data.get("keep", [])))
            if self.path == "/api/score":
                return self._json(game.api_score(data["state"], data["category"]))
            if self.path == "/api/hint":
                return self._json(game.api_hint(data["state"]))
            self.send_error(404)
        except Exception as exc:  # pragma: no cover
            self._json({"error": str(exc)}, code=500)

    def log_message(self, fmt, *args):
        pass  # quiet


def main():
    port = int(os.environ.get("PORT", "8000"))
    print("Warming up the optimal solver...")
    game.get_solver()  # load the EV table now, not on the first click
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print("Yahtzee 'You vs Optimal' running at http://localhost:%d" % port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
