"use strict";

const CATS = ["Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
  "Three of a Kind", "Four of a Kind", "Full House",
  "Small Straight", "Large Straight", "Yahtzee", "Chance"];

// pip positions (1..9 grid) for each die face
const PIPS = {
  1: [5], 2: [1, 9], 3: [1, 5, 9], 4: [1, 3, 7, 9],
  5: [1, 3, 5, 7, 9], 6: [1, 3, 4, 6, 7, 9],
};

// one letter key per category (shown in the scorecard); avoids 1-5/R/H/N/F
const KEYCAT = ["Q", "W", "E", "A", "S", "D", "Z", "X", "C", "V", "B", "Y", "G"];
const CATKEY = {};
KEYCAT.forEach((ltr, i) => { CATKEY[ltr.toLowerCase()] = i; });

let game = null;
let held = new Set();      // indices of currently held dice
let hintCell = null;       // category name the hint points at
let helpOpen = false;

const $ = (id) => document.getElementById(id);

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return res.json();
}

function showOverlay(on) { $("overlay").classList.toggle("hidden", !on); }

// ---- rendering -------------------------------------------------------------

function die(value, index, locked) {
  const kept = held.has(index);
  const d = document.createElement("div");
  d.className = "die" + (kept ? " held" : "") + (locked ? " locked" : "");
  for (const pos of PIPS[value]) {
    const pip = document.createElement("span");
    pip.className = "pip p" + pos;
    d.appendChild(pip);
  }
  if (kept) {
    const badge = document.createElement("span");
    badge.className = "keep-badge";
    badge.textContent = "KEEP";
    d.appendChild(badge);
  }
  const key = document.createElement("span");   // the number key that toggles this die
  key.className = "key-badge";
  key.textContent = index + 1;
  d.appendChild(key);
  if (!locked) d.onclick = () => { kept ? held.delete(index) : held.add(index); render(); };
  return d;
}

function scoreableCat(c) {
  return !!game && game.phase === "rolled"
    && game.you.filled[String(c)] === undefined
    && game.previews && (CATS[c] in game.previews);
}

function renderDice() {
  const wrap = $("dice");
  wrap.innerHTML = "";
  if (!game || !game.dice) return;
  const locked = game.turn !== "you" || game.phase !== "rolled";
  game.dice.forEach((v, i) => wrap.appendChild(die(v, i, locked)));
}

function cell(player, c) {
  const td = document.createElement("td");
  const filled = player.filled[String(c)];
  const isYou = player === game.you;
  const name = CATS[c];
  const preview = game.previews ? game.previews[name] : undefined;
  const scoreable = isYou && game.phase === "rolled" && filled === undefined && preview !== undefined;

  if (filled !== undefined) {
    td.textContent = filled;
    td.className = "filled";
  } else if (scoreable) {
    td.textContent = "+" + preview;
    td.className = "scoreable" + (preview === 0 ? " zero" : "");
    if (hintCell === name) td.classList.add("hintcell");
    td.onclick = () => scoreBox(name);
  } else {
    td.textContent = "·";
    td.className = "empty";
  }
  return td;
}

function bonusText(p) { return p.upper_bonus_earned ? "35" : (p.upper_total + "/63"); }

function renderBoard() {
  const body = $("card-body");
  body.innerHTML = "";
  if (!game) return;
  CATS.forEach((name, c) => {
    if (c === 6) { // insert the upper-bonus row before the lower section
      const tr = document.createElement("tr");
      tr.className = "bonus";
      tr.innerHTML = `<th class="cat">Upper bonus (63)</th><td>${bonusText(game.you)}</td><td>${bonusText(game.ai)}</td>`;
      body.appendChild(tr);
    }
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.className = "cat";
    th.innerHTML = `<kbd class="catkey${scoreableCat(c) ? " on" : ""}">${KEYCAT[c]}</kbd>${name}`;
    tr.appendChild(th);
    tr.appendChild(cell(game.you, c));
    tr.appendChild(cell(game.ai, c));
    body.appendChild(tr);
  });
  // yahtzee bonus row
  const tr = document.createElement("tr");
  tr.className = "bonus";
  tr.innerHTML = `<th class="cat">Yahtzee bonus (+100 each)</th><td>${game.you.yz_bonus || "·"}</td><td>${game.ai.yz_bonus || "·"}</td>`;
  body.appendChild(tr);

  $("you-total").textContent = game.you.total;
  $("ai-total").textContent = game.ai.total;
}

function renderLog() {
  const log = $("log");
  log.innerHTML = "";
  const entries = (game.log || []).slice(-4).reverse();
  for (const e of entries) {
    const div = document.createElement("div");
    if (e.human) {
      div.className = "entry you";
      div.innerHTML = `<b>You</b> scored <b>${e.category}</b> for <b>${e.points}</b>.`;
    } else {
      div.className = "entry";
      const win = e.win !== null && e.win !== undefined
        ? ` <span class="win">(est. win ${e.win}%)</span>` : "";
      div.innerHTML = `<b>Optimal</b> rolled ${e.rolled.join(" ")}, kept [${e.kept.join(" ") || "&ndash;"}], `
        + `final ${e.final.join(" ")} &rarr; <b>${e.category}</b> for <b>${e.points}</b>.${win}`;
    }
    log.appendChild(div);
  }
}

function renderTurnbar() {
  const bar = $("turnbar");
  if (game.phase === "over") { bar.innerHTML = ""; return; }
  if (game.phase === "await_roll") {
    bar.innerHTML = "<b>Your turn.</b> Press Roll.";
  } else if (game.phase === "rolled") {
    bar.innerHTML = game.rolls_left >= 1
      ? "Tap the dice you want to <b>keep</b> (they turn green), then reroll the rest &mdash; or just click a box to score."
      : "No rerolls left &mdash; click a highlighted box to score.";
  }
}

function renderControls() {
  const over = game.phase === "over";
  $("roll-btn").disabled = !(game.phase === "await_roll" && !over);

  const rr = $("reroll-btn");
  const rerollCount = (game.phase === "rolled" && game.dice) ? (5 - held.size) : 0;
  // Label says exactly how many dice will be rerolled (the ones you did NOT keep).
  rr.querySelector(".blabel").textContent = rerollCount === 5 ? "Reroll all 5"
    : rerollCount > 0 ? `Reroll ${rerollCount} (keep ${held.size})`
    : "Keeping all 5";
  rr.disabled = !(game.phase === "rolled" && game.rolls_left >= 1 && rerollCount > 0 && !over);

  $("hint-btn").disabled = !(game.phase === "rolled" && !over);
}

function renderResult() {
  const r = $("result");
  if (game.phase !== "over") { r.className = "hidden"; return; }
  if (game.result === "you") { r.className = "win"; r.textContent = `You win  ${game.you.total} – ${game.ai.total}! 🎉`; }
  else if (game.result === "ai") { r.className = "loss"; r.textContent = `Optimal wins  ${game.ai.total} – ${game.you.total}.`; }
  else { r.className = "tie"; r.textContent = `Tie  ${game.you.total} – ${game.ai.total}.`; }
}

function render() {
  $("play").classList.toggle("hidden", !game);
  if (!game) return;
  renderBoard();
  renderDice();
  renderTurnbar();
  renderControls();
  renderLog();
  renderResult();
  if (game.error) { $("hint").style.color = "#ff7a7a"; $("hint").textContent = game.error; }
}

// ---- actions ---------------------------------------------------------------

async function newGame() {
  const first = $("first").value;
  showOverlay(first === "ai");
  game = await api("/api/new", { first });
  held.clear(); hintCell = null;
  $("hint").textContent = "";
  showOverlay(false);
  render();
}

async function doRoll() {
  game = await api("/api/roll", { state: game });
  held.clear(); hintCell = null; $("hint").textContent = "";
  render();
}

async function doReroll() {
  const keep = [...held].map((i) => game.dice[i]);
  game = await api("/api/reroll", { state: game, keep });
  held.clear(); hintCell = null; $("hint").textContent = "";
  render();
}

async function scoreBox(name) {
  hintCell = null; $("hint").textContent = "";
  showOverlay(true);                       // AI will reply; may take a moment
  game = await api("/api/score", { state: game, category: name });
  held.clear();
  showOverlay(false);
  render();
}

async function doHint() {
  const r = await api("/api/hint", { state: game });
  if (!r.hint) return;
  $("hint").style.color = "";
  if (r.hint.type === "keep") {
    // highlight which dice to hold
    held.clear();
    const need = [...r.hint.keep];
    game.dice.forEach((v, i) => {
      const k = need.indexOf(v);
      if (k !== -1) { held.add(i); need.splice(k, 1); }
    });
    $("hint").textContent = "Optimal would keep [" + r.hint.keep.join(" ") + "] and reroll the rest.";
  } else {
    hintCell = r.hint.category;
    $("hint").textContent = `Optimal would score in ${r.hint.category} (+${r.hint.points}).`;
  }
  render();
}

$("new-btn").onclick = newGame;
$("roll-btn").onclick = doRoll;
$("reroll-btn").onclick = doReroll;
$("hint-btn").onclick = doHint;

// ---- keyboard control ------------------------------------------------------

function toggleHelp(force) {
  helpOpen = force === undefined ? !helpOpen : force;
  $("help").classList.toggle("hidden", !helpOpen);
}
function flipFirst() {
  const s = $("first");
  s.value = s.value === "you" ? "ai" : "you";
}
function toggleDie(i) {
  if (!game || !game.dice || i >= game.dice.length) return;
  held.has(i) ? held.delete(i) : held.add(i);
  render();
}

document.addEventListener("keydown", (e) => {
  const k = e.key;
  const low = k.length === 1 ? k.toLowerCase() : k;
  const tag = (document.activeElement && document.activeElement.tagName) || "";

  if (k === "?") { toggleHelp(); e.preventDefault(); return; }
  if (helpOpen) { if (k === "Escape") toggleHelp(false); return; }

  // don't act while the AI is computing
  if (!$("overlay").classList.contains("hidden")) return;
  // let a focused button/select handle Space/Enter itself (avoid double-fire)
  if ((k === " " || k === "Enter") && (tag === "BUTTON" || tag === "SELECT")) return;

  if (low === "n") { e.preventDefault(); newGame(); return; }

  if (!game || game.phase === "over") {
    if (low === "f") { e.preventDefault(); flipFirst(); }
    return;
  }

  if (game.phase === "await_roll") {
    if (low === "r" || k === " " || k === "Enter") {
      e.preventDefault();
      if (!$("roll-btn").disabled) doRoll();
    }
    return;
  }

  if (game.phase === "rolled") {
    if (low >= "1" && low <= "5") { e.preventDefault(); toggleDie(parseInt(low, 10) - 1); return; }
    if (low === "r" || k === " ") { e.preventDefault(); if (!$("reroll-btn").disabled) doReroll(); return; }
    if (low === "h") { e.preventDefault(); if (!$("hint-btn").disabled) doHint(); return; }
    if (k === "Enter") {
      if (hintCell && scoreableCat(CATS.indexOf(hintCell))) { e.preventDefault(); scoreBox(hintCell); }
      return;
    }
    if (low in CATKEY) {
      const c = CATKEY[low];
      if (scoreableCat(c)) { e.preventDefault(); scoreBox(CATS[c]); }
      return;
    }
  }
});

$("help-btn").onclick = () => toggleHelp();
$("help-close").onclick = () => toggleHelp(false);
$("help").onclick = (e) => { if (e.target.id === "help") toggleHelp(false); };
