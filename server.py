#!/usr/bin/env python3
"""
Pairwise candidate ranking — local-only web app.

Presents two candidate PDFs side by side, asks which you prefer, and builds a
ranking from your (subjective, possibly contradictory) judgements.

Method
------
Phase 1  Complete round-robin: every pair compared exactly once. Pair order is
         shuffled and sides randomised to cancel position/fatigue bias.
Phase 2  Adaptive: repeatedly ask the single most informative remaining pair,
         refitting after each answer, until the ranking is confident.

Ranking is Bradley-Terry (maximum likelihood via the MM algorithm), which
models P(i beats j) = p_i / (p_i + p_j). It handles contradictions gracefully:
an intransitive triad just pulls the strengths together rather than breaking
the ranking. A weak prior (virtual wins/losses against a fixed phantom
opponent) keeps undefeated or winless candidates finite and anchors the scale.

Confidence comes from bootstrap resampling of the comparison list: for each
adjacent pair in the ranking, what fraction of resamples agree on their order.

No third-party dependencies. Nothing leaves this machine.
"""

import http.server
import json
import math
import mimetypes
import os
import random
import re
import socketserver
import threading
import urllib.parse
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_DIR = os.path.join(HERE, "candidates")
STATE_PATH = os.path.join(HERE, "state.json")
PORT = 8765

# --- tuning ---------------------------------------------------------------
PRIOR = 0.5           # virtual wins & losses vs a phantom opponent of strength 1
CONFIDENCE_TARGET = 0.90   # required bootstrap agreement on every adjacent pair
EXTRA_PER_CANDIDATE = 3    # phase-2 budget = seeding + 3 per candidate

# A complete round robin is n(n-1)/2 comparisons — fine at 6 (15), punishing at
# 20 (190). Above this many candidates, phase 1 seeds each candidate with
# SEED_ROUNDS comparisons instead and lets the adaptive phase do the rest.
FULL_ROUND_ROBIN_MAX = 10
SEED_ROUNDS = 5

# Bootstrap refits Bradley-Terry many times after every single click, so its
# cost is what the user feels. Fits are warm-started from the point estimate,
# and the resample count tapers as n grows to keep each click responsive.
def bootstrap_reps(n):
    return max(120, min(400, int(120_000 / max(1, n * n))))

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def pretty_name(filename):
    """Andrea_Piccolo_319307_Candidate_Pack.pdf -> ('Andrea Piccolo', '319307')"""
    stem = re.sub(r"\.pdf$", "", filename, flags=re.I)
    stem = re.sub(r"_?Candidate_?Pack$", "", stem, flags=re.I)
    m = re.search(r"_(\d{4,})$", stem)
    ref = ""
    if m:
        ref = m.group(1)
        stem = stem[: m.start()]
    return stem.replace("_", " ").strip(), ref


def discover_candidates():
    files = sorted(f for f in os.listdir(CANDIDATE_DIR) if f.lower().endswith(".pdf"))
    out = []
    for f in files:
        name, ref = pretty_name(f)
        out.append({"file": f, "name": name, "ref": ref})
    return out


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def round_robin_rounds(n, rng):
    """Circle method: n-1 rounds, each a perfect matching over a shuffled
    ordering. Taking the first k rounds gives every candidate exactly k
    comparisons against k distinct opponents."""
    order = list(range(n))
    rng.shuffle(order)
    if n % 2:
        order.append(None)  # bye
    m = len(order)
    fixed, rot = order[0], order[1:]
    rounds = []
    for _ in range(m - 1):
        ring = [fixed] + rot
        pairs = [
            (ring[i], ring[m - 1 - i])
            for i in range(m // 2)
            if ring[i] is not None and ring[m - 1 - i] is not None
        ]
        rounds.append(pairs)
        rot = rot[1:] + rot[:1]
    return rounds


def connected(n, pairs):
    adj = {i: set() for i in range(n)}
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)
    seen, stack = {0}, [0]
    while stack:
        for nb in adj[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == n


def make_schedule(n, rng):
    """Phase-1 pair list, spread so consecutive comparisons rarely share a
    candidate, with sides randomised.

    Up to FULL_ROUND_ROBIN_MAX candidates this is a complete round robin. Above
    that it is the first SEED_ROUNDS rounds — enough to give every candidate a
    comparable footing and to connect the comparison graph — after which the
    adaptive phase targets only the pairs that still matter.
    """
    rounds = round_robin_rounds(n, rng)
    if n <= FULL_ROUND_ROBIN_MAX:
        take = len(rounds)
    else:
        take = min(len(rounds), SEED_ROUNDS)

    pairs = [p for r in rounds[:take] for p in r]
    # A disconnected comparison graph leaves Bradley-Terry unable to compare
    # the components; add rounds until it is connected.
    while take < len(rounds) and not connected(n, pairs):
        take += 1
        pairs = [p for r in rounds[:take] for p in r]

    rng.shuffle(pairs)
    ordered, pool, prev = [], list(pairs), None
    while pool:
        pick = 0
        for idx, p in enumerate(pool):
            if prev is None or not (set(p) & set(prev)):
                pick = idx
                break
        p = pool.pop(pick)
        ordered.append(p)
        prev = p

    return [list(p) if rng.random() < 0.5 else [p[1], p[0]] for p in ordered]


# ---------------------------------------------------------------------------
# Bradley-Terry
# ---------------------------------------------------------------------------

def fit_bt(n, comparisons, prior=PRIOR, iters=500, tol=1e-10, init=None):
    """Maximum-likelihood Bradley-Terry strengths via the MM algorithm.

    comparisons: list of {"a": i, "b": j, "result": "a"|"b"|"tie"}
    A tie counts as half a win to each side.

    The prior adds `prior` virtual wins and `prior` virtual losses for every
    candidate against a phantom opponent of fixed strength 1.0. This keeps the
    estimates finite under separation and fixes the scale (strength 1 == the
    phantom == a nominal "average"), so no renormalisation is needed.
    """
    wins = [0.0] * n
    counts = [[0.0] * n for _ in range(n)]

    for c in comparisons:
        a, b, r = c["a"], c["b"], c["result"]
        counts[a][b] += 1
        counts[b][a] += 1
        if r == "a":
            wins[a] += 1.0
        elif r == "b":
            wins[b] += 1.0
        else:
            wins[a] += 0.5
            wins[b] += 0.5

    p = list(init) if init else [1.0] * n
    for _ in range(iters):
        new = [0.0] * n
        for i in range(n):
            denom = 2.0 * prior / (p[i] + 1.0)
            for j in range(n):
                if i != j and counts[i][j]:
                    denom += counts[i][j] / (p[i] + p[j])
            new[i] = (wins[i] + prior) / denom if denom > 0 else p[i]
        delta = max(abs(new[i] - p[i]) for i in range(n))
        p = new
        if delta < tol:
            break
    return p, wins, counts


def record(n, comparisons):
    rec = [{"w": 0, "l": 0, "t": 0} for _ in range(n)]
    for c in comparisons:
        a, b, r = c["a"], c["b"], c["result"]
        if r == "a":
            rec[a]["w"] += 1
            rec[b]["l"] += 1
        elif r == "b":
            rec[b]["w"] += 1
            rec[a]["l"] += 1
        else:
            rec[a]["t"] += 1
            rec[b]["t"] += 1
    return rec


def bootstrap_confidence(n, comparisons, order, rng, reps=None, init=None):
    """For each adjacent pair in `order`, the fraction of bootstrap resamples
    that agree with the point estimate's ordering.

    Each resample fit is warm-started from the point estimate and run to a
    looser tolerance: the resamples sit close to it, so this converges in a
    handful of iterations instead of hundreds.
    """
    if len(comparisons) < 2 or n < 2:
        return [0.0] * max(0, n - 1)

    reps = reps or bootstrap_reps(n)
    agree = [0] * (n - 1)
    m = len(comparisons)
    for _ in range(reps):
        sample = [comparisons[rng.randrange(m)] for _ in range(m)]
        ps, _, _ = fit_bt(n, sample, iters=120, tol=1e-6, init=init)
        for k in range(n - 1):
            hi, lo = order[k], order[k + 1]
            if ps[hi] > ps[lo]:
                agree[k] += 1
            elif ps[hi] == ps[lo]:
                agree[k] += 0.5
    return [a / reps for a in agree]


def intransitive_triads(n, comparisons):
    """Count 3-cycles (A>B>C>A) in the majority-preference graph — a direct
    measure of how self-consistent the judgements are."""
    net = [[0.0] * n for _ in range(n)]
    for c in comparisons:
        a, b, r = c["a"], c["b"], c["result"]
        if r == "a":
            net[a][b] += 1
        elif r == "b":
            net[b][a] += 1

    def beats(i, j):
        return net[i][j] > net[j][i]

    cycles, total = [], 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                total += 1
                if (beats(i, j) and beats(j, k) and beats(k, i)) or (
                    beats(i, k) and beats(k, j) and beats(j, i)
                ):
                    cycles.append([i, j, k])
    return cycles, total


def next_adaptive_pair(n, p, counts, last):
    """Most informative unasked-enough pair.

    Fisher information for a Bradley-Terry comparison is proportional to
    q(1-q) where q = P(i beats j) — maximal when the pair is a coin-flip.
    Dividing by (1 + times compared) spreads effort instead of hammering one
    pair, and a small penalty avoids repeating the pair just answered.
    """
    best, best_score = None, -1.0
    for i in range(n):
        for j in range(i + 1, n):
            q = p[i] / (p[i] + p[j])
            score = q * (1 - q) / (1.0 + counts[i][j])
            if last and set(last) == {i, j}:
                score *= 0.25
            if score > best_score:
                best_score, best = score, (i, j)
    return best


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def new_state():
    cands = discover_candidates()
    n = len(cands)
    rng = random.Random()
    schedule = make_schedule(n, rng)
    return {
        "candidates": cands,
        "schedule": schedule,
        "comparisons": [],
        "phase": 1,
        "finished": False,
        # With very few candidates there are barely any distinct pairs to ask,
        # so cap the adaptive budget to avoid asking the same pair over and
        # over. (At n=2 that means one seed comparison plus two re-asks.)
        "extra_budget": min(EXTRA_PER_CANDIDATE * n, 2 * (n * (n - 1) // 2)),
    }


def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as fh:
                st = json.load(fh)
            old = [c["file"] for c in st.get("candidates", [])]
            new = [c["file"] for c in discover_candidates()]
            if old == new:
                return st
            # The candidate set changed under us. Old comparisons refer to
            # candidates by index, so they cannot be carried over safely.
            print(
                f"! candidates/ changed ({len(old)} -> {len(new)} PDFs); "
                f"discarding {len(st.get('comparisons', []))} previous comparison(s) "
                f"and starting a new ranking."
            )
        except Exception:
            pass
    st = new_state()
    save_state(st)
    return st


def save_state(st):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(st, fh, indent=1)
    os.replace(tmp, STATE_PATH)


# ---------------------------------------------------------------------------
# View model
# ---------------------------------------------------------------------------

def build_view(st):
    cands = st["candidates"]
    n = len(cands)
    comps = st["comparisons"]
    rng = random.Random(12345)  # deterministic bootstrap so the number is stable

    p, _, counts = fit_bt(n, comps)
    rec = record(n, comps)

    order = sorted(range(n), key=lambda i: -p[i])
    conf = bootstrap_confidence(n, comps, order, rng, init=p) if comps else [0.0] * max(0, n - 1)
    cycles, total_triads = intransitive_triads(n, comps)

    ranking = []
    for rank, i in enumerate(order):
        ranking.append(
            {
                "rank": rank + 1,
                "index": i,
                "name": cands[i]["name"],
                "ref": cands[i]["ref"],
                "file": cands[i]["file"],
                "strength": p[i],
                "score": round(400 * math.log10(p[i]) + 1500),
                "w": rec[i]["w"],
                "l": rec[i]["l"],
                "t": rec[i]["t"],
                # confidence that this candidate really outranks the next one
                "conf_over_next": conf[rank] if rank < len(conf) else None,
            }
        )

    seed_total = len(st["schedule"])
    done = len(comps)
    min_conf = min(conf) if conf else 0.0

    # --- pick the next comparison ---
    nxt = None
    if not st["finished"]:
        if st["phase"] == 1 and done < seed_total:
            a, b = st["schedule"][done]
            nxt = {"a": a, "b": b}
        else:
            st["phase"] = 2
            spent = done - seed_total
            if min_conf >= CONFIDENCE_TARGET or spent >= st["extra_budget"]:
                st["finished"] = True
            else:
                last = (comps[-1]["a"], comps[-1]["b"]) if comps else None
                i, j = next_adaptive_pair(n, p, counts, last)
                if random.random() < 0.5:
                    i, j = j, i
                nxt = {"a": i, "b": j}

    if st["phase"] == 1:
        kind = "Round robin" if seed_total == n * (n - 1) // 2 else "Seeding"
        progress = {"done": done, "total": seed_total, "label": f"{kind} {done}/{seed_total}"}
    else:
        spent = max(0, done - seed_total)
        progress = {
            "done": spent,
            "total": st["extra_budget"],
            "label": f"Tie-breaking {spent}/{st['extra_budget']} (confidence {min_conf:.0%})",
        }

    return {
        "candidates": cands,
        "next": nxt,
        "phase": st["phase"],
        "finished": st["finished"],
        "progress": progress,
        "ranking": ranking,
        "comparisons": done,
        "min_confidence": min_conf,
        "confidence_target": CONFIDENCE_TARGET,
        "cycles": [[cands[i]["name"] for i in c] for c in cycles],
        "total_triads": total_triads,
        "history": [
            {
                "a": cands[c["a"]]["name"],
                "b": cands[c["b"]]["name"],
                "result": c["result"],
            }
            for c in comps[-12:]
        ][::-1],
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # quiet

    # -- helpers --
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype=None):
        if not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        ctype = ctype or mimetypes.guess_type(path)[0] or "application/octet-stream"

        # Range support — PDF viewers stream large files with partial requests.
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m:
                s, e = m.group(1), m.group(2)
                if s:
                    start = int(s)
                    end = int(e) if e else size - 1
                elif e:
                    start = max(0, size - int(e))
                start = min(start, size - 1)
                end = min(end, size - 1)
                partial = True

        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    # -- routes --
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)

        if path in ("/", "/index.html"):
            return self._file(os.path.join(HERE, "index.html"), "text/html; charset=utf-8")

        if path == "/api/state":
            with _lock:
                st = load_state()
                view = build_view(st)
                save_state(st)
            return self._json(view)

        if path.startswith("/pdf/"):
            name = os.path.basename(path[len("/pdf/"):])
            allowed = {c["file"] for c in discover_candidates()}
            if name not in allowed:
                return self.send_error(404)
            return self._file(os.path.join(CANDIDATE_DIR, name), "application/pdf")

        self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")

        with _lock:
            st = load_state()

            if path == "/api/choose":
                a, b, result = payload["a"], payload["b"], payload["result"]
                if result not in ("a", "b", "tie"):
                    return self.send_error(400)
                st["comparisons"].append({"a": a, "b": b, "result": result})

            elif path == "/api/undo":
                if st["comparisons"]:
                    st["comparisons"].pop()
                st["finished"] = False
                if len(st["comparisons"]) < len(st["schedule"]):
                    st["phase"] = 1

            elif path == "/api/finish":
                st["finished"] = True

            elif path == "/api/more":
                st["finished"] = False
                st["phase"] = 2
                st["extra_budget"] += len(st["candidates"])

            elif path == "/api/reset":
                st = new_state()

            else:
                return self.send_error(404)

            view = build_view(st)
            save_state(st)

        self._json(view)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    cands = discover_candidates()
    if len(cands) < 2:
        raise SystemExit(f"Need at least 2 PDFs in {CANDIDATE_DIR}")

    url = f"http://127.0.0.1:{PORT}/"
    print(f"{len(cands)} candidates found in {CANDIDATE_DIR}")
    for c in cands:
        print(f"  - {c['name']} ({c['ref']})")
    print(f"\nServing at {url}   (Ctrl-C to stop)")
    print(f"Progress is saved to {STATE_PATH} — you can stop and resume.\n")

    threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    with Server(("127.0.0.1", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
