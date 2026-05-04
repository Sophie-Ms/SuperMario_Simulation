import streamlit as st
import time
import heapq
import random
import math
from collections import deque

st.set_page_config(page_title="Mario AI Simulator", layout="wide", page_icon="🍄")

# ── Grid: 7 x 9  ('#'=wall, 'C'=coin, 'E'=enemy) ────────────────
ROWS, COLS = 7, 9
START = (0, 0)
GOAL  = (6, 8)

GRID = [
    ['S', '.', '#', 'C', '.', '.', '#', '.', '.'],
    ['#', '.', '.', '.', '#', '.', 'C', '.', 'E'],
    ['.', 'C', '.', '#', '.', '.', '.', '#', '.'],
    ['.', '#', '.', '.', '.', 'E', '.', '.', 'C'],
    ['.', '.', '#', 'C', '.', '.', '#', '.', '.'],
    ['C', '.', '.', '.', '#', '.', '.', 'E', '.'],
    ['.', '#', '.', '.', '.', 'C', '.', '.', 'G'],
]

# ── Algorithm helpers ─────────────────────────────────────────────
def nbrs(x, y):
    for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < ROWS and 0 <= ny < COLS and GRID[nx][ny] != '#':
            yield (nx, ny)

def h(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def bfs(start, goal):
    q = deque([(start, [start])]); vis = set(); steps = []
    while q:
        node, path = q.popleft(); steps.append(node)
        if node == goal: return steps, path
        if node in vis: continue
        vis.add(node)
        for n in nbrs(*node): q.append((n, path+[n]))
    return steps, []

def dfs(start, goal):
    stack = [(start, [start])]; vis = set(); steps = []
    while stack:
        node, path = stack.pop(); steps.append(node)
        if node == goal: return steps, path
        if node in vis: continue
        vis.add(node)
        for n in nbrs(*node): stack.append((n, path+[n]))
    return steps, []

def astar(start, goal):
    pq = [(0, 0, start, [start])]; vis = set(); steps = []
    while pq:
        _, g, node, path = heapq.heappop(pq); steps.append(node)
        if node == goal: return steps, path
        if node in vis: continue
        vis.add(node)
        for n in nbrs(*node):
            heapq.heappush(pq, (g+1+h(n,goal), g+1, n, path+[n]))
    return steps, []

def greedy(start, goal):
    pq = [(0, start, [start])]; vis = set(); steps = []
    while pq:
        _, node, path = heapq.heappop(pq); steps.append(node)
        if node == goal: return steps, path
        if node in vis: continue
        vis.add(node)
        for n in nbrs(*node): heapq.heappush(pq, (h(n,goal), n, path+[n]))
    return steps, []

def ucs(start, goal):
    pq = [(0, start, [start])]; vis = set(); steps = []
    while pq:
        cost, node, path = heapq.heappop(pq); steps.append(node)
        if node == goal: return steps, path
        if node in vis: continue
        vis.add(node)
        for n in nbrs(*node): heapq.heappush(pq, (cost+1, n, path+[n]))
    return steps, []

def iddfs(start, goal):
    def dls(node, depth, vis, path):
        steps.append(node)
        if node == goal: return path
        if depth == 0: return None
        vis.add(node)
        for n in nbrs(*node):
            if n not in vis:
                res = dls(n, depth-1, set(vis), path+[n])
                if res: return res
        return None
    steps = []
    for d in range(1, 40):
        res = dls(start, d, set(), [start])
        if res: return steps, res
    return steps, []

def hill_climb(start, goal):
    curr = start; path = [curr]; steps = [curr]
    while True:
        ns = list(nbrs(*curr))
        if not ns: return steps, path
        best = min(ns, key=lambda x: h(x, goal))
        if h(best, goal) >= h(curr, goal): return steps, path
        curr = best; path.append(curr); steps.append(curr)
        if curr == goal: return steps, path

def sim_annealing(start, goal):
    curr = start; path = [curr]; steps = [curr]; T = 100
    while T > 1:
        ns = list(nbrs(*curr))
        if not ns: break
        nxt = random.choice(ns)
        delta = h(curr,goal) - h(nxt,goal)
        if delta > 0 or random.random() < math.exp(delta/T):
            curr = nxt; path.append(curr); steps.append(curr)
        if curr == goal: return steps, path
        T *= 0.9
        if len(steps) > 400: break
    return steps, path

def bidirectional(start, goal):
    q1 = deque([(start,[start])]); q2 = deque([(goal,[goal])])
    v1 = {start:[start]}; v2 = {goal:[goal]}; steps = []
    while q1 and q2:
        n1, p1 = q1.popleft(); steps.append(n1)
        if n1 in v2: return steps, p1 + v2[n1][::-1][1:]
        for n in nbrs(*n1):
            if n not in v1: v1[n]=p1+[n]; q1.append((n,p1+[n]))
        n2, p2 = q2.popleft(); steps.append(n2)
        if n2 in v1: return steps, v1[n2] + p2[::-1][1:]
        for n in nbrs(*n2):
            if n not in v2: v2[n]=p2+[n]; q2.append((n,p2+[n]))
    return steps, []

ALGOS = {
    "BFS":                 (bfs,           "Explores level by level — guaranteed shortest path 📏"),
    "DFS":                 (dfs,           "Dives deep first — fast but not always optimal 🏊"),
    "A*":                  (astar,         "Uses heuristic — smart & efficient ⭐"),
    "Greedy":              (greedy,        "Always moves toward goal — fast but risky 💨"),
    "UCS":                 (ucs,           "Cheapest cost first — optimal 💰"),
    "IDDFS":               (iddfs,         "Iterative deepening — memory efficient 🔁"),
    "Hill Climb":          (hill_climb,    "Greedy local search — can get stuck 🏔️"),
    "Simulated Annealing": (sim_annealing, "Random with cooling — escapes traps 🌡️"),
    "Bidirectional":       (bidirectional, "Searches from both ends simultaneously 🔀"),
}

# ── Render grid as HTML table (big cells, colourful bg) ──────────
def render_grid(visited=set(), path=set(), current=None):
    # cell size & background
    CELL = "72px"
    FONT = "44px"
    bg   = "#5c94fc"   # Mario sky blue

    rows_html = ""
    for r in range(ROWS):
        cells_html = ""
        for c in range(COLS):
            pos  = (r, c)
            cell = GRID[r][c]
            # pick emoji
            if cell == '#':
                em = "🧱"
                bg_cell = "#a0522d"
            elif pos == current:
                em = "🔴"
                bg_cell = "#ffe066"
            elif pos == GOAL:
                em = "🏁"
                bg_cell = "#fffde7"
            elif pos == START and pos != current:
                em = "🍄"
                bg_cell = "#c8f7c5"
            elif pos in path:
                em = "🟩"
                bg_cell = "#b9f6ca"
            elif pos in visited:
                em = "🟦"
                bg_cell = "#bbdefb"
            elif cell == 'C':
                em = "🪙"
                bg_cell = bg
            elif cell == 'E':
                em = "👾"
                bg_cell = "#ffcdd2"
            else:
                em = ""
                bg_cell = bg

            cells_html += (
                "<td style='"
                "width:" + CELL + ";"
                "height:" + CELL + ";"
                "font-size:" + FONT + ";"
                "text-align:center;"
                "vertical-align:middle;"
                "background:" + bg_cell + ";"
                "border:2px solid #3a6ad4;"
                "border-radius:6px;"
                "padding:0;"
                "'>" + em + "</td>"
            )
        rows_html += "<tr>" + cells_html + "</tr>"

    table = (
        "<div style='background:#5c94fc;padding:16px;border-radius:12px;"
        "border:4px solid #e8a317;display:inline-block;'>"
        "<table style='border-collapse:separate;border-spacing:4px;'>"
        + rows_html +
        "</table></div>"
    )
    return table

# ── Page header ───────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:16px 0 8px 0;
     background:linear-gradient(135deg,#e8a317 0%,#f7c948 50%,#e8a317 100%);
     border-radius:12px; margin-bottom:12px;
     border:3px solid #c84800;'>
  <span style='font-size:2.4rem; font-weight:900; color:#c84800;
               text-shadow:2px 2px 0 #fff;'>
    🍄 Mario AI Pathfinding Simulator
  </span><br/>
  <span style='color:#7a2d00; font-size:1rem;'>
    Watch AI algorithms find Mario's path!
  </span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar controls ──────────────────────────────────────────────
with st.sidebar:
    st.header("🎮 Controls")
    chosen = st.selectbox("Algorithm", list(ALGOS.keys()))
    fn, desc = ALGOS[chosen]
    st.info(desc)

    speed = st.slider("⚡ Speed", 1, 10, 5, help="Higher = faster animation")
    delay = round(0.5 / speed, 3)

    run_btn   = st.button("▶ Run Simulation", use_container_width=True, type="primary")
    reset_btn = st.button("↺ Reset",          use_container_width=True)

    st.divider()
    st.markdown("**Legend**")
    st.write("🍄 Start (Mario)")
    st.write("🏁 Goal")
    st.write("🔴 Mario moving")
    st.write("🧱 Wall")
    st.write("🪙 Coin")
    st.write("👾 Enemy")
    st.write("🟦 Visited cell")
    st.write("🟩 Final path")

# ── Session state ─────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

if reset_btn:
    st.session_state.result = None
    st.rerun()

# ── Main layout ───────────────────────────────────────────────────
grid_col, stats_col = st.columns([3, 1])

with stats_col:
    st.markdown("### 📊 Stats")
    steps_box   = st.empty()
    path_box    = st.empty()
    coins_box   = st.empty()
    enemies_box = st.empty()
    score_box   = st.empty()
    msg_box     = st.empty()

    def update_stats(s, p, coins, enemies, status=""):
        steps_box.metric("Nodes Explored", s)
        path_box.metric("Path Length", p if p else "—")
        coins_box.metric("Coins", coins)
        enemies_box.metric("Enemies hit", enemies)
        score_box.metric("Score", max(0, coins*100 - enemies*50 - (p or 0)))
        if status == "found":
            msg_box.success("Path Found! Length: " + str(p))
        elif status == "notfound":
            msg_box.error("No path found!")

    update_stats(0, None, 0, 0)

with grid_col:
    grid_display = st.empty()
    grid_display.markdown(render_grid(), unsafe_allow_html=True)

# ── Run ───────────────────────────────────────────────────────────
if run_btn:
    st.session_state.result = None
    msg_box.empty()

    steps, path = fn(START, GOAL)
    path_set = set(map(tuple, path))
    visited  = set()
    coins    = 0
    enemies  = 0

    for i, pos in enumerate(steps):
        visited.add(pos)
        r, c = pos
        if GRID[r][c] == 'C' and pos in path_set: coins += 1
        if GRID[r][c] == 'E' and pos in path_set: enemies += 1

        grid_display.markdown(render_grid(visited, path_set & visited, pos), unsafe_allow_html=True)
        update_stats(i+1, len(path) if path else None, coins, enemies)
        time.sleep(delay)

    # Final frame — full path highlighted
    grid_display.markdown(render_grid(visited, path_set), unsafe_allow_html=True)

    if path:
        update_stats(len(steps), len(path), coins, enemies, "found")
    else:
        update_stats(len(steps), None, coins, enemies, "notfound")

    st.session_state.result = {
        "steps": len(steps), "path": len(path),
        "coins": coins, "enemies": enemies
    }