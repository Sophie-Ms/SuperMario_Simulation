import heapq, random, math
from collections import deque

# -------- COMMON --------
def neighbors(grid, x, y):
    dirs=[(0,1),(1,0),(0,-1),(-1,0)]
    return [(x+dx,y+dy) for dx,dy in dirs
            if 0<=x+dx<len(grid) and 0<=y+dy<len(grid[0]) and grid[x+dx][y+dy] != '#']

def h(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

# -------- BFS --------
def bfs(grid,start,goal):
    q=deque([(start,[start])])
    visited=set()
    steps=[]
    while q:
        node,path=q.popleft()
        steps.append(node)
        if node==goal:
            return steps,path
        if node in visited: continue
        visited.add(node)
        for n in neighbors(grid,*node):
            q.append((n,path+[n]))
    return steps,[]

# -------- DFS --------
def dfs(grid,start,goal):
    stack=[(start,[start])]
    visited=set()
    steps=[]
    while stack:
        node,path=stack.pop()
        steps.append(node)
        if node==goal:
            return steps,path
        if node in visited: continue
        visited.add(node)
        for n in neighbors(grid,*node):
            stack.append((n,path+[n]))
    return steps,[]

# -------- A* --------
def astar(grid,start,goal):
    pq=[(0,0,start,[start])]
    visited=set()
    steps=[]
    while pq:
        f,g,node,path=heapq.heappop(pq)
        steps.append(node)
        if node==goal:
            return steps,path
        if node in visited: continue
        visited.add(node)
        for n in neighbors(grid,*node):
            heapq.heappush(pq,(g+1+h(n,goal),g+1,n,path+[n]))
    return steps,[]

# -------- GREEDY --------
def greedy(grid,start,goal):
    pq=[(0,start,[start])]
    visited=set()
    steps=[]
    while pq:
        _,node,path=heapq.heappop(pq)
        steps.append(node)
        if node==goal:
            return steps,path
        if node in visited: continue
        visited.add(node)
        for n in neighbors(grid,*node):
            heapq.heappush(pq,(h(n,goal),n,path+[n]))
    return steps,[]

# -------- UCS --------
def ucs(grid,start,goal):
    pq=[(0,start,[start])]
    visited=set()
    steps=[]
    while pq:
        cost,node,path=heapq.heappop(pq)
        steps.append(node)
        if node==goal:
            return steps,path
        if node in visited: continue
        visited.add(node)
        for n in neighbors(grid,*node):
            heapq.heappush(pq,(cost+1,n,path+[n]))
    return steps,[]

# -------- IDDFS --------
def iddfs(grid,start,goal):
    def dls(node,depth,visited,path):
        steps.append(node)
        if node==goal: return path
        if depth==0: return None
        visited.add(node)
        for n in neighbors(grid,*node):
            if n not in visited:
                res=dls(n,depth-1,visited,path+[n])
                if res: return res
        return None

    steps=[]
    for d in range(1,20):
        visited=set()
        res=dls(start,d,visited,[start])
        if res:
            return steps,res
    return steps,[]

# -------- HILL CLIMB --------
def hill_climb(grid,start,goal):
    curr=start
    path=[curr]
    steps=[curr]
    while True:
        neigh=neighbors(grid,*curr)
        if not neigh: return steps,path
        best=min(neigh,key=lambda x:h(x,goal))
        if h(best,goal)>=h(curr,goal):
            return steps,path
        curr=best
        path.append(curr)
        steps.append(curr)
        if curr==goal:
            return steps,path

# -------- SIMULATED ANNEALING --------
def simulated_annealing(grid,start,goal):
    curr=start
    path=[curr]
    steps=[curr]
    T=100
    while T>1:
        neigh=neighbors(grid,*curr)
        if not neigh: break
        nxt=random.choice(neigh)
        delta=h(curr,goal)-h(nxt,goal)
        if delta>0 or random.random()<math.exp(delta/T):
            curr=nxt
            path.append(curr)
            steps.append(curr)
        if curr==goal:
            return steps,path
        T*=0.9
    return steps,path

# -------- BIDIRECTIONAL --------
def bidirectional(grid,start,goal):
    q1=deque([(start,[start])])
    q2=deque([(goal,[goal])])
    v1={start: [start]}
    v2={goal: [goal]}
    steps=[]

    while q1 and q2:
        node1,path1=q1.popleft()
        steps.append(node1)
        if node1 in v2:
            return steps, path1 + v2[node1][::-1]

        for n in neighbors(grid,*node1):
            if n not in v1:
                v1[n]=path1+[n]
                q1.append((n,path1+[n]))

        node2,path2=q2.popleft()
        steps.append(node2)
        if node2 in v1:
            return steps, v1[node2] + path2[::-1]

        for n in neighbors(grid,*node2):
            if n not in v2:
                v2[n]=path2+[n]
                q2.append((n,path2+[n]))

    return steps,[]