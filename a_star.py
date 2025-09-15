"""

AStar search

author: Ashwin Bose (@atb033)
Edited by: Daniel Weiner (@danwein8)

"""
import heapq

def _xy_from_start(value):
    # tuple/list (x, y)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    # State-like with .location.x/.y
    loc = getattr(value, "location", None)
    if loc is not None and hasattr(loc, "x") and hasattr(loc, "y"):
        return int(loc.x), int(loc.y)
    # Location directly
    if hasattr(value, "x") and hasattr(value, "y"):
        return int(value.x), int(value.y)
    raise TypeError(f"Unsupported start/goal type: {type(value)}")

def _t_from_start(value):
    # State-like with .time
    if hasattr(value, "time"):
        return int(getattr(value, "time"))
    # default to 0 if we only have a coordinate tuple
    return 0

class Node:
    __slots__ = ("x", "y", "t")
    def __init__(self, x, y, t):
        self.x = int(x); self.y = int(y); self.t = int(t)

    def __eq__(self, other):
        return isinstance(other, Node) and \
               self.x == other.x and self.y == other.y and self.t == other.t

    def __hash__(self):
        # returns a real int, based on the immutable identity (x,y,t)
        return hash((self.x, self.y, self.t))

    def __lt__(self, other):
        # deterministic tie-breaker for heapq
        if not isinstance(other, Node):
            return NotImplemented
        return (self.t, self.x, self.y) < (other.t, other.x, other.y)


class AStar():
    def __init__(self, env):
        # environment class from CBS will pass a lot of information
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        # this get_neighbors will take into account constraints
        self.get_neighbors = env.get_neighbors
        self.Node = Node

    def reconstruct_path(self, parent, node):
        path = []
        while node in parent:
            path.append((node.x, node.y, node.t))
            node = parent[node]
        path.append((node.x, node.y, node.t))
        path.reverse()
        return [{'t': t, 'x': x, 'y': y} for (x, y, t) in path]
    
    # class Node:
    #     __slots__ = ("x", "y", "t") # to save memory, optimization technique
    #     def __init__(self, x, y, t): self.x, self.y, self.t = x, y, t
    #     def __lt__(self, other): return (self.t, self.x, self.y) < (other.t, other.x, other.y)
    #     def __hash__(self): return ((self.x, self.y, self.t)) # return tuple quickly
    #     def __eq__(self, o): return self.x==o.x and self.y==o.y and self.t==o.t

    def search(self, env, agent, constraint_table, T_max):
        """
        env: Environment (has agent_dict, neighbors, in_bounds, is_obstacle, admissible heuristic, etc.)
        agent: str key into env.agent_dict (or object with .start/.goal)
        constraint_table: READ-ONLY per-agent merged constraints (vertex+edge, time-indexed)
        T_max: time horizon (inclusive or exclusive depending on your policy; below assumes inclusive arrival)
        """

        # 1) Look up agent by name (if a string was passed)
        agent_obj = env.agent_dict[agent] if isinstance(agent, str) else agent

        # 2) Normalize start/goal
        if isinstance(agent_obj, dict):
            start_val = agent_obj["start"]
            goal_val  = agent_obj["goal"]
        else:
            start_val = getattr(agent_obj, "start")
            goal_val  = getattr(agent_obj, "goal")

        sx, sy = _xy_from_start(start_val)    # your helper
        gx, gy = _xy_from_start(goal_val)
        t0     = _t_from_start(start_val)

        start  = self.Node(sx, sy, t0)
        goal_xy = (gx, gy)

        # Frontier (min-heap by f = g + h)
        open_heap = []
        g = {start: 0}
        f = {start: self.heuristic(start.x, start.y, goal_xy)}
        parent = {}
        heapq.heappush(open_heap, (f[start], g[start], start))

        closed = set()

        expansions = 0
        MAX_EXPANSIONS = 200_000

        while open_heap:
            expansions += 1
            if expansions > MAX_EXPANSIONS:
                return None
            _, g_curr, curr = heapq.heappop(open_heap)
            if curr in closed:
                continue
            closed.add(curr)

            # Goal test: at goal AND no vertex constraint at current time
            if (curr.x, curr.y) == goal_xy and not self.violates_vertex(constraint_table, curr.x, curr.y, curr.t):
                return self.reconstruct_path(parent, curr)

            # Horizon guard: don’t expand beyond T_max
            if curr.t >= T_max:
                continue

            # Expand 4-neighbors + wait
            for neigh in env.get_neighbors((curr.x, curr.y, curr.t), allow_wait=True):
                nx, ny = neigh.location.x, neigh.location.y
                nt = curr.t + 1

                # Grid validity
                if not env.state_valid(neigh):
                    continue

                # Constraint checks
                if self.violates_vertex(constraint_table, nx, ny, nt):
                    continue
                if self.violates_edge(constraint_table, curr.x, curr.y, nx, ny, nt):
                    continue

                neighbor = self.Node(nx, ny, nt)
                tentative_g = g_curr + 1

                # Optional pruning: if even the best case exceeds horizon, skip
                # h_min = self.heuristic(nx, ny, goal_xy)
                # if nt + h_min > T_max:  # admissible pruning
                #     continue

                if tentative_g < g.get(neighbor, float("inf")):
                    parent[neighbor] = curr
                    g[neighbor] = tentative_g
                    fn = tentative_g + self.heuristic(nx, ny, goal_xy)
                    f[neighbor] = fn
                    heapq.heappush(open_heap, (fn, tentative_g, neighbor))

        # No path found
        return None

    
    def heuristic(self, x: int, y: int, goal_xy: tuple[int, int]) -> float:
        gx, gy = goal_xy
        return abs(x - gx) + abs(y - gy) # Manhattan dist heursitsic
    
    def violates_vertex(self, constraint, x, y, t):
        # constraint.vertex is a dict that holds {(x, y): set of times}
        return t in constraint.vertex.get((x, y), ())

    def violates_edge(self, constraint, x1, y1, x2, y2, t):
        # constr.edge is a dict like {((x1,y1),(x2,y2)): set_of_times}
        return t in constraint.edge.get(((x1, y1), (x2, y2)), ())