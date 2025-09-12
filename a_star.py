"""

AStar search

author: Ashwin Bose (@atb033)
Edited by: Daniel Weiner (@danwein8)

"""
import heapq

class AStar():
    def __init__(self, env):
        # environment class from CBS will pass a lot of information
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        # this get_neighbors will take into account constraints
        self.get_neighbors = env.get_neighbors

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Return reversed path

    def search(self, agent_name):
        """
        low level search 
        """
        s0 = self.agent_dict[agent_name]["start"]
        h0 = self.admissible_heuristic(s0, agent_name)

        # Simple horizon: L1 distance + buffer
        goal = self.agent_dict[agent_name]["goal"]
        dist = abs(s0.location.x - goal.location.x) + abs(s0.location.y - goal.location.y)
        T_max = int(dist + max(20, dist * 2))  # tune if needed

        open_heap = []
        heapq.heappush(open_heap, (h0, 0, s0))  # (f, g, state)

        came_from = {}
        g_score = {s0: 0}
        # Best arrival time seen for each (x,y)
        best_time = {(s0.location.x, s0.location.y): 0}

        closed = set()

        while open_heap:
            f, g, current = heapq.heappop(open_heap)

            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current)

            if current in closed:
                continue
            closed.add(current)

            for nbr in self.get_neighbors(current):
                # prune by horizon
                if nbr.time > T_max:
                    continue

                # dominance pruning by earliest time at cell
                key = (nbr.location.x, nbr.location.y)
                best_arrival = best_time.get(key)
                if best_arrival is not None and best_arrival <= nbr.time:
                    continue

                tentative_g = g + 1
                prev = g_score.get(nbr, float("inf"))
                if tentative_g < prev:
                    came_from[nbr] = current
                    g_score[nbr] = tentative_g
                    best_time[key] = nbr.time
                    h = self.admissible_heuristic(nbr, agent_name)
                    heapq.heappush(open_heap, (tentative_g + h, tentative_g, nbr))

        return False

