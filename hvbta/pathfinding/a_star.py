class AStar():
    def __init__(self, env):
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors

    def reconstruct_path(self, came_from: dict, current: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Reconstructs path from came_from map
        Parameters:
            came_from (dict): map of navigated nodes
            current (tuple[int, int]): current node
        Returns:
            total_path (list[tuple[int, int]]): list of nodes in path from start to goal
        """
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self, agent_name: str) -> list[tuple[int, int]] | bool:
        """
        Perform A* search for a given agent.
        Parameters:
            agent_name (str): The name of the agent to search for.
        Returns:
            path (list[tuple[int, int]] | bool): The path from start to goal as a list of (row, col) tuples, or False if no path is found.
        """
        # set initial state to agents start location
        initial_state = self.agent_dict[agent_name]["start"]
        step_cost = 1
        
        # initialize empty closed set and open set with start location
        closed_set = set()
        open_set = {initial_state}

        # empty backtrack dict
        came_from = {}

        # path cost
        g_score = {} 
        g_score[initial_state] = 0

        # final cost
        f_score = {} 
        # initialize final cost with heuristic value and initial state
        f_score[initial_state] = self.admissible_heuristic(initial_state, agent_name)

        while open_set:
            # get the node in open set with the lowest f_score value
            temp_dict = {open_item:f_score.setdefault(open_item, float("inf")) for open_item in open_set}
            current = min(temp_dict, key=temp_dict.get)

            # if were at goal, reconstruct path and return
            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current)

            # move current from open to closed set
            open_set -= {current}
            closed_set |= {current}
            # get neighbors and check each one
            neighbor_list = self.get_neighbors(current)

            for neighbor in neighbor_list:
                if neighbor in closed_set:
                    continue
                # tentative g score is the g score of current + step cost
                tentative_g_score = g_score.setdefault(current, float("inf")) + step_cost
                # add neighbor to open set if not already there
                if neighbor not in open_set:
                    open_set |= {neighbor}
                # if tentative g score is not better than existing g score, skip
                elif tentative_g_score >= g_score.setdefault(neighbor, float("inf")):
                    continue
                # record best path by setting backtrack to current
                came_from[neighbor] = current
                # update g and f scores
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic(neighbor, agent_name)
        return False