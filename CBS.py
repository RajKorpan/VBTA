import heapq
import random
from collections import defaultdict

class Node:
    """HIGH LEVEL NODES FOR STATES: Each Node represents a state in the search space of CBS algorithm, part of the constraint tree."""
    def __init__(self, constraints: dict, solution: dict, cost: int):
        """Initializes a Node with constraints, solution and cost.
        Parameters:
        - constraints: dict, constraints for each agent in the form of {agent: [(location, time step)]}
        - solution: dict, solution for each agent in the form of {agent: [(location, location, location, ...)]} and the time steps are implied, 
            for example if an agent doesnt move it will have the same location listed twice thus increasing the cost and so on
        - cost: int, cost of the solution"""
        self.constraints = constraints
        self.solution = solution
        self.cost = cost

    def __lt__(self, other: 'Node'):
        """Compares two nodes based on their cost.
        Parameters:
        - other: Node, the other node to compare costs against
        Returns:
        - bool: True if self.cost is less than other.cost, False otherwise"""
        return self.cost < other.cost

    

class AStarNode:
    def __init__(self, position, time, g, h, parent=None):
        """Node class for performing AStar on the solution paths
        Parameters:
        - positionL (row, col)
        - time: int
        - g: cost from start to this node
        - h: heuristic cost to the goal
        - parent: reference to another AStarNode for path reconstruction
        """
        self.position = position # (row, col) in the grid/map
        self.time = time # (row, col, time) for time expanded A*
        self.g = g # path cost so far
        self.h = h # path cost heuristic (underestimate, manhattan distance)
        self.parent = parent # will be none if this is the start node

    @property
    def f(self):
        return self.g + self.h # f = g + h from above

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other: 'Node'):
        """equality check for two nodes based on position, if same node we want to check if we have found a shorter path to it
        Parameters:
        - other: Node, the other node to compare position against
        Returns:
        - bool: True if self.position == other.position"""
        return self.position == other.position
        
    
def cbs(start_positions: dict, goal_positions: dict, map: list, occupied_positions: set):
    """Conflict Based Search algorithm to find a solution to a MAPF problem.
    Parameters:
    - start_positions: dict, tuples representing the start positions of each agent
    - goal_positions: dict, tuples representing the goal positions of each agent
    - map: list, 2D list representing the map where 0 is a free cell and 1 is an obstacle
    - occupied_positions: set, all occupied positions on map
    Returns:
    - dict, a solution to the MAPF problem in the form of {agent: [(location, location, location, ...)]} and the time steps are implied"""
    open_list = []

    # Construct the root constraints and solution
    root_constraints = defaultdict(list)
    root_solution = {}
    total_cost = 0
    print(f"CBS START POSITIONS ISSUE: {start_positions}")
    print(f"CBS GOAL POSITIONS ISSUE: {goal_positions}")
    # Iterate over start positions for all agents and create initial paths using low_level_search()
    # for agent_id in range(len(start_positions)):
    for agent_id in start_positions.keys():
        print(f"CBS AGENT ID ISSUE: {agent_id}")
        path = low_level_search(agent_id=agent_id,
                                # by accessing start_positions and goal_positions with agent_id string, order doesnt matter
                                # for some reason start_positions is always coming in sorted, not sure why, it doesnt happen for goal_positions
                                start_pos=start_positions[agent_id],
                                goal_pos=goal_positions[agent_id],
                                constraints={},
                                grid=map,
                                occupied_positions=occupied_positions)
    # for r_id in start_positions.keys():
    #     path = low_level_search(agent_id=r_id,
    #                             start_pos=start_positions[r_id],
    #                             goal_pos=goal_positions[r_id],
    #                             constraints={},
    #                             grid=map,
    #                             occupied_positions=occupied_positions)
        
        # low_level_search will define what the path is
        root_solution[agent_id] = path
        # cost is sum of all paths which gives us the cheapest state which is what we prioritize in the high level search
        total_cost += len(path)

    root = Node(root_constraints, root_solution, total_cost)
    # push root onto open_list
    heapq.heappush(open_list, root)
    # print(f"ROOT CONSTRAINT DICT: {root_constraints.items()}") # Should be empty at this point
    # print(f"ROOT SOLUTION DICT: {root_solution.items()}") # should have one shortest path for each agent 
    print(f"TOTAL COST: {total_cost}") # lowest possible cost

    # any state nodes in the priority queue are searched in this loop 
    while open_list:
        # pop based on STATE cost priority (state cost is the sum of all paths)
        node = heapq.heappop(open_list)

        # search for conflicts in the node
        conflicts = find_conflicts(node.solution)
        if not conflicts:
            return node.solution # Found conflict free solution
        
        # For each conflict we make 2 new child nodes in the constraint tree
        for conflict in conflicts:
            # NOTE: location can be a single space for vertex conflict or 2 spaces for edge conflict
            agent1, agent2, time, location = conflict

            # Set up constraints for the agents
            if agent1 not in node.constraints:
                node.constraints[agent1] = []
            # Same for child2
            if agent2 not in node.constraints:
                node.constraints[agent2] = []

            # Child 1: constrain agent1 but keep a local copy before creating new node
            child1_constraints = {}
            for k, v in node.constraints.items():
                child1_constraints[k] = v[:] # Shallow copy, for each key copy the entire list in the constraints dict
                # print(f"NODE CONSTRAINT DICT: {node.constraints.items()}")
                # print(f"CHILD1_CONSTRAINTS TYPE: {type(child1_constraints)}")
            child1_constraints[agent1].append((time, location))

            child1_solution = dict(node.solution) # shallow copy existing solution into new child node
            # Replanning step for agent1 path which happens in child1
            new_path_for_agent1 = low_level_search(agent_id=agent1,
                                                    start_pos=start_positions[agent1], 
                                                    goal_pos=goal_positions[agent1], 
                                                    constraints=child1_constraints, 
                                                    grid=map,
                                                    occupied_positions=occupied_positions)

            if new_path_for_agent1 is not None: # Otherwise no solution found
                child1_solution[agent1] = new_path_for_agent1
                # Recompute the total cost of path by counting how many steps it took
                new_cost = sum(len(child1_solution[x]) for x in child1_solution)
                # Create child1
                child1 = Node(child1_constraints, child1_solution, new_cost)
                # push child1 onto the heap
                heapq.heappush(open_list, child1)

            # Child 2: constrain agent2 as the other possible solution to the conflict between the two agents
            child2_constraints = {}
            for k, v in node.constraints.items():
                child2_constraints[k] = v[:] # Shallow copy, for each key copy the entire list in the constraints dict
                # print(f"NODE CONSTRAINT DICT: {node.constraints.items()}")
                # print(f"CHILD2_CONSTRAINTS TYPE: {type(child2_constraints)}")
            child2_constraints[agent2].append((time, location))

            child2_solution = dict(node.solution) # shallow copy existing solution into new child node
            # Replanning step for agent2 path which happens in child2
            new_path_for_agent2 = low_level_search(agent_id=agent2,
                                                    start_pos=start_positions[agent2], 
                                                    goal_pos=goal_positions[agent2],
                                                    constraints=child2_constraints, 
                                                    grid=map,
                                                    occupied_positions=occupied_positions)

            if new_path_for_agent2 is not None: # Otherwise no solution found
                child2_solution[agent2] = new_path_for_agent2
                # Recompute the total cost of path by counting how many steps it took
                new_cost = sum(len(child2_solution[x]) for x in child2_solution)
                # Create child2
                child2 = Node(child2_constraints, child2_solution, new_cost)
                # push child2 onto the heap
                heapq.heappush(open_list, child2)

    return None # No solution found

def find_conflicts(solution: dict):
    """searches the solutions for conflicts between agents, conflicts are defined as two agents being in the same location at the same time or
    two agents crossing each other's paths (this means that the agents essentially swap positions at consecutive time steps)
    This function must:
    1. check all agents paths for conflicts by iterating through every pair of agents
    2. compare their paths time step by time step
        a. if 2 agents are at the same location at the same time, return the conflict
        b. (optional) if 2 agents cross each other between time step t and t+1 return the conflict
    3. For each conflict detected record the details of the conflict in the form of (agent1, agent2, time, location)
    4. return a list of all conflicts detected
    
    solution:   List of paths for all agents
                solution[i] is a list of (row, col) locations representing the path for agent i
                
    Returns:    A list of conflicts.
                Each conflict is a tuple (agent1, agent2, time, conflict_info)
                - agent1, agent2 are the conflicting agents' indices
                - time is the timestep at which the conflict occurs
                - conflict_info could be:
                    * a single (row, col) if its a vertex conflict
                    * a pair ((row1, col1), (row2, col2)) if its an edge conflict
    """
    conflicts = []

    agent_ids = sorted(solution.keys()) # if the agents need to be sorted
    # agent_ids = list(solution.keys()) # if not sorted

    num_agents = len(agent_ids)
    # Find the maximum path length among all agent_ids by looping
    max_time = max(len(solution[x]) for x in agent_ids)

    # Compare each pair of agents
    for i_idx in range(num_agents):
        i = agent_ids[i_idx]
        # list of ith agents solution
        path_i = solution[i]
        for j_idx in range(i_idx + 1, num_agents):
            j = agent_ids[j_idx]
            # list of jth agents solution
            path_j = solution[j]

            # Check times up to the longest path among the two using t as a common index
            for t in range(max_time):
                # get agent i's location at time t (stay at last location if path is ended)
                if t < len(path_i):
                    loc_i_t = path_i[t]
                else:
                    loc_i_t = path_i[-1] # Consider the agent done

                # Get agent j's location at time t
                if t < len(path_j):
                    loc_j_t = path_j[t]
                else:
                    loc_j_t = path_j[-1]

                # ---Vertex Conflict--- #
                if loc_i_t == loc_j_t:
                    # add to conflicts(agent i, agent j, time t, location of agent i at time t)
                    conflicts.append((i, j, t, loc_i_t))

                # ---Edge Conflict--- #
                # Occurs if agent i goes from path_i[t] to path_i[t+1] while
                # agent j goes from path_j[t] to path_j[t+1], and these two moves "swap" locations.
                if (t + 1 < len(path_i)) and (t + 1 < len(path_j)):
                    loc_i_next = path_i[t + 1]
                    loc_j_next = path_j[t + 1]
                    # check if the positions are swapped
                    if loc_i_t == loc_j_next and loc_i_next == loc_j_t:
                        # Store conflict info as pair of edges being traversed
                        conflict_info = (loc_i_t, loc_i_next)
                        conflicts.append((i, j, t + 1, conflict_info))

    return conflicts


def low_level_search(
    agent_id: int,
    start_pos: tuple[int, int],
    goal_pos: tuple[int, int],
    constraints: dict,
    grid: list,
    occupied_positions: set
):
    """A* search in a time-expanded manner for a single agent. 
    - agent_id: the ID to look up constraints in constraints[agent_id]
    - start_pos: (row, col)
    - goal_pos: (row, col)
    - constraints: dict with constraints[agent_id] = [(time, (row, col)), ...]
    - grid: 2D map (0=free, 1=blocked)
    Return: List of positions from start to goal, or None if no path found.
    """
    # NOTE: for the constraints, we can say that on that timestep, that node becomes untraversable? that may eliminate the waiting action the agents sometimes choose since they would
    # just move on trying to find a path elsewhere.

    # Check: if no constraints for agent, use empty dict
    agent_constraints = constraints.get(agent_id, {})

    # Convert constraints to set for constant lookup time
    # these states are a set of (time, (row, col)) the agent cannot occupy
    blocked_states = set(agent_constraints)

    # Priority queue for open nodes, dictionary for visited nodes
    open_list = []
    visited = {} # key: (row, col, time), value: minimal g found so far

    # Heuristic function (manhattan dist)
    def heuristic(pos):
        return abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])

    # Create the start node with g = 0 and h = manhattan dist using start position
    start_node = AStarNode(
        position = start_pos,
        time = 0,
        g = 0,
        h = heuristic(start_pos),
        parent = None
    )

    # push it to the priority queue which is sorted by f
    heapq.heappush(open_list, start_node)
    # mark (start row, start col, time = 0) as visited
    visited[(start_pos[0], start_pos[1], 0)] = 0 # and set g = 0 saying this is the start node

    # Directions for neighbor nodes, allow 8 direction movement with staying still (right, left, up, down, stay still, RU, RD, LU, LD)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    # search loop
    while open_list:
        # pop node with smallest f = g + h
        current = heapq.heappop(open_list)
        
        # Check if goal position has been reached
        if current.position == goal_pos:
            # Reconstruct path
            return reconstruct_path(current)

        # Otherwise expand neighbors
        current_row, current_col = current.position
        current_time = current.time

        # drow, dcol are directions such as going up one row or right one column
        for drow, dcol in directions:
            new_row = current_row + drow
            new_col = current_col + dcol
            new_time = current_time + 1

            # check map boundaries
            if not in_bounds(new_row, new_col, grid):
                # if out of bounds then ignore this direction
                continue

            if (new_row, new_col) in occupied_positions:
                continue

            # check constraints: is (new_time, (new position)) blocked to the agent this move?
            if (new_time, (new_row, new_col)) in blocked_states:
                # if constrained move then ignore this direction
                continue

            # compute the cost of g and h
            new_g = current.g + 1
            new_h = heuristic((new_row, new_col))

            # WE ONLY PUSH A NEW NODE if we didnt visit (new_row, new_col, new_time) or found a cheaper cost to a node
            if (new_row, new_col, new_time) not in visited or new_g < visited[(new_row, new_col, new_time)]:
                visited[(new_row, new_col, new_time)] = new_g
                new_node = AStarNode(
                    position = (new_row, new_col),
                    time = new_time,
                    g = new_g,
                    h = new_h,
                    parent = current
                )
                heapq.heappush(open_list, new_node)
    
    # if open_list is exhausted and we never find a goal
    return None

def reconstruct_path(node):
    """Trace back via parent pointers to build the path from start to goal."""
    path = []
    current = node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return list(reversed(path))

def in_bounds(r, c, grid):
    """Check if row,col is within the grid."""
    rows = len(grid)
    cols = len(grid[0])
    return 0 <= r < rows and 0 <= c < cols

# def low_level_search_initial(start_positions: list, goal_positions: list, map: list):
#     """same as above except for the initial path building, then its not used again.
#     Specifically does not take in any constraints."""
#     """A simpler version of single-agent pathfinding ignoring constraints
#        just to get an initial path for the root node.
#     """
#     # TODO: implement a basic pathfinding (e.g., BFS or A* ignoring other agents).
#     solutions = defaultdict(list)
#     for position in start_positions:
#         start_node = AStarNode(None, position)

#     return solutions  # dummy 2-step path, replace with real path

def load_map(map_filename):
    """Function to load ascii maps from the MAPF benchmark .map files"""
    with open(map_filename, 'r') as f:
        _type = f.readline().strip() # Type octile or similar
        height_line = f.readline().strip() # Height of map e.g. "height 45"
        width_line = f.readline().strip() # Width of map e.g. "width 52"
        map_line = f.readline().strip() # delinates start of map with line that says "map"

        height = int(height_line.split()[1])
        width = int(width_line.split()[1])

        grid = []
        for _ in range(height):
            row_data = f.readline().strip()
            if len(row_data) != width:
                raise ValueError(f"Map file error: row length {len(row_data)} != width {width}") # sanity check for malformed map files
            # periods and G's are traversable terrain, everything else will be unpassable, there are 5 types of unpassable terrains, water will be unpassable
            # row = [0 if c == '.' or c == 'G' else 1 for c in row_data] 
            # 1 means blocked, 0 means free
            row = []
            for c in row_data:
                if c in [".", 'G']:
                    row.append(0)
                else:
                    row.append(1)
            grid.append(row)

    return grid

def load_scenario(scenario_filename):
    """Function to load scenarios from the MAPF benchmark .scen files, not using .scen files for now, just randomly assigning agents"""
    agents = []
    with open(scenario_filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue # skip comments
            parts = line.strip().split()
            # .scen files are in format "0 mapName.map 5 10 30 35 42" in order these are the bucket, map file name, start row, start col, goal row, goal col, optimal distance
            start_row = int(parts[2])
            start_col = int(parts[3])
            goal_row = int(parts[4])
            goal_col = int(parts[5])
            agents.append((start_row, start_col, goal_row, goal_col))
    return agents

def get_random_free_position(grid, occupied_positions):
    """
    Parameters
    - grid: 2D list of 0/1 cells representing the map free/obstacles
    - occupied_positions: exisiting agent positions, places we want to consider
      blocked when choosing a new position

    Returns: a single (row, col) position for one agent,
            randomly from free cells with value = 0 that are not occupied
    """
    free_cells = []
    # iterate thru entire grid and find the free cells, make a list of them for choosing from
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 0 and grid[r][c] not in occupied_positions:
                free_cells.append((r, c))

    if not free_cells:
       raise ValueError("No free cells available to place a robot!")
    
    # randomly sample without replacement
    chosen = random.sample(free_cells, 1)
    return chosen