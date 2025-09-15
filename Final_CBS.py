"""

Python implementation of Conflict-based search

author: Ashwin Bose (@atb033)

"""
import sys
sys.path.insert(0, '../')
import argparse
import yaml
from math import fabs
from itertools import combinations
from collections import defaultdict
from copy import deepcopy

from hvbta.pathfinding.a_star import AStar

# ---- helpers to normalize coordinates regardless of shape ----
def _xy_from_start(value):
    """Accepts tuple/list, State, or object with .location; returns (x, y)."""
    # tuple/list like (x, y)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])

    # State-like: has .location with .x/.y (your State class)
    loc = getattr(value, "location", None)
    if loc is not None and hasattr(loc, "x") and hasattr(loc, "y"):
        return int(loc.x), int(loc.y)

    # Location directly (rare, if you ever store Location instead of State)
    if hasattr(value, "x") and hasattr(value, "y"):
        return int(value.x), int(value.y)

    raise TypeError(f"Unsupported start/goal value type: {type(value)}")


def _get_start_goal(agent_obj):
    """
    Agent can be:
      - dict with 'start'/'goal'
      - object with .start/.goal
    Each of start/goal can be tuple (x,y) or State(Location(x,y), t)
    Returns (sx, sy, gx, gy)
    """
    if isinstance(agent_obj, dict):
        start = agent_obj.get("start")
        goal  = agent_obj.get("goal")
    else:
        start = getattr(agent_obj, "start", None)
        goal  = getattr(agent_obj, "goal", None)

    if start is None or goal is None:
        raise ValueError("Agent missing start/goal")

    sx, sy = _xy_from_start(start)
    gx, gy = _xy_from_start(goal)
    return sx, sy, gx, gy


class Location(object):
    """Location class, represents agent location in environment as an (x, y) coordinate tuple"""
    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __str__(self):
        return str((self.x, self.y))

class State(object):
    """State class, represents the state of the agent as its time step and location
    location is a Location object
    time is an int starting at 0
    """
    def __init__(self, time, location):
        self.time = time
        self.location = location
    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash((self.time, self.location.x, self.location.y))
    def is_equal_except_time(self, state):
        return self.location == state.location
    def __str__(self):
        return str((self.time, self.location.x, self.location.y))

class Conflict(object):
    """Conflict class differentiates between vertex and edge conflicts"""
    VERTEX = 1
    EDGE = 2
    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ''
        self.agent_2 = ''

        self.location_1 = Location()
        self.location_2 = Location()

    def __str__(self):
        return '(' + str(self.time) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
             ', '+ str(self.location_1) + ', ' + str(self.location_2) + ')'

class VertexConstraint(object):
    """Vertex constraint class, imposes vertex constraints on high level search nodes at a specified time and location"""
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location) + ')'

class EdgeConstraint(object):
    """Edge constraint class, imposes edge constraints on high level search nodes at a specified time and location, has 2 locations as opposed to vertex constraints (needs both verticies)"""
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2
    def __eq__(self, other):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2
    def __hash__(self):
        return hash(str(self.time) + str(self.location_1) + str(self.location_2))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location_1) +', '+ str(self.location_2) + ')'

class Constraints(object):
    """Base constraint class, keeps track of sets of constraints and can add new ones as needed"""
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        # union operation for sets
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])

class ConstraintTable:
    """
    Fast time-indexed lookups for CBS constraints.
    Built from your per-agent Constraints() object.
    """
    def __init__(self, constraints):
        # time -> set of (x, y)
        self.vertex = defaultdict(set)
        # time -> set of ((x1,y1),(x2,y2))
        self.edge = defaultdict(set)
        # optional earliest goal time
        self.earliest_goal_time = None

        self._build(constraints)

    def _build(self, constraints):
        # vertex constraints
        for vc in getattr(constraints, "vertex_constraints", []):
            self.vertex[vc.time].add((vc.location.x, vc.location.y))

        # edge constraints
        for ec in getattr(constraints, "edge_constraints", []):
            self.edge[ec.time].add((
                (ec.location_1.x, ec.location_1.y),
                (ec.location_2.x, ec.location_2.y)
            ))

    def blocks_vertex(self, x, y, t) -> bool:
        return (x, y) in self.vertex.get(t, ())

    def blocks_edge(self, x1, y1, x2, y2, t) -> bool:
        return ((x1, y1), (x2, y2)) in self.edge.get(t, ())


class Environment(object):
    """Environment class represents the navigating environment"""
    def __init__(self, dimension, agents, obstacles):
        self.dimension = dimension
        self.obstacles = obstacles

        self.agents = agents
        self.agent_dict = {}

        self.make_agent_dict()

        self.constraints = Constraints()
        self.constraint_dict = {}

        self.a_star = AStar(self)

    def _time_horizon(self, agent_name: str) -> int:
        """
        Conservative upper bound on time.
        Manhattan distance + (width + height) buffer works well in grids.
        """
        agent = self.agent_dict[agent_name]
        sx, sy, gx, gy = _get_start_goal(agent)
        w, h = self.dimension  # (width, height)

        manhattan = abs(sx - gx) + abs(sy - gy)
        return int(manhattan + (w + h))


    def get_neighbors(self, state, allow_wait=False):
        neighbors = []
        if isinstance(state, tuple):
            x, y, t = state
            state = State(t, Location(x, y))

        if allow_wait:
            # Wait action
            n = State(state.time + 1, state.location)
            if self.state_valid(n):
                neighbors.append(n)
        # Up action
        n = State(state.time + 1, Location(state.location.x, state.location.y+1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Down action
        n = State(state.time + 1, Location(state.location.x, state.location.y-1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Left action
        n = State(state.time + 1, Location(state.location.x-1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Right action
        n = State(state.time + 1, Location(state.location.x+1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Diagonal actions
        # Up/Right
        # n = State(state.time + 1, Location(state.location.x+1, state.location.y+1))
        # if self.state_valid(n) and self.transition_valid(state, n):
        #     neighbors.append(n)
        # # Up/Left
        # n = State(state.time + 1, Location(state.location.x-1, state.location.y+1))
        # if self.state_valid(n) and self.transition_valid(state, n):
        #     neighbors.append(n)
        # # Down/Right
        # n = State(state.time + 1, Location(state.location.x+1, state.location.y-1))
        # if self.state_valid(n) and self.transition_valid(state, n):
        #     neighbors.append(n)
        # # Down/Left
        # n = State(state.time + 1, Location(state.location.x-1, state.location.y-1))
        # if self.state_valid(n) and self.transition_valid(state, n):
        #     neighbors.append(n)

        return neighbors


    def get_first_conflict(self, solution):
        """Check for conflicts"""
        max_t = max([len(plan) for plan in solution.values()])
        result = Conflict()
        for t in range(max_t):
            # Check for vertex conflicts
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1 = self.get_state(agent_1, solution, t)
                state_2 = self.get_state(agent_2, solution, t)
                if state_1.is_equal_except_time(state_2):
                    result.time = t
                    result.type = Conflict.VERTEX
                    result.location_1 = state_1.location
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    return result

            # Check for edge conflicts
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t+1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t+1)

                # use is_equal_except_time bc we are checking at specific time steps
                if state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a):
                    result.time = t
                    result.type = Conflict.EDGE
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    result.location_1 = state_1a.location
                    result.location_2 = state_1b.location
                    return result
        return False

    def create_constraints_from_conflict(self, conflict):
        """Create a new constraint based on a conflict"""
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint

        elif conflict.type == Conflict.EDGE:
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(conflict.time, conflict.location_1, conflict.location_2)
            e_constraint2 = EdgeConstraint(conflict.time, conflict.location_2, conflict.location_1)

            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}

            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict

    def get_state(self, agent_name, solution, t):
        """Get the state of an agent at any timestep t"""
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def state_valid(self, state):
        """check location.x is within bounds and location.y is within bounds and we are not violating a vertex constraint or in an obstacle"""
        return state.location.x >= 0 and state.location.x < self.dimension[0] \
            and state.location.y >= 0 and state.location.y < self.dimension[1] \
            and VertexConstraint(state.time, state.location) not in self.constraints.vertex_constraints \
            and (state.location.x, state.location.y) not in self.obstacles

    def transition_valid(self, state_1, state_2):
        """check that the transition did not violate an edge constraint"""
        return EdgeConstraint(state_1.time, state_1.location, state_2.location) not in self.constraints.edge_constraints

    def is_solution(self, agent_name):
        pass

    def admissible_heuristic(self, state, agent_name):
        """Distance heuristic"""
        goal = self.agent_dict[agent_name]["goal"]
        return fabs(state.location.x - goal.location.x) + fabs(state.location.y - goal.location.y)


    def is_at_goal(self, state, agent_name):
        """Check if we are at the goal position"""
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

    def make_agent_dict(self):
        """Create a new agent dictionary with 0 time and start and goal locations"""
        for agent in self.agents:
            start_state = State(0, Location(agent['start'][0], agent['start'][1]))
            goal_state = State(0, Location(agent['goal'][0], agent['goal'][1]))

            self.agent_dict.update({agent['name']:{'start':start_state, 'goal':goal_state}})

    def compute_solution(self):
        """
        Low-level planning for all agents given the current high-level node's constraints.
        Returns dict: {agent_name: path_list_of_dicts}
        Each path entry expected like {'t': t, 'x': x, 'y': y} to match your A* output.
        """
        solution = {}
    
        for agent_name in self.agent_dict.keys():
            # Get (or create) this agent's Constraints() object
            constraints = self.constraint_dict.get(agent_name)
            if constraints is None:
                constraints = Constraints()
                self.constraint_dict[agent_name] = constraints
    
            # Build the per-agent constraint table
            ctab = ConstraintTable(constraints)
    
            # Time horizon
            T_max = self._time_horizon(agent_name)
    
            # Call the new A*
            local_solution = self.a_star.search(self, agent_name, ctab, T_max)
            if not local_solution:
                return None  # Signal failure to the high-level
    
            solution[agent_name] = local_solution
    
        return solution


    def compute_solution_cost(self, solution):
        """compute total solution cost"""
        return sum([len(path) for path in solution.values()])

class HighLevelNode(object):
    """a CBS node that contains a solution, constraint dictionary, and solution cost for a single agent"""
    def __init__(self):
        self.solution = {}
        self.constraint_dict = {}
        self.cost = 0

    def __eq__(self, other):
        """Check for equivalent nodes"""
        if not isinstance(other, type(self)): return NotImplemented
        return self.solution == other.solution and self.cost == other.cost
    
    def __hash__(self):
        # hash with cost and frozenset of per-agent constraints to ensure uniqueness
        vc = [] # vertex constraints
        ec = [] # edge constraints
        for agent, cons in sorted(self.constraint_dict.items()):
            vc.extend((agent, c.time, c.location.x, c.location.y) for c in cons.vertex_constraints)
            ec.extend((agent, c.time, c.location_1.x, c.location_1.y,
                              c.location_2.x, c.location_2.y) for c in cons.edge_constraints)
        return hash((self.cost, tuple(sorted(vc)), tuple(sorted(ec))))

        

    def __lt__(self, other):
        """Compare costs of node solutions"""
        return self.cost < other.cost

class CBS(object):
    """CBS search class"""
    def __init__(self, environment):
        self.env = environment
        self.open_set = set()
        self.closed_set = set()
    def search(self):
        start = HighLevelNode()
        # TODO: Initialize it in a better way
        start.constraint_dict = {}
        for agent in self.env.agent_dict.keys():
            start.constraint_dict[agent] = Constraints()
        # compute solution
        start.solution = self.env.compute_solution()
        if not start.solution:
            return (None, 0, 0)
        # compute cost of solution
        start.cost = self.env.compute_solution_cost(start.solution)

        # count conflicts resolved for a difficulty metric
        total_conflicts = 0

        # set open_set to the solution HighLevelNode
        self.open_set |= {start}

        # Search open_set starting with the lowest cost nodes first, adding them to closed_set as we go
        while self.open_set:
            P = min(self.open_set)
            self.open_set -= {P}
            self.closed_set |= {P}

            
            self.env.constraint_dict = P.constraint_dict
            conflict_dict = self.env.get_first_conflict(P.solution)
            if conflict_dict:
                total_conflicts += 1
            else:
                print(f"Solution found after expanding {len(self.closed_set)} high level nodes")
                print(f"Total conflicts identified {total_conflicts}")

                return self.generate_plan(P.solution), len(self.closed_set), total_conflicts

            constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)

            for agent in constraint_dict.keys():
                new_node = deepcopy(P)
                new_node.constraint_dict[agent].add_constraint(constraint_dict[agent])

                self.env.constraint_dict = new_node.constraint_dict
                new_node.solution = self.env.compute_solution()
                if not new_node.solution:
                    continue
                new_node.cost = self.env.compute_solution_cost(new_node.solution)

                # TODO: ending condition
                if new_node not in self.closed_set:
                    self.open_set |= {new_node}

        return (None, len(self.closed_set), total_conflicts)

    def generate_plan(self, solution):
        plan = {}
        for agent, path in solution.items():
            # dictionary for output
            path_dict_list = [{'t':state.time, 'x':state.location.x, 'y':state.location.y} for state in path]
            plan[agent] = path_dict_list
        return plan


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("param", help="input file containing map and obstacles")
    parser.add_argument("output", help="output file with the schedule")
    args = parser.parse_args(args)

    # Read from input file
    with open(args.param, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param['agents']

    env = Environment(dimension, agents, obstacles)

    # Searching
    cbs = CBS(env)
    solution_data = cbs.search()
    if solution_data:
        solution, nodes_expanded, total_conflicts = solution_data
        print("Solution found")
    
        # Calculate makespan
        if solution:
            makespan = 0
            # NOTE: may have to iterate thru agents then paths
            for agent, path in solution.items():
                if path:
                    makespan = max(makespan, path[-1]['t'])


        # Write to output file
        output = dict()
        # agent movement schedule
        output["schedule"] = solution

        # SOLUTION METRICS
        # Sum of costs, indicates solution quality and correlates with ovferal difficulty
        output["cost"] = env.compute_solution_cost(solution)
        # Makespan indicates last agent to reach goal, indicates congestion or bottlenecks
        output["makespan"] = makespan

        # SEARCH EFFORT METRICS
        # high level nodes expanded indicates that the high-level search had to explore more possibilities in the constraint tree to find a conflict-free solution
        output["high_level_nodes_expanded"] = nodes_expanded
        # total number of conflicts found across all high-level nodes that were checked, each leading to a branching decision in the constraint tree. More conflicts generally implies a more complex interaction between agents
        output["num_of_conflicts_identified"] = total_conflicts
    else:
        print("Solution not found")

    with open(args.output, 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml)


if __name__ == "__main__":
    main()
