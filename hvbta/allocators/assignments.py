import numpy as np
from typing import Callable, List, Tuple, Optional
import random
from itertools import permutations, combinations

# old hint List[List[Tuple[Optional[int], Optional[int]]]]
def generate_random_assignments(num_robots: int, num_tasks: int, num_assignments: int = 5) -> List[Tuple[List[Tuple[Optional[int], Optional[int]]], List[int], List[int]]]:
    """
    Generates a list of random assignments of robots to tasks.
    Each assignment is a list of (robot, task) pairs, where some robots or tasks may be unassigned.
    These assignments are not generated based on suitability scores.
    
    Parameters:
        num_robots: Number of available robots.
        num_tasks: Number of tasks to be assigned.
        num_assignments: Number of different random assignments to generate.
    
    Returns:
        assignments: A list of random assignments, where each assignment is a list of (robot, task) pairs.
    """
    assignments = []

    # Generate random assignments
    for _ in range(num_assignments):
        # Generate random indices for pairing
        available_robots = random.sample(range(num_robots), num_robots)
        available_tasks = random.sample(range(num_tasks), num_tasks)
        
        # Determine the number of pairs to create (limited by the smaller of num_robots or num_tasks)
        num_pairs = min(num_robots, num_tasks)

        # Pair robots and tasks using the precomputed random indices
        assigned_pairs = [(available_robots[i], available_tasks[i]) for i in range(num_pairs)]

        # Determine unassigned robots and tasks
        unassigned_robots = available_robots[num_pairs:] if num_robots > num_pairs else []
        unassigned_tasks = available_tasks[num_pairs:] if num_tasks > num_pairs else []

        assignments.append((assigned_pairs, unassigned_robots, unassigned_tasks))

    return assignments

def generate_all_unique_assignments(num_robots: int, num_tasks: int) -> List[Tuple[List[Tuple[int, int]], List[int], List[int]]]:
    """
    Generates all possible unique assignments of robots to tasks.
    Each assignment is a tuple containing:
      1. A list of (robot, task) pairs.
      2. A list of unassigned robots.
      3. A list of unassigned tasks.
    
    Parameters:
        num_robots: Number of available robots.
        num_tasks: Number of tasks to be assigned.
    
    Returns:
        all_assignments: A list of all possible unique assignments, where each assignment is a tuple (assigned_pairs, unassigned_robots, unassigned_tasks).
    """
    all_assignments = []

    # Determine the number of pairs to create (limited by the smaller of num_robots or num_tasks)
    num_pairs = min(num_robots, num_tasks)

    # Generate all possible combinations of robots for the tasks
    robot_combinations = list(combinations(range(num_robots), num_pairs))
    task_combinations = list(combinations(range(num_tasks), num_pairs))

    # Generate all permutations of those combinations
    for robot_comb in robot_combinations:
        for task_comb in task_combinations:
            # Create all permutations for the selected robot and task pairs
            for permuted_robots in permutations(robot_comb):
                for permuted_tasks in permutations(task_comb):
                    # Pair the robots and tasks
                    assigned_pairs = [(permuted_robots[i], permuted_tasks[i]) for i in range(num_pairs)]

                    # Determine unassigned robots and tasks
                    unassigned_robots = [r for r in range(num_robots) if r not in permuted_robots]
                    unassigned_tasks = [t for t in range(num_tasks) if t not in permuted_tasks]

                    all_assignments.append((assigned_pairs, unassigned_robots, unassigned_tasks))

    return all_assignments

def generate_high_suitability_assignments(num_robots: int, num_tasks: int, suitability_matrix: List[List[float]], num_assignments: int = 5) -> List[Tuple[List[Tuple[int, int]], List[int], List[int]]]:
    """
    Generates assignments for robots with the highest total suitability across all tasks.
    Each assignment includes the top robots paired with tasks based on suitability.
    
    Parameters:
        num_robots: Number of available robots.
        num_tasks: Number of tasks to be assigned.
        suitability_matrix: A 2D list where element [i][j] represents the suitability of robot i for task j.
        num_assignments: Number of different random assignments to generate.
    
    Returns:
        assignments: A list of high-suitability assignments, where each assignment is a tuple containing:
      - A list of (robot, task) pairs.
      - A list of unassigned robots.
      - A list of unassigned tasks.
    """
    assignments = []
    if num_robots == 1 and num_tasks == 1:
        assigned_pairs = [(0,0)]
        unassigned_robots = []
        unassigned_tasks = []
        assignments.append((assigned_pairs, unassigned_robots, unassigned_tasks))
        return assignments
    elif num_robots == 1:
        for task_id in range(num_tasks):
            assigned_pairs = [(0, task_id)]
            unassigned_tasks = list(set(range(num_tasks)) - {task_id})
            assignments.append((assigned_pairs, [], unassigned_tasks))
        return assignments
    elif num_tasks == 1:
        for robot_id in range(num_robots):
            assigned_pairs = [(robot_id, 0)]
            unassigned_robots = list(set(range(num_robots)) - {robot_id})
            assignments.append((assigned_pairs, unassigned_robots, []))
        return assignments
        
    num_assignments = min(num_assignments, num_robots * num_tasks * 10)
    

    # Calculate total suitability score for each robot across all tasks
    robot_suitability_scores = [(i, sum(suitability_matrix[i])) for i in range(num_robots)]
    
    # Sort robots by total suitability score in descending order
    sorted_robots = sorted(robot_suitability_scores, key=lambda x: x[1], reverse=True)
    
    # Select the top robots based on total suitability scores
    top_robot_indices = [robot[0] for robot in sorted_robots[:min(num_robots, num_tasks)]]
    
    # Identify robots with the maximum score for each task
    for task in range(num_tasks):
        max_score = max(suitability_matrix[robot][task] for robot in range(num_robots))
        top_task_robots = [robot for robot in range(num_robots) if suitability_matrix[robot][task] == max_score]
        top_robot_indices.extend(top_task_robots)
    
    # Remove duplicates and keep only as many robots as tasks if robots exceed tasks
    top_robot_indices = list(set(top_robot_indices))[:min(num_robots, num_tasks)]

    for _ in range(num_assignments):
        # Randomly sample tasks to pair with the top robots
        available_tasks = random.sample(range(num_tasks), min(num_tasks, num_robots))
        
        # Randomly sample robots to pair with the tasks
        available_robots = random.sample(top_robot_indices, min(num_tasks, num_robots))

        # Pair each selected robot with a randomly chosen task
        assigned_pairs = [(available_robots[i], available_tasks[i]) for i in range(len(available_robots))]
        
        # Determine unassigned robots and tasks
        unassigned_robots = list(set(range(num_robots)) - set(available_robots))
        unassigned_tasks = list(set(range(num_tasks)) - set(available_tasks))
        # Store each assignment as a tuple
        assignments.append((assigned_pairs, unassigned_robots, unassigned_tasks))

    #***print(assignments)
    return assignments

# Type: a candidate is a list of (robot_idx, task_idx) pairs
Assignment = List[Tuple[int, int]]

def generate_candidates_perturb_and_map(
    S: np.ndarray,
    K: int,
    solve_fn: Callable[[np.ndarray], Assignment],
    noise: str = "gumbel",       # "gumbel" or "gaussian"
    scale: float = 0.10,         # noise strength relative to (S.max - S.min)
    anneal: bool = True,         # gradually reduce noise to mix diversity + quality
    max_tries: int = None,       # attempts to collect K unique candidates
    seed: int = None,
    dedup: bool = True,
) -> List[Assignment]:
    """
    Perturb-and-MAP (a.k.a. Perturb-and-Match) candidate generator.
    - S: suitability matrix (R x T), raw or normalized
    - K: desired number of candidates
    - solve_fn: a function that solves max-assignment on a matrix and returns [(i,j), ...]
                e.g. `lambda M: jv_task_allocation(M)[0]` if your JV returns (pairs, score, time)
    - noise: "gumbel" (recommended) or "gaussian"
    - scale: noise strength as a fraction of (S.max - S.min)
    - anneal: if True, noise scale decreases over iterations (diversity early, quality later)
    - dedup: ensure unique candidates
    """
    rng = np.random.default_rng(seed)
    R, T = S.shape
    spread = float(S.max() - S.min() + 1e-9)
    base_sigma = scale * spread
    max_tries = max_tries or (5 * K)

    def _sample_noise(shape, sigma):
        if noise == "gumbel":
            # Gumbel(0, beta). std ≈ 1.28255*beta → pick beta so that std ≈ sigma
            beta = sigma / 1.28255 if sigma > 0 else 0.0
            return rng.gumbel(loc=0.0, scale=max(1e-12, beta), size=shape)
        elif noise == "gaussian":
            return rng.normal(loc=0.0, scale=max(1e-12, sigma), size=shape)
        else:
            raise ValueError("noise must be 'gumbel' or 'gaussian'")

    # storage
    candidates: List[Assignment] = []
    seen = set()

    tries = 0
    i = 0
    while len(candidates) < K and tries < max_tries:
        tries += 1
        # anneal noise from base_sigma → base_sigma/5 (heuristic)
        if anneal and K > 1:
            frac = i / max(1, K - 1)  # 0..1
            sigma = base_sigma * (0.2 + 0.8 * (1.0 - frac))
        else:
            sigma = base_sigma

        N = _sample_noise((R, T), sigma)
        S_pert = S + N

        # Solve max assignment on perturbed scores
        assignment: Assignment = solve_fn(S_pert)

        # Dedup by making a canonical key
        if dedup:
            key = tuple(sorted(assignment))
            if key in seen:
                continue
            seen.add(key)

        candidates.append(assignment)
        i += 1

    return candidates
