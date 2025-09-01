import time, numpy as np
from typing import List, Tuple, Callable
from .assignments import generate_random_assignments
from hvbta.models import CapabilityProfile, TaskDescription
from hvbta.suitability import calculate_total_suitability, check_zero_suitability, calculate_suitability_matrix

def rank_assignments_range(assignments: List[List[Tuple[int, int]]], suitability_matrix: List[List[float]]) -> Tuple[List[float], List[int]]:
    """
    Ranks the given assignments based on their total suitability scores.
    
    Parameters:
        assignments: A list of assignments, where each assignment is a list of (robot, task) pairs.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
    
    Returns:
        (total_scores, ranking): A tuple where
      1. A list of total suitability scores for each assignment.
      2. A list of indices representing the ranking of the assignments, where the first index corresponds to the
         assignment with the highest total score.
    """
    total_scores = []

    # Calculate total suitability scores for each assignment
    for assignment in assignments:
        total_suitability = sum(suitability_matrix[robot][task] for robot, task in assignment[0])
        total_scores.append(total_suitability)

    # Get the ranking of assignments based on the total scores (higher score is better)
    ranking = sorted(range(len(total_scores)), key=lambda i: total_scores[i], reverse=True)
    #***print (f"ranking: {ranking}")
    return total_scores, ranking

def rank_assignments_borda(assignments: List[List[Tuple[int, int]]], suitability_matrix: List[List[float]]) -> Tuple[List[float], List[int]]:
    """
    Ranks each assignment from the perspective of each robot based on its suitability for its task in each assignment,
    and uses the Borda count method to aggregate these rankings.
    
    Parameters:
        assignments: A list of assignments, where each assignment is a list of (robot, task) pairs.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
    
    Returns:
        (borda_scores, ranked_assignments): A tuple containing:
      1. A list of Borda scores for each assignment.
      2. A list of indices representing the ranking of the assignments based on Borda scores, where the first index corresponds
         to the assignment with the highest total Borda score.
    """
    num_assignments = len(assignments)
    num_robots = len(suitability_matrix)

    # Initialize Borda scores for each assignment
    borda_scores = [0] * num_assignments

    # For each robot, rank the assignments based on its suitability rating
    for robot in range(num_robots):
        # Collect suitability scores for this robot across all assignments
        robot_scores = []
        
        for assignment_index, assignment in enumerate(assignments):
            # Find the task assigned to this robot in the current assignment
            assigned_task = next((task for r, task in assignment[0] if r == robot), None)
            
            # If the robot has no task assigned in this assignment, assign it the lowest score
            if assigned_task is None:
                robot_scores.append((assignment_index, -1))  # Using -1 to rank unassigned lower
            else:
                robot_scores.append((assignment_index, suitability_matrix[robot][assigned_task]))

        # Sort robot's scores for all assignments: unassigned (score -1) at the bottom, then by suitability
        robot_scores.sort(key=lambda x: x[1], reverse=True)

        # Assign Borda points based on the ranking
        for rank, (assignment_index, _) in enumerate(robot_scores):
            borda_points = len(robot_scores) - rank - 1  # Borda count: higher ranks get more points
            borda_scores[assignment_index] += borda_points

    # Rank assignments based on total Borda scores (higher score is better)
    ranked_assignments = sorted(range(num_assignments), key=lambda i: borda_scores[i], reverse=True)

    return borda_scores, ranked_assignments

def rank_assignments_approval(assignments: List[List[Tuple[int, int]]], suitability_matrix: List[List[float]], threshold: float = 0.5) -> Tuple[List[int], List[int]]:
    """
    Uses approval voting where each robot gives 1 point to assignments where its suitability for its task is above a threshold.
    If a robot's suitability is below the threshold or it is unassigned, it gives 0 points for that assignment.
    
    Parameters:
        assignments: A list of assignments, where each assignment is a list of (robot, task) pairs.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
        threshold: The suitability threshold above which a robot will approve an assignment (default is 10).
    
    Returns:
        (approval_scores, ranked_assignments): A tuple containing:
      1. A list of approval scores for each assignment.
      2. A list of indices representing the ranking of the assignments based on approval scores.
    """
    num_assignments = len(assignments)
    num_robots = len(suitability_matrix)

    # Initialize approval scores for each assignment
    approval_scores = [0] * num_assignments

    # For each robot, evaluate each assignment based on the suitability threshold
    for robot in range(num_robots):
        for assignment_index, assignment in enumerate(assignments):
            # Find the task assigned to this robot in the current assignment
            assigned_task = next((task for r, task in assignment[0] if r == robot), None)
            
            # Check if robot's suitability rating for this task is above the threshold
            if assigned_task  and suitability_matrix[robot][assigned_task] > threshold:
                approval_scores[assignment_index] += 1  # Robot approves this assignment

    # Rank assignments based on approval scores (higher score is better)
    ranked_assignments = sorted(range(num_assignments), key=lambda i: approval_scores[i], reverse=True)

    return approval_scores, ranked_assignments

def rank_assignments_majority_judgment(assignments: List[List[Tuple[int, int]]], suitability_matrix: List[List[float]]) -> Tuple[List[float], List[int]]:
    """
    Uses majority judgment to rank assignments based on robot suitability for assigned tasks.
    
    Parameters:
        assignments: A list of assignments, where each assignment is a list of (robot, task) pairs.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
    
    Returns:
        (assignment_ratings, ranked_assignments): A tuple containing:
      1. A list of median scores for each assignment based on majority judgment.
      2. A list of indices representing the ranking of assignments based on median scores.
    """
    # Define qualitative rating categories mapped to suitability score ranges
    rating_scale = {"Excellent": 3, "Good": 2, "Fair": 1, "Poor": 0}
    rating_thresholds = [(.85, "Excellent"), (.7, "Good"), (.5, "Fair"), (0, "Poor")]

    assignment_ratings = []

    for assignment in assignments:
        ratings = []
        
        for robot, task in assignment[0]:
            # Get the suitability score for the robot-task pair
            suitability_score = suitability_matrix[robot][task]

            # Convert suitability score to qualitative rating
            for threshold, rating in rating_thresholds:
                if suitability_score >= threshold:
                    ratings.append(rating_scale[rating])
                    break
        
        # Calculate the median rating for this assignment
        median_rating = np.median(ratings) if ratings else 0
        assignment_ratings.append(median_rating)

    # Rank assignments based on median ratings
    ranked_assignments = sorted(range(len(assignment_ratings)), key=lambda i: assignment_ratings[i], reverse=True)

    return assignment_ratings, ranked_assignments

def rank_assignments_cumulative_voting(assignments: List[List[Tuple[int, int]]], suitability_matrix: List[List[float]], total_votes: int = 10) -> Tuple[List[float], List[int]]:
    """
    Uses cumulative voting to rank assignments, where each robot distributes a fixed number of votes
    across assignments based on suitability scores.
    
    Parameters:
        assignments: A list of assignments, where each assignment is a list of (robot, task) pairs.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
        total_votes: The total number of votes each robot has to distribute (default is 10).
    
    Returns:
        (cumulative_votes, ranked_assignments): A tuple containing:
      1. A list of cumulative votes for each assignment.
      2. A list of indices representing the ranking of assignments based on cumulative votes.
    """
    num_assignments = len(assignments)
    cumulative_votes = [0] * num_assignments
    if total_votes < num_assignments/2:
        total_votes = num_assignments/2
    
    for robot in range(len(suitability_matrix)):
        # Collect suitability scores for this robot across all assignments
        robot_votes = []
        
        for assignment_index, assignment in enumerate(assignments):
            # Find the task assigned to this robot in the current assignment
            assigned_task = next((task for r, task in assignment[0] if r == robot), None)
            
            # If robot is unassigned in this assignment, give it a suitability score of 0
            suitability_score = suitability_matrix[robot][assigned_task] if assigned_task is not None else 0
            robot_votes.append((assignment_index, suitability_score))
        
        # Sort robot votes based on suitability scores in descending order
        robot_votes.sort(key=lambda x: x[1], reverse=True)

        # Distribute the total votes among assignments proportional to suitability scores
        total_score = sum(score for _, score in robot_votes if score > 0)
        for assignment_index, score in robot_votes:
            if total_score > 0:
                cumulative_votes[assignment_index] += total_votes * (score / total_score)

    # Rank assignments based on cumulative votes (higher score is better)
    ranked_assignments = sorted(range(num_assignments), key=lambda i: cumulative_votes[i], reverse=True)

    return cumulative_votes, ranked_assignments

def rank_assignments_condorcet_method(assignments: List[List[Tuple[int, int]]], suitability_matrix: List[List[float]]) -> Tuple[List[int], List[int]]:
    """
    Uses the Condorcet method to rank assignments by comparing each assignment in a pairwise manner.
    
    Parameters:
        assignments: A list of assignments, where each assignment is a list of (robot, task) pairs.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
    
    Returns:
        (pairwise_wins, ranked_assignments): A tuple containing:
      1. A list of pairwise win counts for each assignment.
      2. A list of indices representing the ranking of assignments based on pairwise win counts.
    """
    num_assignments = len(assignments)
    num_robots = len(suitability_matrix)

    # Initialize a matrix to track pairwise wins for each assignment comparison
    pairwise_wins = [0] * num_assignments

    # Compare each pair of assignments
    for i in range(num_assignments):
        for j in range(i + 1, num_assignments):
            # Count the number of robots that prefer assignment i over assignment j and vice versa
            i_wins, j_wins = 0, 0
            
            for robot in range(num_robots):
                # Find the task assigned to the robot in both assignments
                task_i = next((task for r, task in assignments[i][0] if r == robot), None)
                task_j = next((task for r, task in assignments[j][0] if r == robot), None)
                
                # Get suitability scores (or 0 if unassigned)
                suitability_i = suitability_matrix[robot][task_i] if task_i is not None else 0
                suitability_j = suitability_matrix[robot][task_j] if task_j is not None else 0

                # Determine which assignment the robot prefers
                if suitability_i > suitability_j:
                    i_wins += 1
                elif suitability_j > suitability_i:
                    j_wins += 1

            # Update pairwise win counts based on robot preferences
            if i_wins > j_wins:
                pairwise_wins[i] += 1
            elif j_wins > i_wins:
                pairwise_wins[j] += 1

    # Rank assignments based on pairwise win counts
    ranked_assignments = sorted(range(num_assignments), key=lambda k: pairwise_wins[k], reverse=True)

    return pairwise_wins, ranked_assignments

def assign_tasks_with_voting(robots: List[CapabilityProfile], tasks: List[TaskDescription], suitability_matrix: np.ndarray, num_candidates: int, voting_method: Callable) -> Tuple[Tuple[List[Tuple[int, int]], List[int], List[int]], float, float]:
    """
    Assigns tasks to robots using random assignment and ranks the assignments using the specified voting method.
    
    Parameters:
        robots: List of robot profiles.
        tasks: List of task descriptions.
        suitability_matrix: A 2D numpy array with suitability scores for each robot-task pair.
        num_candidates: Number of candidate assignments to generate.
        voting_method: The name of the voting function.
    
    Returns:
        (best_assignment, best_score, length): The best assignment, its suitability score, and the time taken for the voting process.
    """
    num_robots = len(robots)
    num_tasks = len(tasks)
    
    random_assignments = generate_random_assignments(num_robots, num_tasks, num_candidates)
    # def map_with_jv(S_mat):
    #     pairs = jv_task_allocation(S_mat)  # adjust to match your JV return
    #     return pairs

    # candidate_assignments = generate_candidates_perturb_and_map(
    #     S=suitability_matrix,
    #     K=num_candidates,
    #     solve_fn=map_with_jv,
    #     noise="gumbel",
    #     scale=0.10,
    #     anneal=True,
    #     seed=None
    # )
    
    start = time.perf_counter_ns()
    total_scores, assignment_ranking = voting_method(random_assignments, suitability_matrix)
    end = time.perf_counter_ns()
    length = (end - start) / 1000.0

    best_ranking = 0
    while(check_zero_suitability(random_assignments[assignment_ranking[best_ranking]][0], suitability_matrix) and best_ranking < len(assignment_ranking)-1):
        best_ranking += 1
    if best_ranking == num_candidates-1:
        best_ranking = 0

    best_assignment = random_assignments[assignment_ranking[best_ranking]]
    filtered_best_assignments = ([],[],[])
    for robot_id, task_id in best_assignment[0]:
            if suitability_matrix[robot_id][task_id] == 0:
                filtered_best_assignments[1].append(robot_id)
                filtered_best_assignments[2].append(task_id)
            else:
                filtered_best_assignments[0].append((robot_id, task_id))
    print(f"Best assignment in voting {filtered_best_assignments}")

    best_score = calculate_total_suitability(filtered_best_assignments[0], suitability_matrix)

    return filtered_best_assignments, best_score, length

def reassign_robots_to_tasks(
        robots: List[CapabilityProfile], 
        tasks: List[TaskDescription], 
        num_candidates: int, 
        voting_method: Callable, 
        suitability_method: Callable, 
        unassigned_robots: List[str], 
        unassigned_tasks: List[str], 
        start_positions: dict, 
        goal_positions: dict,
        inertia_threshold: float = 0.1):
    """
    Reassigns unassigned robots to unassigned tasks using a voting method.
    Parameters:
        robots: List of all robot profiles.
        tasks: List of all task descriptions.
        num_candidates: Number of candidate assignments to generate.
        voting_method: The name of the voting function to use for ranking assignments.
        suitability_method: The name of the suitability evaluation function.
        unassigned_robots: List of unassigned robot IDs.
        unassigned_tasks: List of unassigned task IDs.
        start_positions: Dictionary mapping robot IDs to their start positions.
        goal_positions: Dictionary mapping robot IDs to their goal positions.
        Inertia threshold: minimum improvement in suitability required to steal an already‐assigned task.

    Returns:
        return_assignments: Dictionary mapping robot IDs to assigned task IDs.
        unassigned_robots: List of unassigned robot IDs after reassignment.
        unassigned_tasks: List of unassigned task IDs after reassignment.
        score: Total suitability score of the best assignment.
        length: Time taken for the voting process in microseconds.
    """
    urobots = [robot for robot in robots if not robot.assigned]
    utasks = [task for task in tasks if not task.assigned]

    suitability_matrix = calculate_suitability_matrix(urobots, utasks, suitability_method)
    output, score, length = assign_tasks_with_voting(
        urobots, utasks, suitability_matrix, num_candidates, voting_method)

    # this assigned_pairs only contains the unassigned robots and tasks, I may have to pass in the actual assigned_pairs to update it
    assigned_pairs = output[0] # list of tuples
    return_assignments = {}
    unassigned_robots = [urobots[i].robot_id for i in output[1]]
    unassigned_tasks = [utasks[j].task_id for j in output[2]]
    for (robot_idx, task_idx) in assigned_pairs:
        # NOTE: THIS IS RIGHT, indexes into the filtered list of assigned pairs with the unassigned robots and tasks
        # print(f"UROBOT: {urobots[robot_idx].robot_id} | UTASK: {utasks[task_idx].task_id}")
        r = urobots[robot_idx]
        t = utasks[task_idx]
        r.current_task = t
        r.tasks_attempted += 1
        t.assigned_robot = r
        r.assigned = True
        t.assigned = True
        # update start and goal positions when robots are assigned
        start_positions[r.robot_id] = r.location
        goal_positions[r.robot_id] = t.location
        return_assignments[r.robot_id] = t.task_id
        # return_assignments.update({urobots[robot_idx].robot_id : utasks[task_idx].task_id})

    # Check for stealing tasks from already assigned robots
    free_robots = [r for r in robots if not r.assigned]
    for task in tasks:
        current = task.assigned_robot
        if current is None:
            continue
        current_suitability = suitability_method(current, task)
        print(f"Better suitability in reassigning: {current_suitability}")
        # find the best free robot for this task
        best, best_suit = None, current_suitability
        for r in free_robots:
            s = suitability_method(r, task)
            if s > best_suit:
                print(f"Better suitability in reassigning: {s}")
                best, best_suit = r, s
        # Inertia check: if the best free robot's suitability is not significantly better, skip stealing
        if best and (best_suit - current_suitability) >= inertia_threshold:
            # unassign the current robot from the task
            current.current_task = None
            current.assigned = False
            if current.robot_id not in unassigned_robots:
                unassigned_robots.append(current.robot_id)
            # update the task's assigned robot
            best.current_task = task
            best.assigned = True
            best.tasks_attempted += 1
            task.assigned_robot = best
            start_positions[best.robot_id] = best.location
            goal_positions[best.robot_id] = task.location
            # remove from free list and unassigned robots
            free_robots.remove(best)
            if best.robot_id in unassigned_robots:
                unassigned_robots.remove(best.robot_id)
    
    print(f"Reassign Score: {score}, Reassign Length: {length}")

    return return_assignments, unassigned_robots, unassigned_tasks, score, length