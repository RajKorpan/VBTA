import time, numpy as np
from typing import List, Tuple
from .base import IAllocator, Assignment
from .assignments import generate_random_assignments

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

class VotingAllocator(IAllocator):
    def __init__(self, method: str = "range", k: int = 50, seed: int = None, threshold: float = 0.5, total_votes: int = 10):
        self.method = method
        self.k = k
        self.seed = seed
        self.threshold = threshold
        self.total_votes = total_votes
    
    def assign(self, suitability_matrix: np.ndarray) -> Tuple[Assignment, float, float]:
        start = time.perf_counter_ns()
        candidates = generate_random_assignments(suitability_matrix, self.k, self.seed)
        totals = [float(sum(suitability_matrix[i, j] for i, j in assignment)) for assignment in candidates]

        # call the appropriate voting method
        if self.method == "range":
            score, ranked_assignments = rank_assignments_range(candidates, suitability_matrix)
        elif self.method == "borda":
            score, ranked_assignments = rank_assignments_borda(candidates, suitability_matrix)
        elif self.method == "approval":
            score, ranked_assignments = rank_assignments_approval(candidates, suitability_matrix, self.threshold)
        elif self.method == "majority_judgment":
            score, ranked_assignments = rank_assignments_majority_judgment(candidates, suitability_matrix)
        elif self.method == "cumulative":
            score, ranked_assignments = rank_assignments_cumulative_voting(candidates, suitability_matrix, self.total_votes)
        elif self.method == "condorcet":
            score, ranked_assignments = rank_assignments_condorcet_method(candidates, suitability_matrix)
        else:
            raise ValueError(f"Unknown voting method: {self.method}")

        end = time.perf_counter_ns()
        elapsed = end - start

        # Select the best assignment based on the chosen voting method
        best_assignment = ranked_assignments[0] if ranked_assignments else None
        return best_assignment, score, elapsed