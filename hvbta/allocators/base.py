from typing import Protocol, Tuple, List
import numpy as np

Assignment = List[Tuple[int, int]]  # List of (robot_index, task_index) pairs

class IAllocator(Protocol):
    def assign(self, suitability_matrix: np.ndarray) -> Tuple[Assignment, float, float]:
        """
        Solve an assignment on suitability matrix (shape: num_robots x num_tasks)
        
        Returns:
            pairs: list of (robot_index, task_index) pairs
            score: total suitability score of the assignment
            time_taken: time taken to compute the assignment
        """
        ...