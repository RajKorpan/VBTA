import time, numpy as np
from scipy.optimize import linear_sum_assignment
from .base import IAllocator

class JVAllocator:
    def assign(self, suitability_matrix: np.ndarray) -> tuple:
        start_time = time.perf_counter_ns()

        max_val = np.max(suitability_matrix)
    
        cost_matrix = max_val - suitability_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignment = [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if suitability_matrix[r, c] > 0]
        
        stop_time = time.perf_counter_ns()
        elapsed_time = stop_time - start_time

        return assignment, elapsed_time