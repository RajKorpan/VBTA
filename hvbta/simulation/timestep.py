import random
from typing import List, Tuple
from hvbta.models import CapabilityProfile, TaskDescription
from hvbta.allocators.misc_assignment import unassign_task_from_robot

def simulate_time_step(
    robots: List[CapabilityProfile],
    tasks: List[TaskDescription],
    unassigned_robots: List[str],
    unassigned_tasks: List[str],
    suitability_method: str,
    occupied_locations: set,
    start_positions: dict,
    goal_positions: dict,
    time_step: float = 1.0,
    total_reward: float = 0.0,
    total_success: int = 0
) -> Tuple[int, int, float, int]:
    """
    Simulates a single time step, updating robot positions, task progress, and handling failures.

    Parameters:
        robots: List of all robots.
        tasks: List of all tasks.
        time_step: The time increment for the simulation step.
        total_reward: Accumulated reward from successfully completed tasks.

    Returns:
        (tasks_completed, count, total_reward, total_success): A count of unassigned robots and the updated total reward.
    """
    unassigned_count = 0  # Count of unassigned robots
    tasks_completed = 0  # Count of tasks completed in this time step

    # Iterate through all robots to update their positions and tasks
    for robot in robots:
        if robot.assigned and robot.current_task is not None: # Check that all assigned robots have a task
            # Get the assigned task
            task = robot.current_task

            # check if there is more path to traverse for the robot
            if robot.current_path and len(robot.current_path) > 1:
                next_position = robot.current_path[1] # gives (x, y) coordinate of next step in path
                occupied_locations.discard(robot.location) # remove current location from occupied set
                robot.location = next_position # update location
                # start position for this robot should be replaced, if not then must index by ID
                start_positions[robot.robot_id] = next_position
                occupied_locations.add(next_position) # update occupied set with robots new current location
                robot.current_path.pop(0) # Move the robot one space by removing the first element from the robots path
                robot.remaining_distance = max(0, len(robot.current_path) - 1) # recalculate the remaining distance and new location

                if robot.remaining_distance <= 1:
                    # The robot has reached the task
                    robot.remaining_distance = 0
                    # this wont work here because once the robot reaches the task it will just keep resetting the time on task, I put it down below after completing the task
                    # robot.time_on_task = 0  # Reset time on task so it can begin work (time on task is a counter for how long it takes to complete a task)


                if robot.battery_life == 0: 
                    # print(f"Robot {robot.robot_id} failed to reach task {task.task_id} due to mechanical failure.")
                    # get task id and task index
                    unassign_task_from_robot(robot, tasks=tasks, unassigned_robots=unassigned_robots, unassigned_tasks=unassigned_tasks)
                    continue

            # If the robot is at the task, increment time on task
            if robot.remaining_distance == 0:
                #robot.time_on_task += time_step
                robot.location = task.location  # Ensure robot is at the task location
                start_positions[robot.robot_id] = robot.location  # Update start position to task location
                robot.battery_life -= time_step
                task.time_left -= time_step
                # suitability = globals()[suitability_method](robot, task)
                # failure_probability = 1 / (100 * (suitability + 1))  # Higher suitability, lower failure rate
                # Check if the task is completed
                if task.time_left <= 0:
                    # Mark task as completed
                    total_reward += task.reward
                    total_success += 1
                    task.assigned = False
                    task.assigned_robot = None
                    robot.current_task = None
                    robot.tasks_successful += 1
                    robot.assigned = False
                    robot.time_on_task = 0  # Reset time on task so it can begin work (time on task is a counter for how long it takes to complete a task)
                    robot.current_path = []
                    robot.remaining_distance = 0
                    # move it to unassigned robots list with check
                    rid = robot.robot_id
                    if rid not in unassigned_robots:
                        unassigned_robots.append(rid)
                    print(f"ROBOT {robot.robot_id} COMPLETED TASK {task.task_id}")
                    tasks_completed += 1
                    try:
                        tasks.remove(task)
                    except ValueError:
                        pass
                elif robot.battery_life <= 0:
                    if getattr(task, "reset_progress", False): # Only resets progress for certain tasks
                        task.time_left = task.time_to_complete
                    else:
                        task.time_to_complete = task.time_left
                    unassign_task_from_robot(
                        robot, tasks=tasks, 
                        unassigned_robots=unassigned_robots, 
                        unassigned_tasks=unassigned_tasks
                    )

                elif task.performance_metrics == "safety compliance":
                    if robot.safety_features:
                        matched_safety = sum(safety in robot.safety_features for safety in task.safety_protocols)
                        safety_score = matched_safety/max(1, len(task.safety_protocols))
                        if safety_score < 0.75 and (random.random() > 0.75):
                            unassign_task_from_robot(
                                robot, tasks=tasks, 
                                unassigned_robots=unassigned_robots, 
                                unassigned_tasks=unassigned_tasks
                            )
                    else:
                        unassign_task_from_robot(
                            robot, tasks=tasks, 
                            unassigned_robots=unassigned_robots, 
                            unassigned_tasks=unassigned_tasks
                        )
        elif not robot.assigned:
            unassigned_count += 1
            
    return tasks_completed, unassigned_count, total_reward, total_success