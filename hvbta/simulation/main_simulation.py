import psutil
import time
import random
import numpy as np
from typing import List
import copy
from hvbta.pathfinding.Final_CBS import CBS, Environment
from hvbta.simulation.timestep import simulate_time_step
from hvbta.allocators.voting import assign_tasks_with_voting, reassign_robots_to_tasks, rank_assignments_range
from hvbta.suitability import calculate_suitability_matrix, evaluate_suitability_new
from hvbta.pathfinding.CBS import load_map, create_obstacle_list, build_cbs_agents
from hvbta.allocators.misc_assignment import add_new_tasks_strict, add_new_robots_strict, remove_random_robots
from hvbta.generation import generate_random_robot_profile_strict, generate_random_task_description_strict
from hvbta.models import CapabilityProfile, TaskDescription
import hvbta.allocators.optimizers as O

def suitability_all_zero(suitability_matrix):
    return all(value == 0 for row in suitability_matrix for value in row)

def state_check(robots: List[CapabilityProfile]):
    """
    Returns a state of the robots for deciding whether to re-plan
    Ignores anything that will cause constant replanning
    Includes who is planned and to which goals
    """
    active = []
    goals = []
    for r in robots:
        if r.assigned and r.current_task:
            active.append(r.robot_id)
            goals.append((r.robot_id, tuple(r.current_task.location), r.current_task.task_id))
    active_signature = tuple(sorted(active))
    goals_signature = tuple(sorted(goals))
    return active_signature, goals_signature

def main_simulation(output: tuple[list[tuple[int, int]],list[int],list[int]], robots: List[CapabilityProfile], tasks: List[TaskDescription], num_candidates: int, voting_method: str, grid: List[List[int]], map_dict: dict, suitability_method: str, suitability_matrix: np.ndarray, max_time_steps: int, add_tasks: bool, add_robots: bool, remove_robots: bool, tasks_to_add: int = 1, robots_to_add: int = 1, robots_to_remove: int = 1):
    print(f"SUITABILITY METHOD: {suitability_method}")
    
    initial_positions = set()
    robot_max_id = len(robots)+1
    task_max_id = len(tasks)+1
    total_reward = 0.0
    total_success = 0.0
    total_tasks = len(tasks)
    total_reassignment_time = 0.0
    total_reassignment_score = 0.0
    total_reassignments = 0

    for r in robots:
        # add the robot's initial position to the occupied positions set
        initial_positions.add(r.location)

    occupied_positions = set(initial_positions) # use the occcupied positions as the current positions for CBS, this is just as a occupation check, not a start position

    assigned_pairs = output[0]
    for robot_idx, task_idx in assigned_pairs:
        r = robots[robot_idx]
        t = tasks[task_idx]

        r.current_task = t
        r.assigned = True
        r.tasks_attempted = 1

        t.assigned_robot = r
        t.assigned = True


    assigned_robots = {r.robot_id: r.current_task.task_id for r in
                       robots if r.assigned and r.current_task is not None}
    unassigned_tasks = [t.task_id for t in tasks if not t.assigned]
    unassigned_robots = [r.robot_id for r in robots if not r.assigned]

    start_positions = {
        r.robot_id: r.location
        for r in robots
        if r.assigned and r.current_task is not None
    }
    goal_positions = {
        r.robot_id: r.current_task.location
        for r in robots
        if r.assigned and r.current_task is not None
    }

    print(f"ROBOTS: {[rob.robot_id for rob in robots]}")
    print(f"TASKS: {[tas.task_id for tas in tasks]}")
    print(f"ASSIGNED PAIRS: {assigned_pairs}")
    print(f"ASSIGNED ROBOTS: {assigned_robots}")
    print(f"UNASSIGNED ROBOTS: {unassigned_robots}")
    print(f"UNASSIGNED TASKS: {unassigned_tasks}")

    agents = build_cbs_agents(robots, start_positions, goal_positions)

    print(f"AGENTS LIST: {agents}")

    # Create the input data dictionary for CBS, this will be passed to the CBS planner
    input_data = {
        'map' : {
            'dimension': map_dict['dimension'],
            'obstacles': map_dict['obstacles']
        },
        'agents': agents,
    }

    env = Environment(
        dimension=map_dict['dimension'],
        agents=input_data['agents'],
        obstacles=map_dict['obstacles'],
    )
    planner = CBS(env)
    solution, nodes_expanded, conflicts = planner.search()

    print(f"SOLUTION: {solution}")

    if solution is None:
        print("CBS could not find a conflict free path assignment for all agents")
        # could possibly fall back on the simple method here if we get a lot of issues
    else:

        # Iterate through the agents and their schedules
        id_to_index = {r.robot_id: idx for idx, r in enumerate(robots)}
        for robot_id, schedule in solution.items():
            ridx = id_to_index[robot_id]
            robots[ridx].current_path = [(p['x'], p['y']) for p in schedule]
            robots[ridx].remaining_distance = max(0, len(schedule) - 1)
            print(f"Robot {robot_id} path: {robots[ridx].current_path}")

    # Keep track of previous state to not run CBS when there are no changes
    previous_active, previous_goals = state_check(robots)
    current_active, current_goals = state_check(robots)
 
    # End the simulation if nothing changes for 3 timesteps and CBS stalling (hopefully the tasks are done and the robots arent moving)
    time_steps_unchanged = 0

    events = {
        "new_tasks": 0,
        "new_robots": 0,
        "completed_tasks": 0,
    }
    idle_steps = {r.robot_id: 0 for r in robots} # track idleness of free robots

    for time_step in range(max_time_steps):
        # print(f"\n--- Time Step {time_step + 1} ---")
        # print(f"OCCUPIED POSITIONS: {occupied_positions}")
        print(f"AMOUNT OF ASSIGNED ROBOTS: {len([rob.current_task for rob in robots])}")
        print(f"ASSIGNED ROBOTS: {[rob.assigned for rob in robots].count(True)}")
        print(f"START POSITIONS: {start_positions}")
        print(f"GOAL POSITONS: {goal_positions}")
        print(f"UNASSIGNED ROBOTS: {unassigned_robots}")
        print(f"UNASSIGNED TASKS: {unassigned_tasks}")
        # print(f"ALL ROBOTS: {len(robots)}")
        # print(f"ALL TASKS: {len(tasks)}")
        print(f"LIST OF ALL ROBOTS: {[rob.robot_id for rob in robots]}")
        print(f"LIST OF ALL TASKS: {[tas.task_id for tas in tasks]}")

        # before each time step, refresh the unassigned robots and tasks lists
        unassigned_robots = [r.robot_id for r in robots if not r.assigned]
        unassigned_tasks = [t.task_id for t in tasks if not t.assigned]

        # Simulate time step
        completed_this_step, unassigned_count, total_reward, total_success = simulate_time_step(
            robots, tasks, unassigned_robots, unassigned_tasks,
            suitability_method, occupied_positions, start_positions, 
            goal_positions, 1.0, total_reward, total_success
        )
        if len(tasks) == 0:
            print(f"All tasks completed in {time_step} time steps!")
            break
        events["completed_tasks"] += completed_this_step # track number of tasks completed
        should_replan_cbs = completed_this_step > 0

        # Periodically add new tasks and robots
        if add_tasks and time_step + 1 <= 2 and random.random() < 0.5: # add tasks only in the first 2 time steps, and randomly
            print(f"ADDING NEW TASKS AT TIME STEP {time_step + 1}")
            num_of_tasks_added = random.randint(0, tasks_to_add)
            task_max_id, total_tasks = add_new_tasks_strict(
                tasks, unassigned_tasks, task_max_id, num_of_tasks_added, total_tasks, grid, occupied_positions
            )
            events["new_tasks"] += num_of_tasks_added # track number of tasks added

        if add_robots and time_step + 1 <= 4 and random.random() < 0.5: # add robots only in the first 4 time steps, and randomly
            print(f"ADDING NEW ROBOTS AT TIME STEP {time_step + 1}")
            num_of_robots_added = random.randint(0, robots_to_add)
            robot_max_id = add_new_robots_strict(
                robots, unassigned_robots, robot_max_id, num_of_robots_added, grid, occupied_positions
            )
            events["new_robots"] += num_of_robots_added # track number of robots added
            for r in robots:
                if r.robot_id not in idle_steps:
                    idle_steps[r.robot_id] = 0

        # Periodically remove robots
        if remove_robots and time_step + 1 <= 4 and random.random() < 0.5: # remove robots only in the first 4 time steps, and randomly
            if len(assigned_robots) > 1: # Otherwise will break CBS, we need at least one agent for things to run smoothly
                print(f"REMOVING RANDOM ROBOTS AT TIME STEP {time_step + 1}")
                remove_random_robots(robots, tasks, unassigned_robots, unassigned_tasks, random.randint(0, robots_to_remove), occupied_positions, start_positions, goal_positions)

        for r in robots:
            if not r.assigned:
                idle_steps[r.robot_id] = idle_steps.get(r.robot_id, 0) + 1 # update idle steps of unassigned robots
            else:
                idle_steps[r.robot_id] = 0
        # stalling_robot = any(value >= 5 for value in idle_steps.values()) # if any robot has been idle for 5 or more steps, consider it stalling
        
        # Update start and goal positions before cbs
        for robot in robots:
            if robot.assigned and robot.current_task:
                start_positions[robot.robot_id] = robot.location
                goal_positions[robot.robot_id] = robot.current_task.location
            else:
                start_positions.pop(robot.robot_id, None)
                goal_positions.pop(robot.robot_id, None)

        # update planning signatures
        current_active, current_goals = state_check(robots)

        # Update assigned robots
        assigned_robots = {r.robot_id: r.current_task.task_id for r in robots if r.assigned and r.current_task}

        # decide to reassign and replan
        should_replan = False
        if unassigned_robots and unassigned_tasks:
            if events["new_tasks"] or events["new_robots"] or events["completed_tasks"]:
                should_replan = True
            elif (current_active != previous_active) or (current_goals != previous_goals):
                should_replan = True
            # elif stalling_robot:
            #     should_replan = True

        if should_replan:
            should_replan_cbs = True

        # Reassign unassigned robots to unassigned tasks
        if should_replan and start_positions and goal_positions:
            print("State change, re-run CBS...")
            total_reassignments += 1
            _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length = reassign_robots_to_tasks(
                robots, tasks, num_candidates, voting_method, suitability_method,
                unassigned_robots, unassigned_tasks, start_positions, goal_positions
            )
            total_reassignment_time  += reassign_length
            total_reassignment_score += reassign_score

            # rebuild starts/goals after potential changes from reassignment
            for robot in robots:
                if robot.assigned and robot.current_task:
                    start_positions[robot.robot_id] = tuple(robot.location)
                    goal_positions[robot.robot_id]  = tuple(robot.current_task.location)
                else:
                    start_positions.pop(robot.robot_id, None)
                    goal_positions.pop(robot.robot_id, None)
            print("***************************AFTER REASSIGNMENT***************************")
            print(f"AMOUNT OF ASSIGNED ROBOTS: {len([rob.current_task for rob in robots])}")
            print(f"ASSIGNED ROBOTS: {[rob.assigned for rob in robots].count(True)}")
            print(f"START POSITIONS: {start_positions}")
            print(f"GOAL POSITONS: {goal_positions}")
            print(f"UNASSIGNED ROBOTS: {unassigned_robots}")
            print(f"UNASSIGNED TASKS: {unassigned_tasks}")
            print(f"LIST OF ALL ROBOTS: {[rob.robot_id for rob in robots]}")
            print(f"LIST OF ALL TASKS: {[tas.task_id for tas in tasks]}")
        
        if should_replan_cbs and start_positions and goal_positions:
            agents = build_cbs_agents(robots, start_positions, goal_positions)

            # duplicate-start validation
            start_locations = [a['start'] for a in agents]
            if len(start_locations) != len(set(start_locations)):
                print("ERROR: Duplicate start locations found in agent list. Aborting CBS.")
                solution = None
            else:
                env = Environment(dimension=map_dict['dimension'], agents=agents, obstacles=map_dict['obstacles'])
                planner = CBS(env)
                solution, nodes_expanded, conflicts = planner.search()
                print(f"CBS COMPLETE. New solution: {solution}")

                if solution:
                    id_to_index = {r.robot_id: idx for idx, r in enumerate(robots)}
                    for robot_id, schedule in solution.items():
                        ridx = id_to_index[robot_id]
                        r = robots[ridx]
                        r.current_path = [(p['x'], p['y']) for p in schedule]
                        r.remaining_distance = max(0, len(schedule) - 1)

                    previous_active, previous_goals = state_check(robots)  # update to the post-replan state
                    events = {k: 0 for k in events}  # reset counters we just consumed
                else:
                    print("CBS could not find a conflict free path assignment for all agents")
                    # could possibly fall back on the simple method here if we get a lot of issues
                    # but for now, we will just skip CBS and continue with the simulation
                    print("Skipping CBS...")
                    time_steps_unchanged += 1
                    if time_steps_unchanged >= 3:
                        print("No state change for 3 time steps, ending simulation.")
                        break
        else:
            print("No state change, skip CBS...")
            print(f"ALL ROBOTS: {len(robots)}")
            print(f"ALL TASKS: {len(tasks)}")
            print(f"LIST OF ALL ROBOTS: {[rob.robot_id for rob in robots]}")
            print(f"LIST OF ALL TASKS: {[tas.task_id for tas in tasks]}")

    overall_success_rate = total_success / total_tasks
    print(f"Voting: Total reward: {total_reward}, Overall success rate: {overall_success_rate:.2%}, Tasks completed: {total_success}, Reassignment Time: {total_reassignment_time}, Reassignment Score: {total_reassignment_score}, total reassignments: {total_reassignments}")
#     for robot in robots:
#         print(f"Robot {robot.robot_id} attempted {robot.tasks_attempted} tasks and successfully completed {robot.tasks_successful} of them.")

def benchmark_simulation(output: tuple[list[tuple[int, int]],list[int],list[int]], robots: List[CapabilityProfile], tasks: List[TaskDescription], num_candidates: int, voting_method: str, grid: List[List[int]], map_dict: dict, suitability_method: str, suitability_matrix: np.ndarray, max_time_steps: int, add_tasks: bool, add_robots: bool, remove_robots: bool, tasks_to_add: int = 1, robots_to_add: int = 1, robots_to_remove: int = 1):
    start_time = time.time()
    main_simulation(output, robots, tasks, num_candidates, voting_method, grid, map_dict, suitability_method, suitability_matrix, max_time_steps, add_tasks, add_robots, remove_robots, tasks_to_add, robots_to_add, robots_to_remove)
    end_time = time.time()
    execution_time = end_time - start_time

    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().used

    print(f"Simulation completed in {execution_time:.2f} seconds.")
    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_usage / (1024 * 1024)} MB")

if __name__ == "__main__":
#     voting_methods = ["rank_assignments_borda", "rank_assignments_approval", "rank_assignments_majority_judgment", "rank_assignments_cumulative_voting", "rank_assignments_condorcet_method", "rank_assignments_range"]
    voting_methods = [rank_assignments_range]
#     suitability_methods = ["evaluate_suitability_loose", "evaluate_suitability_strict", "evaluate_suitability_distance", "evaluate_suitability_priority"]
    # suitability_methods = ["evaluate_suitability_loose","evaluate_suitability_distance"]
    suitability_methods = [evaluate_suitability_new]
    # suitability_methods = ["evaluate_suitability_distance"]
    max_time_steps = 100
    add_tasks = False
    add_robots = False
    remove_robots = False

    map_file = r"test_small_open.map"
    grid = load_map(map_file) # 2D list of 0/1 representing the map
    dims = (len(grid), len(grid[0])) # dimensions of the map grid
    obstacles = create_obstacle_list(grid) # list of obstacle coordinates
    # Create the map dictionary to pass to the CBS planner
    map_dict = {
        'dimension': dims,
        'obstacles': obstacles
    }

    # print(f"MAP DICTIONARY: {map_dict}")
    for i in [5]:
        for j in [5]:
            for nc in [5]:
                for vm in voting_methods:
                    for sm in suitability_methods:
                        for k in range(0,10):
                            # get initial positions list to reuse later
                            # generate_random_robot_profile now takes in the map grid and occupied positions and passes them to get_random_free_position function
                            robots = [generate_random_robot_profile_strict(f"R{i+1}", grid, set()) for i in range(i)]
                            # changed how tasks are made here so there are no overlapping tasks
                            tasks = [generate_random_task_description_strict(f"T{i+1}", grid, set(), []) for i in range(j)]

                            suitability_matrix = calculate_suitability_matrix(robots, tasks, sm)

                            while suitability_all_zero(suitability_matrix):
                                robots = [generate_random_robot_profile_strict(f"R{i+1}", grid, set()) for i in range(i)]
                                tasks = [generate_random_task_description_strict(f"T{i+1}", grid, set(), []) for i in range(j)]
                                suitability_matrix = calculate_suitability_matrix(robots, tasks, sm)

                            output, score, length = assign_tasks_with_voting(robots, tasks, suitability_matrix, nc, vm)
                            cbba_output, cbba_score, cbba_length = O.assign_tasks_with_method(O.cbba_task_allocation,suitability_matrix)
                            ssia_output, ssia_score, ssia_length = O.assign_tasks_with_method(O.ssia_task_allocation,suitability_matrix)
                            ilp_output, ilp_score, ilp_length = O.assign_tasks_with_method(O.ilp_task_allocation,suitability_matrix)
                            jv_output, jv_score, jv_length = O.assign_tasks_with_method(O.jv_task_allocation,suitability_matrix)
                            outputs = [output, cbba_output, ssia_output, ilp_output, jv_output]
                                # param_combinations = []
                                # param_combinations.append((i, j, nc, vm, sm, max_time_steps, add_tasks, add_robots, remove_robots, 10, 10, 10))
                                # with multiprocessing.Pool() as pool:
                                # pool.starmap(main_simulation, param_combinations)
                                # main_simulation(i, j, nc, vm, sm, max_time_steps, add_tasks, add_robots, remove_robots,10,10,10)
                            for o in outputs:
                                benchmark_simulation(o, copy.deepcopy(robots), copy.deepcopy(tasks), nc, vm, grid, map_dict, sm, suitability_matrix, max_time_steps, add_tasks, add_robots, remove_robots,10,10,10)