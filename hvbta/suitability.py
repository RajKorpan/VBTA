import numpy as np
from typing import List, Tuple, Callable
from openai import OpenAI
from .models import CapabilityProfile, TaskDescription

ScoreFn = Callable[[CapabilityProfile, TaskDescription], float]

def suitability_all_zero(suitability_matrix):
    return all(value == 0 for row in suitability_matrix for value in row)

def navigation_suitability(robot_mobility_type: str, robot_size: Tuple[float, float, float], task_constraints: List[str]) -> float:
    """
    Evaluates the suitability of a robot for navigating a task environment based on mobility type, size, and navigation constraints.
    
    Parameters:
        robot_mobility_type: The mobility type of the robot (e.g., "wheeled", "tracked", "legged", "aerial", "hovering", "climbing").
        robot_size: A tuple representing the robot's dimensions (length, width, height).
        task_constraints: A list of navigation constraints for the task environment.
    
    Returns:
        A float score representing the suitability for navigation. Returns 0 if there is a critical mismatch that prevents navigation.
    """

    # Initialize the score
    score = 0.0

    # Define size thresholds for narrow spaces, low ceilings, etc.
    narrow_space_threshold = 2.0  # Width limit for narrow spaces
    low_ceiling_threshold = 2.0   # Height limit for low ceilings

    # Handle each task constraint based on mobility type
    for constraint in task_constraints:
        # Constraint: Elevator access
        if constraint == "elevator":
            score += 1.0

        # Constraint: Stairs
        elif constraint == "stairs":
            if robot_mobility_type in ["legged", "aerial", "hovering", "climbing"]:
                score += 1.0
            else:
                return 0.0

        # Constraint: Shelves
        elif constraint == "shelves":
            if robot_size[2] < low_ceiling_threshold or robot_mobility_type in ["aerial", "climbing", "hovering"]:
                score += 1.0  # Only smaller robots can access shelves effectively

        # Constraint: No loud noises allowed
        elif constraint == "no loud noises allowed":
            if robot_mobility_type in ["legged", "hovering"]:
                score += 1.0  # Quieter mobility types

        # Constraint: Narrow spaces
        elif constraint == "narrow spaces":
            if robot_size[1] <= narrow_space_threshold:
                score += 1.0
            else:
                return 0.0  # Larger robots cannot pass through narrow spaces

        # Constraint: Low ceilings
        elif constraint == "low ceilings":
            if robot_size[2] <= low_ceiling_threshold:
                score += 1.0
            else:
                return 0.0  # Tall robots cannot navigate in areas with low ceilings

        # Constraint: Uneven floors
        elif constraint == "uneven floors":
            if robot_mobility_type in ["tracked", "legged", "climbing", "hovering", "aerial"]:
                score += 1.0  # These types handle uneven floors well

        # Constraint: Low visibility
        elif constraint == "low visibility":
            if robot_mobility_type in ["wheeled", "tracked", "legged"]:
                score += 1.0  # Infrared or LiDAR-equipped robots are suitable
            else:
                score += 0.5

        # Constraint: Slippery surfaces
        elif constraint == "slippery":
            if robot_mobility_type in ["tracked", "hovering", "aerial"]:
                score += 1.0  # Hovering and tracked types handle slippery surfaces better
            elif robot_mobility_type in ["wheeled", "legged"]:
                return 0.0  # Wheeled and legged robots are unsuitable on slippery floors

        # Constraint: Crowded environments
        elif constraint == "crowded":
            if robot_size[0] <= 1.0 and robot_size[1] <= 1.0:
                score += 1.0  # Smaller robots are more suitable in crowded environments
            else:
                score += 0.5  # Larger robots get a lower score

        # Constraint: Loose debris
        elif constraint == "loose debris":
            if robot_mobility_type in ["aerial", "hovering", "tracked"]:
                score += 1.0  # Aerial and hovering robots handle debris better
            elif robot_mobility_type == "legged":
                return 0.0  # Legged robots are unsuitable
            else:
                score += 0.5

        # Constraint: No-fly zone
        elif constraint == "no-fly zone":
            if robot_mobility_type == "aerial":
                return 0.0  # Aerial robots cannot navigate in no-fly zones
            else:
                score += 0.5  # Other mobility types are unaffected

        # Constraint: Windy conditions
        elif constraint == "windy":
            if robot_mobility_type == "aerial":
                return 0.0  # Aerial robots struggle in windy conditions
            else:
                score += 0.5  # All other types are more stable in wind

        # Constraint: Dense obstructions (e.g., tree branches, hanging cables)
        elif constraint == "dense obstructions":
            if robot_mobility_type in ["aerial", "legged"]:
                return 0.0  # Aerial and legged robots are unsuitable in dense obstruction areas
            else:
                score += 0.5  # Other types may navigate dense areas on the ground

        # Constraint: Smooth floors
        elif constraint == "smooth surfaces":
            if robot_mobility_type == "climbing":
                return 0.0  # Climbing robots are less suited for smooth surfaces
            else:
                score += 0.5  # Other types may navigate dense areas on the ground

    # Final suitability score (0 if any constraint returns 0)
    return score if score > 0 else 0.0

def evaluate_suitability_new(robot: CapabilityProfile, task: TaskDescription) -> float:
    score = 0.0
    total_weight = 0.0  # for normalization

    weights = {
        "payload": 3.0,
        "manipulators": 4.0,
        "sensors": 3.0,
        "communication": 0.5,
        "safety": 1.0,
        "environmental": 1.0,
        "reach": 2.0,
        "sensor_range": 1.0,
        "proximity": 1.0,
        "autonomy_match": 0.5,
        "battery_duration": 2.0,
        "special_functions": 2.0,
        "processing_power": 1.0,
        "adaptability": 0.5,
        "navigation": 2.0,
    }

    # Payload
    total_weight += weights["payload"]
    if any("payload capacity" in req and robot.payload_capacity < float(req.split(">= ")[-1]) for req in task.required_capabilities):
        return 0.0
    else:
        score += weights["payload"]

    # Reach 
    total_weight += weights["reach"]
    if any("reach" in req and robot.reach < float(req.split(">= ")[-1]) for req in task.required_capabilities):
        score += 0.0
    else:
        score += weights["reach"]

    # Manipulators (tools_needed[1] is manipulators list)
    total_weight += weights["manipulators"]
    if task.tools_needed:
        if ("cable hoist" in task.tools_needed[1] and "cable hoist" not in robot.manipulators and robot.mobility_type in ["hovering", "aerial"]):
            task.tools_needed[1].remove("cable hoist")
        matched_tools = sum(tool in robot.manipulators for tool in task.tools_needed[1])
        tool_score = matched_tools / len(task.tools_needed[1])
        if tool_score != 1:
            return 0.0
        score += weights["manipulators"] * tool_score

    # Sensors (tools_needed[0] is sensors list)
    total_weight += weights["sensors"]
    if task.tools_needed:
        matched_tools = sum(tool in robot.sensors for tool in task.tools_needed[0])
        tool_score = matched_tools / len(task.tools_needed[0])
        score += weights["sensors"] * tool_score

    # Communication 
    total_weight += weights["communication"]
    if task.communication_requirements:
        matched_comm = sum(proto in robot.communication_protocols for proto in task.communication_requirements)
        comm_score = matched_comm / len(task.communication_requirements)
        score += weights["communication"] * comm_score

    # Safety 
    total_weight += weights["safety"]
    if robot.safety_features and task.safety_protocols:
        matched_safety = sum(safety in robot.safety_features for safety in task.safety_protocols)
        safety_score = matched_safety / len(task.safety_protocols)
        score += weights["safety"] * safety_score

    # Environmental 
    total_weight += weights["environmental"]
    if robot.environmental_resistance and task.environmental_conditions:
        matched_environmental = sum(condition in robot.environmental_resistance for condition in task.environmental_conditions)
        environmental_score = matched_environmental / len(task.environmental_conditions)
        score += weights["environmental"] * environmental_score

    # Navigation 
    total_weight += weights["navigation"]
    if task.navigation_constraints:
        navigation_score = navigation_suitability(robot.mobility_type, robot.size, task.navigation_constraints)
        if navigation_score == 0:
            return 0.0
        score += weights["navigation"] * navigation_score

    # Sensor range 
    total_weight += weights["sensor_range"]
    distance_to_task = len(robot.current_path) - 1
    sensor_score = 1.0 if robot.sensor_range >= distance_to_task else \
                   0.5 if robot.sensor_range >= distance_to_task / 2 else 0.0
    score += weights["sensor_range"] * sensor_score

    # Proximity 
    total_weight += weights["proximity"]
    if distance_to_task < 20.0:
        score += weights["proximity"]
    elif distance_to_task < 50.0:
        score += weights["proximity"] * 0.5

    # Autonomy 
    total_weight += weights["autonomy_match"]
    autonomy_score = 0.0
    if task.priority_level in ["high", "urgent"] and robot.autonomy_level in ["fully autonomous", "teleoperated"]:
        autonomy_score = 1.0
    elif task.priority_level in ["medium", "low"] and robot.autonomy_level in ["semi-autonomous", "fully autonomous"]:
        autonomy_score = 0.5
    score += weights["autonomy_match"] * autonomy_score

    # Battery 
    total_weight += weights["battery_duration"]
    if ((distance_to_task / robot.max_speed) + task.time_to_complete) > robot.battery_life:
        return 0.0
    battery_score = 1.0 if robot.battery_life >= 2 * ((distance_to_task / robot.max_speed) + task.time_to_complete) else 0.5
    score += weights["battery_duration"] * battery_score

    # Special functions 
    total_weight += weights["special_functions"]
    task_function_mapping = {
        "delivery": ["object recognition", "speech output", "facial recognition"],
        "assembly": ["object recognition", "object tracking", "precise alignment"],
        "utilities": ["percise alignment", "balance control"],
        "excavate": [ "terrain leveling", "object recognition", "precise alignment"],
        "debris": ["balance control", "object recognition"],
        "level": ["terrain leveling", "object recognition"],
        "item elevation": ["precise alignment", "object tracking", "balance control"],
        "lay bricks": ["object recognition", "precise alignment"],
        "scaffold": ["precise alignment", "balance control"],
        "remove scaffold": ["object recognition", "object tracking", "precise alignment"],
    }
    required_functions = task_function_mapping[task.task_type]
    matched_special_functions = sum(special in robot.special_functions for special in required_functions)
    special_functions_score = matched_special_functions / len(required_functions)
    score += weights["special_functions"] * special_functions_score

    # Processing power 
    total_weight += weights["processing_power"]
    proc_score = 0.0
    if task.difficulty > 7:
        if robot.processing_power >= 5.0:
            proc_score = 1.0
        elif robot.processing_power >= 3.0:
            proc_score = 0.75
        else:
            proc_score = 0.5
    elif task.difficulty > 4:
        if robot.processing_power >= 3.0:
            proc_score = 1.0
        else:
            proc_score = 0.75
    elif task.difficulty > 2:
        if robot.processing_power >= 1.5:
            proc_score = 1.0
        else:
            proc_score = 0.75
    score += weights["processing_power"] * proc_score

    # Adaptability 
    total_weight += weights["adaptability"]
    score += weights["adaptability"] * (1.0 if robot.adaptability else 0.0)

    # Reward/Difficulty ratio 
    reward_score = (task.reward / max(task.difficulty, 1.0))
    priority_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5, "urgent": 2.0}[task.priority_level]
    score += priority_multiplier * (reward_score / (reward_score + 10.0))  # squash into (0,1)

    # Normalize 
    if total_weight > 0:
        final_score = score / total_weight
    else:
        final_score = 0.0

    return max(0.0, min(1.0, final_score))

def evaluate_suitability_loose(robot: CapabilityProfile, task: TaskDescription) -> float:
    """
    Evaluates the suitability of a robot for a given task.
    A higher score indicates better suitability.
    
    Parameters:
        robot: The CapabilityProfile of the robot.
        task: The TaskDescription of the task.
    
    Returns:
        A float score representing the suitability of the robot for the task. A score of 0 indicates the robot cannot perform the task.
    """
    score = 0.0
    total_weight = 0.0  # for normalization

    weights = {
        "payload": 3.0,
        "manipulators": 4.0,
        "sensors": 3.0,
        "communication": 0.5,
        "safety": 1.0,
        "environmental": 1.0,
        "reach": 2.0,
        "sensor_range": 1.0,
        "proximity": 1.0,
        "autonomy_match": 0.5,
        "battery_duration": 2.0,
        "special_functions": 2.0,
        "processing_power": 1.0,
        "adaptability": 0.5,
        "navigation": 2.0,
    }

    # Payload
    total_weight += weights["payload"]
    if any("payload capacity" in req and robot.payload_capacity < float(req.split(">= ")[-1]) for req in task.required_capabilities):
        score += 0.0
    else:
        score += weights["payload"]

    # Reach
    total_weight += weights["reach"]
    if any("reach" in req and robot.reach < float(req.split(">= ")[-1]) for req in task.required_capabilities):
        score += 0.0
    else:
        score += weights["reach"]

    # Manipulators (tools_needed[1] is manipulators list)
    total_weight += weights["manipulators"]
    if task.tools_needed:
        if ("cable hoist" in task.tools_needed[1] and "cable hoist" not in robot.manipulators and robot.mobility_type in ["hovering", "aerial"]):
            task.tools_needed[1].remove("cable hoist")
        matched_tools = sum(tool in robot.manipulators for tool in task.tools_needed[1])
        tool_score = matched_tools / len(task.tools_needed[1])
        score += weights["manipulators"] * tool_score

    # Sensors (tools_needed[0] is sensors list)
    total_weight += weights["sensors"]
    if task.tools_needed:
        matched_tools = sum(tool in robot.sensors for tool in task.tools_needed[0])
        tool_score = matched_tools / len(task.tools_needed[0])
        score += weights["sensors"] * tool_score

    # Communication
    total_weight += weights["communication"]
    if task.communication_requirements:
        matched_comm = sum(proto in robot.communication_protocols for proto in task.communication_requirements)
        comm_score = matched_comm / len(task.communication_requirements)
        score += weights["communication"] * comm_score

    # Safety
    total_weight += weights["safety"]
    if robot.safety_features and task.safety_protocols:
        matched_safety = sum(safety in robot.safety_features for safety in task.safety_protocols)
        safety_score = matched_safety / len(task.safety_protocols)
        score += weights["safety"] * safety_score

    # Environmental
    total_weight += weights["environmental"]
    if robot.environmental_resistance and task.environmental_conditions:
        matched_environmental = sum(condition in robot.environmental_resistance for condition in task.environmental_conditions)
        environmental_score = matched_environmental / len(task.environmental_conditions)
        score += weights["environmental"] * environmental_score

    # Navigation
    total_weight += weights["navigation"]
    if task.navigation_constraints:
        navigation_score = navigation_suitability(robot.mobility_type, robot.size, task.navigation_constraints)
        score += weights["navigation"] * navigation_score

    # Sensor range
    total_weight += weights["sensor_range"]
    distance_to_task = len(robot.current_path) - 1
    sensor_score = 1.0 if robot.sensor_range >= distance_to_task else \
                   0.5 if robot.sensor_range >= distance_to_task / 2 else 0.0
    score += weights["sensor_range"] * sensor_score

    # Proximity
    total_weight += weights["proximity"]
    if distance_to_task < 20.0:
        score += weights["proximity"]
    elif distance_to_task < 50.0:
        score += weights["proximity"] * 0.5

    # Autonomy
    total_weight += weights["autonomy_match"]
    autonomy_score = 0.0
    if task.priority_level in ["high", "urgent"] and robot.autonomy_level in ["fully autonomous", "teleoperated"]:
        autonomy_score = 1.0
    elif task.priority_level in ["medium", "low"] and robot.autonomy_level in ["semi-autonomous", "fully autonomous"]:
        autonomy_score = 0.5
    score += weights["autonomy_match"] * autonomy_score

    # Battery
    total_weight += weights["battery_duration"]
    if ((distance_to_task / robot.max_speed) + task.time_to_complete) > robot.battery_life:
        return 0.0
    battery_score = (1.0 if robot.battery_life >= 2 * ((distance_to_task / robot.max_speed) + task.time_to_complete)
                    else 0 if ((distance_to_task / robot.max_speed) + task.time_to_complete) > robot.battery_life
                    else 0.5)
    score += weights["battery_duration"] * battery_score

    # Special functions
    total_weight += weights["special_functions"]
    task_function_mapping = {
        "delivery": ["object recognition", "speech output", "facial recognition"],
        "assembly": ["object recognition", "object tracking", "precise alignment"],
        "utilities": ["percise alignment", "balance control"],
        "excavate": [ "terrain leveling", "object recognition", "precise alignment"],
        "debris": ["balance control", "object recognition"],
        "level": ["terrain leveling", "object recognition"],
        "item elevation": ["precise alignment", "object tracking", "balance control"],
        "lay bricks": ["object recognition", "precise alignment"],
        "scaffold": ["precise alignment", "balance control"],
        "remove scaffold": ["object recognition", "object tracking", "precise alignment"],
    }
    required_functions = task_function_mapping[task.task_type]
    matched_special_functions = sum(special in robot.special_functions for special in required_functions)
    special_functions_score = matched_special_functions / len(required_functions)
    score += weights["special_functions"] * special_functions_score

    # Processing power
    total_weight += weights["processing_power"]
    proc_score = 0.0
    if task.difficulty > 7:
        if robot.processing_power >= 5.0:
            proc_score = 1.0
        elif robot.processing_power >= 3.0:
            proc_score = 0.75
        else:
            proc_score = 0.5
    elif task.difficulty > 4:
        if robot.processing_power >= 3.0:
            proc_score = 1.0
        else:
            proc_score = 0.75
    elif task.difficulty > 2:
        if robot.processing_power >= 1.5:
            proc_score = 1.0
        else:
            proc_score = 0.75
    score += weights["processing_power"] * proc_score

    # Adaptability
    total_weight += weights["adaptability"]
    score += weights["adaptability"] * (1.0 if robot.adaptability else 0.0)

    # Reward/Difficulty ratio
    reward_score = (task.reward / max(task.difficulty, 1.0))
    priority_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5, "urgent": 2.0}[task.priority_level]
    score += priority_multiplier * (reward_score / (reward_score + 10.0))  # squash into (0,1)

    # Normalize
    if total_weight > 0:
        final_score = score / total_weight
    else:
        final_score = 0.0

    return max(0.0, min(1.0, final_score))

def evaluate_suitability_strict(robot: CapabilityProfile, task: TaskDescription) -> float:
    """
    Evaluates the suitability of a robot for a given task.
    A higher score indicates better suitability.
    
    Parameters:
        robot: The CapabilityProfile of the robot.
        task: The TaskDescription of the task.
    
    Returns:
        A float score representing the suitability of the robot for the task. A score of 0 indicates the robot cannot perform the task.
    """
    score = 0.0
    total_weight = 0.0  # for normalization

    weights = {
        "payload": 3.0,
        "manipulators": 4.0,
        "sensors": 3.0,
        "communication": 0.5,
        "safety": 1.0,
        "environmental": 1.0,
        "reach": 2.0,
        "sensor_range": 1.0,
        "proximity": 1.0,
        "autonomy_match": 0.5,
        "battery_duration": 2.0,
        "special_functions": 2.0,
        "processing_power": 1.0,
        "adaptability": 0.5,
        "navigation": 2.0,
    }

    # Payload
    total_weight += weights["payload"]
    if any("payload capacity" in req and robot.payload_capacity < float(req.split(">= ")[-1]) for req in task.required_capabilities):
        return 0.0
    else:
        score += weights["payload"]

    # Reach
    total_weight += weights["reach"]
    if any("reach" in req and robot.reach < float(req.split(">= ")[-1]) for req in task.required_capabilities):
        return 0.0
    else:
        score += weights["reach"]

    # Manipulators (tools_needed[1] is manipulators list)
    total_weight += weights["manipulators"]
    if task.tools_needed:
        if ("cable hoist" in task.tools_needed[1] and "cable hoist" not in robot.manipulators and robot.mobility_type in ["hovering", "aerial"]):
            task.tools_needed[1].remove("cable hoist")
        matched_tools = sum(tool in robot.manipulators for tool in task.tools_needed[1])
        tool_score = matched_tools / len(task.tools_needed[1])
        if tool_score != 1:
            return 0.0
        score += weights["manipulators"] * tool_score

    # Sensors (tools_needed[0] is sensors list)
    total_weight += weights["sensors"]
    if task.tools_needed:
        matched_tools = sum(tool in robot.sensors for tool in task.tools_needed[0])
        tool_score = matched_tools / len(task.tools_needed[0])
        if tool_score != 1:
            return 0.0
        score += weights["sensors"] * tool_score

    # Communication
    total_weight += weights["communication"]
    if task.communication_requirements:
        matched_comm = sum(proto in robot.communication_protocols for proto in task.communication_requirements)
        comm_score = matched_comm / len(task.communication_requirements)
        if comm_score != 1:
            return 0.0
        score += weights["communication"] * comm_score

    # Safety
    total_weight += weights["safety"]
    if robot.safety_features and task.safety_protocols:
        matched_safety = sum(safety in robot.safety_features for safety in task.safety_protocols)
        safety_score = matched_safety / len(task.safety_protocols)
        if safety_score != 1:
            return 0.0
        score += weights["safety"] * safety_score

    # Environmental
    total_weight += weights["environmental"]
    if robot.environmental_resistance and task.environmental_conditions:
        matched_environmental = sum(condition in robot.environmental_resistance for condition in task.environmental_conditions)
        environmental_score = matched_environmental / len(task.environmental_conditions)
        if environmental_score != 1:
            return 0.0
        score += weights["environmental"] * environmental_score

    # Navigation
    total_weight += weights["navigation"]
    if task.navigation_constraints:
        navigation_score = navigation_suitability(robot.mobility_type, robot.size, task.navigation_constraints)
        if navigation_score == 0:
            return 0.0
        score += weights["navigation"] * navigation_score

    # Sensor range
    total_weight += weights["sensor_range"]
    distance_to_task = len(robot.current_path) - 1
    sensor_score = 1.0 if robot.sensor_range >= distance_to_task else \
                   0.5 if robot.sensor_range >= distance_to_task / 2 else 0.0
    score += weights["sensor_range"] * sensor_score

    # Proximity
    total_weight += weights["proximity"]
    if distance_to_task < 20.0:
        score += weights["proximity"]
    elif distance_to_task < 50.0:
        score += weights["proximity"] * 0.5

    # Autonomy
    total_weight += weights["autonomy_match"]
    autonomy_score = 0.0
    if task.priority_level in ["high", "urgent"] and robot.autonomy_level in ["fully autonomous", "teleoperated"]:
        autonomy_score = 1.0
    elif task.priority_level in ["medium", "low"] and robot.autonomy_level in ["semi-autonomous", "fully autonomous"]:
        autonomy_score = 0.5
    score += weights["autonomy_match"] * autonomy_score

    # Battery
    total_weight += weights["battery_duration"]
    if ((distance_to_task / robot.max_speed) + task.time_to_complete) > robot.battery_life:
        return 0.0
    battery_score = 1.0 if robot.battery_life >= 2 * ((distance_to_task / robot.max_speed) + task.time_to_complete) else 0.5
    score += weights["battery_duration"] * battery_score

    # Special functions
    total_weight += weights["special_functions"]
    task_function_mapping = {
        "delivery": ["object recognition", "speech output", "facial recognition"],
        "assembly": ["object recognition", "object tracking", "precise alignment"],
        "utilities": ["percise alignment", "balance control"],
        "excavate": [ "terrain leveling", "object recognition", "precise alignment"],
        "debris": ["balance control", "object recognition"],
        "level": ["terrain leveling", "object recognition"],
        "item elevation": ["precise alignment", "object tracking", "balance control"],
        "lay bricks": ["object recognition", "precise alignment"],
        "scaffold": ["precise alignment", "balance control"],
        "remove scaffold": ["object recognition", "object tracking", "precise alignment"],
    }
    required_functions = task_function_mapping[task.task_type]
    matched_special_functions = sum(special in robot.special_functions for special in required_functions)
    special_functions_score = matched_special_functions / len(required_functions)
    score += weights["special_functions"] * special_functions_score

    # Processing power
    total_weight += weights["processing_power"]
    proc_score = 0.0
    if task.difficulty > 7:
        if robot.processing_power >= 5.0:
            proc_score = 1.0
        elif robot.processing_power >= 3.0:
            proc_score = 0.75
        else:
            proc_score = 0.5
    elif task.difficulty > 4:
        if robot.processing_power >= 3.0:
            proc_score = 1.0
        else:
            proc_score = 0.75
    elif task.difficulty > 2:
        if robot.processing_power >= 1.5:
            proc_score = 1.0
        else:
            proc_score = 0.75
    score += weights["processing_power"] * proc_score

    # Adaptability
    total_weight += weights["adaptability"]
    score += weights["adaptability"] * (1.0 if robot.adaptability else 0.0)

    # Reward/Difficulty ratio
    reward_score = (task.reward / max(task.difficulty, 1.0))
    priority_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5, "urgent": 2.0}[task.priority_level]
    score += priority_multiplier * (reward_score / (reward_score + 10.0))  # squash into (0,1)

    # Normalize
    if total_weight > 0:
        final_score = score / total_weight
    else:
        final_score = 0.0

    return max(0.0, min(1.0, final_score))

def evaluate_suitability_distance(robot: CapabilityProfile, task: TaskDescription) -> float:
    """
    Evaluates the suitability of a robot for a given task.
    A higher score indicates better suitability.
    
    Parameters:
        robot: The CapabilityProfile of the robot.
        task: The TaskDescription of the task.
    
    Returns:
        A float score representing the suitability of the robot for the task. A score of 0 indicates the robot cannot perform the task.
    """
    score = 0.0
    
#     print(task.required_capabilities, robot.payload_capacity)
    # Check if robot meets the minimum requirements
    if any(req for req in task.required_capabilities if "payload capacity" in req and robot.payload_capacity < float(req.split(">= ")[-1])):
        score += 0.0  # Suitability is zero if the robot doesn't meet minimum requirements
    else:
        score += 1.0  # Add score if payload meets or exceeds requirements
    
#     print(task.tools_needed, robot.sensors+robot.manipulators)
    # Check if the robot has the necessary tools for the task
    if robot.sensors:
        if task.tools_needed and not all(item in robot.sensors for item in task.tools_needed):
            score += 0.0  # Suitability is zero if the robot lacks necessary tools
        else:
            score += 1.0  # Add score if robot has necessary tools

    if robot.manipulators:
        if task.tools_needed and not all(item in robot.manipulators for item in task.tools_needed):
            score += 0.0  # Suitability is zero if the robot lacks necessary tools
        else:
            score += 1.0  # Add score if robot has necessary tools
    
#     print(task.communication_requirements, robot.communication_protocols)
    # Check if the robot can communicate as required by the task
    if task.communication_requirements and not all(protocol in robot.communication_protocols for protocol in task.communication_requirements):
        score += 0.0  # Suitability is zero if the robot lacks required communication protocols
    else:
        score += 1.0  # Add score if robot has communication requirements
    
#     print(task.safety_protocols, robot.safety_features)
    # Check if the robot can safely perform the task
    if robot.safety_features and task.safety_protocols:
        if task.safety_protocols and not all(safety in robot.safety_features for safety in task.safety_protocols):
            score += 0.0  # Suitability is zero if the robot lacks required safety features
        else:
            score += 1.0  # Add score if robot meets safety requirements
    
#     print(task.environmental_conditions, robot.environmental_resistance)
    # Environmental compatibility: Can the robot operate in the task’s conditions?
    if robot.environmental_resistance and task.environmental_conditions:
        if task.environmental_conditions and not all(condition in robot.environmental_resistance for condition in task.environmental_conditions):
            score += 0.0  # Suitability is zero if the robot can't operate in required environmental conditions
        else:
            score += 1.0  # Add score if robot has required environmental resistances
    
#     print(task.required_capabilities, robot.reach)
    # Check if the robot meets reach requirements
    if any(req for req in task.required_capabilities if "reach" in req and robot.reach < float(req.split(">= ")[-1])):
        score += 0.0  # Suitability is zero if the robot cannot reach the task area as required
    else:
        score += 1.0  # Add score if reach meets or exceeds requirements
    
#     print(task.navigation_constraints, robot.mobility_type, robot.size)
    # Check navigation constraints based on mobility type and robot size
    if task.navigation_constraints:
        navigation_match = navigation_suitability(robot.mobility_type, robot.size, task.navigation_constraints)
        if navigation_match == 0:
            score += 0.0
        else:
            score += navigation_match

    # NOTE: CHANGED TO WORK WITH COORDINATES
    # distance_to_task = ((robot.location[0] - task.location[0]) ** 2 + (robot.location[1] - task.location[1]) ** 2) ** 0.5
    # stop suitability matrix from going negative
    distance_to_task = max(0, len(robot.current_path) - 1)
#     print(robot.sensor_range)
    # Check sensor capabilities for the task
    if robot.sensor_range:
        if robot.sensor_range >= distance_to_task:
            score += 1.0
        elif robot.sensor_range >= distance_to_task/2:
            score += 0.5

    # Battery and distance check: Ensure the robot has sufficient battery to reach and complete the task
#     print(robot.max_speed, robot.battery_life, task.duration, distance_to_task)
    if ((distance_to_task / robot.max_speed)+task.time_to_complete) > robot.battery_life:
        score += 0.0  # Suitability is zero if the robot can't complete the task due to distance, speed, or battery limitations

    # Add to score based on proximity (closer robots get higher scores)
#     if distance_to_task < 20.0:
#         score += 1.0
#     elif distance_to_task < 50.0:
#         score += 0.5

#     print(task.priority_level, robot.autonomy_level)
    # Check if the robot's autonomy level matches the task's priority level
    if task.priority_level in ["high", "urgent"] and robot.autonomy_level in ["fully autonomous", "teleoperated"]:
        score += 1.0
    elif task.priority_level in ["medium", "low"] and robot.autonomy_level in ["semi-autonomous", "fully autonomous"]:
        score += 0.5

#     print(robot.battery_life, task.duration)
    # Evaluate battery life for task duration
    if robot.battery_life >= 2*((distance_to_task / robot.max_speed)+task.time_to_complete):
        score += 1.0
    else:
        score += 0.5

#     print(task.task_type, robot.special_functions)
    task_function_mapping = {
        "delivery": ["object recognition", "speech output", "facial recognition"],
        "inspection": ["object recognition", "object tracking", "gesture recognition"],
        "cleaning": ["object recognition"],
        "monitoring": ["speech output", "object tracking", "facial recognition"],
        "maintenance": ["object recognition", "path planning"],
        "assembly": ["object recognition"],
        "surveying": ["speech output", "facial recognition", "object recognition", "object tracking"],
        "data collection": ["object recognition", "object tracking", "facial recognition", "gesture recognition"],
        "assistance": ["speech output", "facial recognition", "gesture recognition"]
    }

    # Get the relevant functions for this task type
    required_functions = task_function_mapping[task.task_type]

    # Calculate the score based on matches between robot's functions and required functions
    if robot.special_functions:
        for function in robot.special_functions:
            if function in required_functions:
                score += 1.0  # Increase score for each match
    
#     # Dependencies
#     if task.dependencies:
#         # Assume dependencies are represented as tasks that must be completed first
#         score += 0.5 if all(dep in completed_tasks for dep in task.dependencies) else 0.0
    
#     print(task.difficulty, robot.processing_power)
    # Processing power: Certain tasks may benefit from higher processing power if they are computationally demanding
    if task.difficulty > 7 and robot.processing_power >= 5.0:  # Difficulty > 7 indicates a complex task
        score += 1.0
    elif task.difficulty > 4 and robot.processing_power >= 3.0:
        score += 1.0
    elif task.difficulty > 2 and robot.processing_power >= 1.5:
        score += 0.5

#     print(robot.adaptability)
    # Consider robot's adaptability to changing conditions
    if robot.adaptability:
        score += 0.5
    
#     print(task.task_type, robot.preferred_tasks)
    # Preference matching
    #if task.task_type in robot.preferred_tasks:
    #    score += 1.0

    # Score based on priority, reward, and difficulty
    priority_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5, "urgent": 2.0}[task.priority_level]
    reward_to_difficulty_ratio = task.reward / task.difficulty
#     print(task.priority_level, task.reward, task.difficulty, priority_multiplier, reward_to_difficulty_ratio)
    score += priority_multiplier * reward_to_difficulty_ratio

    # Weight score by distance to task
    # NOTE: IF THE ROBOT IS AT THE TASK THIS CAN CAUSE A DIVIDE BY ZERO ERROR
    score = score / (distance_to_task + 1E-5)
    
    # Return the final suitability score
#     print(score)
    return score

def evaluate_suitability_priority(robot: CapabilityProfile, task: TaskDescription) -> float:
    """
    Evaluates the suitability of a robot for a given task.
    A higher score indicates better suitability.
    
    Parameters:
        robot: The CapabilityProfile of the robot.
        task: The TaskDescription of the task.
    
    Returns:
        score: A float score representing the suitability of the robot for the task. A score of 0 indicates the robot cannot perform the task.
    """
    score = 0.0
    
#     print(task.required_capabilities, robot.payload_capacity)
    # Check if robot meets the minimum requirements
    if any(req for req in task.required_capabilities if "payload capacity" in req and robot.payload_capacity < float(req.split(">= ")[-1])):
        score += 0.0  # Suitability is zero if the robot doesn't meet minimum requirements
    else:
        score += 1.0  # Add score if payload meets or exceeds requirements
    
#     print(task.tools_needed, robot.sensors+robot.manipulators)
    # Check if the robot has the necessary tools for the task
    if robot.sensors:
        if task.tools_needed and not all(item in robot.sensors for item in task.tools_needed):
            score += 0.0  # Suitability is zero if the robot lacks necessary tools
        else:
            score += 1.0  # Add score if robot has necessary tools
    
    if robot.manipulators:
        if task.tools_needed and not all(item in robot.manipulators for item in task.tools_needed):
            score += 0.0  # Suitability is zero if the robot lacks necessary tools
        else:
            score += 1.0  # Add score if robot has necessary tools
    
#     print(task.communication_requirements, robot.communication_protocols)
    # Check if the robot can communicate as required by the task
    if task.communication_requirements and not all(protocol in robot.communication_protocols for protocol in task.communication_requirements):
        score += 0.0  # Suitability is zero if the robot lacks required communication protocols
    else:
        score += 1.0  # Add score if robot has communication requirements
    
#     print(task.safety_protocols, r
# obot.safety_features)
    # Check if the robot can safely perform the task
    if robot.safety_features and task.safety_protocols:
        if task.safety_protocols and not all(safety in robot.safety_features for safety in task.safety_protocols):
            score += 0.0  # Suitability is zero if the robot lacks required safety features
        else:
            score += 1.0  # Add score if robot meets safety requirements
    
#     print(task.environmental_conditions, robot.environmental_resistance)
    # Environmental compatibility: Can the robot operate in the task’s conditions?
    if robot.environmental_resistance and task.environmental_conditions:
        if task.environmental_conditions and not all(condition in robot.environmental_resistance for condition in task.environmental_conditions):
            score += 0.0  # Suitability is zero if the robot can't operate in required environmental conditions
        else:
            score += 1.0  # Add score if robot has required environmental resistances
    
#     print(task.required_capabilities, robot.reach)
    # Check if the robot meets reach requirements
    if any(req for req in task.required_capabilities if "reach" in req and robot.reach < float(req.split(">= ")[-1])):
        score += 0.0  # Suitability is zero if the robot cannot reach the task area as required
    else:
        score += 1.0  # Add score if reach meets or exceeds requirements
    
#     print(task.navigation_constraints, robot.mobility_type, robot.size)
    # Check navigation constraints based on mobility type and robot size
    if task.navigation_constraints:
        navigation_match = navigation_suitability(robot.mobility_type, robot.size, task.navigation_constraints)
        if navigation_match == 0:
            score += 0.0
        else:
            score += navigation_match

    # NOTE: CHANGED TO WORK WITH COORDINATES
    # distance_to_task = ((robot.location[0] - task.location[0]) ** 2 + (robot.location[1] - task.location[1]) ** 2) ** 0.5
    distance_to_task = len(robot.current_path) - 1
#     print(robot.sensor_range)
    # Check sensor capabilities for the task
    if robot.sensor_range:
        if robot.sensor_range >= distance_to_task:
            score += 1.0
        elif robot.sensor_range >= distance_to_task/2:
            score += 0.5

    # Battery and distance check: Ensure the robot has sufficient battery to reach and complete the task
#     print(robot.max_speed, robot.battery_life, task.duration, distance_to_task)
    if ((distance_to_task / robot.max_speed)+task.time_to_complete) > robot.battery_life:
        score += 0.0  # Suitability is zero if the robot can't complete the task due to distance, speed, or battery limitations

    # Add to score based on proximity (closer robots get higher scores)
    if distance_to_task < 20.0:
        score += 1.0
    elif distance_to_task < 50.0:
        score += 0.5

#     print(task.priority_level, robot.autonomy_level)
    # Check if the robot's autonomy level matches the task's priority level
    if task.priority_level in ["high", "urgent"] and robot.autonomy_level in ["fully autonomous", "teleoperated"]:
        score += 1.0
    elif task.priority_level in ["medium", "low"] and robot.autonomy_level in ["semi-autonomous", "fully autonomous"]:
        score += 0.5

#     print(robot.battery_life, task.duration)
    # Evaluate battery life for task duration
    if robot.battery_life >= 2*((distance_to_task / robot.max_speed)+task.time_to_complete):
        score += 1.0
    else:
        score += 0.5

#     print(task.task_type, robot.special_functions)
    task_function_mapping = {
        "delivery": ["object recognition", "speech output", "facial recognition"],
        "inspection": ["object recognition", "object tracking", "gesture recognition"],
        "cleaning": ["object recognition"],
        "monitoring": ["speech output", "object tracking", "facial recognition"],
        "maintenance": ["object recognition", "path planning"],
        "assembly": ["object recognition"],
        "surveying": ["speech output", "facial recognition", "object recognition", "object tracking"],
        "data collection": ["object recognition", "object tracking", "facial recognition", "gesture recognition"],
        "assistance": ["speech output", "facial recognition", "gesture recognition"]
    }

    # Get the relevant functions for this task type
    required_functions = task_function_mapping[task.task_type]

    # Calculate the score based on matches between robot's functions and required functions
    if robot.special_functions:
        for function in robot.special_functions:
            if function in required_functions:
                score += 1.0  # Increase score for each match
    
#     # Dependencies
#     if task.dependencies:
#         # Assume dependencies are represented as tasks that must be completed first
#         score += 0.5 if all(dep in completed_tasks for dep in task.dependencies) else 0.0
    
#     print(task.difficulty, robot.processing_power)
    # Processing power: Certain tasks may benefit from higher processing power if they are computationally demanding
    if task.difficulty > 7 and robot.processing_power >= 5.0:  # Difficulty > 7 indicates a complex task
        score += 1.0
    elif task.difficulty > 4 and robot.processing_power >= 3.0:
        score += 1.0
    elif task.difficulty > 2 and robot.processing_power >= 1.5:
        score += 0.5

#     print(robot.adaptability)
    # Consider robot's adaptability to changing conditions
    if robot.adaptability:
        score += 0.5
    
#     print(task.task_type, robot.preferred_tasks)
    # Preference matching
    #if task.task_type in robot.preferred_tasks:
    #    score += 1.0

    # Score based on reward and difficulty
    priority_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5, "urgent": 2.0}[task.priority_level]
    reward_to_difficulty_ratio = task.reward / task.difficulty
#     print(task.priority_level, task.reward, task.difficulty, priority_multiplier, reward_to_difficulty_ratio)
    score += reward_to_difficulty_ratio

    # Weight based on priority
    score = score * priority_multiplier
    
    # Return the final suitability score
#     print(score)
    return score

def evaluate_suitability_with_llm(robots: List[CapabilityProfile], tasks: list[TaskDescription]) -> str:
    """
    Uses a large language model (LLM) to evaluate the suitability of a robot for a given task.


    Parameters:
        robot: The CapabilityProfile of the robot.
        task: The TaskDescription of the task.


    Returns:
        score: A float score (0 to 1) representing the suitability of the robot for the task.
    """
    prompt = f"""
    You are evaluating the suitability of robots for tasks in a multi-robot system.
   
    Robot Details:
    {robots}
    (Note: Maximum Speed in units/sec, Payload Capacity in kg, Reach and Sensor Range in meters, Battery Life in seconds, Size in (length, width, height) in meters.)

    Task Details:
    {tasks}
    (Note: tools_needed[0] is the required sensors list. Tools_needed[1] is the required manipulators list. Difficulty is on a 1-10 scale, Location is (x, y, z) coordinates, Duration is in seconds.)

    Based on the robot capabilities and the task requirements, *please rate the suitability of each robot-task pair on a scale of 0 to 1, where 0 means the robot is completely unsuitable and 1 means it is perfectly suited.*
    When evaluating whether the robot has the CRITICAL requirements, ONLY check the following three fields:

    1. Manipulators (only compare against the tasks's tools_needed[1])
    2. Navigation Constraints (assume aerial and hovering can circumvent ground conditions)
    3. Payload Capacity

    Only the these three fields are critical (Manipulators, Navigation, Payload). Do not consider any other field, like Reach, as critical. The robot should only receive a score of 0 **if one or more of those THREE specific requirements are not satisfied**.
    Once the critical requirements (Tools Needed, Navigation Constraints, and Payload Capacity) are satisfied, evaluate all OTHER attributes. All other capabilities and requirements should still influence the score between 0 and 1 (if the critical requirements are met).
    When comparing multi-value attributes (such as Environmental Resistance, Sensors, Communication Protocols, Special Functions, or Safety Features), treat them as **matched** if the robot satisfies the **majority** of the required values. If it doesn't reach majority, but is still not 100% unmatched, give *partial credit*.  

    *Duration is how much time is required to finish a task, once the robot arrives at the task. Use it to calculate whether or not the robot has enough battery to both arrive at the site, and complete the task*
    *Task priority level should be compared with robot's autonomy level. Task's difficulty should be compared with robot's processing power. If robot is adaptable, add to the score, automatically. A robot's special functions should be compared to the overall requirements of the task*.   

    Output Format:
    Return a 2D suitability matrix:
        Rows = robots.
        Columns = tasks.
        Each entry = suitability score for that robot-task pair (float between 0 and 1).    
    Make sure the output section only has the title "OUTPUT" and the suitability matrix, no additional words in this section

    Example Output (for 3 robots x 3 tasks):
    [
        [0.9, 0.7, 0.0],
        [0.4, 0.8, 0.6],
        [0.0, 0.3, 1.0]
    ]
"""

    
    #Provide ONLY a single number between 0 and 1. Do not include any words, punctuation, or explanations.
    score = 0.0
    try:
        client = OpenAI()
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )
        score = response.output_text
    except Exception as e:
        print(f"Error: {response}")

    return score

def calculate_total_suitability(assignment: List[Tuple[int, int]], suitability_matrix: List[List[float]]) -> float:
    """
    Calculates the total suitability score for a given assignment.
    
    Parameters:
        assignment: A list of (robot, task) pairs representing the assignment.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
    
    Returns:
        total_suitability: The total suitability score for the assignment.
    """
    total_suitability = 0.0
    
    # Sum the suitability ratings for each robot-task pair in the assignment
    for robot, task in assignment:
        total_suitability += suitability_matrix[robot][task]
    
    return total_suitability

def check_zero_suitability(assignment: List[Tuple[int, int]], suitability_matrix: List[List[float]]) -> bool:
    """
    Checks if any robot-task pair in the assignment has a suitability rating of 0.
    
    Parameters:
        assignment: A list of (robot, task) pairs representing the assignment.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
    
    Returns:
        Bool: True if any robot-task pair in the assignment has a suitability of 0, otherwise False.
    """
    for robot, task in assignment:
        if suitability_matrix[robot][task] == 0:
            return True  # Found a zero suitability rating
    
    return False  # No zero suitability ratings found

def calculate_suitability_matrix(robots: List[CapabilityProfile], tasks: List[TaskDescription], scorer: ScoreFn) -> np.ndarray:
    """
    Calculates the suitability matrix for the given robots and tasks.
    
    Parameters:
        robots: List of robot profiles.
        tasks: List of task descriptions.
        suitability_method: The name of the suitability evaluation function.
    
    Returns:
        suitability_matrix: A 2D numpy array representing the suitability scores of each robot-task pair.
    """
    suitability_matrix = np.zeros((len(robots), len(tasks)), dtype=float)

    # Evaluate suitability of each robot for each task
    for i, robot in enumerate(robots):
        for j, task in enumerate(tasks):
            # suitability_score = globals()[suitability_method](robot, task)
            # suitability_matrix[i][j] = suitability_score
            suitability_matrix[i, j] = scorer(robot, task)
            
    return suitability_matrix

