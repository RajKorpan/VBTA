import random
from typing import List

def _sample(items: List[str], k_min: int = 0, k_max: int = None) -> List[str]:
    """Sample between k_min and k_max items from the list."""
    if not items:
        return []
    if k_max is None:
        k_max = len(items)
    k_min = max(0, min(k_min, len(items)))
    k_max = max(k_min, min(k_max, len(items)))
    k = random.randint(k_min, k_max)
    return random.sample(items, k)

# STRICT ROBOT PROFILES FOR ROBOT GENERATION
STRICT_ROBOT_PROFILES = [
    {
        "name": "delivery",
        "mobility_type": "aerial",
        "environmental_resistance": ["weatherproof", "heat-resistant", "cold-resistant"],
        "sensors": ["LiDAR", "GPS", "proximity sensor", "camera"],
        "manipulators": ["gripper"],
        "communication_protocols": ["Wi-Fi", "4G"],
        "special_functions": ["object recognition", "object tracking", "facial recognition"],
        "safety_features": ["obstacle detection", "emergency stop"],
        "sensor_range": 25.0,
        "processing_power": 5.0,
        "autonomy_level": "fully autonomous",
        "payload_capacity": 20.0,
        "reach": 0.0,
        "battery_life": 80.0,
        "size": (2.0, 2.0, 2.0),
        "adaptability": False
    },
    {
        "name": "assembly",
        "mobility_type": "hovering",
        "environmental_resistance": ["dustproof", "heat-resistant", "shock-resistant"],
        "sensors": ["camera", "proximity sensor", "infrared"],
        "manipulators": ["gripper", "drill", "dispenser", "welding tool"],
        "communication_protocols": ["Wi-Fi", "Radio"],
        "special_functions": ["object recognition", "object tracking", "gesture recognition", "precise alignment"],
        "safety_features": ["collision avoidance", "overheat protection", "emergency stop"],
        "sensor_range": 10.0,
        "processing_power": 7.0,
        "autonomy_level": "semi-autonomous",
        "payload_capacity": 20.0,
        "reach": 5.0,
        "battery_life": 75.0,
        "size": (3.0, 2.0, 2.0),
        "adaptability": True
    },
    {
        "name": "excavator",
        "mobility_type": "tracked",
        "environmental_resistance": ["dustproof", "shock-resistant", "weatherproof"],
        "sensors": ["LiDAR", "camera", "proximity sensor", "magnetometer"],
        "manipulators": ["hydraulic bucket", "cable hoist"],
        "communication_protocols": ["Wi-Fi", "Radio"],
        "special_functions": ["terrain leveling", "object recognition"],
        "safety_features": ["collision avoidance", "overload protection", "emergency stop"],
        "sensor_range": 15.0,
        "processing_power": 8.0,
        "autonomy_level": "semi-autonomous",
        "payload_capacity": 20.0,
        "reach": 8.0,
        "battery_life": 75.0,
        "size": (4.2, 2.5, 3.0),
        "adaptability": True
    },
    {
        "name": "bricklayer",
        "mobility_type": "wheeled",
        "environmental_resistance": ["dustproof", "waterproof", "shock-resistant"],
        "sensors": ["infrared", "camera", "proximity sensor"],
        "manipulators": ["mixer drum", "dispenser", "gripper", "cable hoist"],
        "communication_protocols": ["Wi-Fi", "Radio"],
        "special_functions": ["precise alignment", "concrete mixing"],
        "safety_features": ["obstacle detection", "emergency stop", "collision avoidance"],
        "sensor_range": 10.0,
        "processing_power": 5.5,
        "autonomy_level": "semi-autonomous",
        "payload_capacity": 20.0,  
        "reach": 5.5,
        "battery_life": 80.0,
        "size": (4.0, 2.0, 2.0),
        "adaptability": False
    },
    {
        "name": "crane",
        "mobility_type": "tracked", 
        "environmental_resistance": ["weatherproof", "wind-resistant", "dustproof"],
        "sensors": ["camera", "GPS", "proximity sensor", "ultrasonic"],
        "manipulators": ["gripper", "cable hoist"],
        "communication_protocols": ["5G", "Radio"],
        "special_functions": ["precise alignment", "object tracking", "object detection"],
        "safety_features": ["overload protection", "balance control", "collision avoidance", "emergency stop"],
        "sensor_range": 30.0,
        "processing_power": 6.5,
        "autonomy_level": "semi-autonomous",
        "payload_capacity": 35.0,
        "reach": 30.0,
        "battery_life": 70.0,
        "size": (3.5, 3.5, 10.0),
        "adaptability": False
    },
    {
        "name": "scaffolding",
        "mobility_type": "climbing",
        "environmental_resistance": ["dustproof", "shock-resistant", "wind-resistant"],
        "sensors": ["LiDAR", "camera", "ultrasonic", "proximity sensor"],
        "manipulators": ["gripper", "drill", "welding tool", "cable hoist"],
        "communication_protocols": ["Wi-Fi", "Bluetooth"],
        "special_functions": ["precise alignment", "balance control"],
        "safety_features": ["fall detection", "collision avoidance", "emergency stop"],
        "sensor_range": 12.0,
        "processing_power": 7.0,
        "autonomy_level": "fully autonomous",
        "payload_capacity": 20.0,  
        "reach": 10.0,  
        "battery_life": 65.0,
        "size": (1.5, 1.0, 2.0),
        "adaptability": True
    }
]

# STRICT TASK PROFILES FOR TASK GENERATION
STRICT_TASK_PROFILES = [
    {
        "task_type": "utilities", #gripper
        "priority_level": random.choice(["medium", "high"]),
        "reward": random.randint(5, 8),
        "difficulty": random.randint(5, 8),
        "navigation_constraints": _sample(["uneven floors", "loose debris", "slippery"], 0, 3),
        "required_capabilities": _sample(["payload >= 10", "reach >= 3"], 0, 2),
        "environmental_conditions": _sample(["weatherproof", "dustproof", "shock-resistance"], 0, 3),
        "tools_needed": [["LiDAR", "camera", "proximity sensor"], ["gripper"]],
        "communication_requirements": ["Radio", "Wi-Fi"],
        "safety_protocols": ["overload protection", "balance control", "emergency stop"],
        "duration": random.randint(3, 6),
        "performance_metric": "safety compliance"
    },
    {
        "task_type": "utilities", #bucket
        "priority_level": random.choice(["medium", "high"]),
        "reward": random.randint(5, 8),
        "difficulty": random.randint(5, 8),
        "navigation_constraints": _sample(["uneven floors", "loose debris", "slippery"], 0, 3),
        "required_capabilities": _sample(["payload >= 10", "reach >= 3"], 0, 2),
        "environmental_conditions": _sample(["weatherproof", "dustproof", "shock-resistance"], 0, 3),
        "tools_needed": [["LiDAR", "camera", "proximity sensor"], ["hydraulic bucket"]],
        "communication_requirements": ["Radio", "Wi-Fi"],
        "safety_protocols": ["overload protection", "balance control", "emergency stop"],
        "duration": random.randint(3, 6),
        "performance_metric": "safety compliance"
    },
    {
        "task_type": "debris", #gripper
        "priority_level": random.choice(["medium", "high"]),
        "reward": random.randint(4, 7),
        "difficulty": random.randint(4, 7),
        "navigation_constraints": _sample(["loose debris", "uneven floors", "crowded"], 0, 2),
        "required_capabilities": [f"payload >= {random.randint(10,15)}", f"reach >= {random.randint(0,3)}"],
        "environmental_conditions": _sample(["weatherproof", "wind-resistant"], 0, 2),
        "tools_needed": [["LiDAR", "camera", "ultrasonic", "proximity sensor"], _sample(["gripper"], 1, 1)],
        "communication_requirements": ["Radio", "Wi-Fi"],
        "safety_protocols": ["overload protection", "balance control", "emergency stop"],
        "duration": random.randint(3, 6),
        "performance_metric": "safety compliance"
    },
    {
        "task_type": "debris", #bucket
        "priority_level": random.choice(["medium", "high"]),
        "reward": random.randint(4, 7),
        "difficulty": random.randint(4, 7),
        "navigation_constraints": _sample(["loose debris", "uneven floors", "crowded"], 0, 2),
        "required_capabilities": [f"payload >= {random.randint(10,15)}", f"reach >= {random.randint(0,3)}"],
        "environmental_conditions": _sample(["weatherproof", "wind-resistant"], 0, 2),
        "tools_needed": [["LiDAR", "camera", "ultrasonic", "proximity sensor"], ["hydraulic bucket"]],
        "communication_requirements": ["Radio", "Wi-Fi"],
        "safety_protocols": ["overload protection", "balance control", "emergency stop"],
        "duration": random.randint(3, 6),
        "performance_metric": "safety compliance"
    },
    {
        "task_type": "delivery",
        "priority_level": random.choice(["low", "medium"]),
        "reward": random.randint(2, 5),
        "difficulty": random.randint(2, 5),
        "navigation_constraints": _sample(["uneven floors", "crowded", "slippery", "elevator", "stairs"], 0),
        "required_capabilities": _sample(["payload capacity >= 1.0", "payload capacity >= 5.0", "payload capacity >= 10.0"], 1, 1),
        "environmental_conditions": _sample(["weatherproof", "dustproof"], 1, 1),
        "tools_needed": [["camera", "proximity sensor", "GPS"], ["gripper"]],
        "communication_requirements": ["Wi-Fi", "4G"],
        "safety_protocols": ["obstacle detection", "emergency stop"],
        "duration": random.randint(2, 5),
        "performance_metric": "time taken",
    },
    {
        "task_type": "assembly",
        "priority_level": random.choice(["medium", "high"]),
        "reward": random.randint(5, 7),
        "difficulty": random.randint(5, 7),
        "navigation_constraints": _sample(["crowded", "loose debris", "slippery", "low ceilings"], 0, 2),
        "required_capabilities": _sample(["reach >= 1.5", "reach >= 3.0", "reach >= 5.0", "payload capacity >= 10.0"], 1, 1),
        "environmental_conditions": _sample(["dustproof", "shock-resistant", "heat-resistant"], 0, 3),
        "tools_needed": [["camera", "infrared", "dispenser"], ["gripper", "drill", "welding tool"]],
        "communication_requirements": ["Wi-Fi", "Radio"],
        "safety_protocols": ["collision avoidance", "emergency stop"],
        "duration": random.randint(4, 7),
        "performance_metric": "accuracy",
    },
    {
        "task_type": "excavate",
        "priority_level": random.choice(["high", "urgent"]),
        "reward": random.randint(6, 9),
        "difficulty": random.randint(6, 8),
        "navigation_constraints": _sample(["uneven floors", "loose debris", "low visibility"], 0, 3),
        "required_capabilities": [f"payload >= {random.randint(15, 20)}", f"reach >= {random.randint(2,7)}"],
        "environmental_conditions": _sample(["dustproof", "shock-resistant", "weatherproof"], 0, 3),
        "tools_needed": [["LiDAR", "camera", "proximity sensor"], ["hydraulic bucket"]],
        "communication_requirements": ["Radio"],
        "safety_protocols": ["overload protection", "obstacle detection"],
        "duration": random.randint(5, 7),
        "performance_metric": "safety compliance"
    },
    {
        "task_type": "item elevation",
        "priority_level": random.choice(["low", "medium"]),
        "reward": random.randint(4, 8),
        "difficulty": random.randint(4, 8),
        "navigation_constraints": _sample(["low visibility", "crowded", "uneven floors"], 0, 3),
        "required_capabilities": [f"reach >= {random.randint(15, 25)}", "payload >= 15"],
        "environmental_conditions": _sample(["weatherproof", "wind-resistant"], 0, 2),
        "tools_needed": [["camera", "GPS", "proximity sensor", "ultrasonic"], ["cable hoist", "gripper"]],
        "communication_requirements": ["Radio", "Wi-Fi"],
        "safety_protocols": ["overload protection", "emergency stop"],
        "duration": random.randint(3, 5),
        "performance_metric": "safety compliance"
    },
    {
        "task_type": "lay bricks",
        "priority_level": "medium",
        "reward": random.randint(6, 12),
        "difficulty": random.randint(6, 12),
        "navigation_constraints": _sample(["crowded", "low visibility", "windy"], 0, 3),
        "required_capabilities": [f"reach >= {random.randint(2,4)}", f"payload >= {random.randint(8,10)}"],
        "environmental_conditions": _sample(["dustproof", "weatherproof"], 0, 2),
        "tools_needed": [["LiDAR", "camera", "proximity sensor"], ["gripper", "dispenser"]],
        "communication_requirements": ["Wi-Fi"],
        "safety_protocols": ["collision avoidance", "overheat protection"],
        "duration": random.randint(6, 10),
        "performance_metric": "accuracy"
    },
    {
        "task_type": "scaffold",
        "priority_level": "medium",
        "reward": random.randint(5, 8),
        "difficulty": random.randint(5, 8),
        "navigation_constraints": _sample(["narrow spaces", "low ceilings", "windy"], 0, 3),
        "required_capabilities": [f"payload >= {random.randint(8, 10)}", f"reach >= {random.randint(2, 3)}"],
        "environmental_conditions": _sample(["weatherproof", "wind-resistant"], 0, 2),
        "tools_needed": [["LiDAR", "camera", "ultrasonic", "proximity sensor"], ["gripper", "drill"]],
        "communication_requirements": ["Radio", "Wi-Fi"],
        "safety_protocols": ["overload protection", "balance control", "emergency stop"],
        "duration": random.randint(3, 6),
        "performance_metric": "safety compliance"
    }
]