# FEATURE POOLS FOR RANDOM SAMPLING
MOBILITY_TYPES = ["wheeled", "tracked", "legged", "aerial", "hovering", "climbing"]

ENV_RESISTANCES = ["weatherproof", "waterproof", "dustproof", "heat-resistant", 
                   "cold-resistant", "shock-resistant", "wind-resistant"]

SENSORS = ["camera", "microphone", "LiDAR", "GPS", "ultrasonic", "temperature sensor", 
           "infrared", "proximity sensor", "magnetometer"]

MANIPULATORS = ["gripper", "drill", "welding tool", "hydraulic bucket", "dispenser", 
                "mixer drum", "cable hoist"]

COMM_PROTOCOLS = ["Wi-Fi", "Bluetooth", "4G", "5G", "Radio"]

SPECIAL_FUNCS = ["object recognition", "speech output", "facial recognition", "object tracking", 
                 "gesture recognition", "precise alignment", "terrain leveling", "balance control", 
                 "concrete mixing", "object detection"]

SAFETY_FEATS = ["emergency stop", "collision avoidance", "overheat protection", "fall detection",
                "obstacle detection", "speed reduction in crowded areas", "overload protection", 
                "balance control"]

NAV_CONSTRAINTS = ["elevator", "stairs", "shelves", "no loud noises allowed", "narrow spaces", 
                   "low ceilings", "uneven floors", "low visibility", "slippery", "crowded", 
                   "loose debris", "no-fly zone", "windy", "dense obstructions", "smooth surfaces"]

ENV_CONDITIONS = ["weatherproof", "waterproof", "dustproof", "heat-resistant", "cold-resistant", 
                  "shock-resistant", "wind-resistant"]

CAP_REQ_TEMPLATES = [
    "payload capacity >= 1.0",
    "payload capacity >= 5.0",
    "payload capacity >= 10.0",
    "payload capacity >= 20.0",
    "reach >= 1.5",
    "reach >= 3.0",
    "reach >= 5.0",
]