H = 64
W = 64


SOIL_TYPES = ["loam", "clay", "sand", "peat"]
PLANT_TYPES = ["grass_0", "grass_1", "grass_2", "shrub_0", "shrub_1", "shrub_2", "tree_0", "tree_1", "tree_2"]

# Assign channels
CHANNELS = {
    "soil": {soil: i for i, soil in enumerate(SOIL_TYPES)},
    "elevation": 4,
    "shade": 5,
    "plants": {plant: 6 + i for i, plant in enumerate(PLANT_TYPES)},
}
NUM_CHANNELS = max(CHANNELS["plants"].values()) + 1

# --- Plant Rules ---
PLANT_RULES = {
    "grass_0": {"soil": "loam", "elevation": "any",    "shade": "any"},
    "grass_1": {"soil": "clay", "elevation": "any", "shade": "any"},
    "grass_2": {"soil": "clay", "elevation": "any", "shade": "any"},
    "shrub_0": {"soil": "sand", "elevation": "any",    "shade": "any"},
    "shrub_1": {"soil": "peat", "elevation": "any",    "shade": "any"},
    "shrub_2": {"soil": "peat", "elevation": "any", "shade": "any"},
    "tree_0":  {"soil": "loam", "elevation": "any",   "shade": "any"},
    "tree_1":  {"soil": "clay", "elevation": "any",    "shade": "any"},
    "tree_2": {"soil": "clay", "elevation": "any", "shade": "any"},


}
PLANT_GROUPS = {
    "grass": ["grass_0", "grass_1", "grass_2"],
    "shrub": ["shrub_0", "shrub_1", "shrub_2"],
    "tree": ["tree_0", "tree_1", "tree_2"]
}
