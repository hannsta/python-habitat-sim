import torch

H = 64
W = 64


SOIL_TYPES = ["loam", "clay", "sand", "peat"]
# Assign channels


BASE_PLANT_RULES = {
    # === Grasses ===
    "grass_0": {  # Wheatgrass – generalist stabilizer
        "group": [1, 0, 0],              # grass
        "soil_types": [1, 1, 0, 0],      # loam, clay
        "elevation": [1, 1, 0],          # low, med
        "shade": [1, 1, 0],              # low, med
        "moisture_tolerance": 0.3,
        "spread_rate": 0.1,
        "root_competition": 0.4,
        "persistence": 0.2
    },
    "grass_1": {  # Clay Bunchgrass – tough, slow-spreading anchor
        "group": [1, 0, 0],              # grass
        "soil_types": [0, 1, 0, 0],      # clay
        "elevation": [0, 1, 1],          # med, high
        "shade": [1, 0, 0],              # low only
        "moisture_tolerance": 0.5,
        "spread_rate": 0.5,
        "root_competition": 0.6,
        "persistence": 0.5
    },
    "grass_2": {  # Dunegrass – aggressive spreader for dry sand
        "group": [1, 0, 0],              # grass     
        "soil_types": [0, 0, 1, 0],      # sand
        "elevation": [1, 0, 0],          # low only
        "shade": [1, 0, 0],              # low only
        "moisture_tolerance": 0.2,
        "spread_rate": 1.0,
        "root_competition": 0.3,
        "persistence": 0.1
    },

    # === Shrubs ===
    "shrub_0": {  # Coastal Sage – dry ridge colonizer
        "group": [0, 1, 0],              #shrub      
        "soil_types": [0, 0, 1, 0],      # sand
        "elevation": [0, 1, 1],          # med, high
        "shade": [1, 0, 0],              # low only
        "moisture_tolerance": 0.3,
        "spread_rate": 0.6,
        "root_competition": 0.5,
        "persistence": 0.4
    },
    "shrub_1": {  # Bog Willow – wetland buffer species
        "group": [0, 1,  0],             #shrub     
        "soil_types": [0, 0, 0, 1],      # peat
        "elevation": [1, 0, 0],          # low only
        "shade": [0, 1, 1],              # med, high
        "moisture_tolerance": 0.9,
        "spread_rate": 0.3,
        "root_competition": 0.7,
        "persistence": 0.8
    },
    "shrub_2": {  # Thicket Alder – flexible medium shrub
        "group": [0, 1, 0],              # shrub
        "soil_types": [1, 0, 0, 0],      # loam
        "elevation": [1, 1, 0],          # low, med
        "shade": [1, 1, 0],              # low, med
        "moisture_tolerance": 0.6,
        "spread_rate": 0.5,
        "root_competition": 0.6,
        "persistence": 0.6
    },

    # === Trees ===
    "tree_0": {  # Douglas Fir – upland dominant
        "group": [0, 0, 1],              # tree
        "soil_types": [1, 0, 0, 0],      # loam
        "elevation": [0, 1, 1],          # med, high
        "shade": [1, 0, 0],              # low only
        "moisture_tolerance": 0.4,
        "spread_rate": 0.2,
        "root_competition": 1.0,
        "persistence": 1.0
    },
    "tree_1": {  # Black Cottonwood – versatile canopy tree
        "group": [0, 0, 1],              # tree
        "soil_types": [0, 1, 0, 0],      # clay
        "elevation": [1, 1, 0],          # low, med
        "shade": [1, 1, 0],              # low, med
        "moisture_tolerance": 0.6,
        "spread_rate": 0.3,
        "root_competition": 0.8,
        "persistence": 0.9
    },
    "tree_2": {  # Swamp Ash – climax species for wet areas
        "group": [0, 0, 1],              # tree       
        "soil_types": [0, 0, 0, 1],      # peat
        "elevation": [1, 0, 0],          # low only
        "shade": [0, 1, 1],              # med, high
        "moisture_tolerance": 0.85,
        "spread_rate": 0.4,
        "root_competition": 0.9,
        "persistence": 0.95
    },
    "tree_4": {  # Placeholder
        "group": [0, 0, 1],              # tree
        "soil_types": [0, 1, 1, 0],      # peat
        "elevation": [1, 1, 1],          # low only
        "shade": [1, 1, 1],              # med, high
        "moisture_tolerance": 1,
        "spread_rate": 0.5,
        "root_competition": 0.8,
        "persistence": .8
    }
}
PLANT_RULES = BASE_PLANT_RULES

PLANT_GROUPS = {
    "grass": ["grass_0", "grass_1", "grass_2"],
    "shrub": ["shrub_0", "shrub_1", "shrub_2"],
    "tree": ["tree_0", "tree_1", "tree_2","tree_4"]
}
CHANNELS = {
    "soil": {soil: i for i, soil in enumerate(SOIL_TYPES)},
    "elevation": 4,
    "shade": 5,
    "plants": [(plant, 6 + i) for i, plant in enumerate(PLANT_RULES)]
}
NUM_CHANNELS = 16
print(str(NUM_CHANNELS) + 'NUMCHANNELS')
import numpy as np

