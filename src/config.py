TERRAIN_PLAIN    = 0
TERRAIN_MOUNTAIN = 1
TERRAIN_LAKE     = 2
TERRAIN_ROAD     = 3

AGENT_PATROL = 0
AGENT_SUPPLY = 1

TRAFFIC_CLEAR     = 0
TRAFFIC_BUSY      = 1
TRAFFIC_CONGESTED = 2

# Step cost: cost to move WHILE ON this terrain (source-based).
# Road cost depends on traffic status.
STEP_COST = {
    TERRAIN_PLAIN:    2,
    TERRAIN_MOUNTAIN: 3,
    TERRAIN_LAKE:     None,   # impassable
    TERRAIN_ROAD:     {
        TRAFFIC_CLEAR:     1,
        TRAFFIC_BUSY:      2,
        TRAFFIC_CONGESTED: 4,
    },
}

# Fuel cost: cost to move WHILE ON this terrain (source-based, patrol only).
FUEL_COST = {
    TERRAIN_PLAIN:    1,
    TERRAIN_MOUNTAIN: 2,
    TERRAIN_ROAD:     2,
}

N_DIRECTIONS = 6
