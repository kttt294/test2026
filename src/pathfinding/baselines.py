"""Resource-constrained DFS, BFS, Dijkstra and greedy best-first baselines.

All searches use the same legal moves and budgets. BFS minimizes move count;
Dijkstra minimizes step cost. DFS and greedy best-first return a feasible path.
No mid-path refueling is modeled, matching the existing A* pathfinder.
"""
from collections import deque
import heapq
from itertools import count

import config as C
from env.models import AgentAction, CMD_MOVE
from pathfinding.astar import PathResult, find_path

ALGORITHMS = ('dfs', 'bfs', 'dijkstra', 'gbfs', 'astar')


def search_path(grid, terrain, traffic, src, dst, step_budget, fuel_budget=None,
                algorithm='astar'):
    if algorithm not in ALGORITHMS:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    if algorithm == 'astar':
        return find_path(grid, terrain, traffic, src, dst, step_budget, fuel_budget)
    if (step_budget < 0 or (fuel_budget is not None and fuel_budget < 0)
            or terrain.get(src, C.TERRAIN_LAKE) == C.TERRAIN_LAKE
            or terrain.get(dst, C.TERRAIN_LAKE) == C.TERRAIN_LAKE):
        return PathResult.unreachable()

    # Immutable labels include depth so dominance preserves BFS's hop objective.
    start = (src, 0, 0, 0)
    labels = {src: {(0, 0, 0)}}
    parents = {}
    sequence = count()
    frontier = deque([start])
    heap = [(0, next(sequence), start)]
    while frontier if algorithm in ('bfs', 'dfs') else heap:
        if algorithm == 'bfs':
            current = frontier.popleft()
        elif algorithm == 'dfs':
            current = frontier.pop()
        else:
            _, _, current = heapq.heappop(heap)
        cell, steps, fuel, depth = current
        if (steps, fuel, depth) not in labels[cell]:
            continue
        if cell == dst:
            actions = []
            while current in parents:
                current, direction = parents[current]
                actions.append(AgentAction(CMD_MOVE, direction))
            return PathResult(list(reversed(actions)), steps, fuel, True)
        cost = C.STEP_COST[terrain[cell]]
        cost = cost[traffic.get(cell, C.TRAFFIC_CLEAR)] if isinstance(cost, dict) else cost
        new_steps = steps + cost
        new_fuel = fuel + C.FUEL_COST.get(terrain[cell], 0)
        if new_steps > step_budget or (fuel_budget is not None and new_fuel > fuel_budget):
            continue
        for direction, neighbor in grid.neighbors(cell):
            if terrain.get(neighbor, C.TERRAIN_LAKE) == C.TERRAIN_LAKE:
                continue
            costs = labels.setdefault(neighbor, set())
            candidate = (new_steps, new_fuel, depth + 1)

            def dominates(a, b):
                return (a[0] <= b[0] and (fuel_budget is None or a[1] <= b[1])
                        and (algorithm != 'bfs' or a[2] <= b[2]))

            if any(dominates(old, candidate) for old in costs):
                continue
            costs.difference_update({old for old in costs if dominates(candidate, old)})
            costs.add(candidate)
            label = (neighbor, *candidate)
            parents[label] = (current, direction)
            if algorithm in ('bfs', 'dfs'):
                frontier.append(label)
            else:
                priority = new_steps if algorithm == 'dijkstra' else grid.hex_distance(neighbor, dst)
                heapq.heappush(heap, (priority, next(sequence), label))
    return PathResult.unreachable()
