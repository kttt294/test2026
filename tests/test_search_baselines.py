import random

import pytest

from env.hex_grid import HexGrid
from pathfinding.baselines import ALGORITHMS, search_path
import config as C


def test_baselines_match_reachability_and_return_valid_paths():
    rng = random.Random(23)
    grid = HexGrid(6, 6)
    for _ in range(40):
        terrain = {i: rng.choice([0, 0, 1, 2, 3]) for i in range(36)}
        src, dst = rng.sample(range(36), 2)
        terrain[src] = terrain[dst] = 0
        traffic = {i: rng.randrange(3) for i in range(36)}
        results = {name: search_path(grid, terrain, traffic, src, dst, 24, 12, name)
                   for name in ALGORITHMS}
        assert len({r.reachable for r in results.values()}) == 1
        assert results['dijkstra'].total_steps == results['astar'].total_steps
        for result in results.values():
            if not result.reachable:
                continue
            cell, steps, fuel = src, 0, 0
            for action in result.actions:
                cost = C.STEP_COST[terrain[cell]]
                steps += cost[traffic[cell]] if isinstance(cost, dict) else cost
                fuel += C.FUEL_COST.get(terrain[cell], 0)
                cell = grid.neighbor_in_dir(cell, action.direction)
                assert cell is not None and terrain[cell] != 2
            assert cell == dst
            assert steps == result.total_steps <= 24
            assert fuel == result.total_fuel <= 12
            assert len(results['bfs'].actions) <= len(result.actions)


@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_zero_budget_and_lake(algorithm):
    grid = HexGrid(2, 1)
    assert search_path(grid, {0: 0, 1: 0}, {}, 0, 0, 0, 0, algorithm).reachable
    assert not search_path(grid, {0: 0, 1: 0}, {}, 0, 1, 0, 0, algorithm).reachable
    assert not search_path(grid, {0: 0, 1: 2}, {}, 0, 1, 10, 10, algorithm).reachable
