"""Compare pathfinders on identical seeded maps, endpoints and resource budgets."""
import argparse
import csv
from pathlib import Path
import random
import time

from env.hex_grid import HexGrid
from env.map_generator import generate_random_scenario
from pathfinding.baselines import ALGORITHMS, search_path


def run_benchmark(games=100, sizes=(8, 16, 32), seed=42):
    if games < 1:
        raise ValueError('games must be positive')
    rows = []
    for size in sizes:
        for offset in range(games):
            case_seed = seed + offset
            cfg, board, agents = generate_random_scenario(
                case_seed, width_range=(size, size), height_range=(size, size))
            grid = HexGrid(size, size)
            terrain = {c.id: c.terrain for c in board.cells}
            rng = random.Random(case_seed)
            src = agents[0].cell
            dst = rng.choice(board.spots).cell_id
            traffic = {cell: rng.randrange(3) for cell, t in terrain.items() if t == 3}
            for algorithm in ALGORITHMS:
                started = time.perf_counter()
                result = search_path(grid, terrain, traffic, src, dst,
                                     cfg.steps_per_day[0], cfg.fuel_max, algorithm)
                elapsed = (time.perf_counter() - started) * 1000
                rows.append(dict(size=size, seed=case_seed, algorithm=algorithm,
                    src=src, dst=dst, step_budget=cfg.steps_per_day[0], fuel_budget=cfg.fuel_max,
                    reachable=result.reachable, steps=result.total_steps if result.reachable else '',
                    fuel=result.total_fuel if result.reachable else '',
                    moves=len(result.actions) if result.reachable else '', time_ms=elapsed))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--sizes', type=int, nargs='+', choices=[8, 16, 32], default=[8, 16, 32])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/pathfinding.csv')
    args = parser.parse_args()
    if args.games < 1:
        parser.error('--games must be positive')
    rows = run_benchmark(args.games, args.sizes, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} measurements to {output}')
