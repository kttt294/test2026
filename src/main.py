"""
Entry point for HEXA UDON bot.

Modes:
  sim     -- compare Greedy and Lookahead on the simulator

Usage:
  python main.py sim
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from env.models import AgentState, MatchConfig
from env.simulator import HexaUdonSimulator
from strategy.greedy import GreedyPlanner
from strategy.lookahead import LookaheadPlanner


# ------------------------------------------------------------------ #
# Demo / sanity-check data                                            #
# ------------------------------------------------------------------ #

def _demo_config() -> MatchConfig:
    return MatchConfig(
        width=10, height=8,
        total_days=6,
        steps_per_day=[120, 120, 100, 100, 80, 80],
        n_teams=4,
        traffic_threshold_busy=3.0,
        traffic_threshold_congested=7.0,
    )


def _demo_map(cfg: MatchConfig):
    from env.models import Cell, MapData, Spot
    cells = []
    for i in range(cfg.width * cfg.height):
        # Simple: all plain except a few roads and a lake
        if i in (5, 6, 7, 15, 16, 17):
            terrain = 3   # road
        elif i in (22, 32):
            terrain = 2   # lake
        elif i in (11, 21, 31):
            terrain = 1   # mountain
        else:
            terrain = 0   # plain
        cells.append(Cell(id=i, terrain=terrain))

    spots = [
        Spot(cell_id=12, series_id=1, max_inventory=3),
        Spot(cell_id=24, series_id=2, max_inventory=2),
        Spot(cell_id=35, series_id=1, max_inventory=1),
        Spot(cell_id=47, series_id=3, max_inventory=2),
        Spot(cell_id=58, series_id=2, max_inventory=3),
    ]
    return MapData(cells=cells, spots=spots)


def _demo_agents() -> list[AgentState]:
    return [
        AgentState(id=1, type=0, cell=0,  fuel=20),   # patrol
        AgentState(id=2, type=0, cell=70, fuel=20),   # patrol
        AgentState(id=3, type=1, cell=40, fuel=0),    # supply
    ]


# ------------------------------------------------------------------ #
# Modes                                                               #
# ------------------------------------------------------------------ #

def _run_one(label: str, planner, sim, agents_fn):
    state = sim.reset(agents_fn())
    print(f"\n=== {label} ===")
    while not sim.is_done(state):
        orders = planner.plan(state)
        state = sim.apply_day(state, orders)
        print(
            f"  Day {state.day-1}: "
            f"series={sorted(state.collected_series)}  "
            f"udon={state.total_udon}"
        )
    print(f"  Final: unique_series={len(state.collected_series)}  "
          f"total_udon={state.total_udon}")
    return len(state.collected_series), state.total_udon


def run_sim(args):
    """Compare Greedy vs Lookahead on demo map."""
    cfg      = _demo_config()
    map_data = _demo_map(cfg)
    sim      = HexaUdonSimulator(cfg, map_data)

    greedy    = GreedyPlanner(cfg, map_data, sim)
    lookahead = LookaheadPlanner(cfg, map_data, sim)

    g_series, g_udon = _run_one("Greedy",    greedy,    sim, _demo_agents)
    l_series, l_udon = _run_one("Lookahead", lookahead, sim, _demo_agents)

    print(f"\n{'='*40}")
    print(f"Greedy   : {g_series} series, {g_udon} udon")
    print(f"Lookahead: {l_series} series, {l_udon} udon")


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="HEXA UDON Bot")
    sub    = parser.add_subparsers(dest="mode", required=True)

    # sim
    sub.add_parser("sim", help="Compare Greedy and Lookahead on demo map")

    args = parser.parse_args()

    if args.mode == "sim":
        run_sim(args)


if __name__ == "__main__":
    main()
