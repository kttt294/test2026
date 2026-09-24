"""
Game simulator for HEXA UDON.

Cost convention (based on problem statement):
  - Step cost = terrain of SOURCE cell (where agent currently stands).
  - Fuel cost = same (terrain of SOURCE cell).
  - Cannot move INTO lake (terrain=2).
  - Traffic status affects road cost at source.

Traffic model:
  traffic(cell) = sum of steps our agents + opponent agents spent at that road cell
                  over the past 2 days, divided by n_teams.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import config as C
from env.hex_grid import HexGrid
from env.models import (
    AgentAction, AgentState, CMD_MOVE, CMD_STAY,
    DayOrder, DayState, MapData, MatchConfig, Spot,
)


def _schedule_orders(orders, state, map_data, grid, *, pad=False):
    """Schedule arrivals on each car's own timeline; STAY lasts one step.

    Planners may explicitly pad short routes. Execution/validation never do so.
    """
    agents = state.agents_by_id()
    by_id, events, completed = {}, {}, []
    for order in orders:
        if order.agent_id not in agents:
            raise ValueError(f"agent {order.agent_id}: not found in state")
        if order.agent_id in by_id:
            raise ValueError(f"agent {order.agent_id}: duplicate order")
        by_id[order.agent_id] = order
    for aid, agent in agents.items():
        if aid not in by_id and not pad:
            raise ValueError(f"agent {aid}: missing order")
        actions = list(by_id[aid].actions) if aid in by_id else []
        cell, elapsed = agent.cell, 0
        for action in actions:
            if action.cmd == CMD_STAY:
                if action.direction is not None:
                    raise ValueError(f"agent {aid}: stay cannot have a direction")
                elapsed += 1
            elif action.cmd == CMD_MOVE:
                if type(action.direction) is not int or action.direction not in range(6):
                    raise ValueError(f"agent {aid}: invalid direction {action.direction!r}")
                dst = grid.neighbor_in_dir(cell, action.direction)
                if dst is None:
                    raise ValueError(f"agent {aid}: moving off map from {cell}")
                if map_data.cell_map[dst].terrain == C.TERRAIN_LAKE:
                    raise ValueError(f"agent {aid}: moving into lake at {dst}")
                terrain = map_data.cell_map[cell].terrain
                cost = C.STEP_COST[terrain]
                elapsed += cost[state.traffic.get(cell, C.TRAFFIC_CLEAR)] if isinstance(cost, dict) else cost
                events.setdefault(elapsed, []).append((aid, dst, C.FUEL_COST.get(terrain, 0)))
                cell = dst
            else:
                raise ValueError(f"agent {aid}: unknown command {action.cmd!r}")
            if elapsed > state.steps_left:
                raise ValueError(f"agent {aid}: step budget exceeded ({elapsed} > {state.steps_left})")
        if pad:
            actions.extend(AgentAction(CMD_STAY) for _ in range(state.steps_left - elapsed))
        elif elapsed != state.steps_left:
            raise ValueError(f"agent {aid}: step budget incomplete ({elapsed} != {state.steps_left})")
        completed.append(DayOrder(aid, actions))
    return completed, events


def complete_orders(orders, state, map_data, grid):
    """Append explicit one-step waits for every unfinished/missing car route."""
    return _schedule_orders(orders, state, map_data, grid, pad=True)[0]


def execute_timeline(orders, state, map_data, grid, fuel_max=None, trace=None):
    """BTC Q6: consume fuel, move all cars, collect by ID, refill, count occupancy.

    Pure with respect to state/traffic: an invalid plan raises before committing.
    Cars remain at the source until the scheduled arrival step.
    """
    _, events = _schedule_orders(orders, state, map_data, grid)
    agents = {a.id: deepcopy(a) for a in state.my_agents}
    inventory = dict(state.spot_inventory)
    visited, series, road_steps = set(), set(), {}
    udon = 0
    snapshots = []
    capacity = fuel_max if fuel_max is not None else state.fuel_max
    for step in range(1, state.steps_left + 1):
        for aid, dst, cost in events.get(step, []):
            agent = agents[aid]
            if agent.is_patrol():
                if agent.fuel < cost:
                    raise ValueError(f"agent {aid} step {step}: fuel exhausted ({agent.fuel} < {cost})")
                agent.fuel -= cost
            agent.cell = dst
        collected = []
        for aid in sorted(agents):
            agent = agents[aid]
            key = (aid, agent.cell)
            if agent.is_patrol() and key not in visited and inventory.get(agent.cell, 0) > 0:
                spot = map_data.spot_map[agent.cell]
                inventory[agent.cell] -= 1
                visited.add(key)
                series.add(spot.series_id)
                udon += 1
                collected.append(key)
        supply_cells = {a.cell for a in agents.values() if a.is_supply()}
        for agent in agents.values():
            if agent.is_patrol() and agent.cell in supply_cells:
                if capacity is None:
                    raise ValueError("fuel_max required to simulate refueling")
                agent.fuel = capacity
            if map_data.cell_map[agent.cell].terrain == C.TERRAIN_ROAD:
                road_steps[agent.cell] = road_steps.get(agent.cell, 0) + 1
        if trace is not None:
            snapshots.append(dict(step=step, cells=[agents[a.id].cell for a in state.my_agents],
                                  fuel=[agents[a.id].fuel for a in state.my_agents],
                                  collected=collected, road_steps=dict(road_steps)))
    if trace is not None:
        trace.extend(snapshots)
    return list(agents.values()), series, udon, road_steps


class TrafficModel:
    """Tracks road step counts and computes daily traffic status."""

    def __init__(self, n_teams: int, thr_busy: float, thr_congested: float):
        self.n_teams       = n_teams
        self.thr_busy      = thr_busy
        self.thr_congested = thr_congested
        # history[i] = {cell_id: total_steps_all_teams} for day (i+1)
        self._history: List[Dict[int, float]] = []

    def record_day(self, all_team_steps: Dict[int, float]) -> None:
        """Call at end of each day with aggregated step counts."""
        self._history.append(dict(all_team_steps))

    def compute_status(self, for_day: int) -> Dict[int, int]:
        """Return traffic dict {cell_id: status} for the given day number."""
        if for_day <= 1:
            return {}
        past = self._history[max(0, for_day - 3) : for_day - 1]  # up to 2 days
        total: Dict[int, float] = {}
        for day_counts in past:
            for cid, steps in day_counts.items():
                total[cid] = total.get(cid, 0.0) + steps
        result: Dict[int, int] = {}
        for cid, s in total.items():
            val = s / self.n_teams
            if val >= self.thr_congested:
                result[cid] = C.TRAFFIC_CONGESTED
            elif val >= self.thr_busy:
                result[cid] = C.TRAFFIC_BUSY
        return result

    def reset(self) -> None:
        self._history.clear()


def apply_joint_day(simulators, states, orders):
    """Execute teams on the same day, with private stock/fuel and common traffic.

    All plans are decided from the old states and checked before any history is
    committed. Teams interact through road occupancy, never through private stock.
    """
    count = len(simulators)
    if not count or len(states) != count or len(orders) != count:
        raise ValueError('One simulator, state and plan required per team')
    if any(sim.cfg.n_teams != count for sim in simulators):
        raise ValueError('n_teams must equal the actual number of teams')
    if any(s.day != states[0].day or s.traffic != states[0].traffic for s in states):
        raise ValueError('Teams must share the same day and traffic')
    own = [execute_timeline(plan, state, sim.map, sim.grid, sim.cfg.fuel_max)[3]
           for sim, state, plan in zip(simulators, states, orders)]
    total = {}
    for counts in own:
        for cell, steps in counts.items():
            total[cell] = total.get(cell, 0) + steps
    following, rewards = [], []
    for sim, state, plan, counts in zip(simulators, states, orders, own):
        opponents = {cell: steps - counts.get(cell, 0) for cell, steps in total.items()}
        new_state, reward = sim.apply_day(state, plan, opponents)
        following.append(new_state)
        rewards.append(reward)
    for i, state in enumerate(following):
        state.opponent_cells = [a.cell for j, other in enumerate(following) if j != i for a in other.my_agents]
    return following, rewards


class HexaUdonSimulator:
    """
    Simulates one full game of HEXA UDON.

    Usage:
        sim = HexaUdonSimulator(cfg, map_data)
        state = sim.reset(initial_agents)
        while not sim.is_done(state):
            orders = your_strategy(state)
            state, reward = sim.apply_day(state, orders)
    """

    def __init__(self, cfg: MatchConfig, map_data: MapData):
        self.cfg      = cfg
        self.map      = map_data
        self.grid     = HexGrid(cfg.width, cfg.height)
        self.traffic  = TrafficModel(
            cfg.n_teams,
            cfg.traffic_threshold_busy,
            cfg.traffic_threshold_congested,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def reset(self, initial_agents: List[AgentState], time_limit_ms: int = 5000) -> DayState:
        self.traffic.reset()
        if self.cfg.fuel_max is None:
            self.cfg.infer_fuel_max(initial_agents)
        return DayState(
            day=1,
            steps_left=self.cfg.steps_per_day[0],
            time_limit_ms=time_limit_ms,
            traffic={},
            spot_inventory={s.cell_id: s.max_inventory for s in self.map.spots},
            my_agents=deepcopy(initial_agents),
            opponent_cells=[],
            collected_series=set(),
            daily_series=[],
            total_udon=0,
            fuel_max=self.cfg.fuel_max,
        )

    def apply_day(
        self,
        state: DayState,
        orders: List[DayOrder],
        opponent_step_counts: Optional[Dict[int, float]] = None,
        *, trace: Optional[List[dict]] = None,
    ) -> Tuple[DayState, float]:
        """
        Execute one full day's orders and return (next_state, reward).

        opponent_step_counts: {road_cell_id: steps} from opponent agents this day.
        If None, opponents are assumed idle (optimistic; update once we have data).
        """
        agents, today_series, udon_gained, road_steps = execute_timeline(
            orders, state, self.map, self.grid, self.cfg.fuel_max, trace)

        # --- aggregate traffic (our team + opponents) ---
        combined_road_steps: Dict[int, float] = dict(road_steps)
        if opponent_step_counts:
            for cid, s in opponent_step_counts.items():
                combined_road_steps[cid] = combined_road_steps.get(cid, 0.0) + s
        self.traffic.record_day(combined_road_steps)

        # --- build next state ---
        next_day = state.day + 1
        new_collected = set(state.collected_series) | today_series
        new_series_gained = today_series - state.collected_series

        next_state = DayState(
            day=next_day,
            steps_left=(
                self.cfg.steps_per_day[next_day - 1]
                if next_day <= self.cfg.total_days else 0
            ),
            time_limit_ms=state.time_limit_ms,
            traffic=self.traffic.compute_status(next_day),
            # Spots refill to max at start of each new day
            spot_inventory={s.cell_id: s.max_inventory for s in self.map.spots},
            my_agents=agents,
            opponent_cells=[],
            collected_series=new_collected,
            daily_series=state.daily_series + [today_series],
            total_udon=state.total_udon + udon_gained,
            _road_step_counts=road_steps,
            fuel_max=self.cfg.fuel_max,
        )

        # Waiting can collect/refuel; unused movement is not intrinsically wasteful.
        reward = self._reward(new_series_gained, today_series, udon_gained, 0, 0, 0)
        return next_state, reward

    def is_done(self, state: DayState) -> bool:
        return state.day > self.cfg.total_days

    # ------------------------------------------------------------------ #
    # Costs                                                                #
    # ------------------------------------------------------------------ #

    def _step_cost(self, cell_id: int, traffic: Dict[int, int]) -> int:
        terrain = self.map.cell_map[cell_id].terrain
        if terrain == C.TERRAIN_ROAD:
            status = traffic.get(cell_id, C.TRAFFIC_CLEAR)
            return C.STEP_COST[C.TERRAIN_ROAD][status]
        return C.STEP_COST[terrain]

    def _fuel_cost(self, cell_id: int) -> int:
        terrain = self.map.cell_map[cell_id].terrain
        return C.FUEL_COST.get(terrain, 0)

    def step_cost_of(self, cell_id: int, traffic: Dict[int, int]) -> Optional[int]:
        """Public helper for pathfinder: returns cost or None if impassable."""
        terrain = self.map.cell_map[cell_id].terrain
        if terrain == C.TERRAIN_LAKE:
            return None
        return self._step_cost(cell_id, traffic)

    # ------------------------------------------------------------------ #
    # Reward                                                               #
    # ------------------------------------------------------------------ #

    def _reward(
        self,
        new_series: Set[int],
        daily_series: Set[int],
        udon: int,
        fuel_depleted: int,
        steps_used: int,
        steps_budget: int,
    ) -> float:
        r  = C.RW_NEW_SERIES   * len(new_series)
        r += C.RW_DAILY_SERIES * len(daily_series)
        r += C.RW_UDON         * udon
        r += C.RW_FUEL_EMPTY   * fuel_depleted
        wasted = max(0, steps_budget - steps_used)
        r += C.RW_WASTED_STEP  * wasted
        return r
