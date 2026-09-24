"""
Self-play pool for MAPPO training.

Maintains a rolling pool of old model checkpoints used as opponents.
On each game day, the sampled opponent model generates actions, which:
  1. Produces road step counts that influence the traffic model
  2. Updates opponent agent positions visible in DayState.opponent_cells

This forces the RL agent to learn how to handle traffic caused by opponents
and to reason about opponent positions when planning routes.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import random
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import torch

import config as C
from env.hex_grid import HexGrid
from env.models import AgentState, DayState, MapData, MatchConfig
from env.simulator import HexaUdonSimulator


class SelfPlayPool:
    """
    Rolling pool of old ActorCritic checkpoints for opponent simulation.

    Usage in training loop:
        pool = SelfPlayPool()
        ...
        # Each episode: update pool if threshold reached
        pool.step(current_model)

        # Each day: if pool is non-empty, simulate opponent
        if pool.has_opponent():
            opp_model = pool.sample()
            new_opp_cells, road_steps = SelfPlayPool.simulate_day(
                opp_model, opp_agents, state, map_data, cfg, grid
            )
    """

    def __init__(self, pool_size: int = 5, update_every: int = 1000):
        if pool_size < 1 or update_every < 1:
            raise ValueError('pool_size and update_every must be positive')
        self.pool_size    = pool_size
        self.update_every = update_every
        self._pool:     List = []   # stored as CPU models
        self._episodes: int  = 0

    def step(self, model) -> bool:
        """Call once per episode. Returns True when a new checkpoint is added."""
        self._episodes += 1
        if self._episodes % self.update_every == 0:
            self._add(model)
            return True
        return False

    def _add(self, model) -> None:
        snapshot = copy.deepcopy(model).cpu()
        snapshot.eval()
        if len(self._pool) >= self.pool_size:
            self._pool.pop(0)
        self._pool.append(snapshot)
        print(f"[selfplay] Pool: {len(self._pool)}/{self.pool_size} checkpoints @ ep {self._episodes}")

    def has_opponent(self) -> bool:
        return len(self._pool) > 0

    def sample(self):
        """Return a random checkpoint from the pool."""
        return random.choice(self._pool)

    # ------------------------------------------------------------------ #
    # Opponent state construction                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_opponent_view(state: DayState, opp_agents: List[AgentState],
                            opponent_state: Optional[DayState] = None) -> DayState:
        """
        Construct a DayState from the opponent's perspective:
          - my_agents      = opponent agents (so the model sees them as "mine")
          - opponent_cells = our agents' cells
        Traffic is common; spot inventory and score are private to each team.
        """
        our_cells = [a.cell for a in state.my_agents]
        return replace(state, my_agents=copy.deepcopy(opp_agents), opponent_cells=our_cells,
                       spot_inventory=dict(opponent_state.spot_inventory) if opponent_state else dict(state.spot_inventory),
                       collected_series=set(opponent_state.collected_series) if opponent_state else set(),
                       daily_series=copy.deepcopy(opponent_state.daily_series) if opponent_state else [],
                       total_udon=opponent_state.total_udon if opponent_state else 0)

    # ------------------------------------------------------------------ #
    # Opponent simulation                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def simulate_day(
        opp_model,
        opp_agents:  List[AgentState],
        state:       DayState,
        map_data:    MapData,
        cfg:         MatchConfig,
        grid:        HexGrid,
        opponent_state: Optional[DayState] = None,
    ) -> Tuple[List[int], Dict[int, float]]:
        """
        Run the opponent model for one day and return:
          new_opp_cells : updated cell IDs for all opponent agents
          road_steps    : {cell_id: steps} of opponent road usage (for traffic)
        """
        # Reuse the exact training decoder and simulator, including waiting
        # collection and timed refueling. Import here to avoid module-init cycles.
        from rl.mappo import MAPPOTrainer
        opp_view = SelfPlayPool.build_opponent_view(state, opp_agents, opponent_state)
        patrol_ids   = [a.id for a in opp_agents if a.type == C.AGENT_PATROL]

        with torch.no_grad():
            actions, _, _, _ = opp_model.get_action_and_value(
                opp_view, map_data, cfg, patrol_ids, deterministic=True
            )

        sim = HexaUdonSimulator(cfg, map_data)
        sim.grid = grid
        orders = MAPPOTrainer._actions_to_orders(opp_view, map_data, cfg, sim, patrol_ids, actions,
                                                secondary_routes=opp_model.secondary_routes,
                                                reserve_spots=opp_model.reserve_spots)
        following, _ = sim.apply_day(opp_view, orders)
        updated = following.agents_by_id()
        for agent in opp_agents:
            agent.cell = updated[agent.id].cell
            agent.fuel = updated[agent.id].fuel
        if opponent_state is not None:
            opponent_state.spot_inventory = following.spot_inventory
            opponent_state.day = following.day
            opponent_state.collected_series = following.collected_series
            opponent_state.daily_series = following.daily_series
            opponent_state.total_udon = following.total_udon
        return [a.cell for a in opp_agents], following._road_step_counts

    # ------------------------------------------------------------------ #
    # Opponent agent generation                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def make_opponent_agents(
        map_data:  MapData,
        cfg:       MatchConfig,
        n_agents:  int,
        n_patrol:  int,
        our_cells: List[int],
        seed:      int,
    ) -> List[AgentState]:
        """BTC Q38: all teams start with the same car count and cells."""
        if n_agents != len(our_cells):
            raise ValueError("Opponent must have the same starting car count")
        cells = list(our_cells)
        fuel_max = cfg.fuel_max or 20

        agents = []
        for i, cell in enumerate(cells):
            atype = C.AGENT_PATROL if i < n_patrol else C.AGENT_SUPPLY
            fuel  = fuel_max if atype == C.AGENT_PATROL else 0
            agents.append(AgentState(id=1001 + i, type=atype, cell=cell, fuel=fuel))
        return agents

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        return {"episodes": self._episodes, "pool_size": self.pool_size,
                "update_every": self.update_every,
                "models": [model.state_dict() for model in self._pool],
                "target_masking": [model.target_masking for model in self._pool],
                "secondary_routes": [model.secondary_routes for model in self._pool],
                "reserve_spots": [model.reserve_spots for model in self._pool],
                "fuel_max": [model._fuel_max for model in self._pool]}

    def load_state_dict(self, state: dict, model) -> None:
        self._episodes = state.get('episodes', 0)
        self.pool_size = state.get('pool_size', self.pool_size)
        self.update_every = state.get('update_every', self.update_every)
        self._pool.clear()
        for index, weights in enumerate(state.get('models', [])):
            snapshot = copy.deepcopy(model).cpu()
            snapshot.load_state_dict(weights)
            snapshot.target_masking = state.get('target_masking', [False] * len(state['models']))[index]
            snapshot.secondary_routes = state.get('secondary_routes', [False] * len(state['models']))[index]
            snapshot.reserve_spots = state.get('reserve_spots', [False] * len(state['models']))[index]
            snapshot.set_fuel_max(state.get('fuel_max', [20] * len(state['models']))[index])
            snapshot.eval()
            self._pool.append(snapshot)
