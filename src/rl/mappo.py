"""
MAPPO trainer for HEXA UDON.

Episode = 1 game on a randomly generated map.
Step    = 1 day (strategic level; A* handles tactical execution).

Key design decisions:
  - CurriculumEngine controls map difficulty (8x8 -> 32x32).
    Use random maps when no curriculum is provided.
  - SelfPlayPool provides opponent checkpoints. When the pool is non-empty,
    opponent agents are simulated each day to produce realistic traffic.
  - Reward shaping: potential-based phi(s) = -POTENTIAL_SCALE * mean_min_hex_dist
    to nearest uncollected spot (Ng 1999 — provably policy-invariant).
  - TensorBoard logging when available.
  - fuel_max comes from each generated match config, shared with the baseline.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import random
import tempfile
import warnings
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config as C
from env.hex_grid import HexGrid
from env.map_generator import generate_random_scenario
from env.models import (
    AgentState, DayOrder, DayState, MapData, MatchConfig,
)
from env.simulator import HexaUdonSimulator, complete_orders
from env.scoring import compute_score, collection_potential
from pathfinding.astar import multi_waypoint_path, find_path
from rl.actor_critic import ActorCritic
from rl.curriculum import CurriculumEngine
from rl.selfplay import SelfPlayPool
from strategy.greedy import GreedyPlanner

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ------------------------------------------------------------------ #
# Rollout buffer                                                       #
# ------------------------------------------------------------------ #

@dataclass
class Transition:
    state:     DayState
    map_data:  MapData      # episode-level
    cfg:       MatchConfig  # episode-level
    actions:   List[int]
    log_probs: torch.Tensor
    value:     torch.Tensor
    reward:    float        # shaped reward
    done:      bool
    fuel_max:  Optional[int] = None  # episode context used when collecting this transition


class RolloutBuffer:
    def __init__(self):
        self.transitions: List[Transition] = []

    def add(self, t: Transition) -> None:
        self.transitions.append(t)

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)

    def compute_returns(
        self,
        gamma: float = C.GAMMA,
        lam:   float = C.GAE_LAMBDA,
    ) -> Tuple[List[float], List[float]]:
        """GAE-Lambda returns and advantages."""
        T          = len(self.transitions)
        returns    = [0.0] * T
        advantages = [0.0] * T
        gae        = 0.0
        next_val   = 0.0

        for t in reversed(range(T)):
            tr    = self.transitions[t]
            v     = tr.value.item()
            delta = tr.reward + gamma * next_val * (1 - tr.done) - v
            gae   = delta + gamma * lam * (1 - tr.done) * gae
            advantages[t] = gae
            returns[t]    = gae + v
            next_val      = v

        return returns, advantages


# ------------------------------------------------------------------ #
# Trainer                                                              #
# ------------------------------------------------------------------ #

class MAPPOTrainer:
    """
    MAPPO trainer with optional curriculum learning and self-play.

    Args:
        max_spots, max_series, max_width, max_height:
            Fixed network dimensions; maps smaller than max are padded.
        device:     "cpu" or "cuda"
        log_dir:    TensorBoard log directory (None disables logging)
        curriculum: CurriculumEngine instance (None = fully random maps)
        selfplay:   SelfPlayPool instance (None = no opponent simulation)
        seed:       Seeds weight initialization and random generators.
    """

    def __init__(
        self,
        max_spots:  int = 30,
        max_series: int = 28,
        max_width:  int = 32,
        max_height: int = 32,
        device:     str = "cpu",
        log_dir:    str = "runs/mappo",
        curriculum: Optional[CurriculumEngine] = None,
        selfplay:   Optional[SelfPlayPool]     = None,
        seed:       int = 42,
    ):
        # Seed before allocating weights, not after the model was initialized.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.seed = seed
        self.episodes_completed = 0
        self.transitions_per_step = 1
        self.max_spots  = max_spots
        self.max_series = max_series
        self.device     = torch.device(device)
        self.curriculum = curriculum
        self.selfplay   = selfplay

        self.model = ActorCritic(
            max_spots  = max_spots,
            max_series = max_series,
            max_width  = max_width,
            max_height = max_height,
        ).to(self.device)

        self.optimizer = optim.Adam([
            {"params": self.model.map_encoder.parameters(),  "lr": C.LR_ACTOR},
            {"params": self.model.agent_mlp.parameters(),    "lr": C.LR_ACTOR},
            {"params": self.model.global_mlp.parameters(),   "lr": C.LR_ACTOR},
            {"params": self.model.query_mlp.parameters(),    "lr": C.LR_ACTOR},
            {"params": self.model.key_mlp.parameters(),      "lr": C.LR_ACTOR},
            {"params": self.model.stay_head.parameters(),    "lr": C.LR_ACTOR},
            {"params": self.model.critic_head.parameters(),  "lr": C.LR_CRITIC},
        ])

        self.buffer = RolloutBuffer()

        if _TB_AVAILABLE and log_dir:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
            if log_dir and not _TB_AVAILABLE:
                print("[mappo] TensorBoard not available; install tensorboard for logging.")

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #

    def train(
        self,
        n_episodes:          int = 1000,   # additional completed episodes on resume
        seed:                Optional[int] = None,
        log_every:           int = 50,
        eval_baseline_every: int = 10,
        save_every:          int = 500,   # auto-checkpoint every N episodes
        save_path:           str = "",    # path to save to; empty = no auto-save
        games_per_update:    int = 1,     # complete games collected with frozen weights
        transitions_per_step: int = 1,   # mean gradient over this many days per optimizer step
    ) -> None:
        if n_episodes < 0 or log_every < 1 or eval_baseline_every < 1 or save_every < 0 or games_per_update < 1 or transitions_per_step < 1:
            raise ValueError('Invalid episode count or training interval')
        self.transitions_per_step = transitions_per_step
        if seed is not None:
            if self.episodes_completed and seed != self.seed:
                raise ValueError('Resume must use the checkpoint seed')
            self.seed = seed
        seed = self.seed
        self.model.train()
        ep_shaped:  List[float] = []
        ep_raw:     List[float] = []
        ep_series:  List[int]   = []
        ep_udon:    List[int]   = []

        end_episode = self.episodes_completed + n_episodes
        collected_games = 0
        checkpoint_due = False
        for ep in range(self.episodes_completed, end_episode):
            # --- Generate scenario ---
            if self.curriculum:
                cfg, map_data, agents = self.curriculum.generate_scenario(seed=seed + ep)
            else:
                cfg, map_data, agents = generate_random_scenario(seed=seed + ep)

            # --- Collect episode ---
            ep_info = self._collect_episode(cfg, map_data, agents, seed=seed + ep)
            ep_shaped.append(ep_info["shaped_return"])
            ep_raw.append(ep_info["raw_return"])
            ep_series.append(ep_info["unique_series"])
            ep_udon.append(ep_info["total_udon"])
            collected_games += 1

            # --- PPO update ---
            losses: Dict[str, float] = {}
            if len(self.buffer) > 0 and (collected_games >= games_per_update or ep + 1 == end_episode):
                losses = self._update()
                self.buffer.clear()
                collected_games = 0

            # --- Curriculum: compare vs Lookahead baseline ---
            if self.curriculum and (ep + 1) % eval_baseline_every == 0:
                import copy
                baseline = self.curriculum.evaluate_baseline(
                    cfg, map_data, copy.deepcopy(agents)
                )
                policy_score = self._evaluate_policy(cfg, map_data, agents)
                self.curriculum.record(policy_score, baseline)
                advanced = self.curriculum.try_advance()
                if advanced and self.writer:
                    self.writer.add_scalar(
                        "curriculum/level", self.curriculum.level_number, ep + 1
                    )

            # --- Self-play: update pool ---
            if self.selfplay:
                self.selfplay.step(self.model)

            # --- TensorBoard ---
            if self.writer is not None:
                g = ep + 1
                self.writer.add_scalar("episode/shaped_return", ep_info["shaped_return"], g)
                self.writer.add_scalar("episode/raw_return",    ep_info["raw_return"],    g)
                self.writer.add_scalar("episode/unique_series", ep_info["unique_series"], g)
                self.writer.add_scalar("episode/total_udon",    ep_info["total_udon"],    g)
                if self.curriculum:
                    self.writer.add_scalar("curriculum/win_rate",    self.curriculum.win_rate,    g)
                    self.writer.add_scalar("curriculum/level_number", self.curriculum.level_number, g)
                for k, v in losses.items():
                    self.writer.add_scalar(f"train/{k}", v, g)

            # --- Console ---
            if (ep + 1) % log_every == 0:
                n   = log_every
                lvl = f" lv={self.curriculum.level_number}" if self.curriculum else ""
                n_series_max = f"/{self.curriculum.level.n_series}" if self.curriculum else ""
                print(
                    f"Ep {ep+1:5d}{lvl} | "
                    f"shaped={np.mean(ep_shaped[-n:]):.1f}  "
                    f"raw={np.mean(ep_raw[-n:]):.1f}  "
                    f"series={np.mean(ep_series[-n:]):.2f}{n_series_max}  "
                    f"udon={np.mean(ep_udon[-n:]):.1f}"
                    + (f"  loss={losses.get('total', 0):.4f}" if losses else "")
                    + (f"  kl={losses.get('max_policy_kl', 0):.4f}"
                       f" accepted={losses.get('accepted_steps', 0)}"
                       f" rejected={losses.get('rejected_steps', 0)}" if losses else "")
                )

            # --- Auto-checkpoint ---
            self.episodes_completed = ep + 1
            if save_path and save_every > 0 and (ep + 1) % save_every == 0:
                checkpoint_due = True
            # Checkpoints omit rollout data: defer periodic saves until the
            # next update boundary so resume never silently discards a batch.
            if checkpoint_due and len(self.buffer) == 0:
                self.save(save_path)
                checkpoint_due = False

        if save_path:
            self.save(save_path)
        if self.writer is not None:
            self.writer.flush()

    def _evaluate_policy(self, cfg, map_data, agents):
        """Deterministic solo evaluation, matching curriculum's baseline setup."""
        cfg = replace(cfg, n_teams=1)
        sim = HexaUdonSimulator(cfg, map_data)
        state = sim.reset(agents)
        was_training = self.model.training
        previous_fuel_max = self.model._fuel_max
        self.model.set_fuel_max(cfg.fuel_max or max(
            (a.fuel for a in agents if a.is_patrol()), default=20))
        self.model.eval()
        try:
            with torch.no_grad():
                while not sim.is_done(state):
                    ids = [a.id for a in state.patrol_agents()]
                    actions, _, _, _ = self.model.get_action_and_value(
                        state, map_data, cfg, ids, deterministic=True)
                    orders = self._actions_to_orders(state, map_data, cfg, sim, ids, actions,
                                                     secondary_routes=self.model.secondary_routes,
                                                     reserve_spots=self.model.reserve_spots)
                    state, _ = sim.apply_day(state, orders)
            return compute_score(state)
        finally:
            self.model.train(was_training)
            self.model.set_fuel_max(previous_fuel_max)

    # ------------------------------------------------------------------ #
    # Episode collection                                                   #
    # ------------------------------------------------------------------ #

    def _collect_episode(
        self,
        cfg:      MatchConfig,
        map_data: MapData,
        agents:   List[AgentState],
        seed:     int = 0,
    ) -> dict:
        fuel_max = cfg.fuel_max or max((a.fuel for a in agents if a.is_patrol()), default=20)
        self.model.set_fuel_max(fuel_max)
        for a in agents:
            if a.is_patrol():
                a.fuel = fuel_max

        has_opponent = bool(self.selfplay and self.selfplay.has_opponent())
        cfg = replace(cfg, n_teams=2 if has_opponent else 1)
        sim = HexaUdonSimulator(cfg, map_data)

        # --- Set up opponent agents (self-play) ---
        opp_agents: Optional[List[AgentState]] = None
        opp_state = None
        if has_opponent:
            opp_model  = self.selfplay.sample()
            our_cells  = [a.cell for a in agents]
            n_opp      = len(agents)
            n_opp_pat  = sum(a.is_patrol() for a in agents)
            opp_agents = SelfPlayPool.make_opponent_agents(
                map_data, cfg,
                n_agents   = n_opp,
                n_patrol   = n_opp_pat,
                our_cells  = our_cells,
                seed       = seed + 9999,
            )
            for a in opp_agents:
                if a.is_patrol():
                    a.fuel = fuel_max
            opp_model.set_fuel_max(fuel_max)
            opp_state = HexaUdonSimulator(cfg, map_data).reset(opp_agents)

        # --- Initial state ---
        state     = sim.reset(agents)
        if opp_agents:
            state = replace(state, opponent_cells=[a.cell for a in opp_agents])

        raw_total    = 0.0
        shaped_total = 0.0
        phi_s        = self._compute_potential(state, map_data, sim.grid)

        while not sim.is_done(state):
            patrol_ids = [a.id for a in state.patrol_agents()]

            with torch.no_grad():
                actions, log_probs, entropy, value = self.model.get_action_and_value(
                    state, map_data, cfg, patrol_ids
                )

            orders = self._actions_to_orders(state, map_data, cfg, sim, patrol_ids, actions,
                                             secondary_routes=self.model.secondary_routes,
                                             reserve_spots=self.model.reserve_spots)

            # --- Opponent simulation (self-play) ---
            opp_road_steps: Optional[Dict[int, float]] = None
            if opp_agents and self.selfplay:
                new_opp_cells, opp_road_steps = SelfPlayPool.simulate_day(
                    opp_model, opp_agents, state, map_data, cfg, sim.grid,
                    opponent_state=opp_state,
                )
                # Update opponent agent positions
                cell_map = {a.id: c for a, c in zip(opp_agents, new_opp_cells)}
                for a in opp_agents:
                    a.cell = cell_map.get(a.id, a.cell)

            next_state, reward = sim.apply_day(
                state, orders,
                opponent_step_counts=opp_road_steps,
            )

            # Update opponent cells in next state
            if opp_agents:
                next_state = replace(
                    next_state,
                    opponent_cells=[a.cell for a in opp_agents],
                )

            done     = sim.is_done(next_state)
            phi_next = 0.0 if done else self._compute_potential(next_state, map_data, sim.grid)
            shaped_r = reward + C.GAMMA * phi_next - phi_s
            phi_s    = phi_next

            self.buffer.add(Transition(
                state     = state,
                map_data  = map_data,
                cfg       = cfg,
                actions   = actions,
                log_probs = log_probs.detach(),
                value     = value.detach(),
                reward    = shaped_r,
                done      = done,
                fuel_max  = fuel_max,
            ))

            state         = next_state
            raw_total    += reward
            shaped_total += shaped_r

        return {
            "raw_return":    raw_total,
            "shaped_return": shaped_total,
            "unique_series": len(state.collected_series),
            "total_udon":    state.total_udon,
            "score":         compute_score(state),
        }

    # ------------------------------------------------------------------ #
    # PPO update                                                           #
    # ------------------------------------------------------------------ #

    def _update(self) -> Dict[str, float]:
        returns, advantages = self.buffer.compute_returns()
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)
        ret_t = torch.tensor(returns,    dtype=torch.float32, device=self.device)

        total_loss   = 0.0
        total_actor  = 0.0
        total_critic = 0.0
        total_ent    = 0.0
        count        = 0
        rejected_steps = 0
        accepted_kl = 0.0
        reference = self._policy_snapshot()

        for _ in range(C.N_EPOCHS):
            for i, tr in enumerate(self.buffer.transitions):
                if i % self.transitions_per_step == 0:
                    chunk_size = min(self.transitions_per_step, len(self.buffer) - i)
                    chunk_totals = [0.0, 0.0, 0.0, 0.0]
                    self.optimizer.zero_grad()
                self.model.set_fuel_max(tr.fuel_max or tr.cfg.fuel_max or self.model._fuel_max)
                patrol_ids = [a.id for a in tr.state.patrol_agents()]
                _, new_log_probs, new_entropy, new_value = \
                    self.model.get_action_and_value(
                        tr.state, tr.map_data, tr.cfg, patrol_ids, actions=tr.actions
                    )

                # MAPPO: clip each agent's importance ratio for its saved action.
                ratio        = (new_log_probs - tr.log_probs).exp()
                adv          = adv_t[i]
                surr1        = ratio * adv
                surr2        = torch.clamp(ratio, 1 - C.CLIP_EPS, 1 + C.CLIP_EPS) * adv
                actor_loss   = -torch.min(surr1, surr2).mean() if patrol_ids else new_value * 0
                critic_loss  = (new_value - ret_t[i]).pow(2)
                entropy_loss = -new_entropy.mean() if patrol_ids else new_value * 0

                loss = actor_loss + 0.1 * critic_loss + C.ENTROPY_COEF * entropy_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite PPO loss')
                # Accumulate the mean gradient without keeping every day's
                # computation graph in memory (important on a 6 GB GPU).
                (loss / chunk_size).backward()
                for j, term in enumerate((loss, actor_loss, critic_loss, entropy_loss)):
                    chunk_totals[j] += term.item() / chunk_size
                if (i + 1) % self.transitions_per_step and i + 1 < len(self.buffer):
                    continue
                gradient_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError('Non-finite PPO gradient')
                weights = deepcopy(self.model.state_dict())
                optimizer_state = deepcopy(self.optimizer.state_dict())
                self.optimizer.step()

                # Actor clipping is not a hard trust-region constraint. Check
                # all rollout states, since updating one day can change others.
                max_kl = 0.0
                for old, new in zip(reference, self._policy_snapshot()):
                    if old.numel():
                        # Masked slots have zero probability and log-prob -inf.
                        kl = torch.where(old.isfinite(), old.exp() * (old - new), 0.).sum(-1)
                        max_kl = max(max_kl, float(kl.max()) if torch.isfinite(kl).all() else float('inf'))
                if max_kl > C.MAX_POLICY_KL:
                    self.model.load_state_dict(weights)
                    self.optimizer.load_state_dict(optimizer_state)
                    self.optimizer.zero_grad()
                    rejected_steps = 1
                    break
                accepted_kl = max_kl

                total_loss   += chunk_totals[0]
                total_actor  += chunk_totals[1]
                total_critic += chunk_totals[2]
                total_ent    += chunk_totals[3]
                count        += 1
            if rejected_steps:
                break

        denom = max(count, 1)
        return {
            "total":   total_loss   / denom,
            "actor":   total_actor  / denom,
            "critic":  total_critic / denom,
            "entropy": total_ent    / denom,
            "accepted_steps": count,
            "rejected_steps": rejected_steps,
            "max_policy_kl": accepted_kl,
        }

    @torch.no_grad()
    def _policy_snapshot(self):
        """Full action distributions on the fixed rollout (no sampling)."""
        result = []
        for tr in self.buffer.transitions:
            self.model.set_fuel_max(tr.fuel_max or tr.cfg.fuel_max or self.model._fuel_max)
            ids = [a.id for a in tr.state.patrol_agents()]
            _, distributions, _ = self.model.action_distributions(
                tr.state, tr.map_data, tr.cfg, ids, actions=tr.actions)
            result.append(torch.stack([d.logits for d in distributions]) if distributions
                          else torch.empty(0, device=self.device))
        return result

    # ------------------------------------------------------------------ #
    # Reward shaping                                                       #
    # ------------------------------------------------------------------ #

    def _compute_potential(
        self,
        state:    DayState,
        map_data: MapData,
        grid:     HexGrid,
    ) -> float:
        """
        phi(s) = -POTENTIAL_SCALE * mean over patrol agents of
                  min hex_distance to any spot in an uncollected series.
        Returns 0 when all series collected or no patrol agents exist.
        """
        return collection_potential(state, map_data, grid)

    # ------------------------------------------------------------------ #
    # Action -> Orders                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _actions_to_orders(
        state:      DayState,
        map_data:   MapData,
        cfg:        MatchConfig,
        sim:        HexaUdonSimulator,
        patrol_ids: List[int],
        actions:    List[int],
        secondary_routes: bool = False,
        reserve_spots: bool = False,
    ) -> List[DayOrder]:
        """Convert spot-index actions -> DayOrders via A* on each car's own daily timeline."""
        orders:      List[DayOrder] = []
        terrain      = {c.id: c.terrain for c in map_data.cells}
        agents_by_id = state.agents_by_id()
        n_spots      = len(map_data.spots)
        remaining_inventory = dict(state.spot_inventory)
        if secondary_routes:
            from strategy.lookahead import LookaheadPlanner
            lookahead = LookaheadPlanner(cfg, map_data, sim)

        for aid, act in zip(patrol_ids, actions):
            agent = agents_by_id[aid]
            if act >= n_spots:
                path_actions = []
            else:
                target_cell  = map_data.spots[act].cell_id
                waypoints = [target_cell]
                if secondary_routes:
                    planning_state = replace(state, spot_inventory=remaining_inventory) if reserve_spots else state
                    waypoints.extend(lookahead._secondary_waypoints(agent, target_cell, planning_state, {target_cell}))
                path_actions = multi_waypoint_path(
                    sim.grid, terrain, state.traffic,
                    agent.cell, waypoints,
                    step_budget = state.steps_left,
                    fuel_budget = agent.fuel if agent.is_patrol() else None,
                )
            orders.append(DayOrder(agent_id=aid, actions=path_actions))
            if secondary_routes and reserve_spots:
                # Reserve actual pickups along the reachable route, not every
                # proposed waypoint. This is a planning heuristic, not a rule change.
                from env.simulator import execute_timeline
                solo = replace(state, my_agents=[agent], spot_inventory=remaining_inventory)
                trace = []
                execute_timeline(complete_orders([orders[-1]], solo, map_data, sim.grid),
                                 solo, map_data, sim.grid, cfg.fuel_max, trace=trace)
                for tick in trace:
                    for _, cell in tick['collected']:
                        remaining_inventory[cell] -= 1

        greedy = GreedyPlanner(cfg, map_data, sim)
        for agent in state.supply_agents():
            target = greedy._supply_target(agent, state)
            if target is not None:
                result = find_path(
                    sim.grid, terrain, state.traffic,
                    agent.cell, target, step_budget=state.steps_left,
                )
                supply_actions = result.actions if result.reachable else []
            else:
                supply_actions = []
            orders.append(DayOrder(agent_id=agent.id, actions=supply_actions))

        return complete_orders(orders, state, map_data, sim.grid)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        numpy_state = np.random.get_state()
        payload: dict = {
            "format_version": 3,
            "simulation_rules": "btc_even_r_v2",
            "training_scenarios": "btc_ranges_same_start_v1",
            "training_algorithm": "critic_head_only_positions_v2",
            "target_masking": self.model.target_masking,
            "secondary_routes": self.model.secondary_routes,
            "reserve_spots": self.model.reserve_spots,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "max_spots":  self.max_spots,
            "max_series": self.max_series,
            "max_width": self.model.max_width,
            "max_height": self.model.max_height,
            "fuel_max": self.model._fuel_max,
            "episodes_completed": self.episodes_completed,
            "seed": self.seed,
            "python_rng": random.getstate(),
            # Store numpy's array as plain values for weights_only=True loading.
            "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        if self.curriculum:
            payload["curriculum"] = self.curriculum.state_dict()
        if self.selfplay:
            payload["selfplay"] = self.selfplay.state_dict()
        destination = os.path.abspath(path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', dir=os.path.dirname(destination),
                                             prefix='.checkpoint-', suffix='.tmp', delete=False) as stream:
                temporary = stream.name
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        print(f"[mappo] Model saved -> {path}")

    def load(self, path: str) -> None:
        # CPU loading also keeps the saved CPU RNG tensor on the right device.
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        # Reconstruct checkpoint dimensions rather than silently truncating series
        # or forcing legacy 10-series weights into the new 28-series model.
        hidden = ckpt['model']['global_mlp.0.weight'].shape[0]
        series = ckpt.get('max_series', ckpt['model']['global_mlp.0.weight'].shape[1] - hidden//2 - 3)
        if series != self.max_series:
            restored = MAPPOTrainer(max_series=series,
                max_spots=ckpt.get('max_spots', self.max_spots),
                max_width=ckpt.get('max_width', self.model.max_width),
                max_height=ckpt.get('max_height', self.model.max_height),
                device=str(self.device), log_dir=None, seed=self.seed)
            self.model, self.optimizer = restored.model, restored.optimizer
            self.max_series = series
        self.max_spots = ckpt.get('max_spots', self.max_spots)
        if ckpt.get("simulation_rules") != "btc_even_r_v2":
            warnings.warn("Checkpoint predates the official timeline/grid; old training scores are not comparable", RuntimeWarning)
        self.model.load_state_dict(ckpt["model"])
        self.model.target_masking = ckpt.get('target_masking', False)
        self.model.secondary_routes = ckpt.get('secondary_routes', False)
        self.model.reserve_spots = ckpt.get('reserve_spots', False)
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if ckpt.get('training_algorithm') != 'critic_head_only_positions_v2':
                # Legacy moments use the old gradient partition and/or lack
                # position columns. Migrate once; new checkpoints resume fully.
                self.optimizer.state.clear()
                warnings.warn('Migrated legacy training checkpoint: reset optimizer moments once; '
                              'weights, episode, RNG and self-play are preserved', RuntimeWarning)
        else:
            warnings.warn('Legacy checkpoint: weights loaded; optimizer/RNG progress unavailable',
                          RuntimeWarning)
        self.model.max_width = ckpt.get('max_width', self.model.max_width)
        self.model.max_height = ckpt.get('max_height', self.model.max_height)
        self.model.set_fuel_max(ckpt.get('fuel_max', 20))
        self.episodes_completed = ckpt.get('episodes_completed', 0)
        self.seed = ckpt.get('seed', self.seed)
        self.buffer.clear()
        if self.curriculum and "curriculum" in ckpt:
            self.curriculum.load_state_dict(ckpt["curriculum"])
        if self.selfplay and 'selfplay' in ckpt:
            self.selfplay.load_state_dict(ckpt['selfplay'], self.model)
        # Restore RNG after reconstructing opponents, which may allocate weights.
        if 'python_rng' in ckpt:
            random.setstate(ckpt['python_rng'])
            ns = ckpt['numpy_rng']
            np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
            torch.set_rng_state(ckpt['torch_rng'])
            if torch.cuda.is_available() and ckpt.get('cuda_rng'):
                torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        print(f"[mappo] Model loaded <- {path}")
