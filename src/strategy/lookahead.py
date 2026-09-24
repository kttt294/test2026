"""
Look-ahead planner with terminal-score evaluation of daily route candidates.

Compare a greedy day and a multi-spot day using greedy continuation to the
end of the match. This accounts for fuel spent today reducing future daily
collections. Rollouts use private traffic history and never mutate the game.

Improvements over GreedyPlanner:
  1. Global series assignment  — no two patrols waste steps on the same uncollected
                                  series when others remain uncovered.
  2. Multi-spot routing        — each patrol visits as many spots as fuel/steps allow
                                  in one day, not just one.
  3. Per-car timeline         — every car receives the full day duration.
  4. Supply intercept (push)   — supply car predicts which patrol will be fuel-starved
                                  mid-route and moves to intercept proactively.
  5. Hybrid fuel fallback      — if supply is predicted to arrive too late, patrol
                                  adjusts its own route to meet supply halfway,
                                  trading a detour for guaranteed fuel continuity.
  6. End-of-day repositioning  — after primary targets, drift toward tomorrow's target
                                  so next day starts closer to the action.
"""
from __future__ import annotations

import heapq
import sys
import os
import time
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, List, Optional, Set, Tuple

import config as C
from env.models import AgentState, DayOrder, DayState, MapData, MatchConfig, Spot
from env.simulator import HexaUdonSimulator, complete_orders
from env.scoring import compute_score
from pathfinding.astar import find_path, multi_waypoint_path, PathResult
from strategy.planner import BasePlanner
from strategy.greedy import GreedyPlanner


class LookaheadPlanner(BasePlanner):
    MAX_SECONDARY = 5

    def __init__(self, cfg: MatchConfig, map_data: MapData, sim: HexaUdonSimulator):
        super().__init__(cfg, map_data, sim)
        self._terrain = self.terrain_dict()

        # Pre-group spots by series for fast lookup
        self._spots_by_series: Dict[int, List[Spot]] = {}
        for s in map_data.spots:
            self._spots_by_series.setdefault(s.series_id, []).append(s)

    # ------------------------------------------------------------------ #
    # Main entry                                                           #
    # ------------------------------------------------------------------ #

    def plan(self, state: DayState) -> List[DayOrder]:
        # Keep at least half the caller's allowance for submission/fallback.
        # The deadline is cooperative: an individual A* call is not preempted.
        deadline = time.monotonic() + min(0.5, max(0, state.time_limit_ms) / 2000)
        greedy = GreedyPlanner(self.cfg, self.map, self.sim)
        best_orders = greedy.plan(state)
        best_score = None
        for use_greedy in (True, False):
            if time.monotonic() >= deadline:
                break
            first_orders = best_orders if use_greedy else self._plan_routes(state)
            orders = first_orders
            rollout = HexaUdonSimulator(self.cfg, self.map)
            rollout.traffic = deepcopy(self.sim.traffic)
            continuation = GreedyPlanner(self.cfg, self.map, rollout)
            future = state
            while not rollout.is_done(future):
                if time.monotonic() >= deadline:
                    return best_orders
                future, _ = rollout.apply_day(future, orders)
                if not rollout.is_done(future):
                    orders = continuation.plan(future)
            score = compute_score(future)
            if best_score is None:
                best_score = score
            elif score > best_score:
                best_orders = first_orders
                best_score = score
        return best_orders

    def _plan_routes(self, state: DayState) -> List[DayOrder]:
        patrols  = state.patrol_agents()
        supplies = state.supply_agents()

        # Step 1 — global series assignment
        assignments: Dict[int, List[int]] = self._assign_series(patrols, state)

        # Step 2 — allocate base step budget (used as base limit)
        base_budgets: Dict[int, int] = self._allocate_budget(
            patrols, supplies, assignments, state
        )

        # Step 3 — supply intercept: supply moves toward predicted fuel-starved patrol
        supply_targets: Dict[int, Optional[int]] = self._compute_supply_targets(
            supplies, patrols, assignments, state
        )

        # Step 4 — hybrid fallback: if supply can't reach patrol in time, patrol
        #           adjusts its route to converge toward the supply car instead.
        assignments = self._apply_hybrid_fallback(
            patrols, supplies, assignments, supply_targets, base_budgets, state
        )

        # Step 5 — Plan each route within its own daily timeline
        orders: List[DayOrder] = []

        # Sort agents by priority:
        # 1. Patrols targeting uncollected series
        # 2. Other patrols
        # 3. Supply agents
        uncollected = set(self.map.series_ids) - state.collected_series
        
        def agent_priority(a: AgentState) -> float:
            if a.is_patrol():
                wps = assignments.get(a.id, [])
                if not wps:
                    return 1.0
                first = self.map.spot_map.get(wps[0])
                return 100.0 if (first and first.series_id in uncollected) else 10.0
            else:
                return 0.0

        sorted_agents = sorted(state.my_agents, key=agent_priority, reverse=True)
        orders_by_id: Dict[int, DayOrder] = {}

        for agent in sorted_agents:
            act_budget = state.steps_left

            if agent.is_patrol():
                waypoints = assignments.get(agent.id, [])
                if not waypoints:
                    actions = []
                else:
                    actions = multi_waypoint_path(
                        self.grid,
                        self._terrain,
                        state.traffic,
                        agent.cell,
                        waypoints,
                        step_budget=act_budget,
                        fuel_budget=agent.fuel,
                    )
                    # End-of-day repositioning
                    actions = self._append_reposition(agent, waypoints, actions, act_budget, state)
            else:
                # Supply agent
                target = supply_targets.get(agent.id)
                if target is None:
                    actions = []
                else:
                    result = find_path(
                        self.grid, self._terrain, state.traffic,
                        agent.cell, target,
                        step_budget=act_budget,
                        fuel_budget=None,
                    )
                    actions = result.actions if result.reachable else []

            orders_by_id[agent.id] = DayOrder(agent_id=agent.id, actions=actions)

        # Re-assemble orders in the order of state.my_agents for consistency
        return complete_orders(list(orders_by_id.values()), state, self.map, self.grid)

    # ------------------------------------------------------------------ #
    # Step 1 — Global series assignment                                    #
    # ------------------------------------------------------------------ #

    def _assign_series(
        self,
        patrols: List[AgentState],
        state: DayState,
    ) -> Dict[int, List[int]]:
        """
        Assign each patrol to an ordered list of spot waypoints for the day.

        Priority rule:
          - A series not yet collected anywhere (uncollected) scores 100x.
          - Two patrols are never assigned to the same uncollected series unless
            all uncollected series are already covered (then overlap is allowed).
          - Each patrol may receive multiple secondary waypoints if capacity allows.
        """
        uncollected = set(self.map.series_ids) - state.collected_series

        # Build score heap: (-score, patrol_id, series_id, nearest_spot_cell)
        heap: List[Tuple] = []
        for patrol in patrols:
            for series_id, spots in self._spots_by_series.items():
                available = [
                    s for s in spots
                    if state.spot_inventory.get(s.cell_id, 0) > 0
                ]
                if not available:
                    continue

                nearest = min(available, key=lambda s: self.grid.hex_distance(patrol.cell, s.cell_id))
                dist    = self.grid.hex_distance(patrol.cell, nearest.cell_id)

                value   = 100.0 if series_id in uncollected else 5.0
                # Penalise if rough fuel estimate suggests it's barely reachable
                rough_fuel_needed = dist * C.FUEL_COST.get(C.TERRAIN_PLAIN, 1)
                if patrol.fuel > 0 and rough_fuel_needed > patrol.fuel:
                    value *= 0.3

                score = value / (dist + 1)
                heapq.heappush(heap, (-score, patrol.id, series_id, nearest.cell_id))

        assigned_patrols:  Set[int] = set()
        assigned_series:   Set[int] = set()   # uncollected series already claimed
        primary: Dict[int, int] = {}          # patrol_id -> primary spot_cell

        # --- primary assignment ---
        while heap and len(assigned_patrols) < len(patrols):
            _, pid, sid, spot_cell = heapq.heappop(heap)

            if pid in assigned_patrols:
                continue

            # Don't assign two patrols to the same uncollected series when other
            # uncollected series still have no owner.
            if (sid in uncollected
                    and sid in assigned_series
                    and len(uncollected - assigned_series) > 0):
                continue

            primary[pid] = spot_cell
            assigned_patrols.add(pid)
            if sid in uncollected:
                assigned_series.add(sid)

        # Patrols with no primary (all series exhausted or infeasible) → nearest any spot
        for patrol in patrols:
            if patrol.id not in primary:
                fallback = self._nearest_available_spot(patrol.cell, set(), state)
                if fallback is not None:
                    primary[patrol.id] = fallback

        # --- secondary waypoints ---
        # After reaching the primary, each patrol may visit more spots.
        result: Dict[int, List[int]] = {}
        for patrol in patrols:
            wp_cells = []
            if patrol.id in primary:
                wp_cells.append(primary[patrol.id])
                wp_cells.extend(
                    self._secondary_waypoints(patrol, wp_cells[-1], state, set(wp_cells))
                )
            result[patrol.id] = wp_cells

        return result

    def _secondary_waypoints(
        self,
        patrol:  AgentState,
        from_cell: int,
        state:   DayState,
        visited: Set[int],
    ) -> List[int]:
        """
        Greedily pick up to MAX_SECONDARY additional spots reachable after
        the primary target, preferring uncollected series.
        """
        uncollected = set(self.map.series_ids) - state.collected_series
        claimed_series: Set[int] = set()

        secondaries: List[int] = []
        cur = from_cell

        for _ in range(self.MAX_SECONDARY):
            best_score = -1.0
            best_cell  = None

            for spot in self.map.spots:
                if spot.cell_id in visited:
                    continue
                if state.spot_inventory.get(spot.cell_id, 0) == 0:
                    continue

                dist  = self.grid.hex_distance(cur, spot.cell_id)
                value = 60.0 if spot.series_id in uncollected and spot.series_id not in claimed_series else 3.0
                score = value / (dist + 1)

                if score > best_score:
                    best_score = score
                    best_cell  = spot.cell_id
                    best_series = spot.series_id

            if best_cell is None:
                break

            secondaries.append(best_cell)
            visited.add(best_cell)
            claimed_series.add(best_series)
            cur = best_cell

        return secondaries

    def _nearest_available_spot(
        self,
        from_cell: int,
        exclude:   Set[int],
        state:     DayState,
    ) -> Optional[int]:
        candidates = [
            s for s in self.map.spots
            if s.cell_id not in exclude
            and state.spot_inventory.get(s.cell_id, 0) > 0
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda s: self.grid.hex_distance(from_cell, s.cell_id)).cell_id

    # ------------------------------------------------------------------ #
    # Step 2 — Budget allocation                                           #
    # ------------------------------------------------------------------ #

    def _allocate_budget(
        self,
        patrols:     List[AgentState],
        supplies:    List[AgentState],
        assignments: Dict[int, List[int]],
        state:       DayState,
    ) -> Dict[int, int]:
        """Every car has the full day, independent of other cars' routes."""
        return {a.id: state.steps_left for a in patrols + supplies}

    # ------------------------------------------------------------------ #
    # Repositioning & Helper                                               #
    # ------------------------------------------------------------------ #

    def _append_reposition(
        self,
        patrol:     AgentState,
        waypoints:  List[int],
        actions:    List,
        budget:     int,
        state:      DayState,
    ) -> List:
        """
        After primary route, consume leftover budget by moving toward
        the next day's most valuable spot (end-of-day repositioning).
        """
        if not actions:
            return actions

        # Calculate actual steps used so far and find the correct final cell
        steps_used = 0
        fuel_used = 0
        cur_cell = patrol.cell
        for act in actions:
            if act.cmd == "stay":
                steps_used += 1
            if act.cmd == "move":
                steps_used += self.sim._step_cost(cur_cell, state.traffic)
                fuel_used += self.sim._fuel_cost(cur_cell)
                dst = self.grid.neighbor_in_dir(cur_cell, act.direction)
                if dst is not None:
                    cur_cell = dst
        end_cell = cur_cell

        remaining = budget - steps_used
        if remaining < 2:
            return actions

        # Find next best spot not already in route
        already = set(waypoints)
        uncollected = set(self.map.series_ids) - state.collected_series
        next_target = None
        best_val    = -1.0

        for spot in self.map.spots:
            if spot.cell_id in already:
                continue
            if state.spot_inventory.get(spot.cell_id, 0) == 0:
                continue
            dist  = self.grid.hex_distance(end_cell, spot.cell_id)
            value = 50.0 if spot.series_id in uncollected else 2.0
            score = value / (dist + 1)
            if score > best_val:
                best_val    = score
                next_target = spot.cell_id

        if next_target is None:
            return actions

        reposition_result = find_path(
            self.grid,
            self._terrain,
            state.traffic,
            end_cell,
            next_target,
            step_budget=remaining,
            fuel_budget=max(0, patrol.fuel - fuel_used),
        )

        if reposition_result.reachable:
            return actions + reposition_result.actions
        return actions

    # ------------------------------------------------------------------ #
    # Step 3b — Supply intercept targets                                   #
    # ------------------------------------------------------------------ #

    def _compute_supply_targets(
        self,
        supplies:    List[AgentState],
        patrols:     List[AgentState],
        assignments: Dict[int, List[int]],
        state:       DayState,
    ) -> Dict[int, Optional[int]]:
        """
        For each supply car, compute the intercept cell (where to head today).
        Returns {supply_id: target_cell or None}.
        """
        fuel_max = self.cfg.fuel_max or 20
        targets: Dict[int, Optional[int]] = {}
        for supply in supplies:
            targets[supply.id] = self._best_intercept_cell(
                supply, patrols, assignments, fuel_max, state
            )
        return targets

    def _best_intercept_cell(
        self,
        supply:      AgentState,
        patrols:     List[AgentState],
        assignments: Dict[int, List[int]],
        fuel_max:    int,
        state:       DayState,
    ) -> Optional[int]:
        """
        Find the waypoint where the most fuel-needy patrol will hit LOW_FUEL,
        weighted by how urgent the need is vs how far the supply car is.
        """
        LOW_FUEL      = int(fuel_max * 0.35)
        best_target   = None
        best_priority = -1.0

        for patrol in patrols:
            wps = assignments.get(patrol.id, [])
            if not wps:
                continue
            sim_fuel, cur = patrol.fuel, patrol.cell
            for wp in wps:
                hops      = self.grid.hex_distance(cur, wp)
                sim_fuel -= hops * C.FUEL_COST.get(C.TERRAIN_PLAIN, 1)
                cur        = wp
                if sim_fuel <= LOW_FUEL:
                    dist_supply = self.grid.hex_distance(supply.cell, wp)
                    urgency     = (fuel_max - sim_fuel) / (dist_supply + 1)
                    if urgency > best_priority:
                        best_priority = urgency
                        best_target   = wp
                    break

        return best_target

    # ------------------------------------------------------------------ #
    # Step 4 — Hybrid fallback: patrol adjusts route if supply too far    #
    # ------------------------------------------------------------------ #

    def _apply_hybrid_fallback(
        self,
        patrols:        List[AgentState],
        supplies:       List[AgentState],
        assignments:    Dict[int, List[int]],
        supply_targets: Dict[int, Optional[int]],
        budgets:        Dict[int, int],
        state:          DayState,
    ) -> Dict[int, List[int]]:
        """
        For each patrol predicted to run low on fuel, check whether any supply
        car can realistically reach it in time.

        If NOT → insert a "rendezvous" waypoint into the patrol's route so it
        detours toward the supply car before fuel runs out.

        The rendezvous point is chosen as the midpoint of the path between
        patrol's predicted low-fuel position and the supply car's current cell,
        minimising total detour cost.
        """
        if not supplies:
            return assignments

        fuel_max  = self.cfg.fuel_max or 20
        LOW_FUEL  = int(fuel_max * 0.35)
        CRIT_FUEL = int(fuel_max * 0.15)   # if below this, detour is mandatory

        new_assignments = {pid: list(wps) for pid, wps in assignments.items()}

        for patrol in patrols:
            wps = new_assignments[patrol.id]
            if not wps:
                continue

            # Predict fuel level at each waypoint
            sim_fuel, cur = patrol.fuel, patrol.cell
            crisis_cell: Optional[int] = None
            for wp in wps:
                hops      = self.grid.hex_distance(cur, wp)
                sim_fuel -= hops * C.FUEL_COST.get(C.TERRAIN_PLAIN, 1)
                cur        = wp
                if sim_fuel <= LOW_FUEL:
                    crisis_cell = wp
                    break

            if crisis_cell is None:
                continue   # patrol has enough fuel, no action needed

            # Check if any supply can reach crisis_cell in time
            supply_can_help = any(
                self.grid.hex_distance(s.cell, crisis_cell) <= budgets.get(s.id, 0)
                for s in supplies
            )

            if supply_can_help:
                continue   # supply is coming, patrol doesn't need to detour

            # Supply cannot arrive in time → find nearest supply car and insert
            # a rendezvous waypoint BEFORE the crisis point in patrol's route.
            nearest_supply = min(
                supplies,
                key=lambda s: self.grid.hex_distance(patrol.cell, s.cell),
            )

            # Rendezvous = midpoint between patrol current cell and supply cell.
            # We approximate by finding the cell along the patrol→supply path
            # that is reachable within remaining fuel (CRIT_FUEL buffer).
            rendezvous = self._rendezvous_cell(
                patrol.cell,
                nearest_supply.cell,
                patrol.fuel,
                CRIT_FUEL,
            )

            if rendezvous is not None and rendezvous != wps[0]:
                # Insert rendezvous before current first waypoint
                new_assignments[patrol.id] = [rendezvous] + wps

        return new_assignments

    def _rendezvous_cell(
        self,
        patrol_cell:  int,
        supply_cell:  int,
        patrol_fuel:  int,
        fuel_reserve: int,
    ) -> Optional[int]:
        """
        Return a cell on the path from patrol to supply that the patrol can
        reach while keeping at least fuel_reserve units in the tank.
        Approximated by walking hex-distance steps toward supply.
        """
        max_hops  = max(0, (patrol_fuel - fuel_reserve) // C.FUEL_COST.get(C.TERRAIN_PLAIN, 1))
        total_dist = self.grid.hex_distance(patrol_cell, supply_cell)

        if total_dist == 0 or max_hops == 0:
            return supply_cell if total_dist == 0 else None

        # Walk min(max_hops, half the distance) steps toward supply_cell.
        # We use hex_distance as a proxy — actual path would require A*.
        steps = min(max_hops, total_dist // 2)
        if steps == 0:
            return None

        # BFS-style: find the cell that is `steps` hops toward supply_cell
        # (greedy: each step pick the neighbour closest to supply_cell)
        cur = patrol_cell
        for _ in range(steps):
            best_nbr, best_dist = cur, self.grid.hex_distance(cur, supply_cell)
            for _, nbr in self.grid.neighbors(cur):
                d = self.grid.hex_distance(nbr, supply_cell)
                if d < best_dist:
                    best_dist = d
                    best_nbr  = nbr
            if best_nbr == cur:
                break
            cur = best_nbr

        return cur if cur != patrol_cell else None
