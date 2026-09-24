from copy import deepcopy

import pytest
import torch

from env.hex_grid import HexGrid
from env.map_generator import generate_random_scenario
from env.models import AgentState, Cell, MapData, MatchConfig
from rl.curriculum import CurriculumEngine
from rl.mappo import MAPPOTrainer
from rl.selfplay import SelfPlayPool


@pytest.mark.parametrize('level', range(5))
def test_curriculum_contest_ranges_and_reproducibility(level):
    engine = CurriculumEngine(level)
    for seed in range(20):
        scenario = engine.generate_scenario(seed)
        assert scenario == engine.generate_scenario(seed)
        cfg, board, agents = scenario
        perimeter = cfg.width + cfg.height
        assert all(perimeter <= steps <= 4*perimeter for steps in cfg.steps_per_day)
        assert cfg.steps_per_day[0] <= cfg.fuel_max <= 3*cfg.steps_per_day[0]
        assert len(agents) <= len(board.spots) <= max(cfg.width, cfg.height)
        assert {c.terrain for c in board.cells} == {0, 1, 2, 3}
        grid = HexGrid(cfg.width, cfg.height)
        passable = {c.id for c in board.cells if c.terrain != 2}
        seen, pending = {agents[0].cell}, [agents[0].cell]
        while pending:
            for _, cell in grid.neighbors(pending.pop()):
                if cell in passable and cell not in seen:
                    seen.add(cell)
                    pending.append(cell)
        assert seen == passable
        assert all(a.fuel == cfg.fuel_max for a in agents if a.is_patrol())


def test_random_rectangular_training_ranges():
    for seed in range(20):
        cfg, board, agents = generate_random_scenario(seed, (8, 8), (16, 16))
        assert (cfg.width, cfg.height) == (8, 16)
        assert all(24 <= steps <= 96 for steps in cfg.steps_per_day)
        assert cfg.steps_per_day[0] <= cfg.fuel_max <= 3*cfg.steps_per_day[0]
        assert len(agents) <= len(board.spots) <= 16


@pytest.mark.parametrize('with_opponent', [False, True])
def test_rollout_same_start_and_actual_traffic_divisor(monkeypatch, with_opponent):
    cfg = MatchConfig(8, 8, 2, [2, 2], 4, 1.5, 10, 20)
    board = MapData([Cell(i, 3) for i in range(64)], [])
    agents = [AgentState(i, 0, i, 20) for i in range(3)]
    before = deepcopy(agents)
    pool = SelfPlayPool()
    trainer = MAPPOTrainer(log_dir=None, selfplay=pool)
    if with_opponent:
        pool._add(trainer.model)
    trainer._collect_episode(cfg, board, agents)
    first, second = trainer.buffer.transitions
    assert cfg.n_teams == 4  # Local config only; caller unchanged.
    assert agents == before
    assert first.cfg.n_teams == (2 if with_opponent else 1)
    assert first.state.opponent_cells == ([0, 1, 2] if with_opponent else [])
    # Two steps per car, divided by actual team count: always 2, not 0.5/1.
    assert second.state.traffic == {0: 1, 1: 1, 2: 1}
    assert torch.isfinite(second.log_probs).all()


def test_opponent_inventory_is_private():
    cfg, board, agents = CurriculumEngine().generate_scenario(42)
    from env.simulator import HexaUdonSimulator
    state = HexaUdonSimulator(cfg, board).reset(agents)
    opponent = deepcopy(state)
    cell = board.spots[0].cell_id
    opponent.spot_inventory[cell] = 0
    view = SelfPlayPool.build_opponent_view(state, agents, opponent)
    assert view.spot_inventory[cell] == 0
    assert state.spot_inventory[cell] > 0
    view.spot_inventory[cell] = 99
    assert opponent.spot_inventory[cell] == 0
