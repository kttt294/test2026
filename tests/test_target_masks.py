from types import SimpleNamespace

import pytest
import torch

from env.models import AgentState, Cell, MapData, MatchConfig, Spot
from env.simulator import HexaUdonSimulator
from report.benchmark_target_masks import masked_choices
from rl.mappo import MAPPOTrainer
from test_rl_regressions import setup_game


@pytest.mark.parametrize('level', [0, 2, 4])
def test_production_matches_counterfactual_with_traffic_and_per_car_steps(level):
    from rl.curriculum import CurriculumEngine
    from env.validator import validate_orders
    cfg, mp, agents = CurriculumEngine(level).generate_scenario(8000)
    sim = HexaUdonSimulator(cfg, mp)
    state = sim.reset(agents)
    trainer = MAPPOTrainer(log_dir=None, seed=43)
    trainer.model.target_masking = True
    trainer.model.set_fuel_max(cfg.fuel_max)
    with torch.no_grad():
        for _ in range(2):
            for sampled in (False, True):
                torch.manual_seed(101)
                ids, expected, _ = masked_choices(trainer, state, mp, cfg, sim, 'both', sampled)
                torch.manual_seed(101)
                actual, _, _, _ = trainer.model.get_action_and_value(state, mp, cfg, ids, deterministic=not sampled)
                assert actual == expected
            orders = trainer._actions_to_orders(state, mp, cfg, sim, ids, actual)
            valid, errors = validate_orders(orders, state, mp, sim.grid)
            assert valid, errors
            state, _ = sim.apply_day(state, orders)


def test_production_masks_replay_saved_prefix_without_sampling(monkeypatch):
    cfg = MatchConfig(8, 8, 4, [20]*4, 2, 3, 7, 20)
    mp = MapData([Cell(i, 0) for i in range(64)], [Spot(1, 0, 1), Spot(2, 1, 1)])
    state = HexaUdonSimulator(cfg, mp).reset([AgentState(0, 0, 0, 20), AgentState(1, 0, 0, 20)])
    trainer = MAPPOTrainer(log_dir=None)
    trainer.model.target_masking = True
    monkeypatch.setattr(trainer.model, 'forward', lambda *a: (
        [torch.tensor([[5., 3., -1.]]) for _ in range(2)], torch.tensor(0.)))
    chosen, saved, _, _ = trainer.model.get_action_and_value(state, mp, cfg, [0, 1], deterministic=True)
    assert chosen == [0, 0]
    monkeypatch.setattr(torch.distributions.Categorical, 'sample', lambda *a: pytest.fail('PPO must not resample'))
    _, replayed, _, _ = trainer.model.get_action_and_value(state, mp, cfg, [0, 1], actions=chosen)
    torch.testing.assert_close(replayed, saved, rtol=0, atol=0)
    _, distributions, _ = trainer.model.action_distributions(state, mp, cfg, [0, 1], actions=[2, 0])
    assert distributions[1].probs[0] > 0  # First patrol stayed, so stock remains.
    _, distributions, _ = trainer.model.action_distributions(state, mp, cfg, [0, 1], actions=[0, 1])
    assert distributions[1].probs[0] > 0  # Simultaneous arrivals are resolved by the simulator.


@pytest.mark.parametrize('inventory,expected', [(1, [0, 0]), (2, [0, 0])])
def test_mask_does_not_reserve_future_inventory_by_planning_order(inventory, expected):
    cfg = MatchConfig(8, 8, 4, [20]*4, 2, 3, 7, 20)
    mp = MapData([Cell(i, 0) for i in range(64)], [Spot(1, 0, inventory), Spot(2, 1, 1)])
    sim = HexaUdonSimulator(cfg, mp)
    state = sim.reset([AgentState(0, 0, 0, 20), AgentState(1, 0, 0, 20)])
    def forward(*args):
        return [torch.tensor([[5., 3., -1.]]) for _ in range(2)], torch.tensor(0.)
    trainer = SimpleNamespace(model=forward)
    _, choices, _ = masked_choices(trainer, state, mp, cfg, sim, 'both')
    assert choices == expected
    assert state.spot_inventory[1] == inventory
    state.steps_left = 2
    _, choices, _ = masked_choices(trainer, state, mp, cfg, sim, 'both')
    assert choices == [0, 0]  # Each patrol has its own full two-step timeline.


def test_unmasked_counterfactual_matches_original_policy():
    cfg, mp, sim, state = setup_game(n_spots=2)
    trainer = MAPPOTrainer(log_dir=None, seed=43)
    with torch.no_grad():
        for sampled in (False, True):
            torch.manual_seed(100)
            expected, _, _, _ = trainer.model.get_action_and_value(state, mp, cfg, [0], deterministic=not sampled)
            torch.manual_seed(100)
            _, actual, _ = masked_choices(trainer, state, mp, cfg, sim, 'baseline', sampled)
            assert actual == expected


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_masked_ppo_handles_only_stay_without_nan(device, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    monkeypatch.setattr('config.N_EPOCHS', 1)
    cfg, mp, _, state = setup_game(n_spots=2)
    cfg.steps_per_day[:] = [0] * cfg.total_days
    state.my_agents = [AgentState(0, 0, 0, 0)]
    trainer = MAPPOTrainer(log_dir=None, device=device)
    trainer.model.target_masking = True
    trainer._collect_episode(cfg, mp, state.my_agents)
    snapshots = trainer._policy_snapshot()
    assert all(torch.isneginf(p[:, :-1]).all() and (p[:, -1] == 0).all() for p in snapshots)
    result = trainer._update()
    assert result['accepted_steps'] > 0
    assert result['rejected_steps'] == 0
    assert result['max_policy_kl'] == 0
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
