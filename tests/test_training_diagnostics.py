from copy import deepcopy

import torch

from env.models import AgentAction, AgentState, DayOrder
from rl.mappo import MAPPOTrainer
from report.diagnose_training import audit_orders, measured_update
from test_rl_regressions import setup_game


def test_pilot_evaluation_preserves_training_rng_and_fuel():
    import random
    import numpy as np
    from report.pilot_training import measure
    trainer = MAPPOTrainer(log_dir=None, seed=43)
    trainer.model.set_fuel_max(17)
    py_state, np_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    weights = deepcopy(trainer.model.state_dict())
    result = measure(trainer, 5000, games=1)
    assert set(result) == {'greedy', 'argmax', 'sampled'}
    assert trainer.model._fuel_max == 17
    assert random.getstate() == py_state
    current_np = np.random.get_state()
    assert current_np[0] == np_state[0]
    assert np.array_equal(current_np[1], np_state[1])
    assert current_np[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), cpu_state)
    for actual, expected in zip(torch.cuda.get_rng_state_all() if cuda_states else [], cuda_states):
        assert torch.equal(actual, expected)
    for key, expected in weights.items():
        assert torch.equal(trainer.model.state_dict()[key], expected)


def test_idle_audit_preserves_each_cars_budget():
    cfg, mp, sim, state = setup_game()
    state.my_agents = [AgentState(0, 0, 0, 20), AgentState(1, 0, 0, 20)]
    state.steps_left = 2
    rows = audit_orders(state, mp, sim, [DayOrder(0, [AgentAction('move', 2)]), DayOrder(1, [])])
    assert rows[0]['steps_available'] == 2
    assert rows[0]['reason'] == 'reachable_collection'


def test_idle_audit_allows_collection_while_waiting():
    _, mp, sim, state = setup_game()
    state.my_agents = [AgentState(0, 0, 1, 0)]
    assert audit_orders(state, mp, sim, [DayOrder(0, [])])[0]['reason'] == 'reachable_collection'
    state.my_agents[0].fuel = 2
    assert audit_orders(state, mp, sim, [DayOrder(0, [])])[0]['reason'] == 'reachable_collection'
    state.spot_inventory = {1: 0}
    assert audit_orders(state, mp, sim, [DayOrder(0, [])])[0]['reason'] == 'no_inventory'


def test_measurement_preserves_ppo_update(monkeypatch):
    monkeypatch.setattr('config.N_EPOCHS', 1)
    cfg, mp, _, state = setup_game()
    trainer = MAPPOTrainer(max_width=8, max_height=8, log_dir=None)
    trainer._collect_episode(cfg, mp, state.my_agents)
    observed = deepcopy(trainer)
    trainer._update()
    result = measured_update(observed)
    assert len(result['optimizer_steps']) == len(trainer.buffer.transitions)
    for left, right in zip(trainer.model.parameters(), observed.model.parameters()):
        assert torch.equal(left, right)
    assert all('ratio_to_rollout' in agent for row in result['optimizer_steps'] for agent in row['agents'])
