from types import SimpleNamespace
from copy import deepcopy

import pytest
import torch
import config as C

from rl.mappo import MAPPOTrainer, RolloutBuffer
from test_rl_regressions import setup_game


def test_gae_stops_at_each_game_terminal():
    first = [SimpleNamespace(value=torch.tensor(v), reward=r, done=d)
             for v, r, d in [(2., 3., False), (4., 5., True)]]
    second = [SimpleNamespace(value=torch.tensor(999.), reward=10000., done=True)]
    single = RolloutBuffer()
    single.transitions = first
    combined = RolloutBuffer()
    combined.transitions = first + second
    expected = single.compute_returns()
    actual = combined.compute_returns()
    assert actual[0][:2] == pytest.approx(expected[0])
    assert actual[1][:2] == pytest.approx(expected[1])


def test_batch_collects_frozen_policy_and_flushes_tail(monkeypatch, tmp_path):
    cfg, mp, _, state = setup_game()
    trainer = MAPPOTrainer(log_dir=None)
    monkeypatch.setattr('rl.mappo.generate_random_scenario', lambda seed: (cfg, mp, state.my_agents))
    original_collect = trainer._collect_episode
    batches, saves, versions = [], [], []
    def collect(*a, **kw):
        versions.append(len(batches))
        return original_collect(*a, **kw)
    def update():
        batches.append(sum(tr.done for tr in trainer.buffer.transitions))
        return {}
    monkeypatch.setattr(trainer, '_collect_episode', collect)
    monkeypatch.setattr(trainer, '_update', update)
    monkeypatch.setattr(trainer, 'save', lambda path: saves.append(len(trainer.buffer)))
    trainer.train(5, games_per_update=3, save_every=2, save_path=str(tmp_path/'model.pt'))
    assert batches == [3, 2]
    assert versions == [0, 0, 0, 1, 1]
    assert saves and all(n == 0 for n in saves)
    assert trainer.episodes_completed == 5
    assert len(trainer.buffer) == 0


def test_mixed_fuel_games_keep_rollout_action_probabilities():
    cfg, mp, _, state = setup_game()
    trainer = MAPPOTrainer(log_dir=None)
    for fuel_max in (2, 200):
        game_cfg = deepcopy(cfg)
        game_cfg.fuel_max = fuel_max
        trainer._collect_episode(game_cfg, mp, deepcopy(state.my_agents))
    snapshots = trainer._policy_snapshot()
    for tr, logs in zip(trainer.buffer.transitions, snapshots):
        selected = logs[torch.arange(len(tr.actions)), tr.actions]
        torch.testing.assert_close(selected, tr.log_probs, rtol=1e-5, atol=1e-6)


def test_resume_at_batch_boundary_matches_continuous(monkeypatch, tmp_path):
    cfg, mp, _, state = setup_game()
    monkeypatch.setattr('rl.mappo.generate_random_scenario',
                        lambda seed: (deepcopy(cfg), deepcopy(mp), deepcopy(state.my_agents)))
    monkeypatch.setattr('config.N_EPOCHS', 1)
    def create():
        return MAPPOTrainer(max_width=8, max_height=8, log_dir=None)
    continuous = create()
    continuous.train(4, games_per_update=2)
    expected_rng = torch.get_rng_state()
    first = create()
    path = str(tmp_path/'batch.pt')
    first.train(2, games_per_update=2, save_path=path)
    resumed = create()
    resumed.load(path)
    resumed.train(2, games_per_update=2)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for key, value in continuous.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key


@pytest.mark.parametrize('capacity', [None, 64])
def test_accumulated_update_matches_mean_loss_gradient(monkeypatch, capacity):
    monkeypatch.setattr(C, 'N_EPOCHS', 1)
    monkeypatch.setattr(C, 'MAX_POLICY_KL', float('inf'))
    cfg, mp, _, state = setup_game()
    trainer = MAPPOTrainer(max_width=8, max_height=8, log_dir=None)
    trainer._collect_episode(cfg, mp, deepcopy(state.my_agents))
    reference = deepcopy(trainer)
    trainer.transitions_per_step = capacity or len(trainer.buffer)
    returns, advantages = reference.buffer.compute_returns()
    adv = torch.tensor(advantages)
    adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    losses = []
    for i, tr in enumerate(reference.buffer.transitions):
        reference.model.set_fuel_max(tr.fuel_max)
        _, logs, entropy, value = reference.model.get_action_and_value(
            tr.state, tr.map_data, tr.cfg, [a.id for a in tr.state.patrol_agents()], actions=tr.actions)
        ratio = (logs-tr.log_probs).exp()
        actor = -torch.min(ratio*adv[i], ratio.clamp(1-C.CLIP_EPS, 1+C.CLIP_EPS)*adv[i]).mean()
        losses.append(actor + .1*(value-returns[i]).square() - C.ENTROPY_COEF*entropy.mean())
    torch.stack(losses).mean().backward()
    torch.nn.utils.clip_grad_norm_(reference.model.parameters(), .5)
    reference.optimizer.step()
    stats = trainer._update()
    assert stats['accepted_steps'] == 1
    for key, value in reference.model.state_dict().items():
        torch.testing.assert_close(trainer.model.state_dict()[key], value, atol=1e-6, rtol=1e-5)
