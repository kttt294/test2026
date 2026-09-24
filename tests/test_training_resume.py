"""Training regressions: persistence, reproducibility and opponent state."""
import random
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.models import AgentState
from env.scoring import Score
from env.simulator import HexaUdonSimulator
from rl.curriculum import CurriculumEngine
from rl.mappo import MAPPOTrainer
from rl.selfplay import SelfPlayPool
from main import run_train
from test_rl_regressions import setup_game


def trainer(seed=42, device='cpu'):
    return MAPPOTrainer(max_width=8, max_height=8, log_dir=None, seed=seed, device=device,
                        curriculum=CurriculumEngine(),
                        selfplay=SelfPlayPool(pool_size=2, update_every=1))


def test_seed_initializes_model_before_training():
    a, b = trainer(17), trainer(17)
    assert all(torch.equal(x, y) for x, y in zip(a.model.parameters(), b.model.parameters()))
    c = trainer(18)
    assert not torch.equal(a.model.critic_head.weight, c.model.critic_head.weight)


@pytest.mark.parametrize('masking', [False, True])
def test_final_save_and_exact_resume(tmp_path, monkeypatch, masking):
    # Small actual updates, including self-play and curriculum evaluation.
    monkeypatch.setattr('config.N_EPOCHS', 1)
    path = tmp_path / 'nested' / 'model.pt'
    continuous = trainer()
    continuous.model.target_masking = masking
    continuous.model.secondary_routes = masking
    continuous.model.reserve_spots = masking
    continuous.train(2, log_every=1, eval_baseline_every=1)
    expected = deepcopy(continuous.model.state_dict())
    expected_optimizer = deepcopy(continuous.optimizer.state_dict())
    expected_rng = (random.random(), np.random.random(), torch.rand(3))

    first = trainer()
    first.model.target_masking = masking
    first.model.secondary_routes = masking
    first.model.reserve_spots = masking
    first.train(1, log_every=1, eval_baseline_every=1, save_every=500, save_path=str(path))
    assert path.exists()
    resumed = trainer(99)
    resumed.load(str(path))
    assert resumed.model.target_masking == masking
    assert resumed.model.reserve_spots == masking
    assert all(m.reserve_spots == masking for m in resumed.selfplay._pool)
    assert all(m.target_masking == masking for m in resumed.selfplay._pool)
    assert resumed.episodes_completed == 1
    assert resumed.optimizer.state
    assert len(resumed.selfplay._pool) == 1
    assert list(resumed.curriculum._wins) == list(first.curriculum._wins)
    resumed.train(1, log_every=1, eval_baseline_every=1)
    assert resumed.episodes_completed == 2
    for key, tensor in expected.items():
        torch.testing.assert_close(resumed.model.state_dict()[key], tensor, rtol=0, atol=0)
    for key, values in expected_optimizer['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(resumed.optimizer.state_dict()['state'][key][name], value)
    assert random.random() == expected_rng[0]
    assert np.random.random() == expected_rng[1]
    assert torch.equal(torch.rand(3), expected_rng[2])


def test_failed_save_preserves_existing_checkpoint(tmp_path, monkeypatch):
    instance = trainer()
    path = tmp_path / 'model.pt'
    instance.save(str(path))
    before = path.read_bytes()
    def fail_write(payload, stream):
        stream.write(b'incomplete')
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', fail_write)
    with pytest.raises(OSError, match='disk full'):
        instance.save(str(path))
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_selfplay_tracks_own_score_and_waits_on_spot():
    cfg, mp, sim, state = setup_game()
    state.collected_series = {99}
    state.total_udon = 50
    opponents = [AgentState(1001, 0, 1, 20)]
    opponent_state = sim.reset(opponents)
    views = []
    def choose(view, *args, **kwargs):
        views.append(deepcopy(view))
        return [0], None, None, None
    model = SimpleNamespace(secondary_routes=False, reserve_spots=False, get_action_and_value=choose)
    for day in (1, 2):
        state.day = day
        cells, _ = SelfPlayPool.simulate_day(
            model, opponents, state, mp, cfg, sim.grid, opponent_state=opponent_state)
        assert cells == [1]
    assert views[0].collected_series == set()
    assert views[0].total_udon == 0
    assert views[1].collected_series == {0}
    assert opponent_state.total_udon == 2
    assert opponents[0].fuel == 20  # Waiting collects without a leave-and-return detour.
    assert state.collected_series == {99} and state.total_udon == 50


def test_curriculum_uses_separate_solo_evaluation(monkeypatch):
    instance = trainer()
    monkeypatch.setattr(instance, '_collect_episode', lambda *a, **kw: dict(
        shaped_return=0, raw_return=0, unique_series=0, total_udon=0, score=Score(0, 0, 0)))
    monkeypatch.setattr(instance, '_evaluate_policy', lambda *a: Score(3, 4, 5))
    monkeypatch.setattr(instance.curriculum, 'evaluate_baseline', lambda *a: Score(2, 4, 5))
    instance.train(1, log_every=1, eval_baseline_every=1)
    assert instance.curriculum.win_rate == 1


def test_missing_requested_checkpoint_fails_before_training(tmp_path, monkeypatch):
    def unexpected_train(*args, **kwargs):
        pytest.fail('A missing --load must not silently start fresh')
    monkeypatch.setattr(MAPPOTrainer, 'train', unexpected_train)
    with pytest.raises(FileNotFoundError):
        run_train(SimpleNamespace(curriculum=False, selfplay=False, device='cpu',
                  log_dir=None, seed=42, load=str(tmp_path / 'missing.pt')))


def test_legacy_weights_checkpoint_still_loads(tmp_path):
    source = MAPPOTrainer(max_series=10, log_dir=None)
    path = tmp_path / 'old.pt'
    torch.save({'model': source.model.state_dict(), 'max_spots': 30, 'max_series': 10}, path)
    restored = trainer(17)
    with pytest.warns(RuntimeWarning, match='Legacy checkpoint'):
        restored.load(str(path))
    assert torch.equal(source.model.critic_head.weight, restored.model.critic_head.weight)
    assert restored.episodes_completed == 0


def test_selfplay_snapshot_uses_current_match_fuel_capacity(monkeypatch):
    instance = trainer()
    instance.selfplay.step(instance.model)
    opponent = instance.selfplay._pool[0]
    cfg, mp, _, state = setup_game()
    cfg.fuel_max = 30
    monkeypatch.setattr(opponent, 'get_action_and_value', lambda *a, **kw: (
        [len(mp.spots)], None, None, None))
    instance._collect_episode(cfg, mp, state.my_agents)
    assert opponent._fuel_max == 30


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable on this machine')
def test_cuda_training_checkpoint_resume(tmp_path, monkeypatch):
    monkeypatch.setattr('config.N_EPOCHS', 1)
    path = tmp_path / 'cuda.pt'
    first = trainer(device='cuda')
    first.train(1, log_every=1, save_path=str(path))
    restored = trainer(device='cuda')
    restored.load(str(path))
    for parameter, state in restored.optimizer.state.items():
        assert state['exp_avg'].device == parameter.device
    restored.train(1, log_every=1)
    assert restored.episodes_completed == 2
    assert all(torch.isfinite(p).all() for p in restored.model.parameters())
