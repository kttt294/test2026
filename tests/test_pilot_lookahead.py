import argparse
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    'pilot_lookahead', Path(__file__).resolve().parents[1] / 'report' / 'pilot_lookahead.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_resume_uses_saved_checkpoint_and_fills_missing_validation(tmp_path, monkeypatch):
    folder = tmp_path / 'seed_42'
    folder.mkdir()
    trainer = pilot.MAPPOTrainer(log_dir=None, seed=42)
    trainer.save(str(folder/'initial.pt'))
    trainer.episodes_completed = 16
    trainer.save(str(folder/'episode_16.pt'))
    original = (folder/'episode_16.pt').read_bytes()
    evaluation = dict(summary=dict(win_rate=.5, mean_score_delta=dict(unique_series=0)))
    config = dict(seeds=[42], episodes=32, interval=16, selfplay_every=8,
        eval_games=1, holdout_games=2, eval_baseline_every=4, eval_seed=1000000,
        holdout_seed=2000000, sizes=[16], finals=True, secondary_routes=False,
        reserve_spots=False, device='cpu', output_dir=str(tmp_path))
    pilot.write_report(tmp_path/'pilot.json', dict(config=config, completed=False,
        runs=[dict(seed=42, milestones=[], before_validation=evaluation,
            updates=[dict(episode=16, accepted_steps=1), dict(episode=24, accepted_steps=99)])]))
    calls = []

    def train(self, count, **kwargs):
        calls.append((self.episodes_completed, count))
        self.episodes_completed += count-1
        self._update()
        self.episodes_completed += 1
        self.save(kwargs['save_path'])

    monkeypatch.setattr(pilot.MAPPOTrainer, 'train', train)
    monkeypatch.setattr(pilot.MAPPOTrainer, '_update', lambda self: dict(max_policy_kl=0., accepted_steps=1))
    monkeypatch.setattr(pilot.SelfPlayPool, 'has_opponent', lambda self: True)
    monkeypatch.setattr(pilot, 'measure', lambda *a, **k: evaluation)
    monkeypatch.setattr('sys.argv', ['pilot', '--resume', '--output-dir', str(tmp_path)])
    pilot.main()
    result = json.loads((tmp_path/'pilot.json').read_text())
    assert result['completed'] and calls == [(16,16)]
    assert [m['episode'] for m in result['runs'][0]['milestones']] == [16,32]
    assert result['runs'][0]['accepted_steps'] == 2
    assert (folder/'episode_16.pt').read_bytes() == original


def test_seed_ranges_reject_leakage():
    args = argparse.Namespace(episodes=32, interval=16, eval_games=2, selfplay_every=8,
        seeds=[42, 10042, 20042], sizes=[8], eval_seed=1000000, holdout_seed=2000000)
    pilot.validate(args)
    args.holdout_seed = 43
    with pytest.raises(ValueError, match='overlap'):
        pilot.validate(args)
    args.holdout_seed = 1000006
    args.holdout_games = 40
    args.seeds = [1000030, 2000042, 3000042]
    with pytest.raises(ValueError, match='overlap'):
        pilot.validate(args)
    args.holdout_seed = 1000001
    with pytest.raises(ValueError, match='overlap'):
        pilot.validate(args)


def test_paired_matches_preserve_training_state():
    torch.set_num_threads(2)
    trainer = pilot.MAPPOTrainer(log_dir=None, seed=42)
    trainer.model.target_masking = True
    trainer.model.train()
    trainer.model.set_fuel_max(123)
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    result = pilot.measure(trainer, [1000000], [8])
    first, second = result['games']
    assert [first['policy_slot'], second['policy_slot']] == [0, 1]
    assert first['policy_score'] == second['policy_score']
    assert first['lookahead_score'] == second['lookahead_score']
    assert result['summary']['matches'] == 2
    assert result['summary']['distinct_maps'] == 1
    assert first['policy_planning_ms'] and all(t >= 0 for t in first['policy_planning_ms'])
    assert sum(result['summary'][key] for key in ('wins', 'ties', 'losses')) == 2
    assert trainer.model.training and trainer.model._fuel_max == 123
    assert random.getstate() == py_rng
    current = np.random.get_state()
    assert current[0] == np_rng[0] and np.array_equal(current[1], np_rng[1])
    assert current[2:] == np_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_failed_evaluation_restores_model_and_rng(monkeypatch):
    trainer = pilot.MAPPOTrainer(log_dir=None)
    fuel, rng = trainer.model._fuel_max, torch.get_rng_state()

    def fail(*args, **kwargs):
        trainer.model.set_fuel_max(999)
        torch.rand(1)
        raise ValueError('Invalid match')

    monkeypatch.setattr(pilot, 'play_match', fail)
    with pytest.raises(ValueError, match='Invalid match'):
        pilot.measure(trainer, [1000000], [8])
    assert trainer.model.training and trainer.model._fuel_max == fuel
    assert torch.equal(torch.get_rng_state(), rng)


def test_sampled_diagnostics_are_paired_and_replayable():
    from dataclasses import asdict
    from benchmark_match import contest_scenario
    from env.models import AgentAction, DayOrder
    from env.scoring import compute_score
    from env.simulator import HexaUdonSimulator, apply_joint_day

    torch.set_num_threads(2)
    trainer = pilot.MAPPOTrainer(log_dir=None)
    trainer.model.target_masking = True
    result = pilot.measure(trainer, [3000000], [8], mode='sampled', diagnostics=True)
    first, second = result['games']
    assert first['decisions'] == second['decisions']
    assert first['policy_score'] == second['policy_score']
    replay = first['replay']
    cfg, board, agents = contest_scenario(3000000, 8)
    cfg.n_teams = 2
    sims = [HexaUdonSimulator(cfg, board) for _ in range(2)]
    states = [sim.reset(agents) for sim in sims]
    for state in states:
        state.opponent_cells = [a.cell for a in agents]
    for day in replay['days']:
        orders = [[DayOrder(o['agent_id'], [AgentAction(**a) for a in o['actions']])
                   for o in team] for team in day['orders']]
        states, _ = apply_joint_day(sims, states, orders)
        assert [asdict(compute_score(s)) for s in states] == day['scores']
    assert [asdict(compute_score(s)) for s in states] == replay['scores']


def test_novel_inference_reduces_missing_series_on_fixed_map():
    torch.set_num_threads(2)
    trainer = pilot.MAPPOTrainer(log_dir=None, seed=42)
    trainer.model.target_masking = True
    normal = pilot.measure(trainer, [3000000], [8], diagnostics=True)
    novel = pilot.measure(trainer, [3000000], [8], mode='novel_argmax', diagnostics=True)
    assert novel['games'][0]['policy_score']['unique_series'] >= normal['games'][0]['policy_score']['unique_series']
    assert novel['games'][0]['policy_score'] == novel['games'][1]['policy_score']
    assert any(choice['novel_series'] for day in novel['games'][0]['decisions'] for choice in day['choices'])
