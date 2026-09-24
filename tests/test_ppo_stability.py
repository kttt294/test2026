from copy import deepcopy

import pytest
import torch

from rl.mappo import MAPPOTrainer
from test_rl_regressions import setup_game


def test_critic_gradient_does_not_update_actor_features():
    cfg, mp, _, state = setup_game()
    trainer = MAPPOTrainer(max_width=8, max_height=8, log_dir=None)
    _, value = trainer.model.forward(state, mp, cfg, [0])
    (value - 100).square().mean().backward()
    for name, parameter in trainer.model.named_parameters():
        if not name.startswith('critic_head.'):
            assert parameter.grad is None, name
    assert trainer.model.critic_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('old_algorithm', [None, 'critic_head_only_v1'])
def test_old_optimizer_is_migrated_once_without_changing_weights(tmp_path, old_algorithm):
    trainer = MAPPOTrainer(max_width=8, max_height=8, log_dir=None)
    cfg, mp, _, state = setup_game()
    trainer._collect_episode(cfg, mp, state.my_agents)
    trainer._update()
    trainer.buffer.clear()
    path = tmp_path / 'checkpoint.pt'
    trainer.save(str(path))
    payload = torch.load(path, weights_only=True)
    payload.pop('training_algorithm', None)
    if old_algorithm is not None:
        payload['training_algorithm'] = old_algorithm
    payload['format_version'] = 2
    torch.save(payload, path)
    weights = deepcopy(trainer.model.state_dict())
    restored = MAPPOTrainer(max_width=8, max_height=8, log_dir=None)
    with pytest.warns(RuntimeWarning, match='optimizer'):
        restored.load(str(path))
    assert not restored.optimizer.state
    for key, value in weights.items():
        assert torch.equal(restored.model.state_dict()[key], value)
    restored._collect_episode(cfg, mp, state.my_agents)
    restored._update()
    restored.buffer.clear()
    restored.save(str(path))
    again = MAPPOTrainer(max_width=8, max_height=8, log_dir=None)
    again.load(str(path))
    assert again.optimizer.state


@pytest.mark.parametrize('masking', [False, True])
def test_excessive_policy_update_restores_weights_and_optimizer(monkeypatch, masking):
    cfg, mp, _, state = setup_game()
    trainer = MAPPOTrainer(max_width=8, max_height=8, log_dir=None)
    trainer.model.target_masking = masking
    trainer._collect_episode(cfg, mp, state.my_agents)
    trainer._update()
    assert trainer.optimizer.state
    before = deepcopy(trainer.model.state_dict())
    optimizer_before = deepcopy(trainer.optimizer.state_dict())
    step = trainer.optimizer.step
    def destructive_step(*args, **kwargs):
        result = step(*args, **kwargs)
        with torch.no_grad():
            trainer.model.stay_head[-1].bias.add_(1000)
        return result
    monkeypatch.setattr(trainer.optimizer, 'step', destructive_step)
    result = trainer._update()
    assert result['rejected_steps'] == 1
    for key, value in before.items():
        assert torch.equal(trainer.model.state_dict()[key], value)
    optimizer_after = trainer.optimizer.state_dict()
    assert optimizer_after['param_groups'] == optimizer_before['param_groups']
    for key, state_before in optimizer_before['state'].items():
        for name, value in state_before.items():
            assert torch.equal(optimizer_after['state'][key][name], value)
