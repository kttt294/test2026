import pytest
import torch

from env.models import AgentState, Cell, MapData, MatchConfig, Spot
from env.simulator import HexaUdonSimulator
from pathfinding.astar import find_path
from rl.actor_critic import ActorCritic


def test_actor_distinguishes_positions_with_opposite_reachable_targets():
    torch.manual_seed(42)
    cfg = MatchConfig(8, 8, 1, [10], 2, 3, 7, 5)
    mp = MapData([Cell(i, 0) for i in range(64)], [Spot(0, 0, 3), Spot(63, 1, 3)])
    sim = HexaUdonSimulator(cfg, mp)
    actor = ActorCritic()
    actor.set_fuel_max(5)
    logits = []
    reachable = []
    for cell in (36, 27):
        state = sim.reset([AgentState(0, 0, cell, 5)])
        with torch.no_grad():
            outputs, _ = actor.forward(state, mp, cfg, [0])
        logits.append(outputs[0])
        reachable.append([find_path(sim.grid, {i: 0 for i in range(64)}, state.traffic,
                                    cell, spot.cell_id, 10, 5).reachable for spot in mp.spots])
    assert reachable == [[False, True], [True, False]]
    assert not torch.allclose(logits[0], logits[1]), 'Actor cannot observe the change of vehicle position'


def test_actor_distinguishes_identical_spots_at_different_positions():
    torch.manual_seed(42)
    cfg = MatchConfig(32, 32, 1, [30], 2, 3, 7, 30)
    mp = MapData([Cell(i, 0) for i in range(1024)],
                 [Spot(8 * 32 + 8, 0, 3), Spot(23 * 32 + 23, 1, 3)])
    state = HexaUdonSimulator(cfg, mp).reset([AgentState(0, 0, 16 * 32 + 16, 30)])
    actor = ActorCritic()
    with torch.no_grad():
        logits, _ = actor(state, mp, cfg, [0])
    assert not torch.allclose(logits[0][0, 0], logits[0][0, 1])


def test_legacy_position_migration_preserves_outputs():
    from test_rl_regressions import setup_game
    torch.manual_seed(42)
    reference = ActorCritic()
    keys = ('agent_mlp.0.weight', 'key_mlp.0.weight')
    with torch.no_grad():
        reference.agent_mlp[0].weight[:, -2:].zero_()
        reference.key_mlp[0].weight[:, -2:].zero_()
    legacy = reference.state_dict()
    for key in keys:
        legacy[key] = legacy[key][:, :-2].clone()
    restored = ActorCritic()
    restored.load_state_dict(legacy)
    for key in keys:
        assert torch.count_nonzero(restored.state_dict()[key][:, -2:]) == 0
        assert legacy[key].shape[1] == restored.state_dict()[key].shape[1] - 2
    cfg, mp, _, state = setup_game()
    with torch.no_grad():
        expected, expected_value = reference(state, mp, cfg, [0])
        actual, actual_value = restored(state, mp, cfg, [0])
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    from rl.selfplay import SelfPlayPool
    pool = SelfPlayPool()
    pool.load_state_dict({'models': [legacy], 'fuel_max': [20]}, restored)
    with torch.no_grad():
        opponent_logits, _ = pool._pool[0](state, mp, cfg, [0])
    torch.testing.assert_close(opponent_logits[0], expected[0], rtol=0, atol=0)
    malformed = restored.state_dict()
    malformed['key_mlp.0.weight'] = malformed['key_mlp.0.weight'][:, :-1]
    with pytest.raises(RuntimeError, match='size mismatch'):
        restored.load_state_dict(malformed)
