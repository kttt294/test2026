from copy import deepcopy

import pytest

from env.models import AgentAction, AgentState, Cell, DayOrder, MapData, MatchConfig, Spot
from env.simulator import HexaUdonSimulator, apply_joint_day


def fixture(count=2):
    cfg = MatchConfig(2, 1, 2, [2, 2], count, 1, 3, 20)
    board = MapData([Cell(0, 3), Cell(1, 0)], [Spot(1, 0, 1)])
    sims = [HexaUdonSimulator(cfg, board) for _ in range(count)]
    states = [sim.reset([AgentState(0, 0, 0, 5), AgentState(1, 0, 1, 5)]) for sim in sims]
    orders = [[DayOrder(i, [AgentAction('stay')]*2) for i in range(2)] for _ in sims]
    return sims, states, orders


@pytest.mark.parametrize('count', [2, 4])
def test_joint_private_inventory_common_traffic(count):
    sims, states, orders = fixture(count)
    previous = deepcopy(states)
    following, _ = apply_joint_day(sims, states, orders)
    assert states == previous
    assert [s.total_udon for s in following] == [1]*count
    assert [s.traffic for s in following] == [{0: 1}]*count
    assert all(sim.traffic._history == [{0: 2*count}] for sim in sims)
    assert following[0].opponent_cells == [0, 1]*(count-1)


def test_opponent_supply_cannot_refuel_our_patrol():
    sims, states, orders = fixture()
    states[1].my_agents[0].type = 1
    following, _ = apply_joint_day(sims, states, orders)
    assert following[0].my_agents[0].fuel == 5


def test_joint_invalid_plan_does_not_commit_any_team():
    sims, states, orders = fixture()
    orders[1][0].actions.pop()
    with pytest.raises(ValueError):
        apply_joint_day(sims, states, orders)
    assert all(not sim.traffic._history for sim in sims)


@pytest.mark.parametrize('size', [8, 16, 32])
def test_joint_benchmark_uses_contest_ranges(size):
    from benchmark_match import contest_scenario
    for seed in range(42, 47):
        cfg, board, agents = contest_scenario(seed, size)
        assert all(2*size <= s <= 8*size for s in cfg.steps_per_day)
        assert cfg.steps_per_day[0] <= cfg.fuel_max <= 3*cfg.steps_per_day[0]
        assert len(agents) <= len(board.spots) <= size
        assert {c.terrain for c in board.cells} == {0, 1, 2, 3}
