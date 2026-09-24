"""BTC Q&A 1, Q6–8, and action supplement pp. 3–4 (six-step example)."""
from copy import deepcopy

import pytest

from env.models import AgentAction, AgentState, Cell, DayOrder, MapData, MatchConfig, Spot
from env.simulator import HexaUdonSimulator
from env.validator import validate_orders


def example():
    cfg = MatchConfig(4, 1, 1, [6], 1, 3, 10, 3)
    board = MapData([Cell(0, 3), Cell(1, 3), Cell(2, 0), Cell(3, 0)], [Spot(2, 0, 4)])
    sim = HexaUdonSimulator(cfg, board)
    state = sim.reset([AgentState(i, int(i == 8), cell, 0 if i == 8 else 3)
                       for i, cell in enumerate([0, 0, 0, 0, 2, 3, 3, 2], 1)])
    state.traffic = {1: 1}
    # Negative numbers are expanded into one-step waits in our internal model.
    plans = [[2, 2, 2, -1], [-1, 2, 2, 2], [-2, 2, 2, -1], [-3, 2, -2],
             [5, 5, -2], [5, 5, -2], [-2, 5, 5], [5, 5, -2]]
    orders = [DayOrder(i, [a for code in plan for a in
                          ([AgentAction('stay')] * -code if code < 0 else [AgentAction('move', code)])])
              for i, plan in enumerate(plans, 1)]
    return sim, state, orders


def test_btc_six_step_example():
    sim, state, orders = example()
    before = deepcopy(state)
    assert validate_orders(orders, state, sim.map, sim.grid)[0]
    trace = []
    final, _ = sim.apply_day(state, orders, trace=trace)
    assert state == before
    assert [row['cells'] for row in trace] == [
        [1, 0, 0, 0, 2, 3, 3, 2], [1, 1, 0, 0, 1, 2, 3, 1],
        [2, 1, 1, 0, 1, 2, 3, 1], [2, 2, 1, 1, 0, 1, 2, 0],
        [3, 2, 2, 1, 0, 1, 2, 0], [3, 3, 2, 1, 0, 1, 1, 0],
    ]
    assert [a.fuel for a in final.my_agents] == [0, 0, 1, 1, 3, 1, 1, 0]
    assert [row['fuel'] for row in trace] == [
        [1, 3, 3, 3, 3, 3, 3, 0], [3, 3, 3, 3, 3, 2, 3, 0],
        [1, 3, 3, 3, 3, 2, 3, 0], [1, 1, 3, 1, 3, 1, 2, 0],
        [0, 1, 1, 1, 3, 1, 2, 0], [0, 0, 1, 1, 3, 1, 1, 0],
    ]
    assert [row['collected'] for row in trace] == [[(5, 2)], [(6, 2)], [(1, 2)], [(2, 2)], [], []]
    assert final.total_udon == 4
    assert final._road_step_counts == {0: 12, 1: 17}


@pytest.mark.parametrize('change', ['short', 'long', 'fuel', 'missing', 'duplicate'])
def test_invalid_day_rejected_atomically(change):
    sim, state, orders = example()
    if change == 'short':
        orders[0].actions.pop()
    elif change == 'long':
        orders[0].actions.append(AgentAction('stay'))
    elif change == 'fuel':
        state.my_agents[0].fuel = 1  # Road move completes at step 1 before any refill.
    elif change == 'missing':
        orders.pop()
    else:
        orders.append(orders[0])
    before = deepcopy(state)
    assert not validate_orders(orders, state, sim.map, sim.grid)[0]
    with pytest.raises(ValueError):
        sim.apply_day(state, orders)
    assert state == before
    assert sim.traffic._history == []


def test_order_and_state_list_order_do_not_change_tie_break():
    sim, state, orders = example()
    expected, _ = sim.apply_day(state, orders)
    sim.traffic.reset()
    state.my_agents.reverse()
    actual, _ = sim.apply_day(state, list(reversed(orders)))
    assert actual.agents_by_id() == expected.agents_by_id()
    assert actual.total_udon == expected.total_udon


@pytest.mark.parametrize('steps,fuel', [(0, 5), (1, 20), (2, 20)])
def test_refill_needs_a_reflection_step(steps, fuel):
    cfg = MatchConfig(2, 1, 1, [steps], 1, 3, 10, 20)
    sim = HexaUdonSimulator(cfg, MapData([Cell(0, 0), Cell(1, 0)], [Spot(0, 0, 2)]))
    state = sim.reset([AgentState(1, 0, 0, 5), AgentState(2, 1, 0, 0)])
    orders = [DayOrder(i, [AgentAction('stay') for _ in range(steps)]) for i in (1, 2)]
    final, _ = sim.apply_day(state, orders)
    assert final.my_agents[0].fuel == fuel
    assert final.total_udon == int(steps > 0)


def test_swapping_adjacent_cells_is_not_refueling():
    cfg = MatchConfig(2, 1, 1, [1], 1, 3, 10, 20)
    sim = HexaUdonSimulator(cfg, MapData([Cell(0, 3), Cell(1, 3)], []))
    state = sim.reset([AgentState(1, 0, 0, 5), AgentState(2, 1, 1, 0)])
    final, _ = sim.apply_day(state, [DayOrder(1, [AgentAction('move', 2)]),
                                    DayOrder(2, [AgentAction('move', 5)])])
    assert [a.cell for a in final.my_agents] == [1, 0]
    assert final.my_agents[0].fuel == 3
    assert final._road_step_counts == {0: 1, 1: 1}


def test_empty_fuel_patrol_collects_again_next_day_without_moving():
    cfg = MatchConfig(2, 1, 2, [1, 1], 1, 3, 10, 20)
    sim = HexaUdonSimulator(cfg, MapData([Cell(0, 0), Cell(1, 0)], [Spot(0, 0, 1)]))
    state = sim.reset([AgentState(1, 0, 0, 0)])
    for day in (1, 2):
        state, _ = sim.apply_day(state, [DayOrder(1, [AgentAction('stay')])])
        assert state.total_udon == day
        assert state.my_agents[0].fuel == 0
