from collections import deque
from unittest.mock import Mock

import pytest

from client.http_client import ContestClient
from env.hex_grid import HexGrid
from env.models import AgentAction, DayOrder


def test_btc_six_directions_both_parities_and_distance():
    grid = HexGrid(8, 8)
    assert [grid.neighbor_in_dir(18, d) for d in range(6)] == [10, 11, 19, 27, 26, 17]
    assert [grid.neighbor_in_dir(26, d) for d in range(6)] == [17, 18, 27, 34, 33, 25]
    for start in range(64):
        distances, queue = {start: 0}, deque([start])
        while queue:
            cell = queue.popleft()
            for _, neighbor in grid.neighbors(cell):
                if neighbor not in distances:
                    distances[neighbor] = distances[cell] + 1
                    queue.append(neighbor)
        assert all(grid.hex_distance(start, cell) == distance for cell, distance in distances.items())


def setting():
    return dict(startsAt=0, daySeconds=[5, 5], daySteps=[16, 16],
                map=dict(width=8, height=8, cells=[[0, 1, 2, 3, 0, 0, 0, 0]]*8),
                spots=[dict(brand=7, pos=4, stocks=2)], agents=[4, 5, 6],
                fuelLimits=20, players=2, busyThreshold=2, jammedThreshold=4)


def test_official_setting_and_payload(monkeypatch):
    client = ContestClient('http://test', token='fixture-only')
    client._get = Mock(return_value=setting())
    cfg, mp, agents = client.get_match_config()
    client._get.assert_called_once_with('/setting')
    assert client.session.headers['Procon-Token'] == 'fixture-only'
    assert [mp.cell_map[i].terrain for i in range(4)] == [0, 3, 1, 2]
    assert cfg.fuel_max == 20 and cfg.total_days == 2
    assert [a.id for a in agents] == [0, 1, 2]
    assert mp.spots[0].series_id == 7
    client._post = Mock(return_value={'revision': 0})
    client.submit_agent_kinds([0, 1, 0])
    client._post.assert_called_with('/agent', [0, 1, 0])


def test_official_encode_waits_in_agent_order():
    orders = [DayOrder(1, [AgentAction('stay')]*3),
              DayOrder(0, [AgentAction('move', 2), AgentAction('stay'), AgentAction('stay')])]
    assert ContestClient.encode_orders(orders) == [[2, -2], [-3]]
    with pytest.raises(ValueError):
        ContestClient.encode_orders([orders[0], orders[0]])


def test_day_zero_maps_to_day_one_and_deadline(monkeypatch):
    client = ContestClient('http://test')
    client._get = Mock(return_value=setting())
    cfg, mp, _ = client.get_match_config()
    monkeypatch.setattr('client.http_client.time.time', lambda: 100.)
    client._get = Mock(return_value=dict(day=0, endsAt=104, agents=[dict(kind=0, pos=4, fuel=20)],
                                       others=[], traffics=[dict(pos=1, status=2)]))
    state = client.get_day_state(1, cfg, mp)
    assert state.day == 1 and state.steps_left == 16 and state.time_limit_ms == 4000
    assert state.traffic == {1: 2} and state.spot_inventory == {4: 2}


def test_rejected_revision_preserves_accepted_plan_and_score(monkeypatch):
    client = ContestClient('http://test')
    data = setting()
    data['agents'] = [4]
    client._get = Mock(return_value=data)
    cfg, mp, _ = client.get_match_config()
    monkeypatch.setattr('client.http_client.time.time', lambda: 100.)
    wire = dict(day=0, endsAt=104, agents=[dict(kind=0, pos=4, fuel=20)], others=[], traffics=[])
    client._get = Mock(return_value=wire)
    state = client.get_day_state(1, cfg, mp)
    client._post = Mock(side_effect=[{'revision': 2}, {'revision': -1}])
    idle = [DayOrder(0, [AgentAction('stay')]*16)]
    assert client.submit_orders(1, idle)['status'] == 'valid'
    assert client.submit_orders(1, [DayOrder(0, [AgentAction('stay')])])['status'] == 'invalid'
    wire['day'] = 1
    following = client.get_day_state(2, cfg, mp, state)
    assert following.total_udon == 1 and following.collected_series == {7}
    assert following.daily_series == [{7}]


def test_auth_failure_is_not_polled(monkeypatch):
    import requests
    client = ContestClient('http://test')
    client._get = Mock(return_value=setting())
    cfg, mp, _ = client.get_match_config()
    response = Mock(status_code=401)
    client._get = Mock(side_effect=requests.HTTPError(response=response))
    with pytest.raises(requests.HTTPError):
        client.get_day_state(1, cfg, mp)
    assert client._get.call_count == 1
