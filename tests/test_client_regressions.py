from types import SimpleNamespace
from unittest.mock import Mock, patch
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from client.http_client import ContestClient
from env.models import AgentAction, AgentState, Cell, DayOrder, MapData, MatchConfig, Spot
from env.simulator import HexaUdonSimulator
from main import run_play


def test_retry_caps_http_timeout_to_remaining_deadline():
    client = ContestClient('http://test')
    response = Mock()
    response.json.return_value = {'revision': 1}
    client.session.post = Mock(return_value=response)
    with patch('client.http_client.time.monotonic', return_value=10.4):
        result = client.submit_with_retry(1, [], [], 1000, 10.)
    assert result['status'] == 'valid'
    assert 0 < client.session.post.call_args.kwargs['timeout'] <= .6


def test_retry_is_bounded_and_reports_the_actual_accepted_fallback():
    client = ContestClient('http://test')
    original = [DayOrder(0, [AgentAction('move', 2)])]
    client.submit_orders = Mock(side_effect=[{'status': 'invalid'}, {'status': 'valid'}])
    with patch('client.http_client.time.monotonic', return_value=10.):
        result = client.submit_with_retry(1, original, [], 1000, 10.)
    assert result['_accepted_orders'] == []
    assert client.submit_orders.call_count == 2
    client.submit_orders.reset_mock()
    with patch('client.http_client.time.monotonic', return_value=12.):
        assert client.submit_with_retry(1, original, [], 1000, 10.)['status'] == 'failed'
    client.submit_orders.assert_not_called()


def test_play_submits_before_advanced_planner_and_survives_failure():
    cfg = MatchConfig(8, 8, 1, [20], 2, 3, 7, 20)
    mp = MapData([Cell(i, 0) for i in range(64)], [Spot(1, 1, 3)])
    sim = HexaUdonSimulator(cfg, mp)
    state = sim.reset([AgentState(0, 0, 0, 20), AgentState(1, 1, 8), AgentState(2, 1, 9)])
    events = []
    client = Mock()
    client.get_match_config.return_value = (cfg, mp, state.my_agents)
    client.get_day_state.return_value = state
    def submit(*args, **kwargs):
        events.append('submit')
        return {'status': 'valid'}
    client.submit_with_retry.side_effect = submit
    client.submit_orders.side_effect = submit
    def advanced(*args):
        assert events == ['submit']
        events.append('plan')
        raise RuntimeError('planner failed')
    with patch('client.http_client.ContestClient', return_value=client), \
         patch('main.LookaheadPlanner') as planner:
        planner.return_value.plan.side_effect = advanced
        run_play(SimpleNamespace(url='http://test', model=None, mcts=False))
    assert events[:2] == ['submit', 'plan']


def test_play_does_not_submit_invalid_advanced_orders():
    cfg = MatchConfig(8, 8, 1, [20], 2, 3, 7, 20)
    mp = MapData([Cell(i, 0) for i in range(64)], [Spot(1, 1, 3)])
    state = HexaUdonSimulator(cfg, mp).reset([AgentState(0, 0, 0, 20)])
    client = Mock()
    client.get_match_config.return_value = (cfg, mp, state.my_agents)
    client.get_day_state.return_value = state
    client.submit_with_retry.return_value = {'status': 'valid'}
    with patch('client.http_client.ContestClient', return_value=client), \
         patch('main.LookaheadPlanner') as planner:
        planner.return_value.plan.return_value = [DayOrder(0, [AgentAction('move', 99)])]
        run_play(SimpleNamespace(url='http://test', model=None, mcts=False))
    assert client.submit_with_retry.call_args.kwargs['orders'][0].actions[0].direction == 2
    client.submit_orders.assert_not_called()


@pytest.mark.parametrize('reject_first', [False, True])
def test_full_fallback_chain_with_local_http_server(tmp_path, reject_first):
    """Real HTTP transport; injected planner failures exercise every tier."""
    posts, tiers = [], []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            import time
            if self.path == '/setting':
                self.respond(dict(startsAt=0, daySeconds=[5], daySteps=[20],
                    map=dict(width=8, height=8, cells=[[0]*8 for _ in range(8)]),
                    spots=[dict(brand=1, pos=1, stocks=1)], agents=[0],
                    fuelLimits=20, players=2, busyThreshold=3, jammedThreshold=7))
            else:
                assert self.path == '/'
                self.respond(dict(day=0, endsAt=time.time()+5,
                    agents=[dict(kind=0, pos=0, fuel=20)], others=[], traffics=[]))

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/agent':
                assert body == [0]
                self.respond({})
                return
            assert self.path == '/'
            posts.append(body)
            invalid = reject_first and len(posts) == 1
            self.respond(dict(revision=-1 if invalid else len(posts)))

    def fail(tier):
        def invoke(*args, **kwargs):
            tiers.append(tier)
            raise RuntimeError(f'{tier} unavailable')
        return invoke

    checkpoint = tmp_path / 'stub.pt'
    checkpoint.touch()  # Loading is mocked; this is not a trained checkpoint.
    trainer = Mock()
    trainer.model.get_action_and_value.side_effect = fail('rl')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with patch('rl.mappo.MAPPOTrainer', return_value=trainer), \
             patch('strategy.mcts.MCTSPlanner') as mcts, \
             patch('main.LookaheadPlanner') as lookahead:
            mcts.return_value.plan.side_effect = fail('mcts')
            lookahead.return_value.plan.side_effect = fail('lookahead')
            run_play(SimpleNamespace(url=f'http://127.0.0.1:{server.server_port}',
                                     model=str(checkpoint), mcts=True))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert tiers == ['mcts', 'rl', 'lookahead']
    assert len(posts) == (2 if reject_first else 1)
    assert posts[0] == [[2, -18]]
    if reject_first:
        assert posts[1] == [[-20]]
