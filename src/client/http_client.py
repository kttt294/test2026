"""BTC Procon 37 API. Internal days start at 1; wire days start at 0."""
from __future__ import annotations
from copy import deepcopy
import os
import time
import requests
from env.models import AgentState, Cell, DayOrder, DayState, MapData, MatchConfig, Spot
from env.simulator import HexaUdonSimulator, complete_orders


class ContestClient:
    def __init__(self, base_url: str, timeout_s: float = 10., token: str | None = None):
        self.base, self.timeout = base_url.rstrip('/'), timeout_s
        self.session = requests.Session()
        token = token if token is not None else os.environ.get('PROCON_TOKEN')
        if token:
            self.session.headers['Procon-Token'] = token
        self._last_request = 0.
        self._state = self._accepted = None
        self._revision = -1
        self._uncertain = False

    def get_match_config(self):
        data = self._get('/setting')
        board = data['map']
        cfg = MatchConfig(board['width'], board['height'], len(data['daySteps']), data['daySteps'],
                          data['players'], data['busyThreshold'], data['jammedThreshold'], data['fuelLimits'])
        terrain = {0: 0, 1: 3, 2: 1, 3: 2}  # BTC -> existing internal encoding
        mp = MapData([Cell(r*cfg.width+c, terrain[t]) for r, row in enumerate(board['cells']) for c, t in enumerate(row)],
                     [Spot(s['pos'], s['brand'], s['stocks']) for s in data['spots']])
        agents = [AgentState(i, 0, cell, cfg.fuel_max) for i, cell in enumerate(data['agents'])]
        self._agent_count = len(agents)
        self._wait_seconds = max(120, sum(data['daySeconds']) + 120)
        return cfg, mp, agents

    def submit_agent_kinds(self, kinds):
        if len(kinds) != self._agent_count or any(type(k) is not int or k not in (0, 1) for k in kinds):
            raise ValueError('One kind (0 or 1) required per agent')
        self._post('/agent', kinds)

    def get_day_state(self, day, cfg, map_data, prev_state=None):
        stop = time.monotonic() + self._wait_seconds
        while True:
            try:
                data = self._get('/')
                actual_day = data['day'] + 1
                if actual_day > day:
                    raise RuntimeError(f'Missed day {day}; server is at day {actual_day}')
                if actual_day == day:
                    break
            except requests.HTTPError as error:
                if error.response.status_code != 403:
                    raise
            if time.monotonic() >= stop:
                raise TimeoutError(f'Timed out waiting for day {day}')
            time.sleep(.25)
        history = prev_state
        if self._state is not None and day > self._state.day:
            if self._uncertain:
                raise RuntimeError('Submission outcome unknown; cannot reconstruct score safely')
            sim = HexaUdonSimulator(cfg, map_data)
            orders = self._accepted if self._accepted is not None else complete_orders([], self._state, map_data, sim.grid)
            history, _ = sim.apply_day(self._state, orders)
            if len(history.my_agents) != len(data['agents']) or any(
                    a.cell != b['pos'] or (a.is_patrol() and a.fuel != b['fuel'])
                    for a, b in zip(history.my_agents, data['agents'])):
                raise RuntimeError('Server state differs from accepted plan; local score is untrusted')
            self._accepted, self._revision = None, -1
        state = DayState(day=day, steps_left=cfg.steps_per_day[day-1],
                         time_limit_ms=max(0, int((data['endsAt'] - time.time()) * 1000)),
                         traffic={t['pos']: t['status'] for t in data['traffics']},
                         spot_inventory={s.cell_id: s.max_inventory for s in map_data.spots},
                         my_agents=[AgentState(i, a['kind'], a['pos'], a['fuel']) for i, a in enumerate(data['agents'])],
                         opponent_cells=[a['pos'] for team in data['others'] for a in team['agents']],
                         collected_series=set(history.collected_series) if history else set(),
                         daily_series=deepcopy(history.daily_series) if history else [],
                         total_udon=history.total_udon if history else 0, fuel_max=cfg.fuel_max)
        self._day_deadline = time.monotonic() + state.time_limit_ms / 1000
        self._state = deepcopy(state)
        return state

    @staticmethod
    def encode_orders(orders):
        ordered = sorted(orders, key=lambda o: o.agent_id)
        if [o.agent_id for o in ordered] != list(range(len(ordered))):
            raise ValueError('Orders require unique consecutive agent IDs starting at zero')
        payload = []
        for order in ordered:
            actions = []
            for action in order.actions:
                if action.cmd == 'stay' and action.direction is None:
                    if actions and actions[-1] < 0:
                        actions[-1] -= 1
                    else:
                        actions.append(-1)
                elif action.cmd == 'move' and type(action.direction) is int and action.direction in range(6):
                    actions.append(action.direction)
                else:
                    raise ValueError('Invalid action')
            payload.append(actions)
        return payload

    def submit_orders(self, day, orders, timeout_s=None):
        if self._state is not None and day != self._state.day:
            raise ValueError('Cannot submit orders for a different day')
        if hasattr(self, '_agent_count') and len(orders) != self._agent_count:
            raise ValueError('Orders required for every agent')
        if self._state is not None:
            remaining = self._day_deadline - time.monotonic() - .05
            timeout_s = min(self.timeout if timeout_s is None else timeout_s, remaining)
            if timeout_s <= 0:
                raise TimeoutError('Day deadline expired')
        try:
            response = self._post('/', self.encode_orders(orders), timeout_s)
        except requests.RequestException:
            self._uncertain = True
            raise
        revision = response.get('revision')
        if type(revision) is not int:
            self._uncertain = True
            raise ValueError('Server response missing integer revision')
        if revision >= 0 and revision >= self._revision:
            self._revision, self._accepted = revision, deepcopy(orders)
            self._uncertain = False
        return {**response, 'status': 'valid' if revision >= 0 else 'invalid'}

    def submit_with_retry(self, day, orders, fallback_orders, deadline_ms, start_ms):
        for candidate in (orders, fallback_orders):
            remaining = start_ms + deadline_ms / 1000 - time.monotonic() - .05
            if remaining <= 0:
                break
            try:
                result = self.submit_orders(day, candidate, timeout_s=remaining)
                if result['status'] == 'valid':
                    return {**result, '_accepted_orders': candidate}
            except (requests.RequestException, ValueError, TimeoutError) as error:
                print(f'[client] Submission failed: {type(error).__name__}')
        return {'status': 'failed'}

    def _request(self, method, path, body=None, timeout_s=None):
        budget = self.timeout if timeout_s is None else min(self.timeout, timeout_s)
        delay = max(0., .21 - (time.monotonic() - self._last_request))
        if budget <= delay:
            raise TimeoutError('Submission deadline expired')
        if delay:
            time.sleep(delay)
        self._last_request = time.monotonic()
        if method == 'GET':
            response = self.session.get(self.base + path, timeout=budget-delay)
        else:
            response = self.session.post(self.base + path, json=body, timeout=budget-delay)
        response.raise_for_status()
        return response.json() if response.content else {}

    def _get(self, path):
        return self._request('GET', path)

    def _post(self, path, body, timeout_s=None):
        return self._request('POST', path, body, timeout_s)
