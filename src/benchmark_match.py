"""Head-to-head matches with shared traffic; no separate solo score comparison."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from env.map_generator import generate_contest_scenario as contest_scenario
from env.map_generator import generate_finals_scenario
from env.scoring import compute_score
from env.simulator import HexaUdonSimulator, apply_joint_day
from strategy.greedy import GreedyPlanner
from strategy.lookahead import LookaheadPlanner


def play_match(seed, size, strategies, planner_factories=None, record_replay=False, finals=False):
    cfg, board, agents = (generate_finals_scenario if finals else contest_scenario)(seed, size)
    cfg.n_teams = len(strategies)
    sims = [HexaUdonSimulator(cfg, board) for _ in strategies]
    states = [sim.reset(agents) for sim in sims]  # Identical starting cells/types/fuel.
    for state in states:
        state.opponent_cells = [a.cell for _ in range(len(states)-1) for a in agents]
    kinds = {'greedy': GreedyPlanner, 'lookahead': LookaheadPlanner,
             'lookahead1': LookaheadPlanner, 'lookahead3': LookaheadPlanner, 'lookahead5': LookaheadPlanner}
    if planner_factories:
        kinds.update(planner_factories)
    planners = [kinds[name](cfg, board, sim) for name, sim in zip(strategies, sims)]
    for name, planner in zip(strategies, planners):
        if name in ('lookahead1', 'lookahead3', 'lookahead5'):
            planner.MAX_SECONDARY = int(name[-1])
    times, days = [[] for _ in strategies], []
    while not sims[0].is_done(states[0]):
        orders = []
        for i, (planner, state) in enumerate(zip(planners, states)):
            start = time.perf_counter()
            orders.append(planner.plan(state))
            times[i].append((time.perf_counter()-start)*1000)
        states = apply_joint_day(sims, states, orders)
        days.append(dict(day=states[0].day-1, traffic=states[0].traffic,
                         scores=[asdict(compute_score(s)) for s in states]))
        if record_replay:
            days[-1]['orders'] = [[asdict(order) for order in team] for team in orders]
            days[-1]['agents'] = [[asdict(a) for a in state.my_agents] for state in states]
    scores = [compute_score(s) for s in states]
    result = dict(seed=seed, size=size, scores=[asdict(s) for s in scores],
                winners=[i for i, score in enumerate(scores) if score == max(scores)],
                planning_ms=times, days=days)
    if record_replay:
        result.update(config=asdict(cfg), board=asdict(board),
                      initial_agents=[asdict(a) for a in agents], strategies=strategies)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sizes', nargs='+', type=int, choices=[8, 16, 32], default=[8, 16, 32])
    parser.add_argument('--strategies', nargs='+', choices=['greedy', 'lookahead', 'lookahead1', 'lookahead3', 'lookahead5'], default=['greedy', 'lookahead'])
    parser.add_argument('--output', default='report/results/joint_benchmark.json')
    args = parser.parse_args()
    if args.games < 1 or len(args.strategies) < 2:
        parser.error('Positive games and at least two teams required')
    report = dict(rules='btc_even_r_v2', strategies=args.strategies,
                  lookahead_default_secondary=LookaheadPlanner.MAX_SECONDARY,
                  seed=args.seed, games_per_size=args.games, suites={})
    for size in args.sizes:
        games = [play_match(args.seed+i, size, args.strategies) for i in range(args.games)]
        summary = []
        for i, strategy in enumerate(args.strategies):
            times = [t for game in games for t in game['planning_ms'][i]]
            summary.append(dict(team=i, strategy=strategy,
                avg_series=sum(g['scores'][i]['unique_series'] for g in games)/args.games,
                avg_daily=sum(g['scores'][i]['daily_series_sum'] for g in games)/args.games,
                avg_udon=sum(g['scores'][i]['total_udon'] for g in games)/args.games,
                outright_wins=sum(g['winners']==[i] for g in games),
                tied_first=sum(i in g['winners'] and len(g['winners'])>1 for g in games),
                max_day_ms=max(times), avg_day_ms=sum(times)/len(times)))
        report['suites'][size] = dict(summary=summary, games=games)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(size, json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
