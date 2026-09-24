import torch
import pytest

from client.http_client import ContestClient
from env.map_generator import generate_finals_scenario
from env.models import AgentAction, DayOrder
from env.simulator import HexaUdonSimulator
from rl.curriculum import CurriculumEngine
from rl.mappo import MAPPOTrainer


def test_pdf_format_examples(monkeypatch):
    client = ContestClient('http://unused.invalid')
    data = dict(startsAt=1778227200, daySeconds=[5,5,5,10], daySteps=[50,100,150,200],
        map=dict(height=8, width=8, cells=[[3,0,1,2,0,1,2,0]]*8),
        spots=[dict(brand=b,pos=p,stocks=s) for b,p,s in [(0,1,4),(1,9,1),(0,17,1),(1,25,3)]],
        agents=[4,12,20,28], fuelLimits=20, players=8, busyThreshold=2, jammedThreshold=4)
    client._get = lambda _: data
    cfg, board, agents = client.get_match_config()
    assert cfg.total_days == 4 and cfg.steps_per_day == [50,100,150,200]
    assert [c.terrain for c in board.cells[:4]] == [2,0,3,1]
    assert [a.cell for a in agents] == [4,12,20,28]
    assert board.series_ids == [0,1] and board.spots[0].max_inventory == 4
    data = dict(endsAt=1778227205,day=1,
        agents=[dict(kind=k,pos=p,fuel=f) for k,p,f in [(0,1,20),(1,1,20),(0,9,10),(0,9,0)]],
        others=[dict(id=0,agents=[dict(kind=0,pos=1,fuel=i) for i in [2,3,4,5]])],
        traffics=[dict(pos=p,status=s) for p,s in [(1,0),(9,0),(17,1),(25,2)]])
    monkeypatch.setattr('client.http_client.time.time',lambda:1778227200)
    state = client.get_day_state(2,cfg,board)
    assert state.day == 2 and state.steps_left == 100 and state.time_limit_ms == 5000
    assert state.traffic == {1:0,9:0,17:1,25:2}
    assert state.opponent_cells == [1]*4
    orders = [DayOrder(0,[AgentAction('stay')]*15),
              DayOrder(1,[AgentAction('move',0),AgentAction('move',1)]+[AgentAction('stay')]*10)]
    assert client.encode_orders(orders) == [[-15],[0,1,-10]]


def test_finals_ranges_global_encoding_and_legacy_load(tmp_path):
    torch.set_num_threads(2)
    trainer = MAPPOTrainer(log_dir=None, curriculum=CurriculumEngine(finals=True))
    assert trainer.max_series == 28
    for size,count,low,high in [(16,4,10,14),(24,5,14,20),(32,7,20,28)]:
        for seed in (0, 1, 2, 3, 17):
            cfg, board, agents = generate_finals_scenario(seed,size)
            assert len(agents)==count and len(board.spots)==size and low<=board.n_series<=high
            state = HexaUdonSimulator(cfg,board).reset(agents)
            state.collected_series=set(board.series_ids)
            captured=[]
            hook=trainer.model.global_mlp[0].register_forward_pre_hook(lambda _,inputs:captured.append(inputs[0]))
            with torch.no_grad():
                trainer.model(state,board,cfg,[a.id for a in state.patrol_agents()])
            hook.remove()
            offset=trainer.model.hidden//2
            assert captured[0][0,offset:offset+28].sum().item()==board.n_series
            if size == 32 and seed == 17:
                assert board.n_series == 28
    saved=tmp_path/'new.pt'
    trainer.save(str(saved))
    loaded=MAPPOTrainer(max_series=10,log_dir=None,curriculum=CurriculumEngine())
    loaded.load(str(saved))
    assert loaded.max_series==28 and loaded.curriculum.finals
    assert all(torch.equal(x,y) for x,y in zip(trainer.model.parameters(),loaded.model.parameters()))
    old=MAPPOTrainer(max_series=10,log_dir=None)
    old.save(str(tmp_path/'old.pt'))
    loaded.load(str(tmp_path/'old.pt'))
    assert loaded.max_series==10
    with pytest.raises(ValueError,match='capacity'):
        loaded.model(state,board,cfg,[agents[0].id])
