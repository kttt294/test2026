from copy import deepcopy

from env.models import AgentState, Cell, MapData, MatchConfig, Spot
from env.simulator import HexaUdonSimulator
from env.validator import validate_orders
from rl.mappo import MAPPOTrainer


def test_secondary_route_keeps_primary_and_respects_fuel_and_stay():
    cfg=MatchConfig(4,1,1,[20],1,3,7,20)
    board=MapData([Cell(i,0) for i in range(4)],[Spot(1,1,1),Spot(2,2,1),Spot(3,3,1)])
    sim=HexaUdonSimulator(cfg,board)
    state=sim.reset([AgentState(0,0,0,2)])
    before=deepcopy(state)
    old=MAPPOTrainer._actions_to_orders(state,board,cfg,sim,[0],[0])
    new=MAPPOTrainer._actions_to_orders(state,board,cfg,sim,[0],[0],secondary_routes=True)
    assert state==before and validate_orders(new,state,board,sim.grid)[0]
    assert new[0].actions[0]==old[0].actions[0]
    old_state,_=sim.apply_day(state,old)
    sim.traffic.reset()
    new_state,_=sim.apply_day(state,new)
    assert old_state.collected_series=={1}
    assert new_state.collected_series=={1,2} and new_state.my_agents[0].fuel==0
    stay=MAPPOTrainer._actions_to_orders(state,board,cfg,sim,[0],[3],secondary_routes=True)
    assert all(a.cmd=='stay' for a in stay[0].actions)


def test_decoder_setting_survives_checkpoint_and_selfplay(tmp_path):
    from rl.selfplay import SelfPlayPool
    trainer=MAPPOTrainer(log_dir=None,selfplay=SelfPlayPool(update_every=1))
    trainer.model.secondary_routes=True
    trainer.model.reserve_spots=True
    trainer.selfplay.step(trainer.model)
    trainer.save(str(tmp_path/'routes.pt'))
    loaded=MAPPOTrainer(log_dir=None,selfplay=SelfPlayPool())
    loaded.load(str(tmp_path/'routes.pt'))
    assert loaded.model.secondary_routes and loaded.selfplay._pool[0].secondary_routes
    assert loaded.model.reserve_spots and loaded.selfplay._pool[0].reserve_spots
    # Metadata absent in older checkpoints must not inherit the new setting.
    import torch
    old=torch.load(tmp_path/'routes.pt',weights_only=True)
    del old['reserve_spots']
    del old['selfplay']['reserve_spots']
    torch.save(old,tmp_path/'old.pt')
    loaded.load(str(tmp_path/'old.pt'))
    assert not loaded.model.reserve_spots and not loaded.selfplay._pool[0].reserve_spots


def test_inventory_reservation_counts_waiting_and_reachable_pickups():
    cfg=MatchConfig(4,1,1,[20],1,3,7,20)
    board=MapData([Cell(i,0) for i in range(4)], [Spot(1,1,1),Spot(2,2,1),Spot(3,3,1)])
    sim=HexaUdonSimulator(cfg,board)
    state=sim.reset([AgentState(0,0,2,20),AgentState(1,0,0,2)])
    before=deepcopy(state)
    # Car 0 stays and takes spot 2. Car 1 retains primary spot 1, but should
    # not waste its last unit of fuel going to the exhausted secondary spot 2.
    old=MAPPOTrainer._actions_to_orders(state,board,cfg,sim,[0,1],[3,0],secondary_routes=True)
    new=MAPPOTrainer._actions_to_orders(state,board,cfg,sim,[0,1],[3,0],
                                       secondary_routes=True,reserve_spots=True)
    assert state==before and validate_orders(new,state,board,sim.grid)[0]
    assert sum(a.cmd=='move' for a in old[1].actions)==2
    assert sum(a.cmd=='move' for a in new[1].actions)==1
    assert new[1].actions[0]==old[1].actions[0]
    board.spots[1].max_inventory=2
    state.spot_inventory[2]=2
    shared=MAPPOTrainer._actions_to_orders(state,board,cfg,sim,[0,1],[3,0],
                                          secondary_routes=True,reserve_spots=True)
    assert sum(a.cmd=='move' for a in shared[1].actions)==2
