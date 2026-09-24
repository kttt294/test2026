"""
Pre-submission validator for DayOrders.

Catches problems before POST to server:
  - Invalid direction value (not 0-5)
  - Direction goes off the map edge
  - Destination is a lake (impassable)
  - Per-car timeline must cover exactly the day duration
  - Patrol fuel exhausted mid-route

Usage:
    ok, errors = validate_orders(orders, state, map_data, grid)
    if not ok:
        orders = fallback_orders  # use safe alternative

Contest strategy: submit greedy orders (~100ms), then try to compute
better orders and resubmit if validate_orders passes and time allows.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, List, Tuple

import config as C
from env.hex_grid import HexGrid
from env.models import CMD_MOVE, CMD_STAY, DayOrder, DayState, MapData


def validate_orders(
    orders:   List[DayOrder],
    state:    DayState,
    map_data: MapData,
    grid:     HexGrid,
) -> Tuple[bool, List[str]]:
    """Validate a complete day with the same simultaneous timeline as execution."""
    from env.simulator import execute_timeline
    try:
        execute_timeline(orders, state, map_data, grid)
    except ValueError as error:
        return False, [str(error)]
    return True, []
