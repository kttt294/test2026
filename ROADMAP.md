# HEXA UDON — Roadmap (mục tiêu: Top 1)

> Ký hiệu: ✅ Đã xong | ⬜ Chưa làm

---

## THÁNG 1 — Nền tảng vững + Heuristic hoàn chỉnh

> Mục tiêu cuối tháng: có baseline tin cậy để benchmark mọi thứ sau này.
> Nếu simulator còn bug → mọi thứ ở tháng 2 đều vô nghĩa.

### Simulator

- ✅ `src/env/hex_grid.py` — hex grid, adjacency, hex_distance
- ✅ `src/env/models.py` — MatchConfig, MapData, DayState, DayOrder, AgentState
- ✅ `src/env/simulator.py` — apply_day(), traffic model, fuel model, scoring
- ⬜ Unit test simulator — từng rule một:
  - di chuyển vào ao → bị chặn
  - thiếu fuel tại bước di chuyển → từ chối toàn bộ kế hoạch
  - mỗi xe có đủ steps_left trên timeline đồng thời
  - traffic tính đúng từ 2 ngày trước (ngày 1 = clear, ngày 2 = chỉ dùng ngày 1)
  - udon chỉ lấy 1 lần / spot / ngày / xe (dù ghé nhiều lần)
  - inventory refill về max đầu mỗi ngày mới
  - tồn kho riêng từng đội (đối thủ lấy không ảnh hưởng)
- ✅ Random map generator (`src/env/map_generator.py`):
  - sinh map ngẫu nhiên trong range 8×8 → 32×32
  - vary: tỉ lệ terrain, số spot, số series, phân bố spot
  - seed-based để reproduce

### Pathfinding

- ✅ `src/pathfinding/astar.py` — Weighted A\*, multi_waypoint_path

### Heuristic

- ✅ `src/strategy/greedy.py` — greedy baseline
- ✅ `src/strategy/lookahead.py` — look-ahead heuristic:
  - ✅ global series assignment (không 2 xe waste vào cùng uncollected series)
  - ✅ multi-spot routing mỗi ngày
  - ✅ ngân sách thời gian riêng cho từng xe
  - ✅ supply intercept prediction (push model: supply tự di chuyển đến patrol)
  - ✅ hybrid fuel fallback: nếu supply dự báo không kịp đến → patrol chủ động chèn rendezvous waypoint trên đường đến supply
  - ✅ end-of-day repositioning
- ✅ `src/strategy/agent_selector.py` — **free ELO, thường bị bỏ qua**:
  - Input: map layout, vị trí start, series distribution
  - Output: n_patrol vs n_supply tối ưu + vị trí nào làm patrol/supply
  - Approach: score vị trí (patrol = gần nhiều series; supply = trung tâm các xe), simulate top-K combo với LookaheadPlanner → chọn theo unique_series → daily_series_sum → total_udon
- ✅ Benchmark runner (`src/benchmark.py`):
  - chạy N game với random maps, thống kê avg unique_series / avg udon / win rate
  - dùng để so sánh Greedy vs Lookahead
- ⬜ Parameter tuning cho Lookahead (random search, ~500 games):
  - `new_series_bonus`, `secondary_max`, `low_fuel_threshold`, `reposition_weight`

### Tests

- ✅ `tests/test_hex_grid.py`:
  - neighbor đúng 6 hướng, edge cell không out-of-bound
  - `hex_distance` vs expected cube coordinate values
- ✅ `tests/test_astar.py`:
  - path tìm được, cost đúng, lake bị chặn
  - fuel budget cắt đúng chỗ, step budget cắt đúng chỗ
- ✅ `tests/test_simulator.py`:
  - di chuyển vào lake bị block
  - thiếu fuel khiến kế hoạch bị từ chối
  - `steps_left` riêng cho mỗi xe, xử lý đồng thời
  - traffic dùng đúng 2 ngày trước (ngày 1 = clear, ngày 2 = chỉ dùng ngày 1)
  - udon chỉ lấy 1 lần / spot / ngày / xe
  - inventory refill về max đầu ngày mới
  - tồn kho riêng từng đội (đối thủ lấy không ảnh hưởng)
- ✅ `tests/test_validator.py` — action hợp lệ / không hợp lệ

### Validator & Scoring

- ✅ `src/env/validator.py` — kiểm tra `List[DayOrder]` trước khi mô phỏng:
  - cell đích kề ô hiện tại (direction hợp lệ)
  - không đi vào lake
  - tổng thời lượng của từng xe bằng `steps_left`
  - patrol: đủ nhiên liệu tại mỗi bước, có xét tiếp tế trên timeline
  - output: `(is_valid: bool, errors: List[str])`
- ✅ `src/env/scoring.py` — extract pure function từ simulator:
  - `compute_score(state) -> Score` với 3 tiêu chí: unique_series → daily_series_sum → total_udon
  - Dùng để optimizer so sánh 2 plan mà không cần reset simulator

### Visualizer & Replay

- ✅ `visualizer/terminal.py` — terminal ASCII, không cần GUI:
  - Grid: `P1`=patrol, `S1`=supply, `**`=spot còn hàng, `..`=spot hết, `##`=lake, `==`/`=B`/`=C`=road, `^^`=mountain
  - Thanh fuel `[████░░]` từng xe patrol
  - Series đã collect / còn lại
- ✅ `replay/` — ghi JSON mỗi ngày (day, state, orders, agent_positions, collected_series, traffic):
  - `ReplayRecorder.record(state, orders)` → `save(path)`
  - `ReplayPlayer.load(path)` → iterate frames, `agent_trace(id)`, `summary()`
  - Dùng để debug bot sau trận

### Traffic Predictor

- ✅ `src/env/traffic_predictor.py` — dự đoán traffic ngày mai:
  - Heuristic: giả định đối thủ dùng greedy đến spot gần nhất → A\* path → đếm bước ở road cell
  - `predict_and_scale()`: scale theo số đội ẩn (chỉ thấy 1 phần vị trí đối thủ)
  - Cộng vào `TrafficModel` → biết ngày mai đường nào kẹt để lookahead tránh chủ động

### Checkpoint cuối tháng 1

```
✓ Simulator pass toàn bộ unit test
✓ Lookahead > Greedy trên benchmark 100 random games
✓ Random map generator chạy được, tái hiện được
```

---

## THÁNG 2 — Tích hợp + Demo bài tập lớn

### Khi BTC công bố thông số còn thiếu

- ⬜ Cập nhật `steps_per_day` → re-tune budget allocation trong Lookahead
- ⬜ Cập nhật `n_teams`, traffic thresholds → re-calibrate traffic model

### Kiểm thử và replay

- ✅ Debug logger: `replay/recorder.py` + `replay/player.py` lưu JSON mỗi ngày → replay sau trận
- ⬜ Edge case testing:
  - map chỉ 1 series → chiến thuật hoàn toàn khác
  - fuel_max rất thấp → supply car critical
  - map toàn đường → traffic dominates
  - tất cả spot bị đội khác khai thác nhanh → phải linh hoạt

### Checkpoint cuối tháng 2

```
✓ Agent selector chạy được trước trận
✓ Sẵn sàng demo bài tập lớn
```

---

## Điểm quyết định quan trọng

| Thời điểm      | Câu hỏi                            | Nếu NO                                               |
| -------------- | ---------------------------------- | ---------------------------------------------------- |
| Tuần 2 tháng 1 | Simulator pass hết unit test?      | Dừng mọi thứ, fix trước                              |
| Cuối tháng 1   | Lookahead > Greedy trên benchmark? | Debug lookahead                                      |

---

## Thứ tự ưu tiên tuyệt đối

```
1. Simulator đúng        → nền tảng của mọi thứ
2. Benchmark runner      → không có thước đo thì không biết đang đi đúng không
3. Agent type selector   → free ELO, ít team làm
4. Demo và replay       → trình bày kết quả mô phỏng
```
