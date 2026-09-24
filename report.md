# Bàn giao dự án Procon2026 — HEXA UDON

Cập nhật: 24/09/2026. Đây là bản tóm tắt để người tiếp nhận tiếp tục từ trạng
thái hiện tại. Kết quả chi tiết và dữ liệu gốc được giữ trong `report/` và
`runs/`; không cần chạy lại pilot vừa hoàn tất.

## Mục tiêu và trạng thái

Bot dùng MAPPO cho cuộc thi Procon2026 HEXA UDON. Luồng hiện tại hỗ trợ input
BTC đã kiểm tra, curriculum 16×16 → 24×24 → 32×32, self-play, tuyến nhiều
spot và tuỳ chọn phân bổ tồn kho giữa các patrol.

Pilot cuối đã train **3 seed × 256 game** trên RTX 4050, có checkpoint đầu,
giữa và cuối, đánh giá validation tại game 0/128/256 và holdout mới sau khi
train. **Không train thêm ở thời điểm bàn giao**: kết quả chưa cho thấy cải
thiện đồng đều trên mọi seed. Báo cáo pilot là nguồn thông tin chính xác nhất:
[`report/PILOT_RESERVED_256.md`](report/PILOT_RESERVED_256.md).

## Kết quả pilot cuối

Thắng Lookahead trên holdout, mỗi ô đếm 20 map duy nhất. Mỗi map đã đấu ở hai
slot; điểm và kết quả hai slot khớp nhau nên chỉ tính một lần:

| Seed | 16×16 trước → sau | 24×24 trước → sau | 32×32 trước → sau | Tổng trước → sau |
| --- | ---: | ---: | ---: | ---: |
| 42 | 10/20 → 13/20 | 17/20 → 16/20 | 15/20 → 14/20 | 42/60 → 43/60 |
| 10042 | 14/20 → 10/20 | 17/20 → 18/20 | 15/20 → 17/20 | 46/60 → 45/60 |
| 20042 | 0/20 → 12/20 | 0/20 → 15/20 | 0/20 → 16/20 | 0/60 → 43/60 |

Tăng trưởng chủ yếu đến từ seed 20042, vốn thua cả 60 map trước train. Seed
42 tăng 1 trận; seed 10042 giảm 1. Ba seed dùng chung map, vì vậy không
diễn giải tổng 88/180 → 131/180 seed-map pair như 180 map độc lập hay như
tỷ lệ thắng khái quát. Validation gộp đạt 24/45, 40/45, 39/45 tại game
0/128/256. Sau train, chênh lệch `unique_series` so với Lookahead bằng 0
trung bình trên holdout; kết quả thắng còn do điểm daily series và udon.

Ổn định huấn luyện: 324 bước optimizer được chấp nhận, 22 bị KL guard
rollback; KL được chấp nhận lớn nhất 0,048192 < 0,05. Trọng số hữu hạn,
không NaN/Inf. Checkpoint cuối của cả ba seed nạp được, curriculum ở 32×32
và năm self-play snapshots mỗi seed giữ đúng hai cờ tuyến.

Toàn bộ số liệu/mốc/cấu hình: [`runs/pilot_reserved_001/pilot.json`](runs/pilot_reserved_001/pilot.json).
Checkpoint cuối:

- `runs/pilot_reserved_001/seed_42/episode_256.pt`
- `runs/pilot_reserved_001/seed_10042/episode_256.pt`
- `runs/pilot_reserved_001/seed_20042/episode_256.pt`

Một tiến trình train từng bị ngắt. Pilot đã được tiếp tục từ checkpoint 128
của seed 10042 bằng `--resume`; không train lại seed đã xong. `elapsed_seconds`
trong JSON chỉ tính lượt chạy tiếp tục, không phải tổng thời gian thực.

## Phần đã triển khai

- Luật, format input/output, timeline và map mẫu được lưu để tra cứu ở
  `report/results/official_rules_20260919/`. Thông tin số ngày, fuel,
  traffic hoặc phân bố map do generator tự đặt phải được gọi là giả định
  tạo dữ liệu nếu BTC không ấn định, không trình bày như luật thi.
- `src/env/map_generator.py`, `src/rl/curriculum.py`: preset và curriculum
  vòng thi 16/24/32.
- `src/rl/mappo.py`: decoder tuyến phụ và dành tồn kho, giữ nguyên target
  chính; cấu hình được ghi trong checkpoint.
- `src/rl/selfplay.py`: snapshot đối thủ lưu/khôi phục các cờ tuyến.
- `src/main.py`, `src/strategy/mcts.py`, `src/benchmark.py`: dùng chung
  decoder theo cấu hình checkpoint/CLI.
- `report/pilot_lookahead.py`: train, validation, holdout và resume. Bản
  resume hiện xử lý checkpoint đã lưu và bổ sung validation bị thiếu.
- Không thêm dependency. Các cờ `reserve_spots` và `secondary_routes` mặc
  định tắt ở model mới/các checkpoint cũ; pilot cần bật cả hai.

Đối chứng heuristic trước đó cho thấy dành tồn kho giảm lượt xe ghé vượt
tồn kho khoảng 72–75% trên map 24/32, fuel di chuyển gần như không đổi;
trên map 24/32 thắng tăng từ 2/120 lên 92/120 và từ 0/120 lên 102/120.
Đây là lợi ích của decoder so với decoder cũ, không phải bằng chứng PPO đã
học tốt hơn. Chi tiết: [`report/LARGE_MAP_ROUTING.md`](report/LARGE_MAP_ROUTING.md)
và [`report/RESOURCE_ROUTING.md`](report/RESOURCE_ROUTING.md).

## Việc người tiếp nhận nên làm

1. Đọc `report/PILOT_RESERVED_256.md` và `report/FINALS_PIPELINE.md` trước
   khi sửa logic luật hoặc khởi động train mới.
2. Chẩn đoán nguyên nhân seed 20042 khởi tạo thua toàn bộ holdout; so sánh
   `initial.pt`, `episode_128.pt`, `episode_256.pt`, phân bố logits/entropy,
   action/target, unique series, daily series, udon, fuel và các lần lấy spot.
3. Điều tra seed 10042 giảm ở 16×16 dù 24/32 tăng. Tách lỗi khởi tạo, nhiễu
   seed, hiện tượng quên theo kích thước và độ biến thiên của tập map; giữ
   nguyên holdout `14000000..14000019` làm kết quả cuối, không dùng để tune.
4. Chỉ đề xuất pilot mới sau khi có giả thuyết cụ thể và map đánh giá mới,
   không kéo dài train chỉ dựa vào tỷ lệ thắng gộp. Giữ nguyên checkpoint
   hiện tại; chạy thí nghiệm mới vào thư mục kết quả mới.

## Lệnh kiểm tra và chạy

Kiểm tra toàn bộ test:

```powershell
python -m pytest tests -q
```

Kết quả gần nhất trước bàn giao: **201 passed**, một warning dự kiến từ test
nạp checkpoint legacy.

Đánh giá lại checkpoint cuối trên map mới (cần một thư mục output chưa tồn
tại; ví dụ seed map mới `15000000..15000019`):

```powershell
python -u report/experiment_routes.py --pilot runs/pilot_reserved_001 --secondary-only --reserve-spots --sizes 16 24 32 --map-seed 15000000 --output-dir report/results/reserved_fresh_eval_001
```

Nếu sau phân tích có lý do để train tiếp, resume một checkpoint cuối vào
file mới, bật đồng nhất cả hai cờ và dùng thư mục log/checkpoint riêng. Ví dụ:

```powershell
python src/main.py train --finals --load runs/pilot_reserved_001/seed_42/episode_256.pt --episodes 128 --secondary-routes --reserve-spots --device cuda --save runs/continued_seed42/model.pt --log-dir runs/continued_seed42
```

Lưu ý: lệnh trên tiếp tục model seed 42; episode 128 là **số game bổ sung**.
Không ghi đè checkpoint pilot và không xem kết quả holdout cũ như validation
cho bất kỳ thay đổi mới nào.
