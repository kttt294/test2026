# Lập kế hoạch đa tác tử trên bản đồ lục giác

Bài tập lớn Thuật toán ứng dụng: tự cài đặt và đánh giá thuật toán tìm đường,
chiến lược chọn mục tiêu và phối hợp xe dưới ràng buộc thời gian, nhiên liệu.
Bản hiện tại chạy offline, không sử dụng ML/RL; runtime chỉ cần Python stdlib.

## Chạy

Từ thư mục gốc dự án:

```bash
python3 src/main.py sim
python3 -m pip install pytest
python3 -m pytest tests/ -q
python3 src/benchmark_pathfinding.py --games 100 --sizes 8 16 32 --output results/pathfinding.csv
python3 src/benchmark.py --games 100 --sizes 8 16 32
python3 src/benchmark_match.py --games 100 --sizes 8 16 32 --output results/matches.json
```

`--games` là số trường hợp cho **mỗi kích thước**, seed mặc định là 42.
CSV tìm đường lưu từng phép đo, kể cả trường hợp không tìm được đường.

## Hai lớp thuật toán

- `src/pathfinding/baselines.py`: DFS dùng stack, BFS dùng queue, Dijkstra và
  Greedy Best-First dùng priority queue. A* được tái sử dụng từ `astar.py`.
- `src/pathfinding/astar.py`: A* giữ các nhãn chi phí không trội nhau khi xét fuel.
- `src/strategy/greedy.py`: chọn spot ưu tiên series chưa thu.
- `src/strategy/lookahead.py`: phân công nhiều spot và so sánh kế hoạch thông qua
  mô phỏng tiếp đến cuối trận bằng Greedy.
- `src/env/simulator.py`: thực thi timeline đồng thời, thu udon, tiếp tế và giao thông.
- `src/env/validator.py`: kiểm tra kế hoạch với cùng luật thực thi.

BFS tối thiểu số cạnh, Dijkstra/A* tối thiểu chi phí step trong giới hạn nhiên liệu.
DFS và Greedy Best-First tìm đường hợp lệ, không bảo đảm tối ưu. Các baseline tìm
đường giữ nhãn tài nguyên để không loại nhầm một đường chậm hơn nhưng tiết kiệm fuel.
Chúng không mô hình hóa tiếp tế giữa đường; tiếp tế được xử lý ở lớp toàn trận.

Greedy Best-First tìm đường khác với Greedy chọn spot. Hiện planner toàn trận vẫn
sử dụng A*; benchmark tìm đường là thí nghiệm độc lập, không gọi BFS/DFS là một
chiến lược giải toàn trận.

## Quy trình đánh giá và so sánh RL sau này

1. Tìm đường: cùng bản đồ, nguồn/đích, traffic, step và fuel; đo tỷ lệ thành công,
   số cạnh, chi phí step, fuel và thời gian. So sánh chi phí trên cùng trường hợp
   tìm được đường; không tính đường thất bại là chi phí bằng 0.
2. Toàn trận: cùng seed, đội hình, nhiên liệu và luật. So sánh Greedy/Lookahead theo
   thứ tự unique series, tổng daily series, tổng udon; báo cáo thắng/hòa/thua và
   thời gian. `benchmark.py` chạy riêng từng chiến lược; `benchmark_match.py`
   chạy đối đầu có giao thông chung. Không trộn hai loại kết quả.
3. RL: chỉ thêm kết quả khi checkpoint được kiểm chứng với cùng luật timeline/grid.
   Tách seed train/tune khỏi seed đánh giá cuối; giữ cùng phân bố bản đồ, đội hình,
   thông tin quan sát, giới hạn thời gian và phần cứng. Nếu RL chọn mục tiêu rồi
   dùng A*, các baseline chiến lược cũng dùng cùng A*. Báo cáo riêng chi phí train
   và thời gian suy luận. Không sử dụng điểm của simulator cũ để kết luận.

Chưa có kết quả RL tương thích được xác minh trong nhánh này, nên chưa kết luận
rule-based mạnh hơn hay yếu hơn RL. Thời gian chạy có thể thay đổi theo máy;
Lookahead có giới hạn thời gian nên quyết định cũng có thể thay đổi khi máy quá tải.
