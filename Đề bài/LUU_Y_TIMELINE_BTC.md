# Lưu ý riêng: timeline từng xe và thứ tự xử lý step

Cập nhật ngày 15/09/2026, đối chiếu Q&A chính thức của BTC. Mục này đính chính giả định “shared steps của cả đội” từng dùng trong code và báo cáo cũ.

**Cập nhật kiểm chứng:** đã sửa thêm grid và client, kiểm tra trực tiếp sáu hướng ở cả hai loại hàng và luồng nộp hai ngày trên server mẫu. Xem mục riêng [Grid và giao thức BTC](LUU_Y_GRID_VA_GIAO_THUC_BTC.md). Đây chưa phải kiểm chứng toàn bộ trận nhiều đội.

## 1. Mỗi xe có toàn bộ số step của ngày

Nếu một ngày có N step, **mỗi xe đều có N step trên cùng trục thời gian**. Xe chạy đồng thời; không chạy hết xe thứ nhất rồi lấy step còn lại chia cho các xe khác.

Ví dụ: ngày có 2 step, hai xe trên plain đều có thể đi một ô. Tổng thời gian di chuyển cộng qua các xe có thể lớn hơn N; điều đó không làm kế hoạch sai.

Chi phí rời ô phụ thuộc **ô nguồn**, kể cả trạng thái giao thông ở ô nguồn. Rời plain cần 2 step: đặt lịch ở step 0, xe vẫn ở ô nguồn trong step 1 và đến đích ở step 2.

## 2. Thứ tự xử lý

- Step 0: chỉ đặt lịch hành động, chưa nhận udon, chưa tiếp nhiên liệu, chưa tính hiện diện giao thông.
- Step 1 đến N: trừ nhiên liệu của các di chuyển hoàn tất → cập nhật vị trí tất cả xe → nhận udon → tiếp nhiên liệu → tính hiện diện giao thông.
- Sau pha trên, đặt lịch hành động tiếp theo nếu xe đã hoàn tất hành động và chưa hết ngày.
- Step N chỉ có pha phản ánh kết quả, không đặt lịch hành động mới.

Xe chưa đến hạn hoàn tất di chuyển vẫn ở ô nguồn. Việc thu udon, tiếp nhiên liệu và tính giao thông dùng vị trí sau khi cập nhật di chuyển tại step đó.

## 3. Tiếp nhiên liệu

Khi patrol và supply cùng ô tại pha tiếp nhiên liệu của một step, patrol tự động được nạp lên `fuel_max`, kể cả khi chưa hết nhiên liệu. Không có pha tiếp nhiên liệu ở step 0.

Supply không tiêu hao nhiên liệu khi di chuyển. Hai xe đổi chỗ qua một cạnh nhưng không cùng ô tại pha này thì không tiếp nhiên liệu. Hai xe cùng di chuyển và vẫn ở cùng ô có thể tiếp nhiên liệu mỗi step.

## 4. Chờ, udon và giao thông

- Chờ không tiêu hao nhiên liệu nhưng có tiêu hao thời gian.
- Patrol đứng tại spot từ step 1 trở đi vẫn nhận udon nếu còn hàng, không cần rời ô rồi quay lại.
- Mỗi patrol chỉ nhận tối đa một udon ở mỗi spot trong một ngày. Sang ngày mới có thể nhận lại khi chờ tại đó.
- Nếu nhiều patrol tranh phần hàng cuối cùng trong cùng step, giải quyết theo thứ tự ID tăng dần.
- Xe đứng trên road vẫn đóng góp một lượt hiện diện tại mỗi step, kể cả đang chờ hoặc chưa hoàn tất di chuyển. Không chỉ đếm lúc xe rời road.

## 5. Kế hoạch phải phủ đủ ngày

Kế hoạch của từng xe phải có tổng thời lượng **đúng N step**. Nếu di chuyển xong sớm, thêm chờ. Thiếu/thừa step, đi vào ao, ra ngoài bản đồ hoặc không đủ nhiên liệu khiến kế hoạch bị từ chối toàn bộ; không thực thi một phần rồi dừng.

Trong code nội bộ, `AgentAction('stay')` là chờ **1 step**, nên chờ k step được biểu diễn bằng k hành động. Trong ví dụ giao thức BTC, số âm `-k` biểu diễn chờ k step. Đây là hai cách biểu diễn, cần chuyển đổi tại lớp giao thức; sửa timeline không đồng nghĩa client hiện tại đã tương thích server BTC.

## 6. Kiểm chứng và ảnh hưởng tới train

Ví dụ BTC trang 3–4 có 7 patrol A–G và một supply, cùng chạy 6 step. Test `tests/test_official_timeline.py` kiểm tra vị trí và fuel mỗi step, người nhận từng udon và tổng hiện diện road: ô 0 là 12, ô 1 là 17.

Chạy từ gốc repo:

```powershell
python -m pytest tests/test_official_timeline.py -q
python report/check_official_rules.py
python -m pytest -q
python src/benchmark.py --games 100 --strategies greedy lookahead --seed 42
```

Checkpoint, benchmark và ZIP trước đính chính được tạo với luật mô phỏng cũ. Giữ để đối chứng; không dùng điểm số cũ để chứng minh sức mạnh theo luật thi. Chưa bắt đầu train dài chỉ vì test timeline đã pass: vẫn cần đối chiếu client/server và đánh giá chiến thuật lại.

## Nguồn chính thức

- [Q&A 1 — Q6, Q7, Q8, Q10, Q16, Q17](https://www.procon.gr.jp/uploads/download/BcAnkyvgVAA).
- [Tài liệu bổ sung về thứ tự hành động, trang 1–4](https://www.procon.gr.jp/uploads/download/BcAnkz3QVAA).
- Bản lưu PDF và văn bản ở `report/official_rules/`.
