# Lưu ý riêng: grid và giao thức BTC

Cập nhật 15/09/2026. Đã đối chiếu với binary Windows nguyên bản trong [gói server mẫu BTC](https://www.procon.gr.jp/news/2026/08/20/117), không sửa server để khớp client.

## Grid: hàng chẵn lệch sang phải

Đếm hàng từ 0. Hàng chẵn lệch sang phải; bảng trước đây trong code bị đảo. Với bản đồ rộng 8:

- Ô 18 (hàng 2), các hướng 0..5: `10, 11, 19, 27, 26, 17`.
- Ô 26 (hàng 3), các hướng 0..5: `17, 18, 27, 34, 33, 25`.

Khoảng cách dùng tọa độ cube với `q = col - (row + (row & 1)) // 2`. Phải đổi cả bảng hàng xóm và công thức khoảng cách. A* và simulator dùng chung `HexGrid`; CNN lưu feature theo chỉ số hàng/cột, không có bảng offset khác cần thay. Visualizer terminal đã thụt hàng chẵn đúng.

Ví dụ BTC về timeline chỉ đi trái/phải, nên riêng ví dụ đó không phát hiện lỗi bốn hướng chéo. Test mới kiểm tra cả sáu hướng trên hai loại hàng và đối chiếu khoảng cách bằng BFS.

## Giao thức chính thức

| Thao tác | HTTP | Nội dung |
|---|---|---|
| Lấy cấu hình | GET `/setting` | `daySteps`, `daySeconds`, `map`, `spots`, `agents`, `fuelLimits`… |
| Chọn xe trước trận | POST `/agent` | Mảng loại xe, ví dụ `[0,0,1]` |
| Đọc trạng thái | GET `/` | `day`, `endsAt`, `agents`, `others`, `traffics` |
| Nộp lệnh | POST `/` | Mảng các mảng số nguyên, đúng thứ tự xe |

Xác thực bằng header `Procon-Token`. Client lấy từ biến môi trường `PROCON_TOKEN`; không ghi token vào log. Chạy trước khi hết thời gian chọn loại xe. Hiện luồng `play` chọn tất cả patrol mặc định, chưa tune đội hình.

Ngày đầu trên server là 0; trong simulator là 1. Client chuyển đổi ở biên. `endsAt` là Unix time, dùng để giới hạn thời gian nộp; không bắt đầu đếm deadline trước khi chờ server mở ngày.

Mã địa hình BTC: plain=0, road=1, mountain=2, lake=3. Mã nội bộ vẫn là plain=0, mountain=1, lake=2, road=3; client dịch một lần khi đọc cấu hình.

Lệnh đi là 0..5. Chờ k step là `-k`; ví dụ `[[2,-14],[-16]]`. Không gửi object `orders` theo spec dự đoán cũ. `revision >= 0` nghĩa là nhận lệnh; số âm nghĩa là từ chối. Lệnh bị từ chối không thay thế lệnh đã nhận trước đó.

## Theo dõi điểm và giới hạn hiện tại

State của API không có điểm. Client tính lại điểm từ lệnh đã nhận mới nhất, rồi đối chiếu vị trí/fuel với trạng thái ngày sau. Nếu bất đồng, bị lỡ ngày hoặc kết quả nộp không xác định, client dừng với lỗi rõ ràng thay vì tiếp tục bằng điểm đoán. Đây chưa phải cơ chế phục hồi sau mất kết nối dài hoặc khởi động lại giữa trận.

Client giới hạn khoảng cách giữa request, dùng timeout trong deadline và thử tối đa lệnh chính + fallback. Luồng `play` đã được chạy qua hai ngày trên server mẫu; bài kiểm tra không thay thế kiểm chứng nhiều đội, mọi địa hình hoặc mọi tình huống mất mạng.

## Chạy và kiểm tra

Từ thư mục gốc repo trong PowerShell:

```powershell
$env:PROCON_TOKEN = '<token BTC cấp>'
python src/main.py play --url http://127.0.0.1:8080 --model runs/nonexistent.pt
```

Đường dẫn model không tồn tại ở ví dụ trên nhằm chạy heuristic, không dùng checkpoint cũ. Khi dùng server thật, thay URL và token; không chia sẻ token trong báo cáo.

```powershell
python report/check_official_server.py
python -m pytest -q
```

Script kiểm chứng tự chạy server mẫu trên localhost, kiểm tra sáu hướng ở hai loại hàng, thử lệnh sai → fallback, chạy CLI hai ngày, rồi đóng server. Kết quả ở `report/results/official_server_fixed/`.

Checkpoint cũ được giữ nguyên nhưng chưa được xác nhận mạnh trên luật đã sửa. Checkpoint mới mang nhãn `btc_even_r_v2`; không so trực tiếp các số benchmark của bản grid cũ với bản mới.
