# Hướng dẫn train RL trên Google Colab (Free GPU)

**16/09: dữ liệu curriculum và self-play đã đổi theo luật BTC.** Xem [bản sửa và smoke test mới](report/TRAINING_SCENARIOS_BTC.md). ZIP cũ chưa chứa các sửa timeline/grid/generator này; chưa dùng ZIP cũ để train dài. Kết quả smoke chỉ xác nhận pipeline chạy, chưa xác nhận sức mạnh thi đấu.

Cập nhật15/09: đã có tùy chọn `--target-masking` dùng chung cho train và
inference. Xem [kiểm chứng và lệnh mới](report/MASK_INTEGRATION.md). Các ZIP cũ
chưa chứa thay đổi này; không dùng chúng để resume checkpoint bật masking.

**Chưa khuyến nghị train dài từ smoke_gpu:** kiểm tra GPU đã tái hiện lỗi luôn
STAY dù KL thấp. Xem kết quả mới nhất trong
[bản sửa vị trí actor](report/ACTOR_POSITION.md),
[đối chứng khởi tạo mới](report/INITIALIZATION.md),
[thí nghiệm gom game](report/ROLLOUT_BATCHING.md) và
[chẩn đoán STAY](report/STAY_DIAGNOSIS.md) trước khi dùng các lệnh train dài bên dưới.

> Colab free cho T4 GPU (15 GB VRAM) — đủ để train MAPPO với map đến 32×32.  
> TPU free tier không tương thích tốt với PyTorch → dùng GPU.

---

## Mục lục
1. [Chuẩn bị trước khi mở Colab](#1-chuẩn-bị)
2. [Tạo notebook và bật GPU](#2-tạo-notebook-và-bật-gpu)
3. [Upload code lên Google Drive](#3-upload-code)
4. [Cài đặt môi trường](#4-cài-đặt-môi-trường)
5. [Chạy training](#5-chạy-training)
6. [Theo dõi bằng TensorBoard](#6-tensorboard)
7. [Lưu và resume checkpoint](#7-lưu-và-resume)
8. [Xử lý timeout Colab](#8-xử-lý-timeout)

---

## 1. Chuẩn bị

**Trên máy local — push code lên GitHub trước:**
```bash
git status --short
git diff
git add src tests .gitignore COLAB_TRAINING_GUIDE.md
git diff --cached
git commit -m "ready for cloud training"
git push origin Procon2026
```

Nếu repo private, cần tạo **Personal Access Token** trên GitHub:  
`GitHub → Settings → Developer settings → Personal access tokens → Generate new token`  
Tick quyền `repo`. Lưu token lại.

---

## 2. Tạo notebook và bật GPU

1. Mở [colab.research.google.com](https://colab.research.google.com)
2. Tạo notebook mới
3. Bật GPU: **Runtime → Change runtime type → T4 GPU → Save**
4. Kiểm tra GPU đang hoạt động:

```python
!nvidia-smi
```

Output mong đợi:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ...   Driver Version: ...   CUDA Version: 12.x                  |
| GPU Name: Tesla T4        |  15360 MiB Total                               |
+-----------------------------------------------------------------------------+
```

---

## 3. Upload code

### Cách A — Clone từ GitHub (khuyến nghị)

```python
# Nếu repo public:
!git clone --branch Procon2026 https://github.com/kttt294/Procon2026.git procon2026

# Nếu repo private (thay YOUR_TOKEN):
!git clone --branch Procon2026 https://YOUR_TOKEN@github.com/kttt294/Procon2026.git procon2026
```

### Cách B — Upload file zip từ máy local

```python
from google.colab import files
files.upload()   # chọn file zip của repo
```

```python
!unzip Procon2026.zip -d procon2026
```

### Mount Google Drive (để lưu checkpoint không mất khi timeout)

```python
from google.colab import drive
drive.mount('/content/drive')

# Tạo thư mục lưu model trên Drive
!mkdir -p /content/drive/MyDrive/procon2026
```

---

## 4. Cài đặt môi trường

```python
%cd /content/procon2026

# PyTorch đã có sẵn trên Colab, chỉ cần cài thêm:
!pip install tensorboard requests -q

# Kiểm tra phiên bản
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

Output mong đợi:
```
PyTorch: 2.x.x+cu121
CUDA available: True
GPU: Tesla T4
```

---

## 5. Chạy training

### Kiểm tra ngắn trước khi train dài

Sau khi cập nhật code mới, chạy 20 episode và resume thêm 20 episode. Dùng
checkpoint thử riêng để không ghi đè model đang train:

```python
!python src/main.py train --episodes 20 --curriculum --selfplay --selfplay-every 5 --device cuda --seed 42 --log-every 5 --save-every 10 --save /content/drive/MyDrive/procon2026/smoke.pt
!python src/main.py train --episodes 20 --curriculum --selfplay --device cuda --log-every 5 --load /content/drive/MyDrive/procon2026/smoke.pt --save /content/drive/MyDrive/procon2026/smoke.pt
```

Lượt resume phải tiếp tục từ episode 21, không quay lại 1; loss phải hữu hạn.
Checkpoint thử 40 episode chỉ kiểm tra luồng chạy, chưa dùng đánh giá sức chơi.

### Chạy cơ bản (curriculum + selfplay, tự động lưu vào Drive)

```python
%cd /content/procon2026

!python src/main.py train \
  --episodes 20000 \
  --curriculum \
  --selfplay \
  --device cuda \
  --seed 42 \
  --log-every 100 \
  --save-every 500 \
  --log-dir /content/runs/mappo \
  --save /content/drive/MyDrive/procon2026/model.pt
```

`--save-every 500` ghi checkpoint mỗi 500 episode và khi train kết thúc bình thường.
Nếu runtime bị dừng đột ngột, phần sau checkpoint gần nhất vẫn có thể mất.

### Các tham số quan trọng

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `--episodes` | 20000 | Số episode train |
| `--curriculum` | flag | Bắt đầu level 1 (8×8), tự tăng |
| `--selfplay` | flag | Bật self-play pool |
| `--device cuda` | cuda | Dùng GPU |
| `--start-level` | 0–4 | Level bắt đầu (0=8×8, 4=32×32) |
| `--save-every` | 500 | Tự động lưu checkpoint mỗi N episode |
| `--selfplay-every` | 1000 | Thêm checkpoint vào pool mỗi N episode |
| `--seed` | 42 | Seed để reproduce |

### Resume từ checkpoint đã lưu

Checkpoint mới lưu cả optimizer, số episode, seed, RNG, lịch sử curriculum và
trọng số pool self-play. `--episodes` khi resume là **số episode chạy thêm**.

**Chuyển checkpoint PPO cũ sang bản sửa critic:** lần đầu nạp checkpoint chưa có
nhãn `critic_head_only_v1`, chương trình giữ trọng số, episode, RNG và pool nhưng
làm mới momentum Adam, kèm cảnh báo `reset optimizer moments once`. Đây là bước
chuyển đổi có chủ đích do gradient critic cũ gây mất ổn định policy. Checkpoint
lưu bằng bản mới sẽ giữ đầy đủ optimizer khi resume tiếp. Luôn lưu thử sang tên
mới (ví dụ `ppo_fixed.pt`), giữ nguyên `smoke_gpu.pt` làm đối chứng.

Seed và lịch cập nhật pool được khôi phục từ checkpoint; không đặt lại bằng seed
mặc định của lệnh. Giữ `--curriculum --selfplay` nếu phiên trước đã bật hai chế độ này.
Đường dẫn `--load` không tồn tại sẽ báo lỗi thay vì tự train từ đầu.

```python
!python src/main.py train \
  --episodes 20000 \
  --curriculum \
  --selfplay \
  --device cuda \
  --load /content/drive/MyDrive/procon2026/model.pt \
  --save /content/drive/MyDrive/procon2026/model.pt \
  --log-dir /content/runs/mappo
```

---

## 6. TensorBoard

Mở trong một cell riêng **trong khi training đang chạy**:

```python
%load_ext tensorboard
%tensorboard --logdir /content/runs/mappo
```

### Metrics cần theo dõi

| Metric | Kỳ vọng | Nếu không đúng |
|---|---|---|
| `episode/unique_series` | Tăng dần | Kiểm tra reward shaping |
| `curriculum/level_number` | Tăng từ 1 lên 5 | Xem win_rate vs Lookahead |
| `curriculum/win_rate` | Vượt 0.6 mỗi level | Tăng số episodes |
| `train/loss` | Giảm ổn định | Nếu NaN → giảm learning rate |
| `train/entropy` | Giảm chậm | Nếu giảm quá nhanh → tăng `ENTROPY_COEF` |

---

## 7. Lưu và resume checkpoint

`--save-every 500 --save /content/drive/MyDrive/procon2026/model.pt` lưu định kỳ và
lưu cuối phiên. File được ghi tạm trong cùng thư mục rồi thay thế checkpoint cũ
sau khi ghi thành công; vẫn cần kiểm tra file trên Drive sau phiên cloud.
Checkpoint cũ chỉ có trọng số vẫn nạp được, nhưng sẽ cảnh báo không thể khôi phục
optimizer/RNG đã không được lưu. Không thể tái dựng tiến độ bị thiếu từ trọng số.

### Sau mỗi session — kiểm tra checkpoint còn không

```python
import os
path = "/content/drive/MyDrive/procon2026/model.pt"
if os.path.exists(path):
    size = os.path.getsize(path) / 1024 / 1024
    print(f"Checkpoint exists: {size:.1f} MB")
else:
    print("No checkpoint found — training from scratch")
```

---

## 8. Xử lý timeout

### Vấn đề
- Colab free **tự disconnect sau ~90 phút không tương tác**
- Tối đa **~12 tiếng liên tục** mỗi session
- Mất hết `/content/` khi runtime reset — chỉ Google Drive là an toàn

### Giải pháp

**Giữ session sống** (chạy trong browser console, F12):
```javascript
// Dán vào Console của browser (F12 → Console)
function keepAlive() {
  document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(keepAlive, 60000);
```

**Hoặc dùng Colab Pro** nếu cần train dài hơn 12 tiếng.

### Kế hoạch train nhiều session

Session 1 (12h): `--episodes 10000`, save → Drive  
Session 2 (12h): `--load model.pt --episodes 10000`, resume  
Session 3 (12h): tiếp tục...

Tổng 30.000+ episodes ~ 3 ngày với free tier.

---

## Ví dụ notebook hoàn chỉnh

Copy toàn bộ vào một notebook Colab theo thứ tự:

```python
# Cell 1: Kiểm tra GPU
!nvidia-smi

# Cell 2: Clone repo
!git clone --branch Procon2026 https://github.com/kttt294/Procon2026.git procon2026

# Cell 3: Mount Drive
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/procon2026

# Cell 4: Cài dependencies
%cd /content/procon2026
!pip install tensorboard requests -q

# Cell 5: Kiểm tra import
import sys; sys.path.insert(0, 'src')
import torch
from rl.mappo import MAPPOTrainer
from rl.curriculum import CurriculumEngine
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("All imports OK")

# Cell 6: Mở TensorBoard (chạy trước training)
%load_ext tensorboard
%tensorboard --logdir /content/runs/mappo

# Cell 7: Chạy training
import os
model_path = "/content/drive/MyDrive/procon2026/model.pt"
load_arg   = f"--load {model_path}" if os.path.exists(model_path) else ""

!python src/main.py train \
  --episodes 20000 \
  --curriculum \
  --selfplay \
  --device cuda \
  --seed 42 \
  --log-every 100 \
  --log-dir /content/runs/mappo \
  {load_arg} \
  --save {model_path}
```

---

## Khi nào dừng train?

| Tín hiệu | Hành động |
|---|---|
| `curriculum/level_number` đạt 5 (32×32) | Train thêm 5000 ep rồi dừng |
| `episode/unique_series` plateau >2000 ep | Thử tăng `--selfplay-every 500` |
| `train/loss` = NaN | Giảm `LR_ACTOR` trong `src/config.py` xuống còn `1e-4`, retrain |
| Level không tăng sau 5000 ep | Thêm `--start-level 0`, train dài hơn ở level hiện tại |

---

## Sau khi train xong

Download model về máy:

```python
from google.colab import files
files.download('/content/drive/MyDrive/procon2026/model.pt')
```

Đặt `model.pt` vào thư mục gốc của repo, dùng khi thi:

```bash
python src/main.py play \
  --url http://[server-ip]:8080 \
  --model model.pt \
  --mcts
```
# Cập nhật 2026-09-15: tạm dừng train dài

Grid/client cũng đã sửa, 173 test pass và kết nối server mẫu BTC qua hai ngày thành công. [Đánh giá mới](report/GRID_CLIENT_FIX.md) vẫn cho thấy checkpoint cũ kém Greedy về Avg Series. Chưa dùng ZIP cũ để train tiếp.

Đã sửa timeline từng xe, tiếp nhiên liệu, thu udon khi chờ và giao thông khi đứng yên; 167 test pass. Xem [báo cáo kiểm toán](report/OFFICIAL_RULE_AUDIT.md). Vẫn cần đối chiếu client/server và đánh giá lại model. Hướng dẫn train bên dưới và ZIP đã tạo là bản trước kiểm toán; chưa dùng để bắt đầu train dài. Giữ checkpoint cũ để đối chứng.
