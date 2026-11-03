# Cài nvcc / CUDA Toolkit để build RPE của SapiensID

Tài liệu này bổ sung cho [README](README.md). Các lệnh cài bên dưới dành cho **Ubuntu 22.04 x86_64, GPU NVIDIA và cấu hình PyTorch CUDA 12.1** trong README.

## 1. Vì sao cần nvcc?

`nvcc` là compiler trong CUDA Toolkit. Phần RPE của SapiensID có mã C++/CUDA cần được biên dịch trước khi dùng pipeline GPU.

| Thành phần | Vai trò |
|---|---|
| NVIDIA driver | Giúp hệ điều hành và chương trình truy cập GPU |
| PyTorch có CUDA | Cung cấp runtime để chạy PyTorch trên GPU |
| CUDA Toolkit có nvcc | Cung cấp công cụ phát triển, compiler và headers để build extension |

Cài PyTorch có CUDA không đồng nghĩa máy đã có `nvcc`. Dòng `CUDA Version` trong `nvidia-smi` cũng không chứng minh toolkit đã được cài.

## 2. Kiểm tra hệ điều hành, GPU và môi trường Python

```bash
cat /etc/os-release
uname -m
nvidia-smi
```

Các lệnh cài ở bước 4 chỉ áp dụng khi hệ điều hành là Ubuntu 22.04 và kiến trúc là `x86_64`. Nếu dùng Ubuntu khác, Windows, WSL, container hoặc GPU cần CUDA mới hơn, cần chọn bộ cài tương ứng thay vì dùng nguyên lệnh Ubuntu 22.04.

Tại thư mục dự án, nếu đã tạo `.venv` theo README:

```bash
source .venv/bin/activate
python -c "import torch; print('Torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

Hướng dẫn này dùng torch 2.3.1 có CUDA runtime `12.1`. Nếu đang dùng torch/CUDA khác, chọn toolkit tương thích với môi trường đó. Nếu chưa cài PyTorch, làm bước cài môi trường trong README trước khi build RPE.

Nếu `nvidia-smi` không hoạt động, cần kiểm tra driver/quyền truy cập GPU. Chỉ cài toolkit không giải quyết được việc máy chưa truy cập được GPU.

## 3. Kiểm tra nvcc đã có nhưng chưa nằm trong PATH

```bash
command -v nvcc
ls /usr/local/cuda*/bin/nvcc
```

Nếu tìm thấy `/usr/local/cuda-12.1/bin/nvcc`, thử:

```bash
/usr/local/cuda-12.1/bin/nvcc --version
```

Nếu lệnh hoạt động, không cần cài lại; chuyển sang bước 5 để cấu hình đường dẫn. Nếu tìm thấy phiên bản khác, kiểm tra tương thích với PyTorch trước khi chọn toolkit. Toolkit cũng có thể được quản trị viên cài ngoài `/usr/local`; hãy kiểm tra vị trí cài của máy trạm nếu có.

## 4. Cài CUDA Toolkit 12.1 trên Ubuntu 22.04 x86_64

Cần quyền `sudo` và Internet. Nếu không có quyền, nhờ quản trị viên cài toolkit phù hợp.

Thêm kho package chính thức của NVIDIA:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

Cài toolkit và compiler C++:

```bash
sudo apt install cuda-toolkit-12-1 build-essential
```

Gói `cuda-toolkit-12-1` cài toolkit phiên bản 12.1, không bao gồm driver. Máy vẫn cần driver NVIDIA phù hợp và GPU hoạt động.

## 5. Cấu hình đường dẫn

Áp dụng cho terminal hiện tại:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
```

Kiểm tra:

```bash
command -v nvcc
nvcc --version
printf '%s\n' "$CUDA_HOME"
g++ --version
```

`nvcc --version` cần báo CUDA release 12.1, và `CUDA_HOME` trỏ đến `/usr/local/cuda-12.1`.

Để lưu cấu hình cho các phiên Bash sau, mở `~/.bashrc` và thêm hai dòng sau vào cuối file, nếu chưa có:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
```

Nạp lại:

```bash
source ~/.bashrc
```

Nếu cài toolkit ở vị trí khác, thay đường dẫn trong các dòng trên bằng vị trí thực tế.

## 6. Build RPE trong đúng môi trường của dự án

Mở terminal tại thư mục cha có `evaluate.py`, `gallery`, `query` và `sapiensid`. Kích hoạt môi trường rồi xác nhận PyTorch thấy GPU:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available(), 'PyTorch khong nhin thay GPU'"
```

Build extension khi GPU đã được truy cập:

```bash
cd sapiensid/tasks/sapiensID/src/models/sapiens_id/rpe/RPE/rpe_ops
python -m pip install --no-build-isolation .
cd -
```

Kiểm tra import:

```bash
python -c "import rpe_index_cpp; print('RPE OK')"
```

Khi thấy `RPE OK`, quay lại các bước tải trọng số và chạy đánh giá trong [README](README.md). Kiểm tra import chỉ xác nhận extension nạp được, chưa thay thế lượt inference với model thật.

## 7. Lỗi thường gặp

| Biểu hiện | Cách xử lý |
|---|---|
| Cài xong nhưng vẫn `nvcc: command not found` | Kiểm tra `/usr/local/cuda-12.1/bin/nvcc` và thực hiện bước 5 trong terminal đang dùng |
| `CUDA_HOME` chưa đặt | Export CUDA_HOME trước khi chạy pip build |
| Phiên bản CUDA không khớp PyTorch | Kiểm tra `nvcc --version` và `torch.version.cuda`; chọn đúng toolkit và PATH |
| Không thấy GPU trong PyTorch | Kiểm tra driver, bản torch CUDA, môi trường Python và quyền GPU/container |
| Không tìm thấy package cuda-toolkit-12-1 | Kiểm tra hệ điều hành/kiến trúc, bước thêm kho NVIDIA và lỗi từ apt update |
| Lỗi compiler khi build RPE | Kiểm tra compiler được CUDA 12.1 hỗ trợ theo tài liệu NVIDIA; không bỏ qua lỗi build |
| Đã đổi torch hoặc toolkit sau khi build | Build lại extension trong môi trường tương ứng |

Tài liệu chưa được kiểm thử cài đặt trên máy trạm của bạn. Gửi kết quả `cat /etc/os-release`, `uname -m`, `nvidia-smi`, `nvcc --version` và traceback nếu cần xác định lỗi cụ thể.

## Nguồn chính thức

- [NVIDIA CUDA 12.1 Installation Guide for Linux](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-installation-guide-linux/index.html).
- [Kho package NVIDIA cho Ubuntu 22.04 x86_64](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/).
- [Mã build RPE của SapiensID](https://github.com/mk-minchul/sapiensid/blob/a8d8011117db23f20c7ade3bc38e8075810ac886/tasks/sapiensID/src/models/sapiens_id/rpe/RPE/rpe_ops/setup.py).
