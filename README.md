# Đánh giá SapiensID trên dữ liệu gallery/query

Repo này dùng SapiensID pretrained để nhận diện người trong ảnh query bằng cách so sánh với ảnh gốc trong gallery. Chương trình xuất độ chính xác tổng thể, độ chính xác từng người và danh sách ảnh nhận diện sai. Không cần huấn luyện lại model.

**Đặt mã nguồn ngay trong thư mục cha của `gallery` và `query`. Không cần tạo thêm folder `dataset`.**

Hướng dẫn dùng Bash trên Linux/Ubuntu có GPU NVIDIA. Các đường dẫn `./...` được tính từ thư mục dự án, nơi có `evaluate.py`.

## 1. Tải mã nguồn

```bash
git clone https://github.com/Munfond/sapiensid-gallery-query-eval.git my_project
cd my_project
```

Từ đây, thực hiện các bước trong `my_project`.

Nếu đã có thư mục chứa sẵn `gallery` và `query`, bạn có thể tải ZIP của repo rồi copy `evaluate.py`, `requirements-inference.txt`, `test_evaluate.py` và README vào thư mục đó. Mở terminal tại chính thư mục đó; không cần di chuyển dữ liệu sang một folder mới.

Kiểm tra bằng `pwd` và `ls`: phải nhìn thấy `evaluate.py`.

## 2. Bố trí dữ liệu cùng cấp với mã nguồn

Cấu trúc hoàn chỉnh sẽ như sau. Repo model và các trọng số được tải ở những bước tiếp theo:

```text
my_project/
├── evaluate.py
├── requirements-inference.txt
├── test_evaluate.py
├── README.md
├── gallery/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── 003.png
├── query/
│   ├── 001/
│   │   ├── anh1.jpg
│   │   └── anh2.jpg
│   ├── 002/
│   │   └── anh3.jpg
│   └── 003/
│       └── anh4.jpg
├── sapiensid/                  # Repo model chính thức
├── checkpoints/
│   └── sapiensid_wb4m/
│       ├── model.yaml
│       └── model.pth
├── yolov8n-pose.pt
└── results/                    # Chương trình tự tạo
```

### Quy tắc gán nhãn

| Vị trí ảnh | Nhãn người |
|---|---|
| `gallery/001.jpg` | `001` |
| `query/001/anh1.jpg` | `001` |
| `query/001/anh2.jpg` | `001` |
| `gallery/002.jpg` | `002` |
| `query/002/anh3.jpg` | `002` |

- Gallery chứa ảnh trực tiếp, mỗi người đúng **một ảnh**. Không tạo folder người bên trong gallery.
- Tên file gallery bỏ đuôi là ID. Tên folder ngay dưới query phải trùng chính xác ID đó: `001` khác `1`, `An` khác `an`.
- Tên ảnh query tùy ý. `1_0.jpg` hay `2_1.jpg` không được dùng làm ID; nhãn luôn lấy từ folder người.
- Không bắt buộc có thư mục con khác trong folder người. Nếu có, ảnh bên trong vẫn được đọc và mang ID của folder người ngoài cùng.
- Mọi người trong query phải có ảnh gallery tương ứng. Nếu thiếu, chương trình dừng và liệt kê ID thiếu.
- Không đặt cùng `001.jpg` và `001.png` trong gallery vì chúng trùng ID.
- Mỗi ảnh nên chỉ chứa một người cần nhận diện. Nhãn folder không được dùng để chọn người trong ảnh nhiều người.
- Dùng ảnh query khác ảnh gốc gallery. Chương trình cảnh báo file trùng nguyên byte, nhưng chưa phát hiện ảnh trùng đã resize/nén lại.

Đuôi ảnh hỗ trợ: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`.

## 3. Kiểm tra máy trạm

```bash
nvidia-smi
python3.10 --version
nvcc --version
g++ --version
```

| Thành phần | Vai trò |
|---|---|
| GPU NVIDIA và driver | Chạy suy luận CUDA |
| Python 3.10 | Môi trường Python trong hướng dẫn |
| CUDA toolkit có `nvcc` | Biên dịch phần mở rộng RPE |
| Compiler C++ như `g++` | Biên dịch mã C++ của RPE |

Cấu hình khởi điểm: **Python 3.10 + PyTorch 2.3.1 + torchvision 0.18.1 + CUDA toolkit 12.1**. Bộ phiên bản này chưa được kiểm thử inference GPU trong phiên làm việc này. GPU đời mới có thể cần PyTorch/CUDA mới hơn và kiểm tra tương thích lại với upstream.

`CUDA Version` trong `nvidia-smi` không xác nhận toolkit đã cài. Dùng `nvcc --version` để kiểm tra toolkit. Wheel PyTorch cung cấp runtime CUDA nhưng chưa đủ compiler để build RPE.

Nếu thiếu công cụ, cài trước khi tiếp tục. Ví dụ trên Ubuntu 22.04, khi package có sẵn trong apt:

```bash
sudo apt update
sudo apt install git build-essential python3.10-venv
```

Driver và CUDA toolkit cần cài phù hợp hệ điều hành. Xem [hướng dẫn CUDA 12.1 cho Linux của NVIDIA](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-installation-guide-linux/index.html). Cài Python 3.10 riêng nếu chưa có; lệnh apt trên không phải cách cài chung cho mọi phiên bản Ubuntu.

## 4. Tạo môi trường và cài thư viện

Tại `my_project`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-inference.txt
python -m pip check
```

Kiểm tra Python và GPU:

```bash
which python
python -c "import torch; print('Torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'PyTorch khong nhin thay GPU'"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

`which python` cần trỏ vào `.venv`, và `GPU available` phải là `True`.

`requirements-inference.txt` dành cho script này, không phải môi trường train/chạy toàn bộ benchmark upstream. Không cần cài thêm requirements gốc của SapiensID chỉ để chạy wrapper này. Các phụ thuộc RAPIDS, DeepSpeed và MXNet của luồng benchmark không được wrapper sử dụng.

Nếu chọn bộ PyTorch khác, tra cặp torch/torchvision tại [tài liệu PyTorch chính thức](https://pytorch.org/get-started/previous-versions/).

## 5. Tải repo model SapiensID chính thức

Repo đánh giá cung cấp `evaluate.py`; kiến trúc model được nạp từ repo riêng của tác giả. Clone ngay bên trong `my_project`:

```bash
git clone https://github.com/mk-minchul/sapiensid.git sapiensid
git -C sapiensid checkout a8d8011117db23f20c7ade3bc38e8075810ac886
```

Checkout cố định phiên bản upstream mà script đã được đối chiếu mã nguồn. Thông báo detached HEAD là bình thường.

Kiểm tra source và trọng số detector mặt đi kèm:

```bash
ls sapiensid/tasks/sapiensID/src
ls sapiensid/tasks/sapiensID/src/aligners/keypoint_predictor/pretrained_models/aligners/dfa_mobilenetv4_medium/mobilenetv4_Final.pth
```

Không đổi tên hoặc di chuyển các file bên trong repo model.

## 6. Biên dịch RPE

RPE là phần mở rộng C++/CUDA của SapiensID. Build sau khi PyTorch nhìn thấy GPU và toolkit có `nvcc`:

```bash
cd sapiensid/tasks/sapiensID/src/models/sapiens_id/rpe/RPE/rpe_ops
python -m pip install --no-build-isolation .
cd -

python -c "import rpe_index_cpp; print('RPE extension OK')"
```

`cd -` đưa terminal trở lại `my_project`. Phải thấy `RPE extension OK`.

Nếu build lỗi, xử lý lỗi compiler/CUDA trước khi chạy đánh giá. Nếu thay phiên bản torch sau đó, cần build lại extension trong môi trường mới.

## 7. Tải checkpoint pretrained SapiensID

Dùng liên kết do tác giả cung cấp:

- [README pretrained chính thức](https://github.com/mk-minchul/sapiensid/blob/a8d8011117db23f20c7ade3bc38e8075810ac886/tasks/sapiensID/README.md).
- [File pretrained trên Google Drive](https://drive.google.com/file/d/18d8Ogw60zxnaIIjSb99Y8oLvzCEn_L-Z/view).

Tạo folder đích:

```bash
mkdir -p checkpoints
```

Giải nén file tải về. Tìm folder checkpoint chứa cả `model.yaml` và `model.pth`, rồi đặt vào `checkpoints`. Ví dụ dùng WB4M:

```text
checkpoints/
└── sapiensid_wb4m/
    ├── model.yaml
    └── model.pth
```

Kiểm tra:

```bash
ls -lh checkpoints/sapiensid_wb4m/model.yaml checkpoints/sapiensid_wb4m/model.pth
```

Nếu archive có nhiều lớp thư mục, tìm vị trí file bằng:

```bash
find checkpoints -type f -name model.pth
```

`--checkpoint` phải trỏ vào **folder chứa trực tiếp cả hai file**, không trỏ vào file `.pth` hay folder bao ngoài. Dùng YAML đi kèm đúng checkpoint; không thay bằng YAML mẫu trong source.

Có thể dùng WB12M nếu có trọng số tương ứng; đổi `--checkpoint` sang folder chứa bản đó.

## 8. Tải YOLOv8 Pose và hiểu vai trò của trọng số này

```text
Ảnh → YOLOv8 Pose + detector mặt → điểm mốc cơ thể/khuôn mặt
                                      ↓
                                  SapiensID
                                      ↓
                             Vector đặc trưng chuẩn hóa L2
                                      ↓
                         So sánh query với gallery bằng cosine
```

**YOLOv8 Pose tìm điểm mốc cơ thể; SapiensID tạo đặc trưng nhận diện người.** File cần dùng là `yolov8n-pose.pt`, khác với `yolov8n.pt` dùng phát hiện đối tượng.

Tại `my_project`, kích hoạt `.venv` và chạy khi có Internet:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
ls -lh yolov8n-pose.pt
```

Ultralytics sẽ tải trọng số nếu chưa có. Xem thêm [tài liệu pose của Ultralytics](https://docs.ultralytics.com/tasks/pose/).

Tham số `--yolo-weights ./yolov8n-pose.pt` cung cấp file cho wrapper. Wrapper copy vào `~/.cache/yolov8n-pose.pt`, đúng đường dẫn pipeline upstream sử dụng. Chỉ có thể bỏ tham số nếu file cache hợp lệ đã có; giữ tham số như lệnh mẫu sẽ dễ kiểm tra hơn.

Máy offline cần chuẩn bị packages, repo và tất cả trọng số trên máy có Internet rồi chuyển sang trước.

## 9. Kiểm tra dữ liệu trước khi chạy model

```bash
python evaluate.py --data . --out ./results/data_check --prepare-only
```

**`--data .` nghĩa là đọc `./gallery` và `./query` ngay tại thư mục hiện tại.**

Lệnh sẽ đọc danh sách ảnh, kiểm tra ID và xuất:

- `results/data_check/manifest.csv`: đường dẫn và nhãn của từng ảnh để bạn kiểm tra.
- `results/data_check/data_check.json`: số ảnh và các file query trùng nguyên byte với gallery.

Chế độ này không cần GPU hay thư viện model. Nó kiểm tra cấu trúc, tên và byte dữ liệu, chưa giải mã ảnh để xác nhận ảnh đọc được.

Chạy bộ kiểm thử logic đánh giá:

```bash
python -m unittest -v test_evaluate.py
```

Bộ 7 test phải kết thúc bằng `OK`. Các test kiểm tra ID và chỉ số; chưa thay thế inference GPU với checkpoint thật.

## 10. Chạy đánh giá toàn bộ query

Tại `my_project`:

```bash
source .venv/bin/activate

python evaluate.py \
  --data . \
  --repo ./sapiensid \
  --checkpoint ./checkpoints/sapiensid_wb4m \
  --yolo-weights ./yolov8n-pose.pt \
  --out ./results/wb4m_run01
```

| Tham số | Ý nghĩa |
|---|---|
| `--data .` | Folder cha của gallery và query, đồng thời chứa mã đánh giá |
| `--repo ./sapiensid` | Repo model chính thức |
| `--checkpoint ...` | Folder chứa model.yaml và model.pth |
| `--yolo-weights ...` | File trọng số tìm điểm mốc cơ thể |
| `--out ...` | Folder báo cáo, tự tạo nếu chưa có |

Nếu máy có nhiều GPU, chọn GPU 0 bằng:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --data . \
  --repo ./sapiensid \
  --checkpoint ./checkpoints/sapiensid_wb4m \
  --yolo-weights ./yolov8n-pose.pt \
  --out ./results/wb4m_gpu0_run01
```

Terminal lần lượt hiển thị `Gallery [i/N]` rồi `Query [i/M]`. Mỗi query được so với **toàn bộ gallery**. ID có cosine similarity cao nhất là dự đoán; nhãn folder chỉ dùng để chấm đúng/sai sau đó.

Khi hoàn tất, chương trình in thống kê JSON và dòng `Results: ...`. Mã hiện xử lý từng ảnh một, FP32, không có cơ chế tiếp tục lượt chạy bị dừng. Nên dùng `--out` mới cho mỗi thử nghiệm.

## 11. Đọc kết quả

```bash
ls results/wb4m_run01
python -m json.tool results/wb4m_run01/metrics.json
```

### Các chỉ số

| Trường trong metrics.json | Ý nghĩa |
|---|---|
| `top1_accuracy` | Số query đoán đúng ID / tổng query |
| `top5_accuracy` | Tỷ lệ query có ID thật trong 5 kết quả đầu |
| `macro_top1_accuracy` | Trung bình Top-1 từng người; mỗi người có trọng số bằng nhau |
| `mAP` | Trung bình AP; với một gallery/người, AP mỗi query = 1 / thứ hạng gallery đúng |
| `query_images` | Tổng query được đưa vào đánh giá |
| `failed_queries` | Số query lỗi đọc ảnh hoặc lỗi xử lý được ghi nhận |
| `effective_top5_k` | min(5, số ảnh gallery) |
| `exact_duplicate_query_count` | Số query trùng nguyên byte với một file gallery |

Accuracy và mAP được lưu ở khoảng 0–1. **Ví dụ `top1_accuracy: 0.92` nghĩa là nhận diện đúng 92% ảnh query**. Đây là ví dụ minh họa, không phải kết quả trên dữ liệu của bạn.

Nếu gallery chỉ có tối đa 5 người, Top-5 ít có giá trị phân biệt; nên xem Top-1 và danh sách lỗi.

### Các file đầu ra

| File | Nội dung |
|---|---|
| `metrics.json` | Chỉ số tổng hợp của lượt chạy hoàn tất |
| `predictions.csv` | Mỗi query: ID thật, ID dự đoán, score và thứ hạng ID thật |
| `wrong_predictions.csv` | Chỉ các query sai hoặc lỗi |
| `per_person.csv` | Số ảnh, số đúng, số lỗi và accuracy từng người |
| `manifest.csv` | Nhãn/đường dẫn; thứ tự từng split tương ứng thứ tự embedding split đó |
| `gallery_embeddings.npy` | Vector gallery chuẩn hóa L2 |
| `query_embeddings.npy` | Vector query; hàng NaN tương ứng query lỗi |
| `data_check.json` | Kiểm tra dữ liệu trước khi nạp model |

Mở CSV bằng Excel, LibreOffice hoặc pandas. Các cột cần xem trong `predictions.csv`:

- `true_id`: tên folder người trong query.
- `predicted_id`: ID gallery giống nhất.
- `correct`: True nếu ID đúng, False nếu sai hoặc query lỗi.
- `similarity`: cosine similarity đứng đầu, không phải xác suất/phần trăm tin cậy.
- `true_rank`: thứ hạng gallery đúng; bằng 1 tức đúng Top-1.
- `top5_ids`: tối đa 5 ID đứng đầu.
- `error`: nội dung lỗi xử lý ảnh nếu có.

### Phạm vi và cách tính ảnh lỗi

- Query lỗi vẫn nằm trong mẫu số, nhận điểm 0; không bị bỏ âm thầm.
- Gallery lỗi làm dừng lượt chạy để không thay đổi tập đối chiếu. GPU hết VRAM cũng làm dừng lượt chạy.
- Đây là **closed-set**: mỗi người query đã có trong gallery; model luôn chọn một ID. Chưa đánh giá từ chối người lạ bằng threshold.
- Không sửa/xóa ảnh dữ liệu. Chạy lại cùng `--out` ghi đè báo cáo cũ. `metrics.json` cũ được xóa trước khi nạp model, chỉ tạo lại khi hoàn tất. Nếu dừng giữa chừng, CSV có thể chỉ là kết quả một phần.

## 12. Các lần chạy tiếp theo

Không cần tải lại repo/trọng số hoặc build RPE mỗi lần. Mở terminal tại `my_project`:

```bash
source .venv/bin/activate

python evaluate.py \
  --data . \
  --repo ./sapiensid \
  --checkpoint ./checkpoints/sapiensid_wb4m \
  --yolo-weights ./yolov8n-pose.pt \
  --out ./results/wb4m_run02
```

Nếu clone repo đánh giá bằng Git, cập nhật hướng dẫn/mã bằng `git pull` khi thư mục làm việc không có thay đổi cần giữ. `gallery`, `query`, trọng số và các folder kết quả mẫu đã được thêm vào `.gitignore`.

## 13. Xử lý lỗi thường gặp

| Lỗi/biểu hiện | Cách kiểm tra |
|---|---|
| Không thấy gallery/query | Chạy `pwd`, `ls gallery query`; `--data` phải trỏ đến folder cha của cả hai |
| `Query identities missing from gallery` | Đối chiếu `query/001/` với `gallery/001.jpg`; kiểm tra hoa/thường, khoảng trắng, số 0 đầu |
| `Duplicate gallery IDs` | Chỉ để một gallery/ID, tránh cùng `001.jpg` và `001.png` |
| Không thấy model.yaml/model.pth | Kiểm tra lớp folder `--checkpoint` và giải nén pretrained |
| Không thấy mobilenetv4_Final.pth | Kiểm tra repo upstream, checkout và đường dẫn detector ở bước 5 |
| Không thấy yolov8n-pose.pt | Tải theo bước 8, kiểm tra đường dẫn `--yolo-weights` |
| CUDA available = False | Kiểm tra `nvidia-smi`, Python đang dùng, bản PyTorch CUDA và quyền truy cập GPU |
| Không thấy nvcc/CUDA_HOME | Cài/chọn toolkit; nếu nhiều bản CUDA, trỏ CUDA_HOME và PATH đến đúng bản |
| Lỗi build/import rpe_index_cpp | Kiểm tra compiler/toolkit/torch và build lại theo bước 6 |
| `no kernel image is available` | Torch/CUDA hoặc extension có thể chưa hỗ trợ GPU; chọn phiên bản hỗ trợ và build lại |
| `CUDA out of memory` | Xem GPU trống bằng `nvidia-smi`, chọn GPU khác hoặc máy đủ VRAM; wrapper hiện không có tham số batch-size |
| Query lỗi trong CSV | Đọc cột `error` của wrong_predictions.csv, kiểm tra ảnh và khả năng tìm người/điểm mốc |
| Accuracy cao bất thường | Kiểm tra ảnh trùng gallery/query, kể cả ảnh gần trùng đã chỉnh sửa |
| Không có metrics.json | Lượt chạy chưa hoàn tất hoặc đã lỗi; xem traceback cuối terminal |

## Tình trạng kiểm thử và nguồn tham khảo

Đã kiểm tra cú pháp và chạy thành công 7 test logic nhãn/xếp hạng trên CPU. Chưa kiểm thử trọn bộ cài đặt, VRAM và inference GPU với checkpoint thật trên máy trạm của bạn.

- [SapiensID chính thức](https://github.com/mk-minchul/sapiensid).
- [Pipeline inference ở phiên bản được dùng](https://github.com/mk-minchul/sapiensid/blob/a8d8011117db23f20c7ade3bc38e8075810ac886/tasks/sapiensID/src/pipelines/infer_aligner_model_pipeline.py).
- [Bộ lấy điểm mốc cơ thể/khuôn mặt](https://github.com/mk-minchul/sapiensid/blob/a8d8011117db23f20c7ade3bc38e8075810ac886/tasks/sapiensID/src/aligners/keypoint_predictor/__init__.py).
- [Phiên bản PyTorch và lệnh cài](https://pytorch.org/get-started/previous-versions/).
- [CUDA toolkit cho Linux](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-installation-guide-linux/index.html).
- [Ultralytics pose estimation](https://docs.ultralytics.com/tasks/pose/).
