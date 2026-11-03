# Đánh giá SapiensID với gallery/query

Gói này chạy inference bằng pipeline chính thức của SapiensID, sau đó đánh giá nhận diện closed-set: mỗi người trong query phải có đúng một ảnh đối chiếu trong gallery. Không cần train lại.

## 1. Cấu trúc dữ liệu

```text
dataset/
  gallery/
    001.jpg
    002.jpg
    003.png
  query/
    001/
      1_0.jpg
      1_1.jpg
      session2/another.jpg
    002/
      any_name.jpg
    003/
      image.jpg
```

- ID chuẩn của gallery là tên file không có đuôi. ID chuẩn của query là tên folder con ngay dưới query. Phân biệt hoa/thường, giữ số 0 đầu: `001` khác `1`.
- Tên ảnh trong query có thể tùy ý; `1_0.jpg`, `2_1.jpg` không được dùng làm nhãn người.
- Mỗi ID trong gallery chỉ có một ảnh. Hai file `001.jpg` và `001.png` sẽ bị báo trùng ID.
- Người trong query thiếu ảnh gallery sẽ làm chương trình dừng và liệt kê ID thiếu. Đây chưa phải đánh giá người lạ/open-set.
- Mỗi ảnh nên chỉ chứa một người cần nhận diện. Pipeline upstream không dùng nhãn folder để chọn đúng người trong ảnh nhiều người.
- Dùng ảnh query khác ảnh gallery; chương trình cảnh báo file trùng nguyên byte giữa hai tập. Cảnh báo này chưa phát hiện ảnh trùng đã resize/nén lại.

## 2. Chuẩn bị máy trạm

Hướng dẫn môi trường bên dưới dành cho Linux/Ubuntu, Python 3.10, GPU NVIDIA, driver phù hợp và CUDA toolkit 12.1 có `nvcc`. Kiểm tra `nvidia-smi` và `nvcc --version`. CUDA runtime đi kèm wheel PyTorch chưa đủ để biên dịch phần mở rộng; cần toolkit và compiler C++ như `g++`.

Nếu GPU quá mới không được PyTorch 2.3.1 hỗ trợ, phải chọn cặp torch/torchvision và CUDA mới phù hợp phần cứng. Môi trường dưới đây là cấu hình khởi điểm theo mã nguồn upstream cũ, chưa được kiểm thử inference GPU trong phiên làm việc này.

Copy toàn bộ folder `sapiensid_gallery_query` này lên máy trạm. Mở terminal tại folder đó:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-inference.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"

git clone https://github.com/mk-minchul/sapiensid.git
git -C sapiensid checkout a8d8011117db23f20c7ade3bc38e8075810ac886
```

`requirements-inference.txt` chỉ dành cho wrapper inference này, không phải môi trường train/evaluate toàn bộ benchmark upstream. Các thư viện RAPIDS, DeepSpeed, MXNet trong requirements gốc không được wrapper này sử dụng.

Biên dịch RPE khi GPU đã được nhìn thấy trong môi trường:

```bash
cd sapiensid/tasks/sapiensID/src/models/sapiens_id/rpe/RPE/rpe_ops
python -m pip install --no-build-isolation .
cd -
python -c "import rpe_index_cpp; print('RPE extension OK')"
```

Nếu compiler báo phiên bản CUDA không tương thích, dùng toolkit tương thích với torch thay vì bỏ qua lỗi build. Nếu GPU hết VRAM, chương trình dừng; inference hiện chạy từng ảnh một, FP32.

## 3. Tải checkpoint và trọng số pose

Tải pretrained từ liên kết chính thức:

- [README pretrained SapiensID](https://github.com/mk-minchul/sapiensid/blob/main/tasks/sapiensID/README.md)
- [Google Drive do tác giả cung cấp](https://drive.google.com/file/d/18d8Ogw60zxnaIIjSb99Y8oLvzCEn_L-Z/view)

Giải nén và tìm folder tương ứng `sapiensid_wb4m` hoặc `sapiensid_wb12m`, bên trong có cả `model.yaml` và `model.pth`. Truyền chính folder này cho `--checkpoint`, không truyền riêng file `.pth`.

```text
/workspace/checkpoints/sapiensid_wb4m/
  model.yaml
  model.pth
```

Checkpoint YAML phải là cấu hình đi kèm trọng số, không thay bằng YAML mẫu. Trọng số detector mặt `mobilenetv4_Final.pth` có trong repo ở đường dẫn mà wrapper kiểm tra.

Tải YOLO pose bằng Ultralytics (cần Internet lần đầu):

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
```

File `yolov8n-pose.pt` sẽ nằm tại thư mục làm việc; truyền đường dẫn file này bằng `--yolo-weights`. Wrapper copy file vào `~/.cache/yolov8n-pose.pt`, đúng đường dẫn mà upstream sử dụng. Máy offline cần chuẩn bị tất cả trọng số và Python packages trước.

## 4. Kiểm tra dữ liệu trước khi chạy GPU

Thay `/workspace/dataset` bằng folder của bạn, chứa trực tiếp gallery và query:

```bash
python evaluate.py \
  --data /workspace/dataset \
  --out /workspace/sapiensid_results \
  --prepare-only
```

Chế độ này chỉ dùng Python standard library: kiểm tra cấu trúc, ID, tạo `manifest.csv`, và phát hiện file trùng nguyên byte. Nó chưa giải mã ảnh hoặc chạy model.

## 5. Chạy đánh giá

```bash
python evaluate.py \
  --data /workspace/dataset \
  --repo ./sapiensid \
  --checkpoint /workspace/checkpoints/sapiensid_wb4m \
  --yolo-weights ./yolov8n-pose.pt \
  --out /workspace/sapiensid_results
```

Script áp dụng transform chính thức của checkpoint, chạy YOLO + detector mặt để lấy keypoint, trích embedding SapiensID, chuẩn hóa L2 và tìm gallery có cosine similarity cao nhất. Không sử dụng nhãn đúng để chọn dự đoán. Không dùng threshold từ ArcFace hoặc model khác.

## 6. Đọc kết quả

| File | Nội dung |
|---|---|
| `metrics.json` | Tổng hợp Top-1, Top-5, macro Top-1, mAP và số ảnh lỗi |
| `predictions.csv` | Nhãn thật, nhãn dự đoán, score, thứ hạng nhãn thật, top 5 ID của từng ảnh query |
| `wrong_predictions.csv` | Các ảnh đoán sai và ảnh lỗi inference |
| `per_person.csv` | Số ảnh, số đúng, số lỗi và Top-1 của mỗi người |
| `manifest.csv` | Thứ tự gallery/query, đường dẫn tương đối và nhãn; tương ứng thứ tự hàng của embedding |
| `gallery_embeddings.npy` | Embedding L2-normalized của gallery |
| `query_embeddings.npy` | Embedding query; hàng NaN nếu ảnh đó lỗi |
| `data_check.json` | Thống kê dữ liệu và file trùng giữa hai tập |

- **Top-1 accuracy** = số ảnh query đoán đúng người / tổng ảnh query.
- **Top-5 accuracy** = tỷ lệ query có người đúng trong 5 gallery đầu. Nếu gallery ít hơn 5 người, `effective_top5_k` thể hiện số thực dùng; chỉ số này khi đó ít ý nghĩa.
- **Macro Top-1** = trung bình Top-1 của từng người; người có nhiều ảnh không lấn át người có ít ảnh.
- **mAP**: ở cấu trúc một gallery/người này, AP mỗi query bằng `1 / thứ hạng gallery đúng`. mAP là trung bình AP; không bằng Top-1.
- Ảnh query lỗi vẫn nằm trong mẫu số, được tính 0 ở các chỉ số; không bị bỏ âm thầm. Gallery lỗi làm dừng lượt chạy để không thay đổi tập đối chiếu. GPU OOM làm dừng lượt chạy.
- Score cosine không phải xác suất hay phần trăm tin cậy. Đây là closed-set: model luôn chọn một ID trong gallery; chưa đánh giá từ chối người lạ.
- Chương trình không sửa hoặc xóa ảnh dữ liệu. Chạy lại cùng `--out` sẽ ghi đè kết quả cũ. `metrics.json` cũ bị xóa trước khi nạp model và chỉ được tạo lại khi hoàn tất; nếu dừng giữa chừng thì các CSV còn lại có thể chỉ là kết quả một phần. Nên dùng `--out` mới cho mỗi thử nghiệm.

## 7. Kiểm tra logic đánh giá

```bash
python -m unittest -v test_evaluate.py
```

Các test kiểm tra ID, cách xếp hạng, mAP, Top-5, xử lý điểm bằng nhau, ảnh lỗi và mẫu số. Chúng không thay thế kiểm thử GPU với pretrained thật.

Nguồn: [SapiensID](https://github.com/mk-minchul/sapiensid), [pipeline inference](https://github.com/mk-minchul/sapiensid/blob/main/tasks/sapiensID/src/pipelines/infer_aligner_model_pipeline.py), [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).
