# Work order: tái cấu trúc và chiến dịch thí nghiệm PECT

## Phạm vi

Chỉ sửa `src/PECT_JEPA/cross_condition`. Không xóa dữ liệu, checkpoint, kết quả
cũ hoặc thay đổi đang có của người dùng. Không chạy chiến dịch dài trong phiên
triển khai; chạy smoke trên dữ liệu thật và xóa thư mục smoke sau kiểm tra.

## Cơ sở và quyết định

- Graph Q1 đã được tra trước triển khai. FM1/FM6/FM8: không thêm blind invariance.
- FM27: whitening + LR chỉ là một arm, không mặc định là cải thiện.
- FM31: adapter phải tác động vào đường encoder/extract_features.
- FM35: giữ uniform pooling trong loss; không phục hồi radial-pool shortcut.
- FM37: dùng preprocessing canonical fail-closed, đường dẫn tuyệt đối.
- Phân biệt source-only, target adaptation, transductive pretraining; nhãn target
  chỉ vào scoring. Không dùng target để chọn threshold/hyperparameter.
- Một waveform 128 mẫu là một đơn vị; không chia các khoảng thời gian cố định.

## Thay đổi dự kiến

1. `campaign/`: protocol ghép cặp nhiều file, backend SSL, adaptation/readout,
   đánh giá object, orchestration tuần tự, tổng hợp và kiểm tra hoàn tất.
2. `run_campaign.py`, `run_campaign.ps1`: entrypoint độc lập CWD, plan/run/status.
3. `PROJECT_PLAN.md`, `README.md`: ma trận, ngân sách, điều kiện kết luận,
   lệnh chạy và danh mục giữ/ngừng dùng code cũ.
4. Tách lựa chọn phạm vi pretrain khỏi split downstream trong luồng mới.
   Giữ nguyên API và state_dict của model cũ khi không bật tính năng mới.
5. Kết quả mới ở `experiments/campaigns/`, không trộn với kết quả lịch sử.

## Resume / acceptance

- JSON và checkpoint ghi atomic; fingerprint protocol/code/config/input.
- Checkpoint có model, optimizer, scheduler position, RNG Python/NumPy/Torch/CUDA,
  epoch/batch; dừng giữa batch chỉ mất tối đa khoảng save_every_steps.
- Không tự coi exit code 0 là hoàn tất: kiểm tra receipt và artifacts.
- Lỗi dừng nhánh phụ thuộc; tiếp tục các nhánh độc lập nếu yêu cầu.
- Kiểm thử deterministic và mô phỏng failure/resume/corruption/lock.
- `py_compile` tất cả file chạm; smoke 1 epoch/256 samples thật cho objective mới;
  rebuild knowledge graph sau eval thật. Không cập nhật verdict bằng số smoke.

## Phương pháp bên ngoài

TFDA gốc có temporal splitting, TimeSiam/TS-JEPA gốc dùng temporal patches.
Không gọi bản sửa spatial masking là reproduction nguyên bản. Phương pháp
PECT điều chỉnh phải có tên riêng, mô tả khác biệt, control và provenance.
Độ mới của mô hình không được suy ra từ tên hoặc năm công bố.
