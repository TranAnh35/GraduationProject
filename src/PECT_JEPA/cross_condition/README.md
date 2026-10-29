# PECT cross-condition research

Luồng nghiên cứu chính: **`run_all.py`**. Thiết kế, ma trận, nguồn phương
pháp và giới hạn kết luận: [PROJECT_PLAN.md](PROJECT_PLAN.md).

Từ thư mục này, một lệnh lập kế hoạch nếu chưa có rồi chạy toàn ma trận:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\tranl\miniconda3\envs\ndt\python.exe' .\run_all.py
```

Nếu lỗi hoặc tắt máy, chạy lại **đúng lệnh đó** để tiếp tục. Không xóa run dir.
Runner chạy tuần tự, lưu model/optimizer/RNG/batch, kiểm tra artifact trước khi
bỏ qua job completed. Nếu GPU có Python/ML process khác, runner từ chối mở job
GPU mới. Không tự dừng tiến trình khác.

Các lệnh riêng:

```powershell
& 'C:\Users\tranl\miniconda3\envs\ndt\python.exe' .\run_all.py --plan-only
& 'C:\Users\tranl\miniconda3\envs\ndt\python.exe' .\run_all.py --status
& 'C:\Users\tranl\miniconda3\envs\ndt\python.exe' .\run_all.py --keep-going
& 'C:\Users\tranl\miniconda3\envs\ndt\python.exe' .\run_all.py --no-verbose
& 'C:\Users\tranl\miniconda3\envs\ndt\python.exe' .\run_campaign.py aggregate --root experiments/campaigns/pect_v4
```

`--keep-going` tiếp tục nhánh độc lập, không bỏ qua lỗi phụ thuộc. Run directory
mặc định là `experiments/campaigns/pect_v4/`. `state.json` và log từng attempt
cho biết đang ở đâu; `completion.json` xác nhận còn thiếu job nào. Kết quả có
`summary.csv`, `by_condition.csv`, `paired_deltas.csv`, `negative_transfer.csv`,
`REPORT.md`, cùng CSV per-scan/per-object ở từng job.

Tqdm hiển thị tiến độ campaign và pha đang chạy bên trong job; chế độ này bật
mặc định. `--no-verbose` tắt thanh hiển thị, còn tiến độ vẫn được ghi vào
`progress.json` và log để xem trạng thái/resume. `--verbose` bật lại tường minh.

Python cố định theo môi trường dự án:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
& 'C:\Users\tranl\miniconda3\envs\ndt\python.exe' run_campaign.py check --root experiments/campaigns/verification
```

Lập một campaign mới với ngân sách riêng bằng `--settings file.json`; không sửa
plan đã có. Ví dụ dense cache float32 cần nhiều dung lượng hơn:

```json
{"feature_stride": 1, "feature_dtype": "float32"}
```

```powershell
& 'C:\Users\tranl\miniconda3\envs\ndt\python.exe' .\run_all.py --root experiments/campaigns/pect_dense --settings configs/campaign_dense.json
```

Ma trận full có nhiều lần pretrain độc lập, nên là chiến dịch dài. Planner in
số job, peak feature cache dự kiến và tổng nếu giữ tất cả cache; không có cam kết
thời gian trước khi đo throughput. Feature maps được dùng chung giữa các readout,
rồi xóa sau khi mọi job phụ thuộc hoàn tất. Nếu job còn pending, cache được giữ;
nếu cache bị thiếu/hỏng, runner trích xuất lại từ checkpoint trước khi tiếp tục.
Mặc định feature grid stride=2 và cache float16 được ghi rõ trong manifest; có
arm kiểm tra float32. **Training model vẫn FP32.**

`tf_consistency_pect` là bản điều chỉnh ý tưởng time–frequency consistency cho
PECT, không phải TFDA nguyên bản. Không có implementation giả cho TS-JEPA,
TimeSiam hoặc ROID. Chi tiết lý do chọn/bỏ có trong kế hoạch.

`train.py`, `evaluate.py` và các run lịch sử vẫn dùng để replay; không trộn
metric của chúng với campaign mới. `_audit/` đã chuyển sang `archive/`.
Checkpoint và output thí nghiệm vẫn ở local/gitignored.

`run_campaign.ps1` là wrapper tùy chọn. Máy hiện yêu cầu chữ ký số cho `.ps1`;
lệnh Python ở trên đã được kiểm tra và không cần đổi execution policy.
