# Kế hoạch nghiên cứu và thực thi PECT, phiên bản campaign v1

Ngày chốt thiết kế: 2026-10-30. Phạm vi: `cross_condition/`.

## 1. Câu hỏi và giới hạn kết luận

**RQ1:** JEPA dự đoán giữa điều kiện đo có tạo biểu diễn chuyển miền tốt hơn
self-JEPA, spatial MAE và encoder ngẫu nhiên cùng kiến trúc/ngân sách không?

**RQ2:** Khi target chưa tham gia pretraining, adaptation giúp bao nhiêu so với
cùng model nguồn? Lợi ích có khác với cho target tham gia pretraining ngay từ đầu?

**RQ3:** Chuyển sensor, waveform, lift-off ảnh hưởng thế nào đến detection,
định vị và sizing khi tính tới vùng đáp ứng rộng hơn CAD?

**RQ4:** Kết quả còn giữ được tại các vị trí vật lý không dùng để học không?

Các scan hiện tại đo lại ba bố trí specimen. Không coi 81 scan hoặc ba seed
là 81 specimen độc lập. Campaign này không chứng minh foundation model phổ quát,
không chứng minh tổng quát hóa sang specimen được chế tạo mới và không bảo đảm
các thuật toán adaptation sẽ thắng. Hoàn tất ma trận là đủ artifact cho các RQ
trong phạm vi trên, không tự động là đủ bằng chứng cho mọi tuyên bố khoa học.

## 2. Tái cấu trúc: giữ, ngừng dùng, sửa

| Thành phần | Quyết định và lý do |
|---|---|
| `models/`, `configs/config.py` | Giữ model chuẩn và API/state_dict cũ. Không phục hồi các nhánh exp44–69 đã loại bỏ. |
| `data/preprocessing.py`, `data/split.py`, topology | Tái sử dụng canonical registration fail-closed và parser metadata. |
| `data/paired_dataset.py` | Sửa truyền transform CSV rõ ràng; thêm cache mapping và self-pair mặc định tắt. Không thay behavior model/checkpoint. |
| `campaign/` | Luồng nghiên cứu chính: protocol → prepare → train → extract → evaluate → aggregate. |
| `train.py`, `evaluate.py`, trainer cũ | Giữ làm legacy/replay. CLI train cũ vẫn học mọi file; không dùng nó cho nghiên cứu source-only. |
| `_audit/` | Chuyển các script local sang `archive/historical_audits/`; loại khỏi cây thao tác chính. Không dùng làm metric source. |
| `colab/`, `experiments/*` scripts, `verification/`, tests cũ | Giữ để replay, không tự động phát hiện/chạy trong campaign. Không xóa công việc người dùng. |
| Raw/FFT từng pixel | Không dùng làm đối chứng chính. Thay bằng FFT cùng 25 probe; thêm neural dual-view có thời gian và Fourier. |
| Invariance đối xứng, radial VICReg, oracle background | Không đưa lại vào ma trận; FM1/FM6/FM8/FM35 và GT leakage đã có bằng chứng. |
| `knowledge/build.py` | Đọc kết quả campaign đầy đủ từ master_benchmark files; loại smoke, giữ verdict chờ review. |

Trong campaign mới, tokenizer chỉ nhận nội dung visible khi tạo context. Các
probe hidden được thay bằng trung bình visible trước các phép xử lý phụ thuộc
toàn patch. Điều này xử lý structural coupling H37. Cùng cách này áp dụng cho
mọi objective mới; số campaign không so trực tiếp với số checkpoint lịch sử.

## 3. Dữ liệu và vai trò

Inventory bắt buộc đủ 81 tuple duy nhất, loại acquisition Differential Pot.
Khóa SHA256 TDMS, registration CSV và GT. Không dùng fallback identity.

Sáu protocol một trục:

| ID | Source | Target | Số source / target |
|---|---|---|---:|
| `air_tmr` | Hall Air | TMR | 27 / 27 |
| `pot_tmr` | Hall Pot | TMR | 27 / 27 |
| `square_chirp` | Square | Chirp | 27 / 27 |
| `gaussian_chirp` | Gaussian | Chirp | 27 / 27 |
| `z1_z3` | z1 | z3 | 27 / 27 |
| `z3_z1` | z3 | z1 | 27 / 27 |

Mỗi protocol có 27 cặp source–target cùng ba yếu tố còn lại. Một model dùng
chung tất cả cặp. Chấm riêng từng target scan rồi macro-average; không pooling
pixel để các scan lớn/đậm defect lấn át. So adaptation trước/sau trên cùng file.

Encoder SSL source dùng 27 file source không nhãn. Head fit trên 21 source,
validation trên sáu source còn lại, hai file mỗi specimen. Các file validation
có thể đã xuất hiện **không nhãn** trong SSL; vì vậy chỉ gọi là validation cho
head/threshold, không gọi là encoder-held-out validation. Checkpoint SSL lấy
fixed-budget final, không chọn bằng target hoặc loss giảm đơn thuần.

`compound`: source gồm Hall Air/Pot × Square/Gaussian × z1/z3 × ba specimen
(24 file); head fit/val = 18/6, target còn lại 57 file. Đây là kiểm tra vận hành
trên hỗn hợp single/multiple shifts, không diễn giải thành tác động riêng sensor.

`air_tmr_spatial`: cùng protocol sensor, nhưng chia các ô không gian 90 px
bằng quy tắc hình học, không dựa CAD. Giữa tâm train và test có buffer 40 px.
Pretraining, head fit và neural adaptation chỉ dùng tâm train. Chấm trên ô test.
Đây là kiểm tra vị trí mới trên cùng mẫu, chưa thay thế specimen mới.

Target label chỉ vào scoring/matching; không vào target marginal sampling,
checkpoint selection, pseudo-label, adaptation hoặc chọn threshold.

## 4. Ma trận model và adaptation

Mỗi protocol chạy seed 42/43/44, **pretrain độc lập** cho từng seed. Mỗi
checkpoint phục vụ detection, anomaly score và sizing; không chọn riêng encoder
đẹp cho từng task.

| Biểu diễn | Pretraining | Readout/DA |
|---|---|---|
| Paired JEPA source | Context A → latent B, cùng vị trí; chỉ source | LR, Mahalanobis, CORAL, whitening+LR, neural λ=0, DANN |
| Paired JEPA transductive | Cùng objective/ngân sách; source+target không nhãn | LR, Mahalanobis |
| Self-JEPA source | Context → latent trong cùng scan | LR, Mahalanobis |
| Spatial MAE source | Cùng backbone/input/mask; reconstruct waveform và complex Fourier | LR, Mahalanobis |
| Random encoder | Cùng kiến trúc, không pretrain | LR, Mahalanobis |
| Spatial FFT | FFT center + mean 24 probe, cùng trường quan sát | LR, Mahalanobis |
| Dual-view source | Encoder MLP thời gian + Fourier trên center/context, supervised source | Neural source-only control |
| `tf_consistency_pect` | Cùng model dual-view; source-free target adaptation | Teacher, kNN pseudo-label, consistency giữa hai nhánh; không target GT |

Paired source có thêm **encoder adapter** được cập nhật bằng masked latent
prediction target và neo vào source teacher. Backbone/predictor/teacher giữ
cố định. Adapter nằm trên đường tạo feature, tránh FM31. Head và threshold của
arm này fit từ feature nguồn trước adaptation; không âm thầm refit trong hệ
tọa độ mới. So với paired source LR trên cùng target.

Đối chứng DANN dùng cùng kiến trúc, initialization, source batches và số bước
với neural λ=0. Không gán lợi ích neural head so với LR cho alignment.

Ablation condition token OFF chạy ở `air_tmr`, `square_chirp`, `z1_z3`, ba seed.
Không có direct invariance loss trong các arm mới. Spatial MAE là control cùng
kiến trúc, không được gọi là implementation của một paper MAE khác.

### Nguồn phương pháp gần đây và tên gọi chính xác

`tf_consistency_pect` là implementation điều chỉnh cho PECT, lấy ý tưởng dual
time–frequency / teacher / neighborhood từ
[TFDA, Information Sciences 2025](https://doi.org/10.1016/j.ins.2024.121734),
[code tác giả](https://github.com/furqon3009/TFDA). Khác bản gốc: token là toàn
waveform, dùng spatial context, không temporal splitting và không ép class prior
đồng đều. Không trình bày đây là TFDA nguyên bản, SOTA reproduction hoặc một
thuật toán mới đã được xác lập tính mới.

[TS-JEPA, 2025](https://arxiv.org/html/2509.25449v1),
[TimeSiam, ICML 2024](https://proceedings.mlr.press/v235/dong24e.html) và
[TimeDART, ICML 2025](https://proceedings.mlr.press/v267/wang25r.html) dùng
temporal patches/subseries. Không đưa nguyên bản vào default campaign khi
invariant cấm hard temporal slicing còn hiệu lực. ROID nguyên bản cũng không
được gắn tên cho whitening hoặc adapter JEPA. Các reproduction này cần protocol
riêng nếu sau này thay invariant; không có job giả trả success cho chúng.

## 5. Footprint, detection và sizing

- Giữ CAD làm hình học defect thật; không giãn label target để nâng IoU.
- Candidate từ local maxima trên score map, tách nhau tối thiểu theo khoảng
  cách cố định. Matching một–một với CAD, tolerance cố định 10 px, không tune trên test.
- Threshold khóa từ background validation source ở mức pixel FPR yêu cầu 1%.
  Target FPR và false alarms/scan được **đo**, không được bảo đảm bằng threshold này.
- Báo AUROC, AP, prevalence, precision/recall/FPR, IoU, object recall,
  false alarms/scan, missed objects và sai số định vị. Object recall tại threshold
  này không được gọi là đường FROC đầy đủ hay POD có kiểm định.
- Sizing chỉ học đường kính Corrosion vì semantic đường kính/rivet ở các mẫu khác
  không đồng nhất. Input là mean/std latent và score quantiles trong ba vòng
  của ROI bán kính cố định 20 px quanh tâm candidate; không dùng đường kính target
  để cắt ROI. Train dùng tâm source CAD; inference dùng tâm detector dự đoán.
- Ridge dự đoán trực tiếp mm, không expm1 gây ngoại suy vô hạn. Clip theo range
  đường kính source và lưu cả prediction chưa clip. Đối chứng constant = median source.
- Báo MAE trên detected matched objects kèm detection coverage/missed objects.
  Không gọi đây là MAE cho mọi defect, không gán kích thước giả cho defect bị bỏ sót.
- Có `sizing_known_location.csv` dùng tâm CAD làm diagnostic riêng để tách lỗi
  detection khỏi thông tin kích thước. ROI vẫn cố định, không cắt theo đường kính
  target; kết quả này không được trình bày là detection/sizing triển khai tự động.
- ROI radius 10/20/30 px chạy ở `air_tmr` và `z1_z3` để kiểm tra lợi ích bao trùm footprint.

Các ROI/threshold trên được preregister; kết quả chưa biết có tốt hay không.
Không triển khai kernel deconvolution tùy ý: chưa có hiệu chuẩn PSF độc lập để
tách kernel khỏi hình học defect một cách đáng tin cậy.

## 6. Ngân sách và độ chính xác

Mặc định: 100 epoch × 16.384 patch/epoch, batch 64; encoder D=64, depth=4;
AdamW lr=4e-4, lịch cosine, FP32; adapter 10 epoch lr=1e-4. Neural control/DANN:
400 bước source + 400 bước tiếp; dual-view source: 400 bước source, bản adaptation
thêm 400 bước target. Đây là ngân sách cố định, không phải kết quả tuning SOTA.

`grid_stride=4` cho lấy mẫu SSL; epoch dùng lấy mẫu tự nhiên phủ các directed
pair bằng seed/epoch/step, không lặp mãi 256 index đầu và không cân bằng defect 50/50.
Số bước optimization giữ bằng nhau giữa objective/scope; không gọi epoch là một
lượt exhaustive trên hai universe có kích thước khác nhau.

Feature map mặc định stride=2, nên pixel metrics là **sampled canonical grid**.
Cache lưu float16 để vừa ổ đĩa, chuyển về float32 trước readout; training vẫn FP32.
Có arm float32 cùng checkpoint ở `air_tmr`, ba seed, để đo độ nhạy quantization.
Muốn bản dense/float32, tạo campaign mới với settings JSON, không sửa manifest cũ.
Planner tính peak cache theo dependency và báo thêm dung lượng tích lũy nếu mọi
feature map được giữ lại. Runner xóa map sau job tiêu thụ cuối cùng, giữ nguyên khi còn
job pending; nếu map bị thiếu/hỏng thì trích xuất lại từ checkpoint đã lưu. Kiểm
tra disk trước chạy gồm peak cache và canonical cache, chưa gồm weights/logs.
Không hứa thời gian hoàn tất trước khi đo throughput máy.

Không cần chạy tất cả tích Descartes representation × DA: alignment chính ở
paired source; các control SSL dùng cùng LR/Mahalanobis để tách chất lượng biểu
diễn khỏi cơ chế adaptation. Ma trận chính xác, số job và mọi command nằm trong
`experiments/campaigns/pect_v4/plan.json` sau khi lập kế hoạch.

Manifest mặc định có **842 job**: prepare 1, train/khởi tạo 153, extract 172
và evaluate 516. Có 24 encoder ngẫu nhiên không train, nên còn 129 lượt
optimization SSL/adapter. Nếu giữ toàn bộ map, tổng feature cache là 51,5 GiB;
runner chỉ cần khoảng 1,78 GiB peak cache nhờ giải phóng theo dependency, cộng
2,82 GiB canonical cache. Checkpoint/log chưa nằm trong ước lượng. `pect_v1`–
`pect_v3` là manifest nháp chưa chạy; lệnh mặc định lập/tiếp tục `pect_v4`.

## 7. Checkpoint, lỗi và tiếp tục

- `plan.json` bất biến: file roles, seeds, config, dependencies, fingerprint,
  SHA256 data/GT/registration và identity code. `git_info.json` cùng code snapshot
  lưu commit, dirty state, môi trường và source code thực tế.
- Runner một child tại một thời điểm, OS lock tự nhả khi crash; kiểm tra Python/ML
  process đang dùng GPU trước mỗi train/extract. Không kill tiến trình khác.
- `state.json`: pending/running/completed/failed/blocked/interrupted, attempt,
  PID, heartbeat và log. Không retry liên tục sau lỗi.
- `latest.pt`: model, optimizer, RNG Python/NumPy/Torch/CUDA, epoch, batch kế tiếp,
  accumulator và lịch sử. Lưu mỗi 25 bước và cuối epoch; khi bị kill đột ngột
  có thể phải làm lại một khoảng lưu 25 bước. Không hứa resume giữa một optimizer step.
- `best.pt` là alias checkpoint **fixed-budget final**, không phải chọn bằng test.
- Neural readout cũng có step checkpoint. Extraction và scoring tiếp tục theo file.
- File atomic; receipt có SHA256 artifact. Exit code 0 mà thiếu receipt/artifact
  không được đánh dấu completed. Artifact kết quả bị sửa/mất sẽ báo corruption;
  feature map là cache tạm, được kiểm SHA khi cần và có thể tái tạo.
- Chạy lại đúng lệnh `run`/`all`: bỏ qua job đã xác minh, tiếp tục job dang dở.
  `--keep-going` cho phép chạy nhánh độc lập sau lỗi; cuối cùng vẫn exit khác 0
  nếu có job chưa hoàn tất. `--max-jobs N` dùng chia lượt chạy chủ động.
- Thay code/data/GT sau khi khóa plan: tạo campaign mới, không trộn kết quả khác
  implementation. Snapshot cũ cho phép quay lại version cũ để resume.
- Sau mỗi eval thật, rebuild graph. Smoke không vào evidence; không tự đổi
  hypothesis ACCEPTED chỉ vì chạy thành công.

## 8. Đọc kết quả và quyết định

`summary.csv` và `by_condition.csv`: xem chất lượng tuyệt đối theo protocol và
các lát sensor/waveform/lift-off/specimen. Không cộng các scenario chồng target
thành những mẫu độc lập mới.

`paired_deltas.csv`: CORAL/whitening so LR, DANN so neural control,
tf-consistency so dual-view source; adapter so cùng paired source frozen head.
`negative_transfer.csv` lấy trung bình seed trước trên mỗi scan rồi tính tỷ lệ
âm. Không suy độc lập từ scan–seed và không tự sinh p-value/CI population từ ba mẫu vật lý.

`completion.json` và `REPORT.md` phải COMPLETE, đồng thời đọc mọi failure,
coverage và clipping trước kết luận. Nếu JEPA source tốt và DA không tăng,
báo cáo giới hạn của adaptation trong thiết lập đã kiểm soát. Nếu transductive
tăng nhưng source-only kém, lợi ích tiếp cận target là một kết quả riêng.

## 9. Kiểm thử trước training dài

1. `py_compile` file chạm; acceptance deterministic/split/masking/strict-load.
2. Mô phỏng crash/resume, corruption, khóa và dependencies.
3. Smoke dữ liệu thật: mỗi objective 1 epoch/256 samples, kiểm tra CSV; resume
   giữa batch cho JEPA phải khớp model và lịch sử; smoke adapter và mọi readout.
4. Rebuild graph sau eval; giữ bản xác nhận boolean trong
   `verification/campaign_acceptance.json`, xóa checkpoint/cache smoke.
5. Lập full manifest mới, kiểm tra số job, dung lượng, status pending. Người dùng
   khởi chạy training dài bằng `run_all.py`; không khởi chạy toàn bộ trong phiên refactor.

## 10. Dữ liệu cần thêm nếu muốn kết luận rộng hơn

Sau benchmark hiện tại, cần một đợt đo độc lập có tháo/lắp lại và specimen hoặc
bố trí defect mới. Khóa test trước khi tune. Đo từng hướng Hall→TMR với waveform/
lift-off tương ứng, giữ các lần đo lặp trong cùng group khi chia dữ liệu. Chọn
số mẫu qua phân tích độ bất định và nguồn lực đo, không coi các pixel lân cận
là hàng nghìn quan sát độc lập. Đây là công việc thu thập dữ liệu; script hiện
tại không thể tạo ra bằng chứng vật lý còn thiếu.
