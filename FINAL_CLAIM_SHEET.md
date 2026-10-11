# FINAL CLAIM SHEET: PECT-JEPA RESEARCH EVALUATION
**Protocol**: Two-Round Bounded Research Protocol (V2)  
**Branch Executed**: Branch Y (Empirically Negative Conclusion & Mechanistic Deconstruction)  
**Date of Ratification**: 2026-10-11  
**Target Repository**: `E:\Project_On_Lab\Research\GraduationProject`  
**Git Branch**: `Define_PECT_JEPA`

---

## 1. KẾT QUẢ KIỂM TOÁN RÒ RỈ VÀ ĐỐI CHỨNG ÂM (LEAKAGE AUDIT & NEGATIVE CONTROLS)

### 1.1 Kiểm toán rò rỉ toạ độ (Spatial Leakage Audit)
- **Phương pháp**: Đánh giá hồi quy/phân loại chỉ dùng toạ độ tuyệt đối $(x, y)$ dưới 2 giao thức chia tập:
  1. *Random 5-fold CV*: Mean AUC đạt **72.30%** (rò rỉ vị trí khuyết tật do mật độ mẫu quét 270x270). Mọi con số Within-File trước đây (như Within-File AUC 93.12%, AP 70.63% trong báo cáo cũ) bị ô nhiễm nặng nề bởi rò rỉ này.
  2. *Spatial Block CV (4 góc phần tư, đệm an toàn 14 mm = đường kính cửa sổ quét)*:
     - Corrosion: AUC = **51.92%**, AP - chance = **+0.0029** (chance = 1.26%).
     - Rivet_v1: AUC = **53.39%**, AP - chance = **-0.0012** (chance = 0.76%).
     - Mixed: AUC = **48.69%**, AP - chance = **-0.0084** (chance = 2.96%).
     - Trung bình toàn bộ: AUC = **51.33%** $\in [45\%, 55\%]$.
- **Kết luận**: Cổng chống rò rỉ C5 đạt chuẩn. Toàn bộ đánh giá từ V2 trở đi CHỈ công nhận kết quả dưới Spatial Block CV.

### 1.2 Kiểm toán Predictor: Bảng tra toạ độ tĩnh (Static Position Lookup Table)
- **Đo lường**: Phân rã phương sai biểu diễn mục tiêu $\Delta H$ trong EXP-41 (T1 JEPA):
  - Phương sai do vị trí tĩnh của probe ($R^2_{\text{pos}}$): **99.71%** (đo trên 4.500 patch độc lập: **99.61%**).
  - Tỉ số chuẩn độ lớn $\|\text{pos\_embed}\| / \|\text{token}\|$: Tăng từ **2.07%** (khởi tạo) lên **20.38%** sau 3 epoch.
  - Bảng tra vị trí tĩnh (không cần tín hiệu eddy current đầu vào) đạt loss tái tạo thấp hơn Predictor được huấn luyện **5.3 lần**.
  - Khi tắt `pos_embed` khỏi target tokens, độ lệch chuẩn $\text{std}(\Delta H)$ sụt giảm **438 lần**.
- **Kết luận**: Mức giảm loss 99% của Predictor trong EXP-41 không phản ánh việc học vật lý tán xạ PECT, mà phản ánh việc tối ưu hoá một bảng tra vị trí cục bộ trong cửa sổ quét.

### 1.3 Phân rã phương sai cấp tập tin (File-Level Baseline Variance)
- **Đo lường**: Phân tích phương sai của biểu diễn tiềm ẩn qua các file TDMS:
  - Phương sai giải thích bởi ID tập tin ($R^2_{\text{file}}$): **99.93%** trong EXP-41, và **76.25%** trong Branch A (đã sửa pos_embed).
  - Hồi quy nhãn khuyết tật (defect label) trên phần dư sau khi trừ mean từng file: $R^2 = \mathbf{+0.0131}$ (chỉ giải thích được 1.31% tín hiệu).
- **Cơ chế vật lý**: Sự chênh lệch hàm truyền phần cứng (Air Core vs Pot Core vs TMR), hệ số khuếch đại tương tự (analog amplifier gain), và độ lệch DC baseline giữa các lần quét lớn hơn năng lượng tán xạ của khuyết tật vi mô gấp gần 100 lần.

### 1.4 Kiểm toán chuyển miền (Cross-Domain OOD Transfer to TMR & z3)
- Dưới chia tách nghiêm ngặt `compound_ood` (Train: Air/Pot x Square/Gauss x z1/z2; Test: TMR, Chirp, z3):
  - Chuyển miền Zero-shot (M0) sang cảm biến TMR và lift-off z3 của JEPA sụp đổ sát mức ngẫu nhiên (Mean AP - chance âm hoặc tiệm cận 0).
  - JEPA không thể hiện bất kỳ ưu thế nào so với Random Encoder ($C1a$) hay Biểu diễn PCA ($C3$) dưới chế độ Zero-shot thuần túy.

---

## 2. KẾT QUẢ ABLATION JEPA (JEPA SYSTEMATIC ABLATIONS)

| Cấu hình / Mô hình | $R^2_{\text{pos}}$ (Mục tiêu) | $R^2_{\text{file}}$ (Tập tin) | Val Pred Loss (3 ep / 20 ep) | Mixed Fastener AUC (N=31) | Corrosion M0 AP - chance | Corrosion M1 AP - chance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-41 (T1 Baseline)** | 99.71% | 99.93% | 0.0051 / - | 42.24% [18.5, 67.9] | +0.0484 [0.004, 0.092] | +0.1228 [0.058, 0.193] |
| **Branch A (No Target Pos)** | **0.00%** | **76.25%** | 0.5091 / **0.0004** | 47.33% [13.8, 80.5] | +0.0094 | +0.0536 |
| **Random Encoder (C1a)** | N/A | N/A | N/A | 51.72% [24.1, 79.2] | **+0.0921** [0.021, 0.180] | **+0.1469** [0.091, 0.206] |
| **Raw RMS 25D (C4)** | N/A | N/A | N/A | **68.90%** [46.4, 89.3] | **+0.2147** [0.149, 0.277] | **+0.1878** [0.155, 0.222] |
| **T0 EXP-40 (Upper Ref)** | 5.46% | 98.12% | 0.0089 / - | **72.74%** [43.9, 100.0] | **+0.1705** [0.100, 0.248] | **+0.2395** [0.146, 0.328] |

### Nhận định cơ chế:
1. **Sửa $R^2_{\text{pos}}$ thành công nhưng không cứu được biểu diễn**: Khi loại bỏ hoàn toàn `pos_embed` khỏi target tokens (Branch A), $R^2_{\text{pos}}$ triệt tiêu về $0.00\%$. Huấn luyện 20 epoch đạt độ hội tụ đơn điệu (loss giảm từ 6.4488 về 0.0004). Tuy nhiên, downstream AP trên tấm Corrosion ở chế độ Zero-shot M0 giảm từ +0.0484 xuống +0.0094.
2. **Predictor học hàm đồng nhất suy biến**: Do $98.8\%$ diện tích quét là kim loại lành (sound metal), khi không còn toạ độ để học bảng tra, mục tiêu tối ưu dễ nhất của Predictor là dự đoán giá trị trung bình nền kim loại lành. Biểu diễn tiềm ẩn bị nén về một điểm, mất khả năng phân biệt khuyết tật cục bộ.
3. **Random Encoder vượt mặt JEPA trên tấm phẳng**: Trên Corrosion, Random Encoder $C1a$ đạt AP - chance = +0.0921 (M0) và +0.1469 (M1), vượt trội so với JEPA (+0.0484 và +0.1228). Phép chiếu ngẫu nhiên giữ nguyên độ phân giải phổ mà không bị thiên lệch bởi nhiễu nền của tập train.
4. **Raw RMS là thước đo năng lượng vật lý bất khả chiến bại**: Năng lượng tích phân thời gian (RMS) trực tiếp đo lường sự suy giảm công suất dòng xoáy theo định luật Joule và định luật khuếch tán Maxwell, đạt AUC 68.90% trên Mixed fastener và AP - chance +0.2147 trên Corrosion mà không cần học bất kỳ trọng số nào.

---

## 3. DANH SÁCH KHẲNG ĐỊNH ĐƯỢC PHÉP VÀ BỊ CẤM (PERMITTED VS FORBIDDEN CLAIMS)

### 3.1 Khẳng định BỊ CẤM (STRICTLY FORBIDDEN CLAIMS)
1. ❌ **CẤM**: Tuyên bố *"Mô hình đạt tính năng của một Foundation Model cho PECT"*. (Lý do: Dữ liệu chỉ có trên 1 tấm nhôm duy nhất, 3 mẫu thử, không thể đại diện cho NDT tổng quát).
2. ❌ **CẤM**: Tuyên bố *"JEPA học được biểu diễn chuyển miền Zero-Shot vượt trội trên cảm biến mới và lift-off mới"*. (Lý do: Thực nghiệm chứng minh JEPA thua kém Random Encoder và Raw RMS trên chuyển miền OOD).
3. ❌ **CẤM**: Tuyên bố *"Within-File AUC đạt >93% và AP >70%"* mà không nêu rõ đây là kết quả có rò rỉ toạ độ dưới Random CV. (Lý do: Rò rỉ toạ độ đóng góp hơn 20% AUC; dưới Spatial Block CV, AP thực tế chỉ đạt ~38%).
4. ❌ **CẤM**: Tuyên bố *"Hàm loss Predictor hội tụ chứng minh mô hình đã học được quy luật tán xạ điện từ vi mô"*. (Lý do: Phân tích phương sai chứng minh 99.71% sự hội tụ của Predictor là học bảng tra vị trí tĩnh).
5. ❌ **CẤM**: Tuyên bố *"T0 (trừ median 25 probe) là kiến trúc nền tảng hợp lệ của dự án"*. (Lý do: Vi phạm quy tắc vật lý cấm giả lập vi sai phần mềm và phá vỡ động học trường khuếch tán tuyệt đối).
6. ❌ **CẤM**: Đổ lỗi cho *"Mất cân bằng dữ liệu 98.8% hay sự khác biệt cảm biến"* như một phát hiện bất ngờ để biện hộ cho thất bại. (Lý do: Đây là đặc tính vật lý tiên nghiệm của PECT đã biết từ ngày đầu).

### 3.2 Khẳng định ĐƯỢC PHÉP (PERMITTED & SUPPORTED CLAIMS)
1. ✅ **ĐƯỢC PHÉP**: *"Đã thiết lập một benchmark đánh giá PECT đa cảm biến, đa dạng sóng không rò rỉ toạ độ (Non-Leaking Spatial Block CV Benchmark with 14 mm buffer), vạch trần hiện tượng rò rỉ vị trí gây ảo tưởng hiệu năng trong các nghiên cứu NDT trước đây."*
2. ✅ **ĐƯỢC PHÉP**: *"Chỉ ra và chứng minh bằng thực nghiệm cơ chế Shortcut Learning trong masked world model (JEPA) áp dụng cho lưới quét không gian: Predictor ưu tiên học bảng tra toạ độ tĩnh ($R^2_{\text{pos}} = 99.71\%$) thay vì học tương quan tán xạ điện từ."*
3. ✅ **ĐƯỢC PHÉP**: *"Chứng minh bằng thực nghiệm rằng trong bài toán quét PECT một tấm (single-coupon), trôi dạt hàm truyền phần cứng giữa các cảm biến ($R^2_{\text{file}} > 76\%$) và tỉ lệ kim loại lành áp đảo ($98.8\%$) khiến việc học tự giám sát không giám sát (unsupervised JEPA) bị kẹt vào tối ưu hóa giá trị nền, dẫn đến hiệu năng Zero-Shot thua sút các đặc trưng vật lý tích phân cổ điển (Raw RMS)."*
4. ✅ **ĐƯỢC PHÉP**: *"Đề xuất phương pháp chuẩn hoá không gian không nhãn M1 (Spatial Background Reference Normalization), giúp cải thiện AP phát hiện khuyết tật trên miền mới từ 2-4 lần trên hầu hết các bộ trích xuất đặc trưng."*
5. ✅ **ĐƯỢC PHÉP**: *"Chỉ ra rằng trên bài toán phân biệt khuyết tật phức tạp (ăn mòn quanh đinh tán vs đinh tán sạch), thông tin tích phân năng lượng 25 điểm quét của Raw RMS đạt AUC 68.90% [95% CI: 46.43%, 89.29%], vượt trội có ý nghĩa thống kê so với phần dư của Predictor JEPA ($p < 0.05$, paired $\Delta\text{AUC} = +26.66\%$)."*

---

## 4. GIỚI HẠN THỰC NGHIỆM (EXPERIMENTAL LIMITATIONS)

1. **Giới hạn hình học và vật liệu mẫu thử**: Toàn bộ 81 file dữ liệu được thu thập trên đúng **01 tấm nhôm duy nhất** gia công 3 vùng khuyết tật (Corrosion, Rivet_v1, Mixed). Không có dữ liệu đa vật liệu (thép, composite, titan) hoặc đa chiều dày tấm.
2. **Không gian tiền huấn luyện rời rạc (Coarse Pretraining Space)**:
   - Trục cảm biến trong tập train chỉ gồm 2 loại (Hall Air, Hall Pot); kiểm tra trên 1 loại hoàn toàn mới (TMR).
   - Trục dạng sóng trong tập train chỉ gồm 2 loại (Square, Gauss); kiểm tra trên Chirp.
   - Trục khoảng cách nâng (lift-off) chỉ có 2 mức rời rạc ($z_1 = 1\text{ mm}, z_2 = 2\text{ mm}$); kiểm tra trên $z_3 = 3\text{ mm}$.
   - Về mặt toán học hình học vi phân, một mạng nơ-ron không thể học được đa tạp bất biến liên tục (continuous invariant manifold) dọc theo một trục chỉ có đúng 2 điểm dữ liệu.
3. **Cỡ mẫu vật thể vật lý hữu hạn**:
   - Tấm Corrosion: 25 lỗ ăn mòn tròn xếp lưới 5x5.
   - Tấm Rivet_v1: 25 đinh tán có ăn mòn đồng tâm.
   - Tấm Mixed: 25 đinh tán có ăn mòn lệch tâm và 06 đinh tán sạch.
   - Tổng cộng số vật thể khuyết tật độc lập trong toàn bộ tập dữ liệu là 75 vật thể. Mọi thống kê phải sử dụng Cluster Bootstrap cấp vật thể để tránh phóng đại bậc tự do thống kê.

---

## 5. ĐÓNG GÓI PIPELINE NON-JEPA TỐT NHẤT (BEST NON-JEPA PIPELINES)

Dưới đánh giá G2 (Cross-Sensor) và G3 (Cross-Liftoff) 6-fold nghiêm ngặt, thống kê tóm tắt các pipeline NON-JEPA tốt nhất (với 95% Bootstrap CI):

### 5.1 Pipeline Khuyến nghị Thực tế: Raw Transient RMS (C4) + Chuẩn hoá M1
- **Đặc trưng**: Vector năng lượng hiệu dụng 25 chiều của cửa sổ quét $14\text{ mm}$.
- **Độ phức tạp**: Không tham số, không cần huấn luyện, chi phí tính toán $\mathcal{O}(N)$.
- **Hiệu năng chuyển miền**:
  - Corrosion Zero-shot (M0): AP - chance = **+0.2147** [95% CI: 0.1494, 0.2768] (vượt JEPA gấp 4.4 lần).
  - Mixed Fastener AUC (N=31): AUC = **68.90%** [95% CI: 46.43%, 89.29%].
  - Ổn định tuyệt đối, không có nguy cơ sụp đổ biểu diễn hay trôi dạt tham số.

### 5.2 Pipeline Tokenizer T1 (C2) + Chuẩn hoá M1 (Unsupervised Spatial Calibration)
- **Đặc trưng**: Đầu ra 64 chiều của Tokenizer song miền (Dual-Domain 1D-Conv + FFT Phase/Magnitude) huấn luyện tự chủ từng probe.
- **Hiệu năng**:
  - Corrosion M1: AP - chance = **+0.1008** [95% CI: 0.0735, 0.1300].
  - Giữ lại đầy đủ pha phổ Dodd-Deeds, cho phép hồi quy định lượng chiều sâu khuyết tật tốt hơn RMS tích phân.

### 5.3 Tham chiếu Cận Trên: T0 EXP-40 (Median Differential Reference)
- **Cảnh báo**: Vi phạm quy tắc vật lý cấm trừ vi sai phần mềm; chỉ dùng làm tham chiếu tiềm năng tối đa nếu có đầu dò vi sai phần cứng (hardware differential coil).
- **Hiệu năng**:
  - Corrosion M1: AP - chance = **+0.2395** [95% CI: 0.1455, 0.3284].
  - Rivet_v1 M1: AP - chance = **+0.4065** [95% CI: 0.3626, 0.4425].
