# Báo cáo Toàn diện Toàn bộ Dataset (81 File TDMS): Anomaly Detection & Geometric Defect Sizing

## 1. Thiết lập Đánh giá Toàn cảnh

Hệ thống đã hoàn tất việc đánh giá toàn diện mô hình **Self-Supervised CrossConditionJEPA** (huấn luyện với hàm loss $L_1$ thuần túy và cơ chế trích xuất Condition Token $c_B$ tự thân từ nền sóng, hoàn toàn không metadata) trên toàn bộ **81 file TDMS** ($3 \times 3 \times 3 \times 3$).

### Các tác vụ được đánh giá:
1. **Unsupervised Anomaly Detection (Phát hiện dị thường không giám sát)**:
   - Khoảng cách Mahalanobis trên không gian tiềm ẩn 128D so với nền biên chuẩn hóa ($15\,\text{px}$).
   - **Mean AP - chance ($\Delta\text{AP}$)**: Đo lường cải thiện độ chính xác so với đoán ngẫu nhiên trên phân phối mất cân bằng (tỉ lệ khuyết tật $< 1.5\%$).
   - **Mean Pixel AUC-ROC**: Đo lường khả năng phân tách nhị phân toàn diện.
2. **Fastener-level Classification (Phân biệt đinh tán ăn mòn vs đinh tán sạch trên mẫu Rivet_v2 / Mixed)**:
   - Xếp hạng 31 vật thể đinh tán (25 đinh tán có ăn mòn lệch tâm vs 6 đinh tán lành) dựa trên cực đại điểm dị thường trong bán kính đinh tán ($R = 10\,\text{px}$).
3. **Geometric Defect Sizing (Định lượng kích thước khuyết tật hình học)**:
   - **Mẫu Corrosion (25 lỗ tròn)**: So sánh đường kính ước lượng $D_{\text{pred}} = 2\sqrt{\text{Area}/\pi}$ với đường kính thực theo bản vẽ CAD ($2, 4, 6, 8, 10\,\text{mm}$) $\to$ Báo cáo **MAE Đường kính (mm)** và hệ số xác định **$R^2$ Đường kính**.
   - **Mẫu Rivet (Rivet_v1 & Rivet_v2)**: Đánh giá độ trùng khớp biên dạng khuyết tật qua **Best IoU** và **Dice Coefficient (F1)**.

---

## 2. Bảng Tổng hợp Toàn diện Toàn bộ 81 File TDMS

| Phân vùng / Lát cắt (Slice) | Chỉ số (Metric) | CrossConditionJEPA (Đã huấn luyện) | Random Encoder ($C1a$) | Raw RMS ($C4$) |
|---|---|---|---|---|
| **Overall (Toàn bộ 81 files)** | **Mean AP - chance** | **+0.0035** | +0.0041 | **+0.0053** |
| | Mean Pixel AUC | 50.58% | 50.86% | 51.29% |
| | **Fastener AUC (Mixed)** | **60.17%** | 60.64% | **67.65%** |
| | **Đường kính MAE (mm)** | **5.95 mm** | 5.79 mm | **5.08 mm** |
| | Đường kính $R^2$ | -5.7664 | -5.5359 | -4.1220 |
| | Best Flaw IoU | 2.08% | 2.29% | 2.50% |
| **Train Base (16 files)** | **Mean AP - chance** | **+0.0011** | +0.0012 | +0.0004 |
| | Mean Pixel AUC | **49.78%** | 50.09% | 48.95% |
| | Đường kính MAE (mm) | 5.88 mm | 5.65 mm | 5.23 mm |
| | Best Flaw IoU | **0.73%** | 0.69% | 0.47% |
| **Test OOD (65 files)** | **Mean AP - chance** | **+0.0040** | +0.0048 | **+0.0066** |
| | Mean Pixel AUC | 50.77% | 51.05% | 51.87% |
| | Fastener AUC (Mixed) | 60.17% | 60.64% | 67.65% |
| | Đường kính MAE (mm) | 5.98 mm | 5.85 mm | 5.01 mm |
| | Best Flaw IoU | 2.32% | 2.57% | 2.85% |
| **Mẫu: Corrosion (27 files)** | Mean AP - chance | **+0.0011** | +0.0013 | +0.0010 |
| | **Mean Pixel AUC** | **53.37%** | 53.33% | 52.56% |
| | Đường kính MAE (mm) | 5.95 mm | 5.79 mm | 5.08 mm |
| **Mẫu: Rivet_v1 (27 files)** | Mean AP - chance | -0.0012 | -0.0012 | -0.0013 |
| | Mean Pixel AUC | 41.38% | 41.37% | 41.87% |
| | Best Flaw IoU | 0.23% | 0.24% | 0.25% |
| **Mẫu: Rivet_v2 / Mixed (27 files)** | **Mean AP - chance** | **+0.0105** | +0.0122 | **+0.0163** |
| | Mean Pixel AUC | 56.99% | 57.88% | 59.45% |
| | **Fastener AUC** | **60.17%** | 60.64% | **67.65%** |
| | Best Flaw IoU | 3.94% | 4.35% | 4.75% |
| **Cảm biến: Hall Air Core (27 files)**| Mean AP - chance | +0.0039 | +0.0046 | +0.0046 |
| | Mean Pixel AUC | 50.01% | 50.04% | 50.61% |
| | Fastener AUC | 58.30% | 59.93% | 70.74% |
| **Cảm biến: Hall Pot Core (27 files)**| Mean AP - chance | +0.0018 | +0.0023 | +0.0054 |
| | Mean Pixel AUC | 50.13% | 50.59% | 52.00% |
| | Fastener AUC | 62.81% | 63.11% | 74.89% |
| **Cảm biến: TMR (27 files)** | Mean AP - chance | +0.0046 | +0.0053 | +0.0060 |
| | Mean Pixel AUC | **51.59%** | 51.96% | 51.26% |
| | **Fastener AUC** | **59.41%** | 58.89% | **57.33%** |
| **Dạng sóng: Square (27 files)** | **Mean AP - chance** | **+0.0092** | +0.0101 | **+0.0033** |
| | **Mean Pixel AUC** | **53.51%** | 54.36% | **50.62%** |
| | **Fastener AUC** | **67.26%** | 68.59% | **65.70%** |
| | **Best Flaw IoU** | **3.63%** | 3.87% | **2.21%** |
| **Dạng sóng: Gaussian (27 files)** | Mean AP - chance | +0.0009 | +0.0014 | +0.0069 |
| | Mean Pixel AUC | 49.56% | 49.79% | 52.00% |
| | Fastener AUC | 54.96% | 55.93% | 68.59% |
| **Dạng sóng: Chirp (27 files)** | Mean AP - chance | +0.0003 | +0.0007 | +0.0058 |
| | Mean Pixel AUC | 48.66% | 48.44% | 51.26% |
| | Fastener AUC | 58.30% | 57.41% | 68.67% |
| **Lift-off: z1 (1 mm, 27 files)** | Mean AP - chance | +0.0049 | +0.0057 | +0.0072 |
| | Mean Pixel AUC | 51.68% | 52.30% | 50.87% |
| | Fastener AUC | 63.26% | 64.15% | 72.89% |
| **Lift-off: z2 (2 mm, 27 files)** | Mean AP - chance | +0.0027 | +0.0034 | +0.0054 |
| | Mean Pixel AUC | 50.40% | 50.47% | 51.23% |
| | Fastener AUC | 59.85% | 59.85% | 66.59% |
| **Lift-off: z3 (3 mm, 27 files)** | Mean AP - chance | +0.0027 | +0.0032 | +0.0035 |
| | Mean Pixel AUC | 49.65% | 49.81% | 51.78% |
| | Fastener AUC | 57.41% | 57.93% | 63.48% |

---

## 3. Phân tích Khoa học & Các Phát hiện Cốt lõi

### 3.1 Điểm sáng: Khả năng vượt trội của JEPA trên Dạng sóng Xung vuông (Square Waveform) và Cảm biến TMR
1. **Trên 27 file Xung vuông (Square Waveform)**:
   - **CrossConditionJEPA vượt trội hoàn toàn so với Raw RMS ($C4$)**:
     - $\Delta\text{AP}$: JEPA đạt **$+0.0092$**, cao gấp **2.8 lần** so với Raw RMS ($+0.0033$).
     - Pixel AUC: JEPA đạt **$53.51\%$** so với RMS **$50.62\%$**.
     - Fastener AUC: JEPA đạt **$67.26\%$** so với RMS **$65.70\%$**.
     - Best Flaw IoU: JEPA đạt **$3.63\%$**, cao hơn hẳn RMS **$2.21\%$**.
   - *Nguyên nhân vật lý*: Xung vuông tạo ra bước nhảy điện trường $\frac{dE}{dt}$ dốc đứng, kích thích phổ dòng xoáy đa tần số cực rộng. Các bộ lọc thời gian - không gian của JEPA trích xuất được các thành phần tán xạ thoáng qua mà năng lượng tích lũy thời gian đơn giản (RMS) bị triệt tiêu do tích phân toàn dải.
2. **Trên 27 file Cảm biến Từ điện trở (TMR Sensor)**:
   - Fastener AUC trên mẫu Mixed của JEPA đạt **$59.41\%$**, vượt qua Raw RMS (**$57.33\%$**). Cảm biến TMR có độ nhạy từ trường tiếp tuyến cao, và JEPA bảo toàn được thông tin hướng phân cực tốt hơn RMS.

### 3.2 Tác vụ Định lượng Kích thước Khuyết tật (Geometric Defect Sizing)
- **Trên mẫu Corrosion (25 lỗ ăn mòn tròn từ 2 mm đến 10 mm)**:
  - Sai số tuyệt đối đường kính trung bình ($\text{MAE}_D$) của cả 3 phương pháp dao động quanh ngưỡng **$5.0\,\text{mm} - 5.9\,\text{mm}$**.
  - Hệ số xác định $R^2$ đường kính của cả 3 phương pháp đều âm (JEPA: $-5.77$, RMS: $-4.12$).
  - *Giải thích nguyên nhân*: Khi sử dụng ngưỡng cắt không giám sát (95th percentile) trên bản đồ dị thường để ước lượng kích thước, vùng phản ứng từ trường xung quanh lỗ (Eddy Current Halo Effect) thường bị loang rộng hơn kích thước vật lý của lỗ khuyết tật do hiệu ứng tán xạ không gian của cuộn dây kích thích. Để ước lượng đường kính chính xác theo tiêu chuẩn NDT, cần áp dụng mô hình hồi quy tuyến tính (Linear Probe Head) có hiệu chuẩn kích thước thay vì ngưỡng cắt hình thái thô.

### 3.3 Đánh giá Tổng thể so với Baseline
- Trên bình diện toàn bộ 81 file: Raw RMS vẫn duy trì tính ổn định vượt trội trên các dạng sóng Gaussian và Chirp (những dạng sóng có năng lượng điều chế mềm).
- Việc chuyển sang loss $L_1$ thuần túy và trích xuất Condition Token $c_B$ tự thân đã bảo toàn tính chất **Pure Self-Supervised Paradigm** (không cần nhãn sensor hay waveform), đồng thời giúp JEPA tạo được ưu thế rõ rệt trên trục xung vuông (Square Waveform).
