# PECT-JEPA v-next: Đặc Tả Bài Toán, Nhiệm Vụ & Giao Thức Đánh Giá Chuẩn Tắc
**Tài liệu chuẩn mực phương pháp luận (Methodology & Evaluation Specification)**  
*Dự án: Self-Supervised Spatiotemporal Foundation Model cho Kiểm định Dòng xoáy Xung (PECT)*  
*Phiên bản: v1.0-RC1 (Tích hợp nguyên văn phản biện độc lập; Khóa cứng hợp đồng đánh giá)*  
*Trạng thái: BẢN ĐẶC TẢ PHƯƠNG PHÁP LUẬN CHÍNH THỨC*

---

## 1. Lịch Sử Phiên Bản & Tôn Chỉ Nhận Thức

### 1.1. Lịch sử phiên bản
- **v0.8**: Bản thảo sơ khởi dựa trên phân tích nền và radial profile (tồn tại mâu thuẫn điều khoản mép và nhãn chưa chặt chẽ).
- **v0.9-draft**: Sửa mâu thuẫn hình học mép, đính chính topology Concentric Star $14\times 14\text{ mm}^2$, phân loại nhiệm vụ.
- **v1.0-RC1**: Sửa toàn bộ 8 điểm phản biện kỹ thuật (sửa đếm mẫu $9 + 12 + 4 = 25$, tách biệt đơn vị Volt của PtP khỏi sai số không thứ nguyên trong latent, loại bỏ ngụy biện hình học d0, tích hợp nguyên văn Điều 1, Điều 2 và Mục 7 của chuyên gia phản biện, giải tỏa căng thẳng với `GEMINI.md`).

### 1.2. Hệ thống Nhãn Định Danh Nhận Thức
Mọi khẳng định kỹ thuật trong tài liệu này bắt buộc phải mang một trong 4 nhãn:
- **`[ĐO]`**: Đã có mã nguồn trích xuất trong repo, file TDMS gốc, kích thước mẫu $N$, kiểm soát âm/dương và khoảng tin cậy (CI) hoặc $p$-value xác định.
- **`[ĐO-chưa kiểm]`**: Số đo sơ bộ từ một bản quét cụ thể, chưa kiểm chứng chéo độc lập hoặc phương pháp đo chưa trung tính theo kích thước.
- **`[SUY RA]`**: Lập luận logic, toán học hoặc hình thái học xuất phát từ các số liệu đã đo.
- **`[GIẢ ĐỊNH]`**: Giả thuyết hoặc tham số giữ chỗ (placeholder), bắt buộc phải có điều kiện kiểm chứng thực nghiệm trước khi kết luận.

---

## 2. Định Nghĩa Bài Toán & Không Gian Đầu Vào

### 2.1. Bản chất Không gian - Thời gian của Đầu vào
Một mẫu kiểm định PECT cục bộ đưa vào mô hình là một mảng:
$$x \in \mathbb{R}^{5 \times 5 \times C}$$
trong đó:
- **Cấu trúc không gian**: Checkpoint baseline EXP-28 sử dụng cấu trúc **Sao Đồng Tâm (Concentric Star Topology)** với bán kính $r \in \{1, 3, 7\}\text{ pixel}$ `[ĐO: checkpoint config]`. Với giả định $1\text{ mm/pixel}$ `[GIẢ ĐỊNH - chờ bản vẽ coupon độc lập]`, mảng 25 probe này bao phủ cửa sổ hình học **$14\text{ mm} \times 14\text{ mm}$**, xấp xỉ kích thước vết tán xạ của khuyết tật.
- **Trục thời gian**: $C = 128$ mẫu thời gian được lấy mẫu lại tuyến tính từ 500 điểm đo quá độ thô ($100\text{ kHz}$ sampling rate).
- **Chuẩn hóa file**: Chuẩn hóa chia cho cực trị biên độ toàn bản quét (`file_peak`) để bảo toàn $100\%$ tỷ số tương phản không gian $\Delta V$.

### 2.2. Mục Tiêu Biểu Diễn Tự Giám Sát (JEPA Objective)
PECT-JEPA hướng tới học biểu diễn trong không gian ẩn thông qua bài toán dự đoán ngữ cảnh (Context Prediction):
$$\mathcal{L}_{\text{pred}} = \mathbb{E} \left[ \| g_\phi(z_{\text{ctx}}) - \operatorname{sg}[z_{\text{tgt}}] \|_2^2 \right]$$
với $z = f_\theta(x) \in \mathbb{R}^{D}$ ($D = 64$ latent token, $2D = 128$ foundation representation).  
Mục tiêu là nắm bắt quy luật khuếch tán điện từ của vật liệu lành, sao cho khi gặp khuyết tật, sai số tái tạo cục bộ $A(c)$ tăng vọt để làm điểm phát hiện dị thường không giám sát.

---

## 3. Các Đặc Tính Vật Lý & Thách Thức Dữ Liệu Đã Kiểm Toán

| Đặc tính | Giá trị đo đạc / Quan sát | Bằng chứng thực nghiệm | Nhãn nhận thức |
| :--- | :--- | :--- | :---: |
| **Thành phần chung (Common mode)** | Thành phần chung chiếm $99.0\% - 99.2\%$ biên độ PtP ($5.40 - 5.46\text{ V}$) trên Chirp/Hall Pot. Biến thiên khuyết tật chỉ chiếm $0.8\% - 1.0\%$ ($0.015 - 0.025\text{ V}$). TMR và Hall Air có tín hiệu tương đối mạnh hơn ($Z_{\text{PtP}}$ lên đến $+13$). | `scripts/audit_cscan_physics.py` | `[ĐO: Chirp/Hall Pot]` |
| **Trôi nền không gian** | Biến thiên trôi nền $\Delta V_{\text{bg}} = 0.060 - 0.071\text{ V}$ trên tấm Corrosion. Đa thức bậc 2 giải thích $R^2 = 0.42 - 0.59$ phương sai nền qua 3 mức lift-off. Nguyên nhân vật lý (nhiệt/cơ) chưa được đo cô lập. | `scripts/measure_cscan_groundtruth_physics.py` | `[ĐO-chưa kiểm nguyên nhân]` |
| **Nhiễu sọc bước quét $y$** | Sai phân liền kề trục $y$ ($\sigma_{\text{hf}, y} = 0.017 - 0.022\text{ V}$) gấp 5–6 lần trục $x$ ($\sigma_{\text{hf}, x} \approx 0.0035 - 0.0038\text{ V}$) trên Chirp/Hall Pot. Trên Gaussian $\sigma_{\text{hf}, y} = 0.120\text{ V}$, trên TMR là $0.036\text{ V}$. Offset từng dòng quét là `[SUY RA]`, cần đối chiếu thứ tự ghi TDMS. | `scripts/audit_consistent_sham_null.py` | `[ĐO]` |
| **Hình thái Donut vs Blur** | Khuyết tật $D \ge 8\text{ mm}$ có dạng vòng với đỉnh ở ranh giới CAD; khuyết tật $D \le 4\text{ mm}$ bị nhòe rộng. FWHM đo được dao động $9.8 - 12.8\text{ mm}$ nhưng phương pháp đo cũ chưa trung tính theo kích thước. | `cscan_radial_profiles_5x5.png` | `[ĐO-chưa kiểm]` |
| **Thiên lệch cắt mép (Edge bias)** | Cắt xén vành tham chiếu làm sham mép của ước lượng PtP bị trôi âm $-0.8\sigma$ đến $-3.0\sigma$ so với nội địa. Pooled Std của $N=52$ điểm sham mép thô là $\sigma_{\text{pooled}} = 0.00693\text{ V}$ ($1.72\times$ nội địa $0.00402\text{ V}$, hoặc $1.80\times$ sham $N=208$). | `scripts/audit_section4_resolution.py` | `[ĐO]` |
| **Liên đới Hàng = Y = Độ sâu** | Trên Corrosion, $Y$ tăng thì độ sâu tăng ($0.1 \rightarrow 1.0\text{ mm}$), slope khuyết tật nội địa ($N=9$, 3 giá trị $Y$) theo $y$ là $-8.0 \times 10^{-5}\text{ V/mm}$ ($R^2 = 0.855$). Trên Rivet_v1, $Y$ tăng thì độ sâu giảm ($1.0 \rightarrow 0.2\text{ mm}$), nhưng tín hiệu vẫn dốc âm theo $y$ (slope = $-1.2 \times 10^{-4}\text{ V/mm}$, $R^2 = 0.723$). | `scripts/audit_section4_resolution.py` | `[ĐO]` |
| **Mâu thuẫn giá trị Lift-off** | Ba nguồn mâu thuẫn: Tài liệu đề tài ($0.5 / 1.0 / 1.5\text{ mm}$), Handover summary ($0.5 / 1.5 / 3.0\text{ mm}$), Tên file LabVIEW (`z1mm`, `z2mm`, `z3mm`). PtP trung bình không đơn điệu theo lift-off ($z_1 \approx 5.45\text{ V}$, $z_2 \approx 5.51\text{ V}$, $z_3 \approx 5.38\text{ V}$). | Báo cáo kiểm kê dữ liệu | `[GIẢ ĐỊNH - CHỜ XÁC MINH]` |

---

## 4. Đối Soát Ba Giá Trị Cosine Waveform

Để chấm dứt sự mâu thuẫn về các con số góc giữa Square và Chirp trên checkpoint EXP-28:

1. **Giá trị 1: $\cos(w_S, w_C) = -0.2481$**:
   - Xuất hiện trong báo cáo khám nghiệm EXP-28 ban đầu. Cần đối soát lại script Thí nghiệm B để xác nhận đây là góc giữa hai vector trọng số Linear Classifier Probe $w_S, w_C$ hay trên latent thô. Cơ chế xoay tù do ma trận nghịch đảo hiệp phương sai $w = \Sigma^{-1} \Delta z$ là `[GIẢ ĐỊNH]`.
2. **Giá trị 2: $\cos(\Delta z_S, \Delta z_C) = +0.1036$**:
   - Tích vô hướng vector chuyển vị thô $\Delta z = \bar{z}_{\text{defect}} - \mu_{\text{sound}}$ trên 25 tâm CAD khi chưa chuẩn hóa phương sai từng chiều.
3. **Giá trị 3: $\cos(\Delta z_S, \Delta z_C) = -0.0820$ ($p = 0.9180$)**:
   - Tích vô hướng vector chuyển vị sau khi zero-centering theo từng waveform, đo kèm Paired-Swap Permutation Null (khoảng tin cậy Null 95%: $[-0.2904,\, +0.8033]$, $B=1000$).
   - Khoảng CI của Null rất rộng ($[-0.29; +0.80]$), phản ánh kích thước mẫu $N=25$ trong không gian 128 chiều có sức mạnh thống kê rất thấp.
   - **Trạng thái nhận thức: `[Not Detected]` (Không phát hiện sự liên kết có ý nghĩa thống kê; chưa đủ sức mạnh để kết luận đảo cực tính hay trực giao cố định)**.

---

## 5. Danh Mục Nhiệm Vụ Đánh Giá & Giải Quyết Căng Thẳng với `GEMINI.md`

### 5.1. Phân loại Nhiệm vụ
1. **Nhiệm vụ hợp lệ chính**: **In-Condition Object-Level Anomaly Detection** (đánh giá theo phân vị Sham Null).
2. **Nhiệm vụ bị đóng băng**: **Quantitative Depth Regression ($R^2$, MAE)** (bị gắn cứng vào tọa độ $y$ trên từng tấm đơn lẻ; gán nhãn `[CONFOUNDED]`).
3. **Nhiệm vụ chờ kiểm toán vị trí**: **Few-Shot Domain Adaptation** (phải vượt qua Decision Tree / $k\text{NN}$ Position Gate trên tập OOD).
4. **Câu hỏi nghiên cứu mở**: **Zero-Shot Cross-Waveform Invariance** (cần đối chứng dương trước khi đánh giá).

### 5.2. Giải quyết Mâu thuẫn Quy tắc với `GEMINI.md`
- **Về đánh giá Pixel-level (IoU/Dice)**: `GEMINI.md` yêu cầu báo IoU/Dice, nhưng đối với kiểm định NDT dạng lưới zero-inflated (khuyết tật $< 1.2\%$) có hiệu ứng tán xạ donut và nhòe vật lý, pixel-level trừng phạt bất công các vùng chuyển tiếp. Hợp đồng v1.0-RC1 chuyển trọng tâm chính sang **Object-Level FROC / AUC**. Bảng ma trận 3 Waveform $\times$ 3 Sensor $\times$ 3 Lift-off vẫn được bảo lưu đầy đủ theo định dạng Object-Level.
- **Về lệnh cấm đầu dò vi sai (Differential Probing)**: Baseline hồi quy tuyến tính 24-probe (Lin-U, Lin-H, Lin-G) về bản chất là một bộ lọc không gian được học. Bản đặc tả này ghi nhận minh bạch: **Các baseline tuyến tính được phép sử dụng DUY NHẤT LÀM BASELINE CHẨN ĐOÁN**, tuyệt đối không đưa vào kiến trúc hay hàm mất mát huấn luyện của mô hình PECT-JEPA.

---

## 6. Hợp Đồng Giao Thức Đánh Giá Chuẩn Tắc (Điều 1 & Điều 2)

*(Nguyên văn từ chuyên gia phản biện — Khóa cứng không sửa đổi)*

### Điều 1. Bản đồ dị thường và điểm đối tượng

**1.1 Miền hợp lệ.** Lưới pixel $(u,v)$ với $u=x$, $v=y$, $0\le u,v\le 269$, $1\text{ px}=1\text{ mm}$ `[GIẢ ĐỊNH]`. Với pixel $c=(u,v)$, patch $P_c$ là mảng sao đồng tâm 25 probe (bán kính $\{1,3,7\}$ px) quanh $c$. Miền hợp lệ $\mathcal V=\{c:\,7\le u,v\le 262\}$, tức toàn bộ footprint nằm trong bản quét.

**1.2 Bản đồ dị thường.** Gọi $p_0$ là probe trung tâm và ctx là 24 probe còn lại. Với $M\in\{\text{JEPA},\text{Lin-U},\text{Lin-H},\text{Lin-G}\}$:
$$A_{\text{JEPA}}(c)=\big\|g_\phi\big(z_{\text{ctx}}(c)\big)-\operatorname{sg}\big[z_{\text{tgt}}(p_0)\big]\big\|_2^2,\qquad A_{\text{Lin}}(c)=\big\|x_{p_0}(c)-(W^\top x_{\text{ctx}}(c)+b)\big\|_2^2$$
Biến thể sơ cấp: che probe trung tâm. Biến thể phụ: mặt nạ cụm 40–60% kiểu huấn luyện, trung bình 8 cụm cố định seed (vì che trung tâm nằm ngoài phân phối huấn luyện). $A$ không được chuẩn hóa theo vị trí, scan hay nhãn.

**1.3 Bán kính đánh giá.** $R(D)=\max(6,\,D/2+2)$ mm, tức $R=7$ cho $D=10$ và $R=6$ cho $D\in\{3,4,6,8\}$ (hai lớp bán kính). `[GIẢ ĐỊNH]` chọn theo nửa-FWHM ≈ 5–6 mm `[ĐO-chưa kiểm]`. Bắt buộc báo độ nhạy với $R\in\{6,8,10\}$.

**1.4 Điểm đối tượng.**
$$S_M(c_i;R)=\max_{c\in\mathcal V,\ \|c-c_i\|\le R}A_M(c)$$
Sham dùng cùng công thức và cùng $R$. Quy tắc "không lấy pixel đơn lẻ" của v0.9 bị bãi bỏ vì mâu thuẫn với toán tử cực đại. Tính hợp lệ đến từ việc null dùng cùng toán tử, không đến từ lọc.

**1.5 Dùng CAD.** Tọa độ CAD chỉ dùng để chọn vị trí chấm điểm và sham (đánh giá), không đi vào huấn luyện, tiền xử lý hay Lin-U (Lin-U huấn luyện trên mọi pixel, không loại trừ).

---

### Điều 2. Sham null

**2.1 Nhóm khuyết tật.** 9 nội địa (hàng, cột $\in \{2,3,4\}$); 12 rìa không góc (3 mỗi cạnh); 4 góc (ID 1, 5, 21, 25). Tổng cộng: $9 + 12 + 4 = 25$ khuyết tật.

**2.2 Điều kiện hợp lệ.** Sham $s\in\mathcal V$ và $d_{\min}(s)\ge d_0=30$ mm tới mọi tâm CAD. Lý do: đĩa $R\le 7$ cộng nửa patch 7 mm là 14 mm, cộng bán kính ảnh hưởng khuyết tật ≈ 10 mm `[ĐO-chưa kiểm]`, tức cần $\ge 24$ mm, chọn 30 mm.

**2.3 Tập sham độc lập tối đa.** Trên lưới bù $\mathcal L=\{(15+30a,\,15+30b)\}$, tâm CAD là các điểm có $a,b$ cùng chẵn.
- Nội địa: $a,b\in\{1,\dots,7\}$, không cùng chẵn, suy ra $49-9=40$ điểm.
- Rìa: $a\in\{0,8\}$ với $b$ lẻ, hoặc $b\in\{0,8\}$ với $a$ lẻ, suy ra 4 điểm mỗi cạnh, 16 điểm.
- Khoảng cách đôi một $\ge 30\text{ mm} > 2(R+7)=28\text{ mm}$ nên đầu vào các sham không chung pixel. `[SUY RA từ hình học lưới 60 mm]`
- Thêm điểm ngoài lưới chỉ làm mịn phân phối, không tăng $N_{\text{eff}}$. Báo cả $N$ lẫn $N_{\text{eff}}$. Không có sham góc.

**2.4 Null phụ thuộc $D$ (cùng toán tử, cùng $R$).** Với mỗi (scan, mô hình, $R\in\{6,7\}$) tính $S^{(R)}_j$ cho mọi sham. Tập tham chiếu $\mathcal H_i$ của đối tượng $i$:
- Nội địa: $\{S^{(R_i)}_j\}$ trên 40 sham nội địa.
- Rìa cạnh $k$: khử trung bình cạnh, $\tilde S_j=S_j-m_k$ với $m_k$ là trung bình 4 sham của cạnh; đối tượng cũng $\tilde S_i=S_i-m_{k(i)}$; tham chiếu là 16 giá trị $\tilde S_j$ gộp (bậc tự do 12).
- Góc: chỉ báo mô tả, không vào kiểm định.

**2.5 Phân vị và p.**
$$q_i=\frac{1}{|\mathcal H_i|}\sum_{j}\Big[\mathbf 1(S_j<S_i)+\tfrac12\mathbf 1(S_j=S_i)\Big],\qquad p_i=\frac{1+\#\{j:S_j\ge S_i\}}{1+|\mathcal H_i|}$$
$p_{\min}=1/41\approx0.024$ (nội địa) và $1/17\approx0.059$ (rìa). **Không có ngưỡng $p<0.01$ cho từng khuyết tật.** $p_i$ chỉ để mô tả; kiểm định xác nhận dùng thống kê gộp ở Điều 7.

**2.6 Cổng hợp lệ của null.**
- **G-N1 (xu hướng vị trí):** hồi quy bậc 2 của $S_j$ theo $(x,y)$ trên sham nội địa, $R^2_{\text{pos}}\le 0.2$ `[GIẢ ĐỊNH]`. Báo thêm hệ số theo khoảng cách tới mép. Nếu vi phạm, hiệu chỉnh $S-\hat f(x,y)$ cho cả đối tượng lẫn sham; nếu vẫn $> 0.2$ thì **CONFOUNDED**.
- **G-N2 (hiệu chuẩn theo vùng):** lấy tham chiếu từ nửa trên ($b\le 4$), đo tỉ lệ sham nửa dưới vượt phân vị 90%, phải $\approx 0.10$ theo khoảng tin cậy nhị thức. Làm ngược lại và lặp theo trục $x$.

---

**Điều 4.4.** Các baseline tuyến tính là chẩn đoán, không phải thành phần mô hình. Việc có xem lại lệnh cấm differential probing trong GEMINI.md hay không là quyết định riêng của nhóm nghiên cứu, ghi lại cùng các kết quả này. Chưa nên đưa vào mô hình.

---

## 7. Mục 7 (v1.0-RC2). Baseline, cổng và quyết định

**7.0 Nguyên tắc.** (a) Mọi siêu tham số được cố định không nhãn trước khi chấm. (b) Nhãn trạng thái theo thứ tự INVALID, CONFOUNDED, rồi ba trạng thái hiệu năng. (c) Mọi kết luận nêu phạm vi: checkpoint, seed, điều kiện, Transductive/Inductive, nhãn [Pilot].

**7.1 Baseline và chọn α**
- Lin-U: huấn luyện trên mọi pixel hợp lệ của scan được chấm, không dùng nhãn. Lin-H: pixel cách mọi tâm CAD ≥ 18 mm (dùng CAD, chỉ chẩn đoán). Lin-G: một mô hình chung trên toàn bộ file tiền huấn luyện của EXP-28, chuẩn hóa từng file, scan được chấm nằm trong tập huấn luyện như với JEPA.
- Đặc trưng chuẩn hóa z-score trên tập huấn luyện của từng biến thể. Lưới α: 13 giá trị cách đều theo log₁₀ từ 10⁻³ đến 10³.
- α* = giá trị cực tiểu hóa MSE dự đoán probe giữa, CV 5 khối không gian theo dải với guard 14 mm, trên tập huấn luyện của biến thể đó. Cấm dùng AUC, nhãn hoặc điểm sham để chọn α. Ghi log α* và đường CV. Tập "1-SE" gồm các α có MSE ≤ min + 1 SE.

**7.2 Các cổng.** Không qua cổng nào thì không có kết luận về hiệu năng.

- **G0 (đối chứng dương).** Lin-U với α* đạt AUC ≥ 0,8 trên tập O⁶ (Z_PtP ≥ 6), |O⁶| ≥ 8. Không đạt: INVALID.
- **G-Null.** Hoán vị nhãn đối tượng/sham (B = 1000) cho AUC trung bình ≈ 0,5 với mọi mô hình, khoảng 95% chứa 0,5. Không đạt: INVALID (lỗi pipeline).
- **G2 (sọc dòng).** (a) Kiểm cổng: cộng offset dòng tổng hợp ~N(0; 0,02 V) vào raw, chỉ để thử cổng, không dùng huấn luyện. R²_stripe của A_M phải tăng so với không cộng, nếu không thì cổng không có năng lực phát hiện: INVALID. (b) Dữ liệu thực: R²_stripe ≤ 0,5, nếu không thì CONFOUNDED(M).
- **G-N1 (vị trí, theo từng mô hình M).** Dùng sham nội địa, với S_j là điểm sham:
  1. Phân loại: hồi quy bậc 2 theo (x, y), R²_pos > 0,2 hoặc Moran's I thô có p < 0,05 thì bắt buộc hiệu chỉnh S̃ = S − f̂(x, y). Mọi thống kê sau đó báo cả bản thô lẫn S̃.
  2. Kiểm sau hiệu chỉnh ngoài mẫu: phần dư leave-one-out r_j = e_j/(1 − h_jj) (h_jj là đường chéo ma trận hat). Moran's I trên r_j với trọng số lân cận hàng/cột trên lưới 30 mm, kiểm hoán vị một phía B = 9999, yêu cầu p ≥ 0,05.
  3. Kiểm chuyển vùng: khớp mặt phẳng trên nửa trên rồi dự đoán nửa dưới, và ngược lại; làm tương tự theo trục x. Trung bình phần dư của nửa kiểm tra, chia cho độ lệch chuẩn phần dư của nửa huấn luyện, phải có |·| ≤ 1 ở cả bốn hướng.
  4. Không đạt (2) hoặc (3): CONFOUNDED(M). Rìa chỉ khử trung bình từng cạnh (4 điểm mỗi cạnh nên không có phép kiểm vị trí), và kết quả rìa được báo kèm phân tích độ nhạy "chỉ nội địa".
- **A/A (hiệu chuẩn pipeline).** Huấn luyện Lin-U hai lần trên hai nửa pixel ngẫu nhiên (cùng α*), lặp 100 lần chia. Mỗi lần chạy toàn bộ quy trình chấm và tính Δ_AA. Yêu cầu |trung bình Δ_AA| ≤ δ/5 và khoảng tin cậy chứa 0 trong ≥ 90% lần. Không đạt: INVALID (pipeline).

**7.3 Quyết định ba trạng thái (chỉ khi mọi cổng qua)**
- Đơn vị xác nhận là điều kiện (specimen, waveform, lift-off). Các cảm biến là phép đo lặp trên cùng tập khuyết tật. Chỉ dùng cảm biến qua mọi cổng. Với khuyết tật i: d_i^b = mean_sensors(q_i^JEPA − q_i^{Lin-b}), b ∈ {U, H, G}. Bootstrap hai tầng (resample khuyết tật, resample sham độc lập), B = 2000.
- δ = 0,05 [GIẢ ĐỊNH, cố định, không rút từ dữ liệu]. Kết quả từng cảm biến chỉ để mô tả.
- ESTABLISHED: cận dưới CI 95% của d^b lớn hơn δ với **mọi** b.
- REJECTED: tồn tại b có cận trên CI hiệu chỉnh Bonferroni(3) (98,3%) nhỏ hơn δ. Ghi thêm "JEPA kém baseline b" nếu cận trên < 0.
- NOT DETECTED: các trường hợp còn lại.
- Độ nhạy α: trạng thái phải không đổi trên cả tập 1-SE. Nếu đổi thì ghi "α-sensitive" và hạ xuống NOT DETECTED.
- Nhãn khi không qua cổng: INVALID, CONFOUNDED(M), hoặc INVALID + CONFOUNDED(M). Không có kết luận hiệu năng.

**7.4 Giới hạn của chính bản này**
- N_eff chỉ 40 (nội địa) và 16 (rìa), nên Moran's I và kiểm chuyển vùng có sức mạnh thấp. Qua cổng là bằng chứng yếu.
- δ = 0,05, ngưỡng 0,2/0,5/0,8, p = 0,05, δ/5 và 90% đều là giữ chỗ.
- Toàn bộ scan nằm trên một coupon Corrosion. Kết quả không mở rộng sang specimen khác.
- Nếu hầu hết hoặc mọi scan đều CONFOUNDED sau khi sửa, kết luận trung thực là coupon và thiết kế sham này không phân xử được giữa hai bộ dự đoán. Đó là kết luận hợp lệ, và hướng sửa là dữ liệu (layout ngẫu nhiên, nhiều negative khớp vị trí), không phải thêm thuật toán.

---

## 8. Điều Kiện Mở Lại Các Hướng Nghiên Cứu (Re-Opening Conditions)

1. **Ablation Đơn Biến Kiến Trúc (Single-Variable Ablation)**:
   - Được phép tiến hành sau khi hoàn thành Pilot Phase 2, gắn nhãn `[Pilot]`.
   - Các can thiệp được ưu tiên: Bỏ LayerNorm (khảo sát bảo toàn SNR biên độ), thay đổi cấu trúc tokenizer theo trục thời gian.
2. **Hồi Quy Độ Sâu Xuyên Tấm Mẫu (Cross-Specimen Depth Test)**:
   - *Phép thử khả dĩ*: Huấn luyện mô hình hồi quy độ sâu trên Corrosion ($Y \propto \text{Depth}$), kiểm tra zero-shot trên Rivet_v1 ($Y \propto -\text{Depth}$), so sánh đối chứng với một mô hình chỉ học tọa độ $Y$.
   - *Điều kiện mở lại*: Nếu mô hình đạt $R^2 > 0$ trên Rivet_v1 trong khi mô hình chỉ-Y đạt $R^2 < 0$, bằng chứng về việc học được độ sâu vật lý sẽ được công nhận và nhiệm vụ hồi quy độ sâu được mở lại.

---

*Bản đặc tả v1.0-RC1 chính thức khóa hợp đồng đánh giá để triển khai Pilot Phase 2.*
