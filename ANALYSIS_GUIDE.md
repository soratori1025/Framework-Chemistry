# Hướng dẫn Phân tích Đặc trưng và Đọc Báo cáo Dữ liệu (Analysis Guide)

Tài liệu này cung cấp hướng dẫn toàn diện về cách sử dụng công cụ **Dataset & Feature Analysis Advisor** trong `chemistry-framework`, cách chạy phân tích từ dòng lệnh (CLI) hoặc Python API, và cách diễn giải chi tiết từng chỉ số trong **Báo cáo Terminal** cũng như **Báo cáo Markdown/HTML**.

---

## Mục lục
1. [Tổng quan về Hệ thống Phân tích](#1-tổng-quan-về-hệ-thống-phân-tích)
2. [Cách chạy Phân tích Dữ liệu](#2-cách-chạy-phân-tích-dữ-liệu)
   - [Chạy nhanh từ dòng lệnh (CLI)](#chạy-nhanh-từ-dòng-lệnh-cli)
   - [Chạy từ mã nguồn Python](#chạy-từ-mã-nguồn-python)
3. [Diễn giải Báo cáo Terminal (Console Report)](#3-diễn-giải-báo-cáo-terminal-console-report)
   - [Điểm sức khỏe dữ liệu (Health Score)](#điểm-sức-khỏe-dữ-liệu-health-score)
   - [Đặc tính cấu trúc hóa học (Chemical Profile)](#đặc-tính-cấu-trúc-hóa-học-chemical-profile)
   - [Đặc trưng dạng bảng người dùng đưa vào (Tabular Features)](#đặc-trưng-dạng-bảng-người-dùng-đưa-vào-tabular-features)
   - [Phân phối nhãn mục tiêu (Target Analysis)](#phân-phối-nhãn-mục-tiêu-target-analysis)
   - [Bảng cảnh báo rủi ro (Risk Factors)](#bảng-cảnh-báo-rủi-ro-risk-factors)
   - [Khuyến nghị huấn luyện mô hình (Prescriptive Recommendations)](#khuyến-nghị-huấn-luyện-mô-hình-prescriptive-recommendations)
4. [Diễn giải Báo cáo Chi tiết Markdown & JSON](#4-diễn-giải-báo-cáo-chi-tiết-markdown--json)
5. [Phân tích Bằng chứng Đặc trưng Huấn luyện (Feature Evidence Scorecard)](#5-phân-tích-bằng-chứng-đặc-trưng-huấn-luyện-feature-evidence-scorecard)
6. [Tự động sinh File Cấu hình YAML và Bắt đầu Train](#6-tự-động-sinh-file-cấu-hình-yaml-và-bắt-đầu-train)

---

## 1. Tổng quan về Hệ thống Phân tích

Trong hóa tin học và học máy phân tử, hiệu năng và độ tin cậy của mô hình phụ thuộc mật thiết vào chất lượng dữ liệu đầu vào. Các vấn đề phổ biến như:
- Chuỗi SMILES không hợp lệ hoặc chứa các phân tử trùng lặp.
- Độ đa dạng giàn khung (Scaffold Diversity) thấp dẫn đến hiện tượng mô hình "học vẹt" (memorization) thay vì khái quát hóa sinh học.
- Mất cân bằng lớp nghiêm trọng (Severe Class Imbalance) trong sàng lọc ảo (Virtual Screening).
- Phân phối nhãn hồi quy bị lệch nặng (Skewness) hoặc chứa giá trị ngoại lai cực trị (Outliers).
- Các cột đặc trưng người dùng đưa vào bị khuyết thiếu (Missingness), phương sai bằng không (Constant Features), hoặc đa cộng tuyến cao ($|r| > 0.90$).

Framework cung cấp module **`DatasetAdvisor`** nhằm tự động kiểm tra toàn bộ các khía cạnh trên, chấm điểm sức khỏe dữ liệu (0–100 điểm), đưa ra cảnh báo rủi ro và kê đơn cấu hình huấn luyện tối ưu nhất.

---

## 2. Cách chạy Phân tích Dữ liệu

### Chạy nhanh từ dòng lệnh (CLI)

Lệnh `chemistry-framework inspect` nhận đầu vào là file CSV dữ liệu hoặc file cấu hình YAML:

```bash
# 1. Phân tích trực tiếp từ file CSV
chemistry-framework inspect data/my_compounds.csv

# 2. Phân tích với chỉ định rõ cột SMILES và cột nhãn
chemistry-framework inspect data/my_compounds.csv \
  --smiles-column smiles \
  --target pIC50

# 3. Phân tích, tự động xuất file cấu hình YAML tối ưu và lưu báo cáo chi tiết
chemistry-framework inspect data/my_compounds.csv \
  --target pIC50 \
  --generate-config configs/auto_pipeline.yaml \
  --output-dir reports/dataset_inspection

# 4. Phân tích tập dữ liệu đã khai báo trong file YAML
chemistry-framework inspect benchmarks/configs/caco2_dmpnn.yaml
```

**Các tùy chọn dòng lệnh chính:**
- `input`: Đường dẫn file `.csv` hoặc `.yaml`.
- `--smiles-column`: Tên cột chứa SMILES (mặc định tự động nhận diện nếu tên là `smiles`, `canonical_smiles`, `Drug`, `molecule`,...).
- `--target`: Tên cột nhãn mục tiêu cần dự đoán (tự nhận diện cột số nếu để trống).
- `--generate-config <path>`: Đường dẫn lưu file cấu hình YAML sẵn sàng chạy train.
- `--output-dir <path>`: Thư mục lưu file báo cáo định dạng Markdown (`.md`) và JSON (`.json`).

---

### Chạy từ mã nguồn Python

Bạn có thể tích hợp việc phân tích vào Jupyter Notebook hoặc pipeline Python:

```python
from chemistry_framework import DatasetAdvisor

advisor = DatasetAdvisor()

# Cách 1: Phân tích file CSV
report = advisor.profile_csv(
    csv_path="data/my_compounds.csv",
    smiles_column="smiles",
    target_columns=["pIC50"],
)

# Cách 2: Phân tích trực tiếp từ danh sách SMILES và Dict dữ liệu
report = advisor.profile_dataset(
    smiles_list=["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"],
    targets={"activity": [0.0, 1.0, 1.0]},
    tabular_features={"assay_conc": [10.0, 50.0, 50.0]},
    dataset_name="virtual_screen_batch1",
)

# In báo cáo ra Terminal
print(advisor.format_console_report(report))

# Lưu báo cáo và file cấu hình YAML
report.save_markdown("reports/analysis_report.md")
report.save_json("reports/health_report.json")
report.save_config("configs/recommended_train.yaml")
```

---

## 3. Diễn giải Báo cáo Terminal (Console Report)

Dưới đây là một mẫu báo cáo Terminal đầy đủ:

```text
==============================================================================
 🧪 CHEMISTRY FRAMEWORK: DATASET HEALTH & ADVISORY REPORT
==============================================================================
 Dataset: custom_assay_screen             Health Score: 85/100 [🟢 EXCELLENT]
------------------------------------------------------------------------------
 • Total Compounds:      1250       • Valid RDKit SMILES: 1248 (99.8%)
 • Unique Scaffolds:     612        • Scaffold Diversity: 0.49
 • Duplicate Molecules:   12         • Invalid SMILES:     2
------------------------------------------------------------------------------
 📊 KEY CHEMICAL PROPERTY DISTRIBUTIONS (Mean ± Std [Min, Max]):
   - Molecular Weight (MW)   :  382.4 ± 112.5 [110.2,  890.1]
   - LogP                    :    2.8 ± 1.6   [ -3.1,    7.4]
   - Heavy Atoms             :   26.5 ±  8.2  [  8.0,   65.0]
   - H-Bond Donors           :    1.8 ±  1.4  [  0.0,   10.0]
   - H-Bond Acceptors        :    4.9 ±  2.3  [  1.0,   18.0]
   - TPSA (Å²)               :   78.6 ± 42.1  [ 12.0,  312.4]
   - Rotatable Bonds         :    4.8 ±  2.9  [  0.0,   21.0]
------------------------------------------------------------------------------
 🔬 INPUT TABULAR FEATURES (4 features detected):
   - Constant features:     1 (['batch_id_const'])
   - High missing (>20%):   1 (['ic50_replicate'])
   - Collinear pairs (>0.9): 1 pairs (feat_mw vs feat_heavy_atoms)
   - Retained for modeling: 2 / 4
   - Preprocessing advice:  Imputation='median', Scaling='standard'
------------------------------------------------------------------------------
 🎯 TARGET ANALYSIS:
   - Target 'bioactivity' [Binary Classification]:
     Class distribution: {'0': 1050, '1': 198} | Imbalance Ratio: 5.30:1
------------------------------------------------------------------------------
 ⚠️  RISK FACTORS & WARNINGS (3 identified):
   [HIGH] Severe Class Imbalance (5.3:1): Class ratio is 1050:198. Standard accuracy will be misleading.
          ↳ Advice: Use BCE with pos_weight=5.30. Prioritize AUPRC and AUROC over Accuracy.
   [MEDIUM] Constant Features Detected (1): Columns ['batch_id_const'] have zero variance across all compounds.
          ↳ Advice: Remove constant features before model ingestion to reduce noise.
   [LOW] Multicollinear Feature Pairs (1): High pairwise correlation (|r| > 0.90) detected: feat_mw vs feat_heavy_atoms (|r|=0.96).
          ↳ Advice: Prune redundant features or apply PCA dimensionality reduction to prevent coefficient inflation.
------------------------------------------------------------------------------
 💡 PRESCRIPTIVE TRAINING RECOMMENDATIONS:
   1. Splitting:      SCAFFOLD (Prevents test memorization)
   2. Architecture:   DMPNN with HYBRID representation
   3. Head & Loss:    Head='classification', Loss='bce'
   4. Scheduler:      Cosine Annealing with 3 warmup epochs
   5. Ensembling:     Ensemble size = 1
   6. Metrics:        AUROC, AUPRC, ACCURACY
==============================================================================
```

### Điểm sức khỏe dữ liệu (Health Score)

Điểm bắt đầu từ **100 điểm** và bị trừ dần dựa trên mức độ rủi ro phát hiện được:
- **85 - 100 [🟢 EXCELLENT]**: Dữ liệu có chất lượng cao, cân bằng tốt, cấu trúc hợp lệ, phù hợp cho deep learning.
- **70 - 84 [🟡 GOOD]**: Dữ liệu tốt, có một số vấn đề nhẹ (như mất cân bằng vừa phải hoặc mẫu vừa phải), có thể huấn luyện hiệu quả sau khi áp dụng cấu hình khuyến nghị.
- **50 - 69 [🟠 FAIR]**: Có các rủi ro đáng kể (mất cân bằng lớp nặng, ngoại lai cao, nhiều đặc trưng khuyết thiếu), cần xử lý theo hướng dẫn.
- **< 50 [🔴 NEEDS ATTENTION]**: Tỷ lệ SMILES lỗi cao (>5%), kích thước mẫu quá bé ($N < 300$) hoặc dữ liệu có quá nhiều thuộc tính hằng số.

---

### Đặc tính cấu trúc hóa học (Chemical Profile)

- **Total Compounds**: Tổng số dòng phân tử trong file.
- **Valid RDKit SMILES**: Số lượng và tỷ lệ phân tử parse thành công qua RDKit. Nếu tỷ lệ lỗi > 1%, hệ thống sẽ cảnh báo để người dùng sanitize dữ liệu trước khi train.
- **Unique Scaffolds & Scaffold Diversity**: Tỷ lệ số khung Bemis-Murcko độc nhất trên tổng số phân tử hợp lệ.
  - *Ý nghĩa*: Nếu tỷ lệ này $> 0.10$ (10%), chứng tỏ dữ liệu có tính đa dạng cấu trúc cao. Bắt buộc phải dùng **Scaffold Split** (thay vì Random Split) để bảo đảm mô hình học được đặc trưng hóa học thực sự thay vì nhớ khung phân tử đã xuất hiện ở tập train.
- **Duplicate Molecules**: Số lượng phân tử bị trùng SMILES. Phân tử trùng cần được gộp và lấy trung bình nhãn để tránh rò rỉ dữ liệu giữa train và test.
- **Phân phối thuộc tính hóa lý (MW, LogP, Heavy Atoms, HBD, HBA, TPSA, RotB)**:
  - Giúp đánh giá không gian hóa học (Chemical Space) của tập dữ liệu có tuân thủ quy tắc Lipinski (Rule of 5) dành cho thuốc đường uống hay không.
  - Phát hiện các phân tử quá lớn (MW > 800) hoặc quá thân dầu (LogP > 5) có thể gây bất ổn định khi tính toán gradient.

---

### Đặc trưng dạng bảng người dùng đưa vào (Tabular Features)

Nếu dataset có chứa các cột thuộc tính bổ sung (metadata nồng độ, kết quả assay trước đó, hoặc descriptor tính toán sẵn):
- **Constant features**: Cột có phương sai bằng 0 hoặc chỉ có 1 giá trị duy nhất $\rightarrow$ Hệ thống đề xuất loại bỏ để tránh gây nhiễu cho mạng nơ-ron.
- **High missing (>20%)**: Cột thiếu hơn 20% dữ liệu $\rightarrow$ Đề xuất bỏ nếu $> 50\%$, hoặc điền giá trị trung vị (`Imputation='median'`).
- **Collinear pairs (>0.90)**: Các cặp đặc trưng có hệ số tương quan Pearson $|r| > 0.90$ $\rightarrow$ Hệ thống tự động chọn giữ lại 1 đặc trưng đại diện và loại bỏ đặc trưng dư thừa nhằm tránh hiện tượng bùng nổ trọng số (multicollinearity).
- **Retained for modeling**: Số lượng đặc trưng đạt tiêu chuẩn được giữ lại để ghép nối vào nhánh MLP Descriptors.
- **Preprocessing advice**:
  - `Imputation`: Phương pháp bù khuyết (`median` nếu có giá trị NaN).
  - `Scaling`: Phương pháp chuẩn hóa (`standard` cho z-score normalization).

---

### Phân phối nhãn mục tiêu (Target Analysis)

#### 1. Bài toán Phân loại (Binary Classification)
- **Class distribution**: Số lượng mẫu nhãn 0 và nhãn 1.
- **Imbalance Ratio**: Tỷ lệ giữa lớp đa số và lớp thiểu số.
  - *Xử lý*: Nếu tỷ lệ $> 5:1$, hệ thống đánh dấu rủi ro `HIGH` và tự động tính `pos_weight = N_neg / N_pos`. Khi huấn luyện với `BCEWithLogitsLoss(pos_weight=...)`, mô hình sẽ phạt nặng hơn khi đoán sai lớp thiểu số.
  - *Đánh giá*: Khuyến nghị ưu tiên metric **AUPRC** (Area Under Precision-Recall Curve) và **AUROC** thay vì dùng Accuracy thông thường.

#### 2. Bài toán Hồi quy (Scalar Regression)
- **Mean ± Std & Range**: Phân phối giá trị nhãn.
- **Skewness (Độ lệch)**:
  - Nếu $|\text{Skewness}| > 1.5$: Phân phối bị lệch nặng (ví dụ phân phối đuôi dài của hoạt tính $\text{IC}_{50}$). MSE Loss sẽ bị chi phối bởi các giá trị cực lớn $\rightarrow$ Hệ thống khuyến nghị dùng **Huber Loss** hoặc log-transformation.
- **Outliers (Giá trị ngoại lai)**:
  - Tính theo công thức IQR: $x < Q_1 - 1.5\text{IQR}$ hoặc $x > Q_3 + 1.5\text{IQR}$.
  - Nếu tỷ lệ ngoại lai $> 3\%$, hệ thống kích hoạt chế độ chống ngoại lai với **Huber Loss**.

---

### Bảng cảnh báo rủi ro (Risk Factors)

Các mục rủi ro được phân loại thành 4 cấp độ:
- **`[HIGH]`**: Rủi ro nghiêm trọng làm sai lệch kết quả mô hình (ví dụ: SMILES lỗi $> 5\%$, mất cân bằng lớp $> 5:1$, kích thước mẫu $< 300$).
- **`[MEDIUM]`**: Cần chú ý xử lý tiền xử lý (ví dụ: tồn tại cột constant, phân phối hồi quy lệch cao, ngoại lai nhiều).
- **`[LOW]`**: Cảnh báo tối ưu hóa (ví dụ: mất cân bằng nhẹ $2:1$, tồn tại cặp đặc trưng tương quan $> 0.9$).
- **`[INFO]`**: Thông tin chỉ dẫn thêm.

Mỗi mục rủi ro đều đi kèm giải pháp hành động cụ thể (`↳ Advice`).

---

### Khuyến nghị huấn luyện mô hình (Prescriptive Recommendations)

Dựa trên phân tích tổng hợp, hệ thống đưa ra công thức huấn luyện tối ưu:
1. **Splitting**: `SCAFFOLD` nếu độ đa dạng scaffold cao, `RANDOM` nếu mẫu quá ít hoặc scaffold đơn điệu.
2. **Architecture**:
   - `DMPNN` (Directed Message Passing Neural Network): Tối ưu cho đồ thị phân tử với truyền tin theo liên kết hóa học.
   - `GIN` (Graph Isomorphism Network): Tối ưu khi cần phân biệt đẳng cấu phân tử chi tiết.
   - `HYBRID`: Tự động kích hoạt khi có bảng đặc trưng tabular hoặc khi dữ liệu ít cần bổ sung descriptor vật lý để tránh overfitting.
3. **Head & Loss**:
   - Classification: `classification` head + `bce` loss (kèm `pos_weight`).
   - Standard Regression: `standard` head + `huber` loss.
   - Large Regression ($N \ge 1000$): `evidential` head + `evidential` loss để dự đoán kèm **Độ bất định (Uncertainty Quantification)**.
4. **Scheduler**: `Cosine Annealing` với 3 epochs khởi động nhẹ (`warmup_epochs: 3`) để gradient ổn định ở giai đoạn đầu.
5. **Ensembling**: Nếu tập dữ liệu nhỏ ($N < 800$), tự động đề xuất **Deep Ensemble ($K=3$)** để ổn định phương sai và giảm rủi ro quá khớp.

---

## 4. Diễn giải Báo cáo Chi tiết Markdown & JSON

Khi truyền cờ `--output-dir reports/`, hai file sẽ được tạo ra:

### File 1: `<name>-advisory-report.md`
Là báo cáo Markdown hoàn chỉnh, có thể đọc trực tiếp trên GitHub, VSCode, hoặc nhúng vào báo cáo dự án:
- Bảng tóm tắt Executive Summary & Health Check.
- Bảng phân phối tính chất hóa lý phân tử 25%, Median, 75%.
- Bảng kiểm tra chất lượng từng cột đặc trưng Tabular (Status: `✅ Pass`, `⚠️ Constant`, `⚠️ High missing`).
- Bảng các cặp đặc trưng đa cộng tuyến $|r| > 0.90$.
- Block YAML hoàn chỉnh sẵn sàng copy để huấn luyện.

### File 2: `<name>-health-report.json`
Chứa toàn bộ dữ liệu thống kê có cấu trúc JSON dành cho việc tích hợp vào CI/CD pipeline hoặc ML dashboard:
```json
{
  "dataset_name": "custom_assay",
  "health_score": 85,
  "health_status": "EXCELLENT",
  "chemical_profile": {
    "total_records": 1250,
    "valid_count": 1248,
    "unique_scaffolds": 612,
    "scaffold_diversity": 0.49
  },
  "prescriptions": {
    "split": "scaffold",
    "encoder": "dmpnn",
    "model_type": "hybrid",
    "loss": "bce",
    "metrics": ["auroc", "auprc", "accuracy"],
    "scheduler": "cosine"
  }
}
```

---

## 5. Phân tích Bằng chứng Đặc trưng Huấn luyện (Feature Evidence Scorecard)

Bên cạnh phân tích dữ liệu trước huấn luyện (`inspect`), framework còn cung cấp lệnh phân tích chuyên sâu các đặc trưng hóa học được trích xuất trong quá trình `prepare`:

```bash
chemistry-framework analyze config.yaml
```

Lệnh này sẽ sinh ra 4 artifacts trong thư mục `analysis.output_dir`:
1. `*-feature-evidence.csv`: Bảng điểm đánh giá từng đặc trưng (Feature Evidence Scorecard) gồm:
   - **Support**: Số lượng phân tử trong tập train kích hoạt đặc trưng này. Nếu support $< 20$, đặc trưng bị xếp vào nhóm `unsupported` hoặc `ood_extrapolation`.
   - **Target Association**: Hệ số tương quan Spearman giữa đặc trưng và nhãn mục tiêu.
   - **Evidence Class**: Phân loại mức độ tin cậy (`supported`, `fixed_analytical`, `weak_uncertain`, `redundant`, `ood_unsupported`).
2. `*-feature-report.html`: Báo cáo HTML trực quan tương tác với bảng màu và nhóm chức hóa học.
3. `*-correlation-heatmap.png`: Ma trận tương quan trực quan dạng ảnh PNG độ phân giải cao giữa tất cả các đặc trưng hóa học.
4. `*-analysis.json`: Bản ghi chi tiết phân phối binning và khoảng giá trị trên tập train vs validation vs test.

---

## 6. Tự động sinh File Cấu hình YAML và Bắt đầu Train

Điểm mạnh của `chemistry-framework inspect` là bạn không cần phải tự viết tay file cấu hình YAML từ đầu. Hãy để Advisor sinh file tối ưu cho bạn:

```bash
# Bước 1: Khảo sát dữ liệu và sinh cấu hình tối ưu
chemistry-framework inspect data/my_assay.csv \
  --target IC50 \
  --generate-config configs/my_assay_pipeline.yaml

# Bước 2: Kiểm tra tính hợp lệ của file cấu hình
chemistry-framework validate configs/my_assay_pipeline.yaml

# Bước 3: Huấn luyện mô hình ngay lập tức theo đúng khuyến nghị
chemistry-framework train configs/my_assay_pipeline.yaml
```

Mô hình sẽ tự động áp dụng đúng cấu trúc (D-MPNN/GIN, Hybrid, Huber/BCE loss, Cosine scheduler) phù hợp nhất với dữ liệu của bạn!
