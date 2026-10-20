# Modular Chemistry Machine-Learning Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2023+-green.svg)](https://www.rdkit.org/)

**A modular, feature-aware machine-learning framework for chemistry and drug discovery.** 
Designed as an extensible platform for molecular property prediction, ADMET profiling, and thermochemistry — seamlessly combining **Graph Neural Networks (D-MPNN, GIN)**, **Handcrafted & Tabular Descriptors**, **Physics/Empirical Priors**, **Deep Ensembles**, **Evidential Uncertainty Quantification**, and an **Automated Data Quality Advisor**.

---

## 📄 Bài báo & tái lập kết quả (Publication & reproducibility)

Kho mã này là bản phát hành đi kèm bài báo **"Evidence before prediction: a configuration-driven framework for testing how molecular descriptors shape property models"** (gửi *Journal of Cheminformatics*):

- 📦 **Kết quả bài báo**: kết quả thô theo từng split nằm trong [`benchmarks/results/`](benchmarks/results/); split manifests (20 seed/tập, bài báo dùng seed 1–10) ở [`benchmarks/splits/`](benchmarks/splits/).
- 🔬 **Cách tái lập từng bảng/hình**: [`REPRODUCING.md`](REPRODUCING.md) — mỗi artifact đều có lệnh tương ứng, kèm cảnh báo về phiên bản RDKit cần pin.
- 📈 **Thống kê so sánh cặp**: `python benchmarks/paper_statistics.py` (hiệu chỉnh Nadeau–Bengio + Holm–Bonferroni) và `python benchmarks/generate_si_tables.py`.
- ✅ **Kiểm chứng**: `python benchmarks/validate_paper_statistics.py` tái tính 2.024 đại lượng từ điểm per-split (184 so sánh cặp) và kiểm tra tính nhất quán.
- 📚 **Trích dẫn / DOI**: [`CITATION.cff`](CITATION.cff); các bước tạo DOI Zenodo: [`RELEASE_NOTES.md`](RELEASE_NOTES.md).

---

## 📑 Tài liệu liên quan
- 📊 [**Hướng dẫn Phân tích Đặc trưng & Đọc Báo cáo (Analysis Guide)**](ANALYSIS_GUIDE.md): Chi tiết cách khảo sát dữ liệu, phân tích đặc trưng, tính điểm sức khỏe dữ liệu (Health Score), và đọc báo cáo.
- 🏗️ [**Kiến trúc Chuyên sâu của Hệ thống (Architecture & Deep Dive)**](ARCHITECTURE.md): Chi tiết kiến trúc từng module, thuật toán truyền tin đồ thị, hàm mất mát bất định, và kết quả benchmark ADMET.

---

## ✨ Điểm nổi bật của Framework

1. **Kiến trúc Lai Đa phương thức (Hybrid Multi-Modal Modeling):**
   - Hỗ trợ kết hợp song song biểu diễn đồ thị phân tử (**D-MPNN**, **GIN**) và nhánh mạng nơ-ron đa tầng (**MLP Descriptors**) qua các chiến lược hợp nhất (*Concat*, *Additive*, *Gated*).
2. **Kiến trúc GNN Mở rộng với Global Virtual Supernode & Attention Pooling:**
   - Tích hợp `virtual_node: true` cung cấp kênh truyền tin toàn cục $O(1)$ giữa các nguyên tử, giúp giảm thiểu hiện tượng nghẽn truyền tin (**information bottleneck / over-squashing**) trên các khung phân tử lớn, kết hợp cơ chế **Graph Attention Pooling** (`aggregation: attention`) học phân bố trọng số tương ứng với các phân vùng cấu trúc quan trọng.
3. **Loss Functions & Metrics Phục vụ Sàng lọc Phân tử:**
   - **Focal Loss** (`loss: focal`): Giảm thiểu tác động của mất cân bằng lớp nặng (imbalanced bioassay datasets) trong phân loại hoạt tính.
   - **Pairwise Ranking Loss** (`loss: pairwise_ranking`): Hỗ trợ tối ưu hóa trực tiếp thứ tự xếp hạng tương đối giữa các phân tử trong bài toán sàng lọc ảo (virtual screening), đồng thời tích hợp metric hệ số tương quan hạng **Spearman** (`metrics: [spearman]`).
4. **Phân tích Đặc trưng & Chọn Lọc Dựa trên Bằng chứng (Feature Evidence Selection):**
   - Hỗ trợ trích xuất Morgan Fingerprint ECFP4/ECFP6 vector hóa (1024/2048-bit), hơn 200 RDKit 2D physicochemical descriptors và khối `AutoPharmacophoreBlock` chọn lọc Top-$K$ đặc trưng có độ phân tách cao trên tập huấn luyện (train-only fit) để ngăn chặn data leakage.
5. **Cố vấn Dữ liệu Hóa học (Dataset Advisor):**
   - Tự động khảo sát độ đa dạng không gian hóa học qua độ tương đồng Tanimoto và Bemis-Murcko Scaffold, phân tích phân phối target, rủi ro ngoại lai, và đề xuất cấu hình thử nghiệm ban đầu (Virtual Supernode, Attention readout, loss function, scaling).
6. **Định lượng Độ bất định (Uncertainty Quantification):**
   - Hỗ trợ **Evidential Deep Learning** (phân phối Normal-Inverse-Gamma) và **Deep Ensembles** ($K \ge 3$) cho phép ước lượng độ bất định nhận thức (epistemic uncertainty) của mô hình.
7. **Hệ thống Đặc trưng Có khả năng Mở rộng (Config-Driven Feature Registry):**
   - Khai báo các khối đặc trưng qua YAML: đếm nguyên tố, cặp liên kết, mẫu SMARTS, nhóm Benson, hoặc hàm Python tùy biến.
8. **Đánh giá Bằng chứng Đặc trưng (Feature Evidence Scorecard):**
   - Sinh báo cáo phân tích trực quan về độ hỗ trợ (support), tương quan (Spearman association), độ dư thừa (redundancy) và khoảng giá trị train/test của từng đặc trưng.

---

## 🚀 Cài đặt

### Cài đặt qua Git:
```bash
git clone https://github.com/soratori1025/Framework-Chemistry.git
cd Framework-Chemistry
pip install -e ".[chemistry,test]"
```

### Yêu cầu môi trường:
- **Python**: $\ge 3.10$ (khuyến nghị 3.11 hoặc 3.12)
- **PyTorch**: $\ge 2.0$ (CPU hoặc CUDA GPU)
- **RDKit**: pin **2023.09.6** để tái lập đúng kết quả bài báo. Khối 210 descriptor được sinh từ `rdkit.Chem.Descriptors.descList`, mà số lượng descriptor thay đổi theo phiên bản RDKit (RDKit 2026.09.1 có 217 descriptor) — xem [REPRODUCING.md](REPRODUCING.md).
- **PyTDC**: (tùy chọn, dùng để tải tập dữ liệu chuẩn Therapeutics Data Commons)

Kiểm tra cài đặt:
```bash
pytest tests/
chemistry-framework --help
```

---

## 🔄 Quy trình Làm việc Trọn vẹn (End-to-End Workflow)

```mermaid
flowchart LR
    A[User CSV Dataset] -->|1. inspect| B[Advisor Report & Auto YAML]
    B -->|2. validate| C[Schema Validation]
    C -->|3. prepare| D[Cache & Feature Evidence]
    D -->|4. train| E[Trained Checkpoint .pt]
    E -->|5. predict| F[Predictions + Uncertainty]
```

### Bước 1: Khảo sát dữ liệu & Tự động sinh cấu hình (`inspect`)
Thay vì phải tự viết file YAML từ đầu, hãy để Advisor khảo sát dữ liệu của bạn, đánh giá rủi ro và tự động xuất ra file cấu hình tối ưu:
```bash
chemistry-framework inspect data/my_compounds.csv \
  --smiles-column smiles \
  --target activity \
  --generate-config configs/my_experiment.yaml \
  --output-dir reports/inspection
```

### Bước 2: Kiểm tra tính hợp lệ của cấu hình (`validate`)
```bash
chemistry-framework validate configs/my_experiment.yaml
```

### Bước 3: Chuẩn bị dữ liệu, chia tập & trích xuất scorecard (`prepare`)
Chia dữ liệu (Scaffold/Random), tính toán ma trận đặc trưng, lưu cache, và xuất báo cáo bằng chứng đặc trưng (HTML, CSV, Heatmap PNG):
```bash
chemistry-framework prepare configs/my_experiment.yaml
```

### Bước 4: Huấn luyện mô hình (`train`)
Huấn luyện mô hình Hybrid, D-MPNN, GIN hoặc Deep Ensemble theo đúng cấu hình:
```bash
chemistry-framework train configs/my_experiment.yaml
```
Kết quả huấn luyện và checkpoint `.pt` sẽ được lưu trong thư mục `checkpoint_dir`.

### Bước 5: Dự đoán cho các phân tử mới (`predict`)
Dự đoán nhãn và độ bất định (variance) cho một tập hợp các phân tử SMILES mới:
```bash
chemistry-framework predict \
  --checkpoint outputs/checkpoints/my_task.pt \
  --input data/screen_candidates.csv \
  --smiles-column smiles \
  --output data/screen_results.csv
```

---

## 🛠️ Hướng dẫn Thiết lập File Cấu hình YAML Cho Dữ Liệu Của Riêng Bạn

Một file cấu hình thử nghiệm chuẩn bao gồm 8 khối chính:

```yaml
name: my-custom-experiment

# 1. KHỐI DATASET: Định nghĩa nguồn dữ liệu
dataset:
  path: data/molecules.csv          # Đường dẫn tới file CSV dữ liệu của bạn
  smiles_column: smiles             # Tên cột chứa chuỗi SMILES
  targets:
    my_target:
      column: target_value          # Tên cột nhãn trong file CSV

# 2. KHỐI FEATURES: Khai báo các khối đặc trưng hóa học bổ sung
features:
  composition:
    type: elements
    elements: [H, C, N, O, S, P, F, Cl, Br, I]
  bonds:
    type: bond_pairs
    elements: [H, C, N, O, S, P, F, Cl, Br, I]
    bond_orders: ["1", "2", "3", "a"]
  aromatic_rings:
    type: smarts
    pattern: "a"
    mode: count

# 3. KHỐI ROUTES: Chỉ định đặc trưng nào đi vào nhánh nào
routes:
  my_target: [composition, bonds, aromatic_rings]

# 4. KHỐI TASKS: Khai báo loại bài toán
tasks:
  my_target:
    target: my_target
    kind: scalar                     # hoặc binary_classification
    route: my_target

# 5. KHỐI SPLIT: Chiến lược phân chia tập dữ liệu
split:
  method: scaffold                  # scaffold, random, size, hoặc smarts
  fractions: [0.8, 0.1, 0.1]        # train / val / test
  seed: 42

# 6. KHỐI MODEL: Kiến trúc mạng nơ-ron
model:
  structure:
    encoder: dmpnn                  # dmpnn hoặc gin
    hidden_dim: 128
    depth: 3
    aggregation: mean
    dropout: 0.1
  descriptors:
    enabled: true                   # Bật nhánh MLP xử lý đặc trưng bảng
    hidden_dim: 64
    depth: 1
    dropout: 0.1
  fusion: concat                    # concat, gated hoặc sum
  ffn:
    hidden_dim: 128
    depth: 2
    dropout: 0.1
  head: auto                        # auto, regression, classification, evidential, gaussian

# 7. KHỐI TRAINING: Tham số huấn luyện
training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  loss: huber                       # huber, bce, mse, evidential
  scheduler: cosine                 # cosine hoặc none
  warmup_epochs: 3
  grad_clip: 1.0
  patience: 15                      # Early stopping patience
  ensemble_size: 1                  # > 1 để bật Deep Ensemble (ví dụ: 3)
  device: auto                      # auto, cuda, hoặc cpu
  checkpoint_dir: outputs/checkpoints

# 8. KHỐI EVALUATION & ANALYSIS
evaluation:
  metrics: [mae, rmse, r2]          # hoặc [auroc, auprc, accuracy]
analysis:
  enabled: true
  output_dir: outputs/analysis
cache:
  enabled: true
  directory: .chemistry_cache/features
```

---

## 📋 Mẫu Cấu hình Tiêu biểu Cho Các Bài Toán Thực Tế

### Mẫu 1: Phân loại Nhị phân Mất cân bằng Lớp (Binary Classification)
Phù hợp cho các bài toán phân loại độc tính (Tox) hoặc vượt hàng rào máu não (BBB):
```yaml
name: admet-bbb-classification
dataset:
  path: data/bbb_dataset.csv
  smiles_column: Drug
  targets:
    penetration:
      column: Y
features:
  composition:
    type: elements
    elements: [H, C, N, O, S, P, F, Cl, Br, I]
tasks:
  penetration:
    target: penetration
    kind: binary_classification
split:
  method: scaffold
  fractions: [0.8, 0.1, 0.1]
model:
  structure:
    encoder: dmpnn
    hidden_dim: 128
    depth: 3
    dropout: 0.1
  descriptors:
    enabled: true
  head: classification
training:
  epochs: 35
  batch_size: 64
  learning_rate: 0.001
  loss: bce
  scheduler: cosine
  warmup_epochs: 3
evaluation:
  metrics: [auroc, auprc, accuracy]
```

---

### Mẫu 2: Hồi quy Kháng Ngoại lai với Huber Loss (Robust Regression)
Phù hợp cho dự đoán tính thấm Caco-2 hoặc Độ hòa tan (Solubility):
```yaml
name: admet-caco2-regression
dataset:
  path: data/caco2_dataset.csv
  smiles_column: Drug
  targets:
    permeability:
      column: Y
features:
  composition:
    type: elements
    elements: [H, C, N, O, S, P, F, Cl, Br, I]
tasks:
  permeability:
    target: permeability
    kind: scalar
split:
  method: scaffold
  fractions: [0.8, 0.1, 0.1]
model:
  structure:
    encoder: dmpnn
    hidden_dim: 64
    depth: 3
    dropout: 0.2
  descriptors:
    enabled: true
    hidden_dim: 64
  head: auto
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  loss: huber                       # Huber Loss triệt tiêu gradient của outliers
  scheduler: cosine
  warmup_epochs: 3
evaluation:
  metrics: [mae, rmse, r2]
```

---

### Mẫu 3: Định lượng Độ bất định với Evidential Head (Uncertainty Quantification)
Phù hợp cho các tập dữ liệu lớn ($N \ge 1000$) như Lipophilicity:
```yaml
name: lipophilicity-evidential
dataset:
  path: data/lipophilicity.csv
  smiles_column: Drug
  targets:
    logD:
      column: Y
features:
  composition:
    type: elements
    elements: [H, C, N, O, S, P, F, Cl, Br, I]
tasks:
  logD:
    target: logD
    kind: scalar
model:
  structure:
    encoder: dmpnn
    hidden_dim: 128
    depth: 3
  descriptors:
    enabled: true
  head: evidential                   # Tự động xuất dự đoán (mean) và độ bất định (variance)
training:
  epochs: 40
  batch_size: 64
  learning_rate: 0.001
  loss: evidential
evaluation:
  metrics: [mae, rmse, r2]
```

---

### Mẫu 4: Deep Ensemble Giảm Phương sai cho Dữ liệu Nhỏ (Deep Ensemble)
Phù hợp cho tập dữ liệu ít mẫu ($N < 800$) hoặc cần độ tin cậy tối đa:
```yaml
name: herg-ensemble
dataset:
  path: data/herg_dataset.csv
  smiles_column: Drug
  targets:
    blocker:
      column: Y
features:
  composition:
    type: elements
    elements: [H, C, N, O, S, P, F, Cl, Br, I]
tasks:
  blocker:
    target: blocker
    kind: binary_classification
model:
  structure:
    encoder: gin                    # Sử dụng Graph Isomorphism Network
    hidden_dim: 128
    depth: 4
  descriptors:
    enabled: true
  head: classification
training:
  epochs: 30
  batch_size: 64
  learning_rate: 0.0005
  loss: bce
  ensemble_size: 3                  # Huấn luyện độc lập 3 mô hình và lấy trung bình dự đoán
evaluation:
  metrics: [auroc, auprc, accuracy]
```

---

## 🐍 Sử dụng Framework qua Python API

```python
from chemistry_framework import (
    load_config,
    prepare_experiment,
    build_hybrid_model,
    make_experiment_dataloaders,
    train_model,
    evaluate_model,
)

# 1. Nạp file cấu hình
config = load_config("configs/my_experiment.yaml")

# 2. Chuẩn bị dữ liệu và trích xuất đặc trưng
prepared = prepare_experiment(config)

# 3. Tạo DataLoader
loaders = make_experiment_dataloaders(prepared, "my_target")

# 4. Khởi tạo mô hình Hybrid
model = build_hybrid_model(
    config.model,
    task_kind="scalar",
    out_dim=1,
    descriptor_dim=len(prepared.features.names),
)

# 5. Huấn luyện mô hình
result = train_model(
    model,
    loaders.train,
    loaders.validation,
    config.training,
    checkpoint_path="outputs/model.pt",
)

# 6. Đánh giá trên tập kiểm thử
metrics, actual, predicted = evaluate_model(
    model,
    loaders.test,
    config.evaluation.metrics,
    device=config.training.device,
)
print("Kết quả kiểm thử:", metrics)
```

---

## 📊 Kết quả Benchmark của bài báo

Kết quả chính được báo cáo trong bài báo, tính trên **10 split Bemis–Murcko scaffold** (FreeSolv dùng split ngẫu nhiên như trong bài). Giá trị là mean ± độ lệch chuẩn (population SD) trên test set, tổng hợp từ [`benchmarks/results/`](benchmarks/results/):

| Tập dữ liệu | Nhiệm vụ | $N$ | Metric | Graph only | Graph + 210 descriptors | RDKit 2D + RF |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **ESOL** | Độ tan $\log S$ (hồi quy) | 1,128 | RMSE ↓ | $0.623\pm0.092$ | $\mathbf{0.623\pm0.054}$ | $0.637\pm0.061$ |
| **FreeSolv** | Năng lượng hydrate hoá (hồi quy) | 642 | RMSE ↓ | $1.251\pm0.153$ | $1.099\pm0.249$ | $1.089\pm0.175$ |
| **Lipophilicity** | $\log D$ (hồi quy) | 4,200 | RMSE ↓ | $0.607\pm0.031$ | $\mathbf{0.606\pm0.019}$ | $0.748\pm0.025$ |
| **BBBP** | Thấm qua hàng rào máu–não (phân loại) | 2,039 | AUROC ↑ | $0.914\pm0.034$ | $\mathbf{0.916\pm0.029}$ | $0.912\pm0.019$ |
| **BACE** | Ức chế BACE-1 (phân loại) | 1,513 | AUROC ↑ | $\mathbf{0.802\pm0.038}$ | $0.785\pm0.033$ | $0.822\pm0.027$ |

Kết luận chính: **không có biểu diễn hay cấu hình nào thắng tuyệt đối** trên cả 5 tập; việc thêm 210 descriptor RDKit đổi điểm theo *cùng một hướng* ở cả framework lẫn Chemprop v2.3.1. Sau hiệu chỉnh Nadeau–Bengio + Holm, chỉ so sánh Hybrid vs RDKit 2D+RF trên Lipophilicity còn kết luận được ("conclusive"); các khác biệt còn lại là **inconclusive** (xem `benchmarks/results/paper_statistics.json`).

Phân tích độ nhạy BBBP sau khi loại 11 nhóm nhãn xung đột và gộp 53 nhóm trùng lặp (1,960 cấu trúc) nằm trong `benchmarks/run_bbbp_sensitivity_study.py` + `benchmarks/results/bbbp_sensitivity_study.json`.

> Kết quả ADMET giai đoạn phát triển ban đầu (BBB Martins, Caco-2, hERG) trong [ARCHITECTURE.md](ARCHITECTURE.md) là từ phiên bản trước của framework (một seed, cấu hình khác) và **không** thuộc bài báo; số liệu chính thức là bảng ở trên.

Chi tiết trực quan hoá xem tại [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 🧪 Chạy Kiểm thử (Unit Tests)

Bộ kiểm thử bao gồm 61 bài test tự động bao phủ toàn bộ các module:
```bash
pytest tests/ -v
```

---

## 📄 Bản quyền
Phát hành theo giấy phép [MIT License](LICENSE).
Mọi đóng góp, báo lỗi (Issues) và yêu cầu kéo (Pull Requests) đều được hoan nghênh.
