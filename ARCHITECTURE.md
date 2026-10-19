# Kiến Trúc Chuyên Sâu Của Repository (Architecture & Technical Deep Dive)

Tài liệu này trình bày chi tiết về **kiến trúc phần mềm**, **thiết kế thuật toán**, **các module cốt lõi**, và **khả năng mở rộng** của `chemistry-framework`.

---

## Mục lục
1. [Triết lý Thiết kế Hệ thống](#1-triết-lý-thiết-kế-hệ-thống)
2. [Cấu trúc Thư mục và Các Module Cốt lõi](#2-cấu-trúc-thư-mục-và-các-module-cốt-lõi)
3. [Chi tiết Kỹ thuật Từng Thành phần](#3-chi-tiết-kỹ-thuật-từng-thành-phần)
   - [Đồ thị Phân tử & Bộ đệm Dữ liệu (data)](#đồ-thị-phân-tử--bộ-đệm-dữ-liệu-data)
   - [Bộ mã hóa Cấu trúc Đồ thị (encoders)](#bộ-mã-hóa-cấu-trúc-đồ-thị-encoders)
   - [Mô hình Lai & Đa phương thức (models)](#mô-hình-lai--đa-phương-thức-models)
   - [Đầu ra Chuyên biệt & Định lượng Độ bất định (heads)](#đầu-ra-chuyên-biệt--định-lượng-độ-bất-định-heads)
   - [Mô hình Tiền nghiệm Vật lý/Hóa học (priors)](#mô-hình-tiền-nghiệm-vật-lýhóa-học-priors)
   - [Hệ thống Phân tích Đặc trưng & Cố vấn Dữ liệu (analysis)](#hệ-thống-phân-tích-đặc-trưng--cố-vấn-dữ-liệu-analysis)
   - [Động cơ Huấn luyện & Đánh giá (training & evaluation)](#động-cơ-huấn-luyện--đánh-giá-training--evaluation)
4. [Công thức Toán học & Thuật toán Nền tảng](#4-công-thức-toán-học--thuật-toán-nền-tảng)
5. [Hệ thống Benchmark ADMET](#5-hệ-thống-benchmark-admet)
6. [Hướng dẫn Mở rộng Framework (Extensibility Guide)](#6-hướng-dẫn-mở-rộng-framework-extensibility-guide)

---

## 1. Triết lý Thiết kế Hệ thống

Hầu hết các công cụ học máy hóa học hiện nay (như Chemprop, DeepChem hay các mô hình Transformer phân tử) thường gặp phải các hạn chế sau:
- **Ghép nối chặt chẽ (Tight Coupling):** Đồ thị phân tử bị khóa chặt vào một kiến trúc duy nhất, khó bổ sung thêm các đặc trưng vật lý hoặc metadata thực nghiệm.
- **Thiếu định lượng độ bất định:** Dự đoán điểm (point prediction) không cho biết mức độ tin cậy của mô hình đối với các phân tử ngoài vùng phân phối huấn luyện (out-of-distribution).
- **Thiếu kiểm tra chất lượng tự động:** Người dùng phải tự phân tích rủi ro dữ liệu, dễ dẫn đến hiện tượng rò rỉ dữ liệu hoặc chọn sai hàm mất mát.

`chemistry-framework` được xây dựng nhằm giải quyết triệt để các vấn đề trên dựa trên 4 nguyên lý:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       NGUYÊN LÝ THIẾT KẾ CỦA FRAMEWORK                      │
├───────────────────────┬─────────────────────────┬───────────────────────────┤
│ 1. Tách biệt Vai trò  │ 2. Mô hình Lai Đa thức  │ 3. Định lượng Bất định    │
│    (Role Separation)  │    (Hybrid Multi-Modal) │    (Uncertainty by Design)│
│ Khối đặc trưng, vai   │ Kết hợp GNN biểu diễn   │ Tích hợp Evidential Deep  │
│ trò học (prior/       │ cấu trúc + MLP biểu     │ Learning & Deep Ensemble  │
│ residual), và mô hình │ diễn đặc trưng hóa lý + │ xuất ra dự đoán kèm       │
│ độc lập hoàn toàn.    │ Physics Priors.         │ độ bất định tin cậy.      │
├───────────────────────┴─────────────────────────┴───────────────────────────┤
│ 4. Trí tuệ Phân tích Dữ liệu (Automated Advisory & Evidence Tracking)      │
│ Tự động profiling dữ liệu, chấm điểm sức khỏe và kiểm chứng bằng chứng     │
│ thực nghiệm của từng đặc trưng qua scorecard trực quan.                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Cấu trúc Thư mục và Các Module Cốt lõi

```text
chemistry_framework/
├── analysis/               # Cố vấn dữ liệu tự động (DatasetAdvisor) và báo cáo scorecard
│   ├── advisor.py          # Profiling hóa học, tabular features, target, risks, và auto config
│   ├── report.py           # Phân tích phân phối binning và khoảng giá trị train vs test
│   └── prediction.py       # Kiểm tra phạm vi ngoại suy cho phân tử suy luận mới
├── config/                 # Định nghĩa schema cấu hình YAML và parser
│   ├── schema.py           # Pydantic/Dataclass schema kiểm tra tính hợp lệ
│   └── model_spec.py       # Cấu hình kiến trúc mạng (ModelSpec, StructureSpec, DescriptorsSpec)
├── core.py                 # Định nghĩa FeatureRole (PRIOR, RESIDUAL, AUXILIARY) và ArchitectureRecipe
├── data/                   # Biểu diễn đồ thị phân tử và tạo DataLoader
│   ├── featurization.py    # Trích xuất đặc trưng nguyên tử (133 chiều) và liên kết (14 chiều)
│   ├── graph.py            # Cấu trúc MolGraph, BatchMolGraph đóng gói tensor thưa
│   ├── dataset.py          # Lớp ChemistryDataset và MolecularDataset
│   └── loaders.py          # Tạo PyTorch DataLoaders hỗ trợ cả Graph lẫn Tensors
├── diagnostics/            # Đo lường khoảng giá trị, độ hỗ trợ và phương sai đặc trưng
├── encoders/               # Các khối mạng nơ-ron học biểu diễn cấu trúc đồ thị & phân tử
│   ├── gnn.py              # D-MPNN (Directed Message Passing) và GIN (2D Graph Isomorphism) tích hợp Virtual Node & Attention Readout
│   ├── mlp.py              # MLPEncoder học biểu diễn phi tuyến từ đặc trưng hóa lý bảng
│   └── graph_features.py   # Lớp nhúng đặc trưng nguyên tử và liên kết
├── evaluation/             # Đo lường hiệu năng và metrics đánh giá
│   └── metrics.py          # AUROC, AUPRC, Accuracy, F1, MAE, RMSE, R2, Spearman
├── features/               # Đăng ký và tính toán đặc trưng hóa học
│   ├── registry.py         # FeatureRegistry quản lý tên và nguồn gốc đặc trưng
│   ├── blocks.py           # Khối đếm nguyên tố, cặp liên kết hóa học
│   ├── smarts.py           # Khối khớp mẫu SMARTS dưới dạng đếm hoặc cờ nhị phân
│   └── cache.py            # Lưu cache đĩa theo hàm băm phân tử và fingerprint cấu hình
├── heads/                  # Các đầu ra chuyên biệt cho mạng nơ-ron
│   ├── specialized.py      # EvidentialRegressionHead (NIG) và GaussianNLLHead
│   └── base.py             # ScalarOutputHead, IdentityOutputHead, MultiTaskHead, NASA7OutputBlock
├── inputs/                 # Hỗ trợ đa định dạng đầu vào hóa học
│   ├── objects.py          # MoleculeObject, ReactionObject, MixtureObject, ComplexObject
│   └── adapters.py         # Bộ chuyển đổi nạp dữ liệu từ CSV/SDF/JSON
├── models/                 # Lắp ráp mô hình hoàn chỉnh
│   ├── hybrid.py           # HybridChemistryModel kết hợp GNN + Descriptors MLP
│   ├── ensemble.py         # EnsembleModel quản lý huấn luyện và suy luận Deep Ensemble
│   └── residual.py         # ResidualPropertyModel kết hợp Base Prior + Neural Residual
├── priors/                 # Các mô hình tiền nghiệm vật lý/hóa lý
│   └── linear.py           # LinearAdditivePrior, Ridge Prior, CompositePrior
├── splits/                 # Các giải thuật chia tập dữ liệu
│   └── scaffold.py         # Bemis-Murcko Scaffold Split, Random Split, Size Split, SMARTS Holdout
├── tasks/                  # Quản lý metadata của bài toán dự đoán (PropertyTask)
└── training/               # Vòng lặp huấn luyện tối ưu hóa
    ├── engine.py           # train_model, train_ensemble, Cosine Annealing, Early Stopping
    └── losses.py           # HuberLoss, EvidentialLoss (Normal-Inverse-Gamma NLL), BCEWithLogits
```

---

## 3. Chi tiết Kỹ thuật Từng Thành phần

### Đồ thị Phân tử & Bộ đệm Dữ liệu (`data`)

Khác với các thư viện đóng gói cồng kềnh, `chemistry-framework` sử dụng cơ chế biểu diễn đồ thị thuần PyTorch tối ưu bộ nhớ:
- **Đặc trưng nguyên tử (`Atom Features` - 133 chiều):**
  - Số hiệu nguyên tử (One-hot 100 nguyên tố).
  - Bậc liên kết (Degree 0–10).
  - Điện tích hình thức (Formal charge -5 đến +5).
  - Trạng thái lai hóa (Hybridization: $sp, sp^2, sp^3, sp^3d, sp^3d^2$).
  - Số lượng hydro liên kết, tính thơm (aromaticity), và khối lượng nguyên tử chuẩn hóa.
- **Đặc trưng liên kết (`Bond Features` - 14 chiều):**
  - Loại liên kết (Đơn, Đôi, Ba, Thơm).
  - Tính liên hợp (Conjugation).
  - Nằm trong vòng (In ring).
  - Cấu hình lập thể (Stereo cis/trans/none).
- **`BatchMolGraph`:**
  - Gom các đồ thị phân tử rời rạc vào một đồ thị khổng lồ (Disjoint Union Graph) với ma trận trỏ liên kết `a2b`, `b2a`, và `b2revb` (chỉ số liên kết đảo ngược), loại bỏ hoàn toàn chi phí padding.

---

### Bộ mã hóa Cấu trúc Đồ thị (`encoders`)

#### 1. D-MPNN (`DMPNNEncoder`)
Triển khai thuật toán truyền tin định hướng (Directed Message Passing) lấy cảm hứng từ Chemprop:
- Thay vì truyền tin từ nguyên tử sang nguyên tử (dễ bị lặp tin không cần thiết), D-MPNN truyền tin trên các **liên kết định hướng** $(v, w)$:
  $$h_{vw}^{(t)} = \text{ReLU}\left(h_{vw}^{(0)} + W_m \sum_{k \in \mathcal{N}(v) \setminus \{w\}} h_{kv}^{(t-1)}\right)$$
- Sau $T$ bước truyền tin, thông điệp trên các liên kết tới nguyên tử $v$ được gom lại để tạo vector nhúng nguyên tử $h_v$. Vector phân tử $h_{\text{mol}}$ được tính bằng phép gom (Readout/Aggregation: `mean`, `sum`, hoặc `vnode`).

#### 2. GIN (`GINEncoder`)
Triển khai mạng đẳng cấu đồ thị (Graph Isomorphism Network) dựa trên định lý Weisfeiler-Lehman 1-WL:
$$h_v^{(k)} = \text{MLP}^{(k)}\left(\left(1 + \epsilon^{(k)}\right) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}\right)$$
GIN có khả năng phân biệt hình học phân tử cao, đặc biệt phù hợp cho các bài toán phân loại độc tính phân tử phức tạp (như hERG).

---

### Mô hình Lai & Đa phương thức (`models`)

#### `HybridChemistryModel`
Mô hình kết hợp 2 luồng biểu diễn:
1. **Luồng đồ thị 2D (Structure Encoder):** Nhận đồ thị phân tử `BatchMolGraph`, xử lý qua D-MPNN hoặc GIN để xuất ra vector đặc trưng học sâu $z_{\text{graph}} \in \mathbb{R}^{d_g}$.
2. **Luồng đặc trưng bảng (Descriptors Encoder):** Nhận vector đặc trưng số (nhóm chức, thành phần nguyên tố, hoặc metadata người dùng) $x_{\text{desc}} \in \mathbb{R}^{d_d}$, xử lý qua mạng MLP với LayerNorm và Dropout để xuất ra $z_{\text{desc}} \in \mathbb{R}^{d_m}$.
3. **Bộ hợp nhất (Fusion Block):**
   - Hợp nhất nối tiếp (Concatenation): $z_{\text{fused}} = [z_{\text{graph}} \,\|\, z_{\text{desc}}]$
   - Hợp nhất cộng tích lũy (Additive): $z_{\text{fused}} = z_{\text{graph}} + W_p z_{\text{desc}}$
4. **Mạng dự đoán sau hợp nhất (Predictive FFN):** Đưa $z_{\text{fused}}$ qua 2 tầng ẩn MLP để truyền vào Output Head.

#### `EnsembleModel`
- Đóng gói $K$ mô hình độc lập được huấn luyện với các seed khởi tạo khác nhau.
- Khi suy luận, ensemble tính toán:
  - Giá trị dự đoán trung bình: $\bar{\mu}(x) = \frac{1}{K} \sum_{k=1}^K \mu_k(x)$
  - Phương sai mẫu đại diện cho độ bất định nhận thức (Epistemic Uncertainty):
    $$\sigma_{\text{ensemble}}^2(x) = \frac{1}{K} \sum_{k=1}^K (\mu_k(x) - \bar{\mu}(x))^2$$

---

### Đầu ra Chuyên biệt & Định lượng Độ bất định (`heads`)

#### 1. `EvidentialRegressionHead`
Sử dụng lý thuyết **Evidential Deep Learning** để mô hình hóa tham số của phân phối liên hợp Normal-Inverse-Gamma (NIG) gồm 4 đại lượng $(\gamma, v, \alpha, \beta)$:
- Dự đoán trung bình: $\mathbb{E}[y] = \gamma$
- Độ bất định tự thân (Aleatoric Uncertainty): $\sigma^2_{\text{aleatoric}} = \frac{\beta}{\alpha - 1}$
- Độ bất định nhận thức (Epistemic Uncertainty): $\sigma^2_{\text{epistemic}} = \frac{\beta}{v(\alpha - 1)}$
- Hàm mất mát Evidential:
  $$\mathcal{L}(y) = \mathcal{L}_{\text{NLL}}(y) + \lambda_{\text{reg}} \, |y - \gamma| \, (2v + \alpha)$$
  trong đó $\mathcal{L}_{\text{NLL}}$ tối ưu hóa log-likelihood của phân phối Student-t, và số hạng regularization phạt mô hình khi đưa ra độ tin cậy cao trên các dự đoán sai.

#### 2. `GaussianNLLHead`
- Đầu ra gồm 2 neuron: $\mu(x)$ và $\log \sigma^2(x)$. Huấn luyện bằng hàm mất mát Gaussian Negative Log-Likelihood, cho phép mô hình học phương sai phụ thuộc vào từng phân tử (Heteroscedastic noise).

---

### Hệ thống Phân tích Đặc trưng & Cố vấn Dữ liệu (`analysis`)

Module `DatasetAdvisor` bao gồm các engine phân tích tự động:
1. **Chemical Profiling:** Sử dụng RDKit để tính toán các đặc trưng phân tử, kiểm tra cấu trúc hợp lệ, tính tỷ lệ Bemis-Murcko Scaffold Diversity.
2. **Tabular Feature Profiling:** Phân tích các cột số ngoài SMILES:
   - Tính phương sai và gắn cờ `Constant` (phương sai = 0) hoặc `Near-Zero Variance` (`std < 1e-4`).
   - Tính ma trận tương quan Pearson giữa tất cả các cặp đặc trưng, phát hiện đa cộng tuyến $|r| > 0.90$.
   - Lọc ra danh sách `recommended_keep_features`.
3. **Target Profiling:** Nhận diện loại bài toán (Binary Classification, Scalar Regression).
   - Với phân loại: Tính tỷ lệ mất cân bằng lớp và `pos_weight`.
   - Với hồi quy: Tính Skewness, Kurtosis và tỷ lệ ngoại lai IQR ($Q_1 - 1.5\text{IQR}$, $Q_3 + 1.5\text{IQR}$).
4. **Health Score Engine:** Chấm điểm dữ liệu từ 100 điểm ban đầu, trừ điểm theo các trọng số rủi ro:
   - Trừ 25 điểm nếu tỷ lệ SMILES lỗi $> 5\%$.
   - Trừ 15 điểm nếu mẫu quá nhỏ ($N < 300$).
   - Trừ 15 điểm nếu mất cân bằng lớp $> 5:1$.
   - Trừ 10 điểm nếu target bị lệch nặng ($|\text{Skewness}| > 1.5$).
   - Trừ 10 điểm nếu có cột hằng số hoặc khuyết thiếu $> 20\%$.

---

### Động cơ Huấn luyện & Đánh giá (`training` & `evaluation`)

- **Bộ lập lịch Cosine Annealing kèm Warmup:** Khởi động learning rate tuyến tính trong $N_{\text{warmup}}$ epochs đầu, sau đó giảm theo đường cong Cosine đến $\eta_{\text{min}} = 10^{-6}$.
- **Cắt xén Gradient (Gradient Clipping):** Giới hạn chuẩn gradient tối đa (`grad_clip: 1.0`) để ngăn chặn bùng nổ gradient trong GNN.
- **Dừng sớm (Early Stopping):** Theo dõi hàm loss trên tập validation với bộ đệm kiên nhẫn (`patience: 15`). Tự động khôi phục lại trọng số của epoch có validation loss thấp nhất.

---

## 4. Công thức Toán học & Thuật toán Nền tảng

### 1. Phân chia Phân tử theo Bemis-Murcko Scaffold (Scaffold Split)
Thuật toán phân chia scaffold ngăn ngừa hiện tượng học vẹt cấu trúc:
1. Với mỗi phân tử $m_i \in \mathcal{D}$, trích xuất khung carbon cốt lõi (loại bỏ các nhóm thế mạch nhánh):
   $$S_i = \text{MurckoScaffold}(m_i)$$
2. Gom các phân tử có chung khung $S_i$ vào cùng một cụm $\mathcal{C}_k$.
3. Sắp xếp các cụm $\mathcal{C}_k$ theo thứ tự giảm dần về kích thước.
4. Phân bổ các cụm lần lượt vào tập Train, Validation, và Test sao cho đạt tỷ lệ mong muốn (ví dụ 80% / 10% / 10%) mà không làm phân rã các phân tử có cùng khung cấu trúc sang hai tập khác nhau.

### 2. Hàm mất mát Huber (Huber Loss)
Đối với bài toán hồi quy chứa giá trị ngoại lai, Huber Loss đóng vai trò chuyển tiếp mượt giữa $L_2$ và $L_1$:
$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{với } |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{ngược lại} \end{cases}$$
với ngưỡng mặc định $\delta = 1.0$.

---

## 5. Hệ thống Benchmark ADMET

Thư mục [`benchmarks/`](benchmarks/) chứa toàn bộ mã nguồn kiểm thử và đối sánh trên 4 tập dữ liệu chuẩn của Therapeutics Data Commons (TDC):

```bash
# Chạy toàn bộ benchmark suite
python benchmarks/run_benchmarks.py
```

### Kết quả Thực nghiệm Thăm dò (Exploratory Benchmarks):
> [!NOTE]
> Bảng dưới đây phản ánh kết quả thử nghiệm thăm dò đơn lẻ (single-run exploratory partition) trên phân chia Bemis-Murcko Scaffold Split tỷ lệ 80/10/10 (Seed 42).
> Đối với đối sánh chuẩn mực khoa học (statistical benchmarks), framework tách biệt thành protocol 5 seeds độc lập với chỉ số phân chia cố định (canonical indices manifest), báo cáo mean ± SD và paired confidence intervals tại `benchmarks/results/manifest.json`.

| Tập dữ liệu | Nhiệm vụ | Số mẫu | Kiến trúc Mô hình | Hàm Loss | Kết quả Đạt được (Single-Run 80/10/10) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BBB Martins** | Khả năng vượt hàng rào máu não | 2,030 | D-MPNN Hybrid | BCE | **AUROC: 0.884**<br>AUPRC: 0.957<br>Accuracy: 0.867 |
| **Caco-2 Wang** | Tính thấm qua tế bào biểu mô ruột | 910 | D-MPNN Hybrid | Huber | **MAE: 0.354**<br>RMSE: 0.528<br>$R^2$: 0.551 |
| **Lipophilicity** | Ái lực phân bố n-octanol/nước | 4,200 | D-MPNN + Evidential | Evidential | **MAE: 0.551**<br>RMSE: 0.749<br>$R^2$: 0.613 |
| **hERG Blockers**| Khả năng ức chế kênh kali hERG | 648 | GIN Ensemble ($K=3$) | BCE | **AUROC: 0.841**<br>AUPRC: 0.730<br>Accuracy: 0.769 |

---

## 6. Hướng dẫn Mở rộng Framework (Extensibility Guide)

### Thêm một Bộ mã hóa Đồ thị mới (Custom Graph Encoder)
Tạo file mới hoặc kế thừa từ `torch.nn.Module` trong `chemistry_framework/encoders/`:

```python
import torch
import torch.nn as nn
from chemistry_framework.data.graph import BatchMolGraph

class MyGraphAttentionEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, depth: int = 3):
        super().__init__()
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        # Khai báo các lớp Attention...

    def forward(self, batch_graph: BatchMolGraph) -> torch.Tensor:
        # Nhận ma trận nút và liên kết từ batch_graph
        x = self.node_embedding(batch_graph.atom_features)
        # Thực hiện phép tính truyền tin...
        # Trả về tensor đại diện phân tử: [batch_size, hidden_dim]
        return pooled_features
```

Đăng ký tên encoder mới vào `build_structure_encoder` trong [`chemistry_framework/encoders/gnn.py`](file:///home/nvsinh1/brats_segmentation/teb-net-model/framework/chemistry_framework/encoders/gnn.py).

### Thêm một Khối Đặc trưng Hóa học mới (Custom Feature Block)
Trong file cấu hình YAML:
```yaml
features:
  my_custom_feature:
    type: python
    fn: my_module.calculators.compute_charge_density
    dim: 8
```
Hoặc đăng ký hàm tính toán vào `FeatureRegistry`:
```python
from chemistry_framework.features import FeatureRegistry, FeatureSpec

registry = FeatureRegistry()
registry.register(
    FeatureSpec(
        name="custom.charge",
        calculator=lambda mol: compute_custom_metric(mol),
        domain="molecular",
        description="Điện tích bề mặt tùy biến",
    )
)
```
