# 魁星 (KuiXing) — 繁體中文預訓練語言模型

<p align="center">
  <img src="https://img.shields.io/badge/語言-繁體中文-red?style=flat-square" />
  <img src="https://img.shields.io/badge/架構-Decoder--Only Transformer-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/參數量-1.07B-green?style=flat-square" />
  <img src="https://img.shields.io/badge/框架-PyTorch%20%7C%20MLX-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/授權-CC BY--NC 4.0-lightgrey?style=flat-square" />
</p>

**魁星（KuiXing）** 是一個從零開始、以繁體中文語料預訓練的 Decoder-Only 大型語言模型。取名自中國傳統文化中掌管文章與科舉的神祇「魁星」，象徵對中文語言理解能力的追求。本專案包含完整的訓練程式碼，可在 Apple Silicon（MLX）或 NVIDIA GPU（CUDA）上執行，並輸出與 HuggingFace `transformers` 相容的模型格式。

---

## 目錄

- [模型概覽](#模型概覽)
- [模型架構詳情](#模型架構詳情)
- [參數量統計](#參數量統計)
- [訓練資料](#訓練資料)
- [訓練超參數](#訓練超參數)
- [環境需求](#環境需求)
- [安裝](#安裝)
- [訓練程式用法](#訓練程式用法)
- [CLI 參數說明](#cli-參數說明)
- [輸出格式與載入方式](#輸出格式與載入方式)
- [程式限制](#程式限制)
- [目錄結構](#目錄結構)
- [授權事項](#授權事項)
- [引用](#引用)

---

## 模型概覽

| 項目 | 內容 |
|------|------|
| 模型名稱 | KuiXing（魁星） |
| 模型類型 | Decoder-Only Transformer（自迴歸語言模型） |
| 主要語言 | 繁體中文（Traditional Chinese） |
| 參數量 | **1.07B**（10.7 億） |
| 詞彙量 | 99,384（SentencePiece BPE） |
| 最大序列長度 | 2,048 tokens |
| 訓練框架 | PyTorch（CUDA）／MLX（Apple Silicon） |
| 輸出格式 | HuggingFace `safetensors` + `config.json` |
| 授權 | CC BY-NC 4.0 |

---

## 模型架構詳情

KuiXing 採用標準 Pre-Norm Decoder-Only Transformer 架構，設計重點在於繁體中文的高效表示與訓練穩定性。

### 整體架構

```
輸入 token IDs
    ↓
Token Embedding（vocab_size × d_model）
    + Position Embedding（max_seq_len × d_model）
    + Embedding Dropout
    ↓
× 12 Transformer Blocks（Pre-Norm）
    ├── RMSNorm
    ├── Multi-Head Self-Attention（Causal Mask）
    │     ├── Q / K / V Projection（無 bias）
    │     ├── Scaled Dot-Product（float32 精度）
    │     ├── Causal Mask（上三角 -1e4，非 -inf）
    │     ├── Softmax → Attention Dropout
    │     └── Output Projection（無 bias）
    ├── Residual + Dropout
    ├── RMSNorm
    ├── Feed-Forward Network（GELU）
    │     ├── Linear: d_model → d_ff（無 bias）
    │     ├── GELU Activation
    │     ├── Dropout
    │     └── Linear: d_ff → d_model（無 bias）
    └── Residual + Dropout
    ↓
Final RMSNorm
    ↓
LM Head（d_model → vocab_size，**與 Token Embedding 共享權重**）
    ↓
Logits（float32）
```

### 關鍵設計決策

**Pre-Norm（前置正規化）**
Norm 層置於 Attention 與 MLP 之前，訓練更穩定，梯度流動更順暢，特別適合深層網路。

**RMSNorm 取代 LayerNorm**
Root Mean Square Normalization 省去均值計算，計算效率更高，且在語言模型中表現與 LayerNorm 相當。

**Causal Mask 使用 -1e4 而非 -inf**
避免 bfloat16 下 `-inf` 經過 softmax 產生 `NaN` 的數值不穩定問題。

**Attention Score 以 float32 計算**
即使在 bfloat16 訓練模式下，Q·Kᵀ 的縮放點積與 softmax 仍升型至 float32 進行，確保精度。

**Weight Tying（權重綁定）**
LM Head 與 Token Embedding 共享同一組權重矩陣，減少約 2.39 億參數，並有助於語意一致性。

**無 Bias 的線性層**
所有 Q/K/V/O Projection 及 FFN 的線性層均不使用 bias，符合現代大型語言模型的主流做法。

**Dropout 正則化**
Embedding dropout、Attention dropout 及殘差連接處均加入 dropout（預設 0.1），有效防止過擬合。

### 架構超參數

| 參數 | 數值 | 說明 |
|------|------|------|
| `n_layers` | 12 | Transformer Block 層數 |
| `d_model` | 2,400 | 隱藏層維度 |
| `n_heads` | 32 | 注意力頭數 |
| `d_head` | 75 | 每個注意力頭的維度（d_model / n_heads） |
| `d_ff` | 9,600 | Feed-Forward 中間層維度（4× d_model） |
| `max_seq_len` | 2,048 | 最大序列長度 |
| `vocab_size` | 99,384 | BPE 詞彙量 |
| `dropout` | 0.1 | Dropout 比率 |
| `activation` | GELU | FFN 激活函數 |
| `norm` | RMSNorm | 正規化層類型 |
| `pos_encoding` | Learned | 可學習的位置嵌入 |

---

## 參數量統計

| 模組 | 參數量 |
|------|--------|
| Token Embedding | 238,521,600（238.5M） |
| Position Embedding | 4,915,200（4.9M） |
| Attention（×12 層） | 276,480,000（276.5M） |
| Feed-Forward（×12 層） | 552,960,000（553.0M） |
| RMSNorm（×25 個） | 62,400 |
| LM Head | 0（與 Token Embedding 共享） |
| **合計** | **1,072,936,800（≈ 1.07B）** |

> **儲存大小估算：**
> - float32（訓練 / safetensors 輸出）：≈ **4.3 GB**
> - bfloat16（推理建議）：≈ **2.1 GB**

---

## 訓練資料

| 項目 | 內容 |
|------|------|
| 主要語料 | [jslin09/wikipedia_tw](https://huggingface.co/datasets/jslin09/wikipedia_tw)（台灣維基百科） |
| 語言 | 繁體中文 |
| Tokenizer | SentencePiece BPE，從語料訓練，詞彙量 99,384 |
| 資料處理 | 全文 tokenize → 串接為長序列 → 切成固定長度 chunk（2,049 tokens/chunk）→ 打散 |
| BOS / EOS | 每篇文章前後分別加入 `<s>` / `</s>` token |

訓練支援多資料集接續（Continual Training），可在訓練完成後以不同語料繼續微調，無需重新初始化模型。

---

## 訓練超參數

| 超參數 | 數值 | 說明 |
|--------|------|------|
| `batch_size` | 4 | 每步實際 mini-batch 大小 |
| `accum_steps` | 32 | 梯度累積步數 |
| 有效 Batch Size | 128 | `batch_size × accum_steps` |
| `learning_rate` | 3×10⁻⁴ | AdamW 峰值學習率 |
| LR Schedule | Linear Warmup + Cosine Decay | |
| `warmup_steps` | 250 | 線性暖身步數 |
| `weight_decay` | 0.1 | AdamW L2 正則化係數 |
| `betas` | (0.9, 0.95) | AdamW 動量係數 |
| `eps` | 1×10⁻⁶ | AdamW 數值穩定項 |
| `grad_clip` | 1.0 | 全局梯度裁剪閾值（L2 norm） |
| `epochs` | 3 | 訓練回合數 |
| `steps` | 30,000 | 每 epoch 最大步數 |
| 混合精度 | bfloat16（CUDA）／float32（MPS）／bfloat16（MLX） | |

**Weight Decay 策略：** 僅對 `dim ≥ 2` 的權重矩陣施加 weight decay；bias、RMSNorm 參數不衰減。

---

## 環境需求

### 必要條件

本程式**不支援純 CPU 執行**，需要以下其中一種硬體加速環境。以下規格為**最低需求**，不符合者將無法完成訓練：

| 環境 | 最低需求 | 建議機型範例 |
|------|----------|-------------|
| Apple Silicon Mac | 統一記憶體（RAM）**≥ 128 GB**，需安裝 MLX | Mac Studio / Mac Pro（M2 Ultra 192GB、M3 Ultra 192GB） |
| NVIDIA GPU | 單卡顯存 **≥ 92 GB**，CUDA 11.8 或以上 | H100 NVL（94 GB）、H200（141 GB） |
| 主機記憶體（RAM） | **≥ 64 GB**（兩種環境皆適用） | — |

> ⚠️ **重要：** 訓練環境低於以上任一最低需求，程式將因記憶體不足而無法執行完整訓練。
>
> **需求說明：**
> - float32 模型權重 ≈ 4.3 GB；加上梯度與 AdamW 狀態（各一份，共 3× 權重大小），訓練峰值顯存約 **90 GB 以上**。
> - Apple Silicon 的統一記憶體由 CPU 與 GPU 共用，128 GB 為能在 MLX bfloat16 模式下穩定訓練的最低配置。
> - 主機記憶體（系統 RAM）需 ≥ 64 GB，以容納分詞器訓練資料、資料集 tokenize、chunk 建構及 PyTorch 的系統側暫存空間。

### Python 套件

```
torch >= 2.2.0
mlx >= 0.12.0          # 僅 Apple Silicon 需要
sentencepiece >= 0.1.99
datasets >= 2.14.0
transformers >= 4.38.0
safetensors >= 0.4.0
numpy >= 1.24.0
matplotlib >= 3.7.0
tqdm >= 4.65.0
```

---

## 安裝

```bash
# 1. 複製專案
git clone https://github.com/your-username/kuixing.git
cd kuixing

# 2. 建立虛擬環境（建議）
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. 安裝 PyTorch（依據您的硬體選擇）
# CUDA 12.1：
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Apple Silicon（CPU/MPS fallback）：
pip install torch

# 4. 安裝 MLX（Apple Silicon 專用，建議安裝以獲得最佳性能）
pip install mlx

# 5. 安裝其餘相依套件
pip install sentencepiece datasets transformers safetensors numpy matplotlib tqdm
```

---

## 訓練程式用法

### 互動模式（無參數，推薦初次使用）

```bash
python KuiXing_Trainer_MLT.py
```

程式會自動偵測硬體環境，若有既有模型則詢問是否接續訓練及使用哪個資料集；若無既有模型則從頭開始訓練。

### 常用指令

```bash
# 從頭訓練（強制忽略已存在的模型）
python KuiXing_Trainer_MLT.py --from-scratch

# 直接接續訓練（不互動詢問，使用 config 預設資料集）
python KuiXing_Trainer_MLT.py --resume

# 接續訓練並切換到新資料集
python KuiXing_Trainer_MLT.py --resume --dataset jslin09/other_dataset --column text

# 以新資料集從頭訓練
python KuiXing_Trainer_MLT.py --from-scratch --dataset your_org/dataset_name --column article

# 獨立繪圖模式（讀取訓練記錄後生成曲線圖，不啟動訓練）
python KuiXing_Trainer_MLT.py --plot

# 訓練摘要模式（顯示 loss / perplexity 統計後結束，不啟動訓練）
python KuiXing_Trainer_MLT.py --summary
```

---

## CLI 參數說明

| 參數 | 類型 | 說明 |
|------|------|------|
| `--from-scratch` | flag | 強制從頭訓練，忽略已存在的 checkpoint 與模型 |
| `--resume` | flag | 直接接續上次訓練，跳過互動詢問 |
| `--dataset NAME` | string | 指定 HuggingFace 資料集名稱（如 `jslin09/wikipedia_tw`） |
| `--column COL` | string | 指定資料集中的文章欄位名稱（如 `article`、`text`） |
| `--plot` | flag | 讀取 JSONL 訓練記錄，生成四格訓練曲線圖後結束 |
| `--summary` | flag | 讀取 JSONL 訓練記錄，顯示訓練摘要統計後結束 |

> **注意：** `--from-scratch` 與 `--resume` 互斥，不可同時使用。  
> `--plot` 與 `--summary` 為獨立模式，不觸發硬體偵測或訓練流程。

### 訓練摘要輸出範例（`--summary`）

```
============================================================
  🏁  KuiXing 訓練完成摘要
============================================================
  總記錄步數         : 2,811  步（optimizer update）
  訓練 Epoch        : 3
  學習率範圍         : 0.00e+00  →  3.00e-04
  梯度被裁剪次數      : 51  次（佔 1.8%）
============================================================
  最終 Loss         : 2.741243
  最終 Perplexity   : 15.5062
============================================================
  最佳 Loss         : 2.639037  （第 81,311 步）
  最佳 Perplexity   : 13.9997
============================================================
  末 100 步平均 Loss : 2.784486
  末 100 步平均 PPL  : 16.1915
============================================================
```

---

## 輸出格式與載入方式

訓練完成後，程式自動將模型輸出至 `./kuixing_model/`，包含以下檔案：

```
kuixing_model/
├── model.safetensors      # float32 模型權重（HuggingFace 格式）
├── config.json            # 模型架構設定
├── modeling_kuixing.py    # 自訂架構定義（含 AutoModel 支援）
├── tokenizer_config.json  # Tokenizer 設定
└── tokenizer.model        # SentencePiece BPE tokenizer
```

### 載入方式

**方式一：直接使用自訂類別（推薦）**

```python
from modeling_kuixing import KuiXingForCausalLM

model = KuiXingForCausalLM.from_pretrained("./kuixing_model")
model = model.eval()
```

**方式二：bfloat16 推理（節省記憶體）**

```python
import torch
from modeling_kuixing import KuiXingForCausalLM

model = KuiXingForCausalLM.from_pretrained("./kuixing_model")
model = model.to(torch.bfloat16).eval()
```

**方式三：透過 HuggingFace AutoModel**

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "./kuixing_model",
    trust_remote_code=True,
)
```

**方式四：從 HuggingFace Hub 載入**

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "jslin09/kuixing",
    trust_remote_code=True,
)
```

---

## 程式限制

使用本訓練程式前，請了解以下限制：

**硬體限制**
- 不支援純 CPU 執行。程式於啟動時偵測硬體，若未找到 MPS 或 CUDA 裝置，將直接終止並提示錯誤。純 CPU 模式因速度過慢（預估為 GPU 的 50–200 倍）而刻意排除。

**記憶體需求為硬性限制**
- 訓練峰值顯存（含模型權重、梯度、AdamW 狀態）約需 **90 GB 以上**。NVIDIA GPU 需單卡顯存 ≥ 92 GB（如 H100 NVL 94 GB），Apple Silicon 需統一記憶體 ≥ 128 GB，主機 RAM 需 ≥ 64 GB。低於上述任一門檻者將因記憶體不足而無法完成訓練，此為硬性限制，無法透過縮減 `batch_size` 解決（梯度累積步數已補償批次大小，顯存瓶頸在於模型本身與優化器狀態）。

**MPS 平台限制**
- Apple Silicon 使用 MPS 後端（無 MLX）時，MPS 尚不支援 `torch.autocast`，因此以 float32 全精度訓練，速度與記憶體效率低於 MLX 路徑。建議安裝 MLX 以獲得最佳性能。

**多 GPU 不支援**
- 本程式目前僅支援單一 GPU 訓練，未實作 DDP（DistributedDataParallel）或 FSDP，無法直接用於多卡並行訓練。

**Checkpoint 跨平台相容性**
- PyTorch（CUDA/MPS）與 MLX（Apple Silicon）的 checkpoint 格式不同，無法直接互換。MLX checkpoint 以 `.npz` 儲存，PyTorch checkpoint 以 `.pt` 儲存。

**Tokenizer 字型依賴**
- 訓練曲線繪圖功能依賴本地安裝的繁體中文字型（Noto Sans CJK TC），字型路徑硬編碼於程式中。若路徑不符，繪圖將失敗或顯示亂碼。**此設定不應修改**（程式設計刻意保留）。

**資料集格式**
- 目前僅支援 HuggingFace `datasets` 格式的文字資料集，需指定文字欄位名稱。不支援本地 jsonl、txt、csv 直接輸入（需先上傳至 HuggingFace 或自行修改 `build_chunks()` 函式）。

**推理功能**
- 本程式為**預訓練專用訓練器**，不包含文字生成（inference）功能。推理請使用輸出的 HuggingFace 格式模型搭配 `transformers` 的 `generate()` 方法。

---

## 目錄結構

執行訓練後，工作目錄將產生以下結構：

```
./
├── KuiXing_Trainer_MLT.py     # 主訓練程式
│
├── kuixing_tokenizer/          # Tokenizer 檔案（自動生成）
│   ├── tokenizer.model
│   └── tokenizer.vocab
│
├── kuixing_checkpoints/        # 訓練 Checkpoint（自動生成）
│   ├── ckpt_ep0_step500.pt
│   ├── ckpt_ep0_step1000.pt
│   └── ...（保留最新 3 個）
│
├── kuixing_logs/               # 訓練記錄（自動生成）
│   ├── training_log.jsonl      # 每步指標（loss, lr, grad_norm, ...）
│   └── training_curves.png    # 訓練曲線圖（每 200 步更新）
│
└── kuixing_model/              # 最終輸出（訓練完成後生成）
    ├── model.safetensors
    ├── config.json
    ├── modeling_kuixing.py
    ├── tokenizer_config.json
    └── tokenizer.model
```

---

## 授權事項

### 程式碼授權

本專案訓練程式碼（`KuiXing_Trainer_MLT.py` 及相關檔案）以 **MIT License** 授權，允許自由使用、修改與散布，包含商業用途，惟需保留原始版權聲明。

```
MIT License

Copyright (c) 2025 Chun-Hsien Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

### 模型權重授權

訓練完成的模型權重（`model.safetensors`）以 **Creative Commons Attribution-NonCommercial 4.0 International（CC BY-NC 4.0）** 授權發布。

**您可以：**
- ✅ 分享：以任何媒介或格式複製、散布本模型
- ✅ 改作：修改、轉換本模型，以其為基礎進行創作（如微調）
- ✅ 學術研究與個人使用

**但需遵守以下條件：**
- 📌 **姓名標示（Attribution）：** 使用本模型時，需標示原始作者（jslin09 / KuiXing）及授權連結
- 🚫 **非商業性（NonCommercial）：** 不得將本模型或其衍生物用於商業目的
- 📌 **相同方式分享（ShareAlike）：** 若散布衍生模型，需採用相同的 CC BY-NC 4.0 或相容授權

完整授權條款請見：https://creativecommons.org/licenses/by-nc/4.0/

### 訓練資料聲明

本模型以台灣維基百科（`jslin09/wikipedia_tw`）為主要訓練語料，該語料源自維基媒體基金會，依據 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 授權。使用者在引用本模型產出內容時，亦應留意上游授權要求。

### 免責聲明

本模型為研究性質的預訓練語言模型，**不保證輸出內容的正確性、完整性或安全性**。使用者需自行評估並承擔模型輸出的風險。作者不對因使用本模型造成的任何直接或間接損失負責。

---

## 引用

若您在研究或作品中使用了 KuiXing，請引用本專案：

```bibtex
@misc{kuixing2026,
  author       = {Chun-Hsien Lin},
  title        = {KuiXing: A Traditional Chinese Pre-trained Language Model},
  year         = {2026},
  publisher    = {HuggingFace},
  howpublished = {\url{https://huggingface.co/jslin09/kuixing}},
}
```

---

<p align="center">
  以繁體中文為本，從零開始。<br>
  <em>Built from scratch, for Traditional Chinese.</em>
</p>
