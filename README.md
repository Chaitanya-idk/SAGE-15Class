# Autonomous Visual AI Agent for Precision Agriculture: SAGE 15-Class Training Pipeline

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers%204.45+-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20GPU%20%7C%20Linux%20%7C%20Windows-green.svg)]()

A modular, production-ready computer vision & multimodal LLM training pipeline fine-tuning **Qwen2.5-VL-3B-Instruct** using **QLoRA / LoRA (4-bit NF4)** on the SAGE dataset (`tirtho149/SAGE`). The pipeline targets **15 verified Indian-relevant crop–disease combinations** across 10 agricultural crops with balanced per-class sampling, leak-proof group stratification, and atomic checkpointing.

---

## 1. Project Purpose & Architecture

In precision agriculture, smallholder farmers and automated field agents require reliable, zero-shot and grounded visual diagnosis of plant pathologies. Generalist vision models often fail on subtle foliar symptoms or confuse similar diseases across crops. 

This repository fine-tunes `Qwen/Qwen2.5-VL-3B-Instruct` into a specialized agricultural diagnostic agent that outputs deterministic, structured classifications:
```text
Crop: <Crop>
Disease: <Disease>
```

```
                 SAGE Dataset (820,504 records)
                             │
                             ▼
              Metadata-First Discovery & Audit
                             │
                             ▼
             Group-Level Deduplication & Split
      (Zero Leakage Across Train 70% / Val 15% / Test 15%)
                             │
                             ▼
              Deterministic Sample Manifest
                             │
                             ▼
       Lazy Multimodal DataLoader (Aspect Preserving)
                             │
                             ▼
         Qwen2.5-VL-3B-Instruct + 4-Bit NF4 Quantization
                             │
                             ▼
           PEFT LoRA Adapters (r=16, alpha=32)
                             │
                             ▼
          2-Epoch Training Loop & Atomic Checkpoints
                             │
                             ▼
       Independent Test Evaluation & Multi-Tier Metrics
```

---

## 2. Verified 15 Crop–Disease Classes

From the SAGE dataset's 820,504 records and 335 crops, 15 high-impact, verified combinations were selected:

| Class ID | Crop | Canonical Disease | SAGE Available Records | Target Train Images | Status |
|---|---|---|---|---|---|
| `0` | Tomato | Tomato Leaf Curl Virus | 50,027 | 800 | Verified & Approved |
| `1` | Tomato | Bacterial Leaf Spot | 27,205 | 800 | Verified & Approved |
| `2` | Tomato | Late Blight | 23,469 | 800 | Verified & Approved |
| `3` | Rice | Blast | 7,411 | 800 | Verified & Approved |
| `4` | Rice | Brown Spot | 5,495 | 800 | Verified & Approved |
| `5` | Rice | Bacterial Leaf Blight | 4,696 | 800 | Verified & Approved |
| `6` | Corn | Common Rust | 20,487 | 800 | Verified & Approved |
| `7` | Corn | Northern Leaf Blight | 10,764 | 800 | Verified & Approved |
| `8` | Potato | Late Blight | 11,615 | 800 | Verified & Approved |
| `9` | Wheat | Rust | 5,579 | 800 | Verified & Approved |
| `10` | Mango | Anthracnose | 2,600 | 800 | Verified & Approved |
| `11` | Cauliflower | Black Rot | 3,245 | 800 | Verified & Approved |
| `12` | Papaya | Papaya Mosaic Virus | 2,732 | 800 | Verified & Approved |
| `13` | Sugarcane | Yellow Leaf | 1,528 | 800 | Verified & Approved |
| `14` | Banana | Panama Disease | 1,465 | 800 | Verified & Approved |

*Note: SAGE categorizes Maize under the canonical crop name `Corn`. Crop aliases (`Maize` -> `Corn`, `Paddy` -> `Rice`) and disease aliases (`Bacterial Blight` -> `Bacterial Leaf Blight`, `Mosaic Virus` -> `Papaya Mosaic Virus`) are automatically normalized.*

---

## 3. Repository Structure

```text
sage-agri-agent/
├── README.md                           # Comprehensive documentation
├── requirements.txt                     # Pinned cloud & local dependencies
├── pyproject.toml                       # Package setup and build specifications
├── .gitignore                           # Ignores large datasets, weights, and caches
├── .env.example                         # Environment template
├── configs/
│   ├── dataset.yaml                     # SAGE source, split ratios, cache settings
│   ├── classes.yaml                     # 15 approved classes, aliases, and IDs
│   ├── training.yaml                    # QLoRA hyperparameters, 2-epoch schedule
│   └── evaluation.yaml                  # Evaluation metrics, batch size, report paths
├── src/
│   └── sage/
│       ├── __init__.py
│       ├── config.py                    # Typed dataclass config loader
│       ├── data/
│       │   ├── schema.py                # Schema validation & column projection
│       │   ├── inventory.py             # Dataset inventory & anomaly tracking
│       │   ├── labels.py                # String normalization & prediction parser
│       │   ├── dedup.py                 # Duplicate grouping to prevent split leakage
│       │   ├── sampling.py              # Balanced sampling (800 train images/class)
│       │   ├── manifest.py              # Parquet manifest generator & validator
│       │   ├── streaming.py             # Lazy batch DataLoader with cache
│       │   └── image_validation.py      # Aspect-ratio preserving resize & padding
│       ├── models/
│       │   ├── load_model.py            # Qwen2.5-VL-3B-Instruct 4-bit NF4 loader
│       │   └── lora.py                  # PEFT LoRA target modules & parameter audit
│       ├── training/
│       │   ├── train.py                 # 2-epoch training loop & multimodal collator
│       │   ├── checkpointing.py         # Atomic checkpoint saver & RNG state manager
│       │   ├── progress.py              # Visible progress tracker & throughput
│       │   └── resume.py                # Resumable checkpoint discovery & validator
│       ├── evaluation/
│       │   ├── evaluate.py              # Multi-tier evaluation on validation & test
│       │   ├── metrics.py               # Crop acc, disease acc, exact match, F1
│       │   ├── confusion.py             # Confusion matrix & top confusions finder
│       │   └── report.py                # Markdown, JSON, and CSV report generator
│       └── utils/
│           ├── logging.py               # Structured logger
│           ├── hashing.py               # SHA256 / MD5 hashing for integrity
│           ├── hardware.py              # GPU, VRAM, and BF16 detection
│           └── reproducibility.py       # Deterministic seeds across PyTorch/CUDA
├── notebooks/
│   └── kaggle_train.ipynb               # Complete 22-cell interactive Kaggle notebook
├── scripts/
│   ├── audit_dataset.py                 # Run schema audit & create dataset_inventory
│   ├── build_manifest.py                # Sample 15 classes & create manifest
│   ├── validate_manifest.py             # Verify zero leakage across duplicate groups
│   ├── train.py                         # Launch 2-epoch QLoRA fine-tuning
│   ├── evaluate.py                      # Evaluate checkpoints on test split
│   └── export_results.py                # Package checkpoints & reports for export
├── tests/
│   ├── test_label_mapping.py            # Normalization and alias tests
│   ├── test_class_counts.py             # Verification of 15 classes and minimum counts
│   ├── test_manifest.py                 # Manifest creation and schema validation
│   ├── test_split_leakage.py            # Verification of zero group leakage
│   ├── test_image_processing.py         # EXIF orientation and aspect ratio tests
│   ├── test_resume.py                   # Atomic checkpoint saving and resume tests
│   └── test_metrics.py                  # Precision, recall, and multi-tier accuracy
└── artifacts/
    ├── metadata/                        # Schema, class registry, sample manifest
    ├── checkpoints/                     # Periodic and best LoRA adapters
    ├── predictions/                     # Detailed test predictions Parquet
    └── reports/                         # Final markdown report & confusion matrix
```

---

## 4. Installation & Environment Setup

### Local / Virtual Environment
```bash
git clone <YOUR_GIT_URL>
cd sage-agri-agent
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## 5. Metadata-First Workflow & Manifest Generation

Before decoding any image, generate the dataset inventory and sample manifest:

```bash
# 1. Audit dataset schema and inventory
python scripts/audit_dataset.py --config configs/dataset.yaml

# 2. Build balanced sample manifest (800 train images per class)
python scripts/build_manifest.py --config configs/dataset.yaml

# 3. Validate manifest for completeness and zero split leakage
python scripts/validate_manifest.py --config configs/dataset.yaml
```

The resulting manifest (`artifacts/metadata/sample_manifest.parquet`) contains:
- `sample_id`: Stable unique identifier.
- `class_id`: Deterministic integer from `0` to `14`.
- `class_name`: Canonical `Crop::Disease` identifier.
- `split`: `train` (70%), `val` (15%), or `test` (15%).
- `group_id`: Duplicate group key ensuring identical specimens never cross splits.

---

## 6. How to Run on Kaggle (Step-by-Step)

The primary execution workflow is **notebook-driven**: open `notebooks/kaggle_train.ipynb` and run the numbered cells sequentially.

1. **Create Notebook**: Go to Kaggle -> **Create** -> **New Notebook**.
2. **Enable GPU**: Settings -> **Accelerator** -> **GPU T4 x2** or **P100** or **L4**.
3. **Enable Internet**: Settings -> **Internet** -> **Internet on**.
4. **Upload Notebook**: File -> **Upload Notebook** -> Select `notebooks/kaggle_train.ipynb`.
5. **Run All Cells**: Execute cells 1 through 22.

### CLI Alternative Inside Kaggle
If you prefer running via terminal inside Kaggle:
```bash
python scripts/train.py --training-config configs/training.yaml
python scripts/evaluate.py --eval-config configs/evaluation.yaml --split test
python scripts/export_results.py --output-tar artifacts/sage_15class_run.tar.gz
```

---

## 7. Checkpointing & Resuming Interrupted Sessions

Kaggle sessions terminate after a timeout or idle disconnect. The pipeline includes an explicit `CheckpointManager`:

- **Atomic Writes**: Each checkpoint is written to `.temp_` first and atomically renamed, preventing corrupted states.
- **Saved State**: Adapter weights, optimizer state, scheduler state, current epoch, global step, and RNG states.
- **Hash Verification**: Resuming verifies that the `manifest_hash` matches the current dataset manifest.
- **Automatic Resume**: When restarting, `train.py` discovers the latest checkpoint and resumes from the exact step.

---

## 8. Evaluation Metrics & Artifacts

After training, the model is evaluated on the independent test split:

- **Exact Match Accuracy**: Both crop and disease correctly predicted.
- **Crop Accuracy**: Crop identified correctly.
- **Disease Accuracy**: Disease diagnosed correctly.
- **Macro & Weighted F1**: Across all 15 classes.
- **Artifacts Saved**:
  - `artifacts/reports/final_report.md`
  - `artifacts/reports/final_report.json`
  - `artifacts/reports/classification_report.csv`
  - `artifacts/reports/class_metrics.csv`
  - `artifacts/reports/top_confusions.csv`
  - `artifacts/reports/confusion_matrix.png`
  - `artifacts/predictions/test_predictions.parquet`

---

## 9. Running Automated Tests

Run the test suite across all modules:
```bash
pytest tests/
```
All 12 unit tests verify:
- Label normalization & alias mapping
- Class count integrity across 820,504 SAGE records
- Deterministic sampling and zero split leakage
- Aspect-ratio preserving image preprocessing
- Atomic checkpoint saving and resume validation
- Multi-tier metric computation and failure handling.
