# Kaggle Runbook — SAGE 15-Class Crop Disease Training Pipeline

This runbook provides exact, step-by-step instructions for training the **Autonomous Visual AI Agent for Precision Agriculture** (`Qwen2.5-VL-3B-Instruct` with QLoRA) on Kaggle with GPU acceleration.

---

## 1. Storage & Environment Architecture on Kaggle

Before executing the notebook, understand Kaggle's storage layout:

| Storage Location | Path | Mutability | Persistence | Description |
|---|---|---|---|---|
| **Kaggle Input** | `/kaggle/input/...` | Read-Only | Permanent | Attached datasets (e.g. SAGE dataset if attached from Kaggle). |
| **Kaggle Working** | `/kaggle/working` | Read-Write | Ephemeral | Scratch directory where code is cloned, logs are written, and micro-batches process. Wiped when session disconnects. |
| **Durable Checkpoints** | Kaggle Dataset / Drive / Cloud Storage | Read-Write | Permanent | Checkpoint directory or artifact archives exported via `scripts/export_results.py`. |
| **Git Source Code** | `c:\Projects\SAGE-15Class` / GitHub | Versioned | Permanent | Modular source code, configs, test suite, and scripts. |

---

## 2. Step-by-Step Kaggle Setup

### Step 1: Create a New Kaggle Notebook
1. Go to [kaggle.com](https://www.kaggle.com) and click **Create** -> **New Notebook**.
2. Set notebook title: `SAGE-15Class-Qwen2.5-VL-Training`.

### Step 2: Configure Accelerator (GPU) & Internet
1. In the right-hand panel under **Notebook options**:
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100** (or **L4** if available).
   - **Internet**: Toggle to **Internet on** (required for Hugging Face weights and repository cloning).
   - **Persistence**: Select **Variables and files** if available.

### Step 3: Upload or Clone the Notebook
1. In the Kaggle notebook menu, select **File** -> **Upload Notebook**.
2. Select `notebooks/kaggle_train.ipynb` from this repository.
3. Alternatively, create a fresh cell and run:
   ```bash
   !git clone https://github.com/<YOUR_USERNAME>/sage-agri-agent.git
   %cd sage-agri-agent
   ```

---

## 3. Running the Pipeline (Cell-by-Cell Flow)

The notebook contains 22 numbered, self-contained cells:

- **Cell 1–3: Diagnostics & Hardware Check**: Verifies GPU VRAM (>= 15 GB recommended), RAM, and scratch disk capacity.
- **Cell 4: Dependency Installation**: Installs verified compatible versions of `transformers>=4.45.0`, `peft>=0.10.0`, `bitsandbytes>=0.43.0`, `accelerate`, and `pyarrow`.
- **Cell 5–6: Git Synchronization**: Clones latest repository branch and prints exact commit hash for auditability.
- **Cell 7: Configuration Loading**: Loads YAML configs from `configs/dataset.yaml`, `configs/classes.yaml`, `configs/training.yaml`, and `configs/evaluation.yaml`.
- **Cell 8–9: SAGE Metadata & Persistent Directories**: Downloads or verifies `sage_metadata.parquet` (~3.7 MB) and sets up `artifacts/` subdirectories.
- **Cell 10–11: Metadata Audit**: Generates `dataset_schema.json` and `dataset_inventory.json` auditing all 820,504 records across 335 crops.
- **Cell 12–14: Sample Manifest & Zero-Leakage Validation**: Executes balanced sampling targeting 800 training images per class across 15 classes. Validates zero duplicate group leakage.
- **Cell 15–16: Model Loading & LoRA Attachment**: Loads `Qwen/Qwen2.5-VL-3B-Instruct` with 4-bit NF4 quantization, double quantization, gradient checkpointing, and attaches PEFT LoRA adapters.
- **Cell 17: Preflight Multimodal Smoke Test**: Runs a micro-batch forward and backward pass, measuring throughput before starting full training.
- **Cell 18: Full 2-Epoch Training**: Executes 2 complete training epochs with visible progress bar, step loss, samples/sec, and periodic checkpointing.
- **Cell 19: Validation Evaluation**: Evaluates on held-out validation set and saves best validation checkpoint.
- **Cell 20: Independent Test Evaluation**: Evaluates held-out test split, generates multi-tier accuracy metrics, confusion matrix, and top confusions table.
- **Cell 21–22: Export & Packaging**: Compresses adapter weights and reports into `artifacts/sage_15class_run.tar.gz`.

---

## 4. CLI Execution Inside Kaggle Terminal (Alternative)

If you prefer terminal-driven execution inside Kaggle, open the Kaggle Terminal and run:

```bash
# 1. Audit dataset
python scripts/audit_dataset.py --config configs/dataset.yaml

# 2. Build reproducible manifest
python scripts/build_manifest.py --config configs/dataset.yaml

# 3. Validate zero split leakage
python scripts/validate_manifest.py --config configs/dataset.yaml

# 4. Run 2-epoch QLoRA training
python scripts/train.py --training-config configs/training.yaml

# 5. Evaluate on held-out test set
python scripts/evaluate.py --eval-config configs/evaluation.yaml --split test

# 6. Export results bundle
python scripts/export_results.py --output-tar artifacts/sage_15class_run.tar.gz
```

---

## 5. Resuming from Interrupted Kaggle Sessions

If your Kaggle session disconnects or reaches the time limit:

1. Re-open notebook and ensure checkpoints are intact or restored from persistent storage.
2. The `CheckpointManager` and `find_resumable_checkpoint` utility automatically:
   - Detects the latest valid checkpoint directory (`artifacts/checkpoints/epoch_*_step_*`).
   - Verifies the `manifest_hash` matches the current manifest to avoid dataset corruption.
   - Restores optimizer state, learning rate scheduler state, epoch, and RNG seed.
   - Resumes training from the exact completed step without reprocessing completed batches.

---

## 6. Downloading Deliverables

Once training finishes, locate the **Data** panel on the right side of Kaggle:
- Navigate to `/kaggle/working/artifacts/`
- Download:
  - `artifacts/sage_15class_run.tar.gz` (Complete bundle containing reports, confusion matrix, and trained LoRA adapter weights).
  - `artifacts/reports/final_report.md`
  - `artifacts/reports/confusion_matrix.png`
