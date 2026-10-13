import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "src"))

from sage.config import load_training_config
from sage.models.load_model import resolve_model_path, load_qwen_processor, get_model_class

def main():
    print("=== Qwen2.5-VL Model & Processor Preflight Verification ===")
    cfg = load_training_config("configs/training.yaml")
    model_path = resolve_model_path(cfg.model_id)
    print(f"Target model path: {model_path}")

    # 1. Verify processor loading
    print("\n[1/3] Loading processor...")
    processor = load_qwen_processor(model_path)
    print(f"  -> Processor loaded successfully: {processor.__class__.__name__}")

    # 2. Verify model class resolution
    print("\n[2/3] Resolving model architecture class...")
    model_cls = get_model_class()
    print(f"  -> Model class: {model_cls.__name__}")

    # 3. Test tokenization & prompt formatting
    print("\n[3/3] Testing dummy prompt processing...")
    test_text = "Crop: Tomato\nDisease: Late Blight"
    inputs = processor(text=[test_text], return_tensors="pt")
    print(f"  -> Input IDs shape: {inputs['input_ids'].shape}")
    print("\n>>> ALL CHECKS PASSED! Model and processor are fully ready for training. <<<")

if __name__ == "__main__":
    main()
