import os
from typing import Any, Dict, Tuple
import torch
from sage.config import TrainingConfig
from sage.utils.hardware import get_hardware_info
from sage.utils.logging import get_logger

logger = get_logger("sage.models")


def resolve_model_path(model_id_or_path: str) -> str:
    """Resolves local model path in Kaggle input or environment, otherwise returns configured ID."""
    # 1. Direct path check
    if os.path.exists(model_id_or_path):
        logger.info(f"Using existing model path: {model_id_or_path}")
        return model_id_or_path

    # 2. Known Kaggle input paths for Qwen2.5-VL-3B-Instruct
    candidate_paths = [
        "/kaggle/input/models/qwen-lm/qwen2.5-vl/transformers/3b-instruct/2",
        "/kaggle/input/models/qwen-lm/qwen2.5-vl/transformers/3b-instruct/1",
        "/kaggle/input/qwen2.5-vl/transformers/3b-instruct/2",
        "/kaggle/input/qwen2.5-vl/transformers/3b-instruct/1",
        "/kaggle/input/qwen2.5-vl-3b-instruct",
        "/kaggle/input/qwen2.5-vl",
    ]
    for cp in candidate_paths:
        if os.path.exists(cp) and os.path.isfile(os.path.join(cp, "config.json")):
            logger.info(f"Auto-detected Kaggle model weights at: {cp}")
            return cp

    # 3. Dynamic search in /kaggle/input
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            if "config.json" in files:
                low = root.lower()
                if "qwen2.5-vl" in low or "qwen2_5_vl" in low or "qwen2.5" in low:
                    logger.info(f"Dynamically discovered Kaggle model at: {root}")
                    return root

    return model_id_or_path


def load_qwen_processor(path: str) -> Any:
    """Robustly loads Qwen2.5-VL processor handling preprocessor_config naming discrepancies."""
    import json
    import os
    import transformers
    from transformers import AutoProcessor, AutoTokenizer

    # Attempt 1: Standard AutoProcessor
    try:
        return AutoProcessor.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Standard AutoProcessor.from_pretrained failed ({e}). Assembling processor directly...")

    # Attempt 2: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Attempt 3: Build image processor from raw config dict (avoids re-reading unrecognized type)
    _SKIP_KEYS = {"image_processor_type", "processor_class", "model_type", "auto_map"}
    preprocessor_cfg: Dict[str, Any] = {}
    cfg_file = os.path.join(path, "preprocessor_config.json")
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as _f:
                preprocessor_cfg = {k: v for k, v in json.load(_f).items() if k not in _SKIP_KEYS}
        except Exception as _e:
            logger.warning(f"Could not parse preprocessor_config.json: {_e}")

    image_processor = None
    for _cls_name in ("Qwen2_5_VLImageProcessor", "Qwen2VLImageProcessor"):
        _Cls = getattr(transformers, _cls_name, None)
        if _Cls is None:
            continue
        try:
            image_processor = _Cls(**preprocessor_cfg)
            logger.info(f"Instantiated {_cls_name} from config dict.")
            break
        except Exception as _e:
            logger.warning(f"{_cls_name}(**cfg) failed ({_e}), retrying with defaults.")
            try:
                image_processor = _Cls()
                logger.info(f"Instantiated {_cls_name} with defaults.")
                break
            except Exception:
                pass

    # Attempt 4: Assemble unified processor and return immediately
    for _proc_name in ("Qwen2_5_VLProcessor", "Qwen2VLProcessor"):
        _ProcCls = getattr(transformers, _proc_name, None)
        if _ProcCls is None:
            continue
        try:
            _kw: Dict[str, Any] = {"tokenizer": tokenizer}
            if image_processor is not None:
                _kw["image_processor"] = image_processor
            assembled = _ProcCls(**_kw)
            logger.info(f"Successfully assembled {assembled.__class__.__name__} from components.")
            return assembled
        except Exception as _e:
            logger.warning(f"Could not instantiate {_proc_name}: {_e}")

    # Attempt 5: Minimal wrapper so training can continue
    class _SimpleProcessor:
        def __init__(self, tok: Any, img: Any = None):
            self.tokenizer = tok
            self.image_processor = img
        def __call__(self, text=None, images=None, return_tensors=None, padding=True, **kw):
            out: Dict[str, Any] = {}
            if text is not None:
                out.update(self.tokenizer(text, return_tensors=return_tensors, padding=padding, **kw))
            if images is not None and self.image_processor is not None:
                img_out = self.image_processor(images=images, return_tensors=return_tensors)
                for k, v in img_out.items():
                    out.setdefault(k, v)
            return out
        def decode(self, *a, **k):
            return self.tokenizer.decode(*a, **k)
        def batch_decode(self, *a, **k):
            return self.tokenizer.batch_decode(*a, **k)

    logger.warning("Using _SimpleProcessor fallback wrapper.")
    return _SimpleProcessor(tok=tokenizer, img=image_processor)


def get_model_class() -> Any:
    """Returns the appropriate Qwen2.5-VL conditional generation model class."""
    for _name in ("Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration", "AutoModelForImageTextToText"):
        import transformers as _t
        _cls = getattr(_t, _name, None)
        if _cls is not None:
            return _cls
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM


def get_quantization_config(training_config: TrainingConfig) -> Any:
    """Creates BitsAndBytesConfig for 4-bit NF4 quantization."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.warning("BitsAndBytesConfig not available in transformers. Using standard precision.")
        return None

    hw = get_hardware_info()
    compute_dtype = torch.bfloat16 if hw["bf16_supported"] else torch.float16

    return BitsAndBytesConfig(
        load_in_4bit=training_config.load_in_4bit,
        bnb_4bit_quant_type=training_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=training_config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_qwen_vl_model(
    training_config: TrainingConfig,
    is_training: bool = True
) -> Tuple[Any, Any]:
    """Loads Qwen2.5-VL model and processor with 4-bit NF4 quantization and path resolution."""
    hw = get_hardware_info()
    device_map = "auto" if hw["cuda_available"] else None
    compute_dtype = torch.bfloat16 if hw["bf16_supported"] else (torch.float16 if hw["cuda_available"] else torch.float32)

    resolved_path = resolve_model_path(training_config.model_id)
    logger.info(f"Resolved model source: {resolved_path}")

    # Load processor using robust loader
    processor = load_qwen_processor(resolved_path)

    bnb_config = get_quantization_config(training_config) if (hw["cuda_available"] and training_config.load_in_4bit) else None

    logger.info(f"Loading weights from {resolved_path} (dtype={compute_dtype}, 4bit={bnb_config is not None})...")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": compute_dtype,
    }

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = device_map
    elif hw["cuda_available"]:
        model_kwargs["device_map"] = device_map

    model_cls = get_model_class()
    model = model_cls.from_pretrained(
        resolved_path,
        **model_kwargs
    )

    if is_training:
        model.config.use_cache = False
        if training_config.use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

    return model, processor
