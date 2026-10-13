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
    from transformers import AutoProcessor

    # Attempt 1: Standard AutoProcessor
    try:
        return AutoProcessor.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Standard AutoProcessor.from_pretrained failed ({e}). Assembling processor components directly...")

    # Attempt 2: Direct component assembly (ImageProcessor + Tokenizer -> Qwen2_5_VLProcessor)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    image_processor = None
    # Check for Qwen2_5_VLImageProcessor
    try:
        from transformers import Qwen2_5_VLImageProcessor
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(path, trust_remote_code=True)
    except Exception:
        pass

    # Check for Qwen2VLImageProcessor (internal backend class)
    if image_processor is None:
        try:
            from transformers import Qwen2VLImageProcessor
            image_processor = Qwen2VLImageProcessor.from_pretrained(path, trust_remote_code=True)
        except Exception:
            pass

    # Fallback to AutoImageProcessor
    if image_processor is None:
        try:
            from transformers import AutoImageProcessor
            image_processor = AutoImageProcessor.from_pretrained(path, trust_remote_code=True)
        except Exception:
            pass

    # Assemble unified processor
    try:
        from transformers import Qwen2_5_VLProcessor
        logger.info("Successfully assembled Qwen2_5_VLProcessor from components.")
        return Qwen2_5_VLProcessor(image_processor=image_processor, tokenizer=tokenizer)
    except Exception:
        pass

    try:
        from transformers import Qwen2VLProcessor
        logger.info("Successfully assembled Qwen2VLProcessor from components.")
        return Qwen2VLProcessor(image_processor=image_processor, tokenizer=tokenizer)
    except Exception:
        pass

    # Final retry with AutoProcessor
    return AutoProcessor.from_pretrained(path, trust_remote_code=True)


def get_model_class() -> Any:
    """Returns the appropriate Qwen2.5-VL conditional generation model class."""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    except ImportError:
        pass

    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    except ImportError:
        pass

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
