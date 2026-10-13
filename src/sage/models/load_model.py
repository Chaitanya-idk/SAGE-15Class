from typing import Any, Dict, Tuple
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from sage.config import TrainingConfig
from sage.utils.hardware import get_hardware_info
from sage.utils.logging import get_logger

logger = get_logger("sage.models")


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
    """Loads Qwen2.5-VL-3B-Instruct model and processor with 4-bit NF4 QLoRA quantization.
    
    Returns:
        (model, processor)
    """
    hw = get_hardware_info()
    device_map = "auto" if hw["cuda_available"] else None
    compute_dtype = torch.bfloat16 if hw["bf16_supported"] else (torch.float16 if hw["cuda_available"] else torch.float32)

    logger.info(f"Loading processor for {training_config.model_id}...")
    processor = AutoProcessor.from_pretrained(
        training_config.model_id,
        trust_remote_code=True,
    )

    bnb_config = get_quantization_config(training_config) if (hw["cuda_available"] and training_config.load_in_4bit) else None

    logger.info(f"Loading {training_config.model_id} (dtype={compute_dtype}, 4bit={bnb_config is not None})...")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": compute_dtype,
    }

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = device_map
    elif hw["cuda_available"]:
        model_kwargs["device_map"] = device_map

    model = AutoModelForVision2Seq.from_pretrained(
        training_config.model_id,
        **model_kwargs
    )

    if is_training:
        model.config.use_cache = False
        if training_config.use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

    return model, processor
