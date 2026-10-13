from typing import Any, Dict, Tuple
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sage.config import TrainingConfig
from sage.utils.logging import get_logger

logger = get_logger("sage.lora")


def apply_lora_to_model(model: Any, training_config: TrainingConfig) -> Tuple[Any, Dict[str, Any]]:
    """Applies PEFT LoRA configuration to the model and prepares for k-bit training."""
    if training_config.load_in_4bit:
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_config.use_gradient_checkpointing
        )

    lora_config = LoraConfig(
        r=training_config.lora_r,
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=training_config.target_modules,
    )

    logger.info(f"Applying LoRA adapter (r={training_config.lora_r}, alpha={training_config.lora_alpha})...")
    model = get_peft_model(model, lora_config)

    # Calculate parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params if total_params > 0 else 0

    stats = {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percentage": round(trainable_pct, 4),
        "target_modules": training_config.target_modules,
    }

    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)"
    )

    return model, stats
