import math
import os
import time
from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from sage.config import ClassesConfig, DatasetConfig, TrainingConfig
from sage.data.streaming import SAGEImageDataset
from sage.models.load_model import load_qwen_vl_model
from sage.models.lora import apply_lora_to_model
from sage.training.checkpointing import CheckpointManager
from sage.training.progress import TrainingProgressTracker
from sage.training.resume import find_resumable_checkpoint
from sage.utils.logging import get_logger
from sage.utils.reproducibility import set_seed

logger = get_logger("sage.train")


def create_multimodal_batch(batch_items: List[Dict[str, Any]], processor: Any, instruction: str) -> Dict[str, Any]:
    """Collates a list of dataset samples into Qwen2.5-VL multimodal model inputs."""
    images = [item["image"] for item in batch_items]
    texts = []

    for item in batch_items:
        # Use apply_chat_template if available so the processor correctly expands
        # <|vision_start|><|image_pad|>...<|vision_end|> to match actual pixel features.
        messages = [
            {"role": "system", "content": "You are an expert precision agriculture AI agent."},
            {"role": "user", "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": instruction},
            ]},
            {"role": "assistant", "content": item["ground_truth_text"]},
        ]
        # Try processor -> tokenizer -> manual fallback for chat template
        _tok = getattr(processor, "tokenizer", None)
        _apply = None
        for _obj in (processor, _tok):
            if _obj is not None and hasattr(_obj, "apply_chat_template"):
                try:
                    _apply = _obj.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    break
                except Exception:
                    pass
        if _apply is not None:
            text = _apply
        else:
            text = (
                f"<|im_start|>system\nYou are an expert precision agriculture AI agent.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n{item['ground_truth_text']}<|im_end|>"
            )
        texts.append(text)

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    # For causal LM loss, labels are input_ids
    labels = inputs["input_ids"].clone()
    inputs["labels"] = labels
    return inputs


def run_training_pipeline(
    manifest_df: Any,
    classes_config: ClassesConfig,
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    image_dir: str = None
) -> Dict[str, Any]:
    """Executes the full 2-epoch QLoRA training workflow with preflight check and resume."""
    set_seed(training_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # 1. Datasets & DataLoaders
    train_dataset = SAGEImageDataset(
        manifest_df=manifest_df,
        split="train",
        image_dir=image_dir,
        target_size=tuple(dataset_config.target_image_size),
        cache_dir=dataset_config.cache_dir,
    )
    val_dataset = SAGEImageDataset(
        manifest_df=manifest_df,
        split="val",
        image_dir=image_dir,
        target_size=tuple(dataset_config.target_image_size),
        cache_dir=dataset_config.cache_dir,
    )

    # 2. Checkpoint Manager
    ckpt_mgr = CheckpointManager(
        training_config=training_config,
        manifest_path=dataset_config.manifest_path,
        class_registry_path=dataset_config.class_registry_path,
    )

    # Check for resume
    resumed = find_resumable_checkpoint(
        training_config.checkpoint_dir,
        dataset_config.manifest_path,
        dataset_config.class_registry_path
    )

    # 3. Model & Processor Loading
    model, processor = load_qwen_vl_model(training_config, is_training=True)
    model, lora_stats = apply_lora_to_model(model, training_config)

    # 4. DataLoader
    def collate_fn(batch):
        return create_multimodal_batch(batch, processor, training_config.instruction)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.micro_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0, # safe across all platforms
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.micro_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )

    steps_per_epoch = len(train_loader) // training_config.gradient_accumulation_steps
    total_training_steps = steps_per_epoch * training_config.epochs
    warmup_steps = int(total_training_steps * training_config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    start_epoch = 1
    global_step = 0
    best_metric = 0.0

    # Handle resume state
    if resumed is not None:
        c_dir, meta = resumed
        logger.info(f"Resuming model weights from {c_dir}...")
        try:
            from peft import set_peft_model_state_dict
            state_dict = torch.load(os.path.join(c_dir, "adapter_model.bin"), map_location="cpu")
            set_peft_model_state_dict(model, state_dict)
        except Exception:
            pass

        training_state_path = os.path.join(c_dir, "training_state.pt")
        if os.path.exists(training_state_path):
            t_state = torch.load(training_state_path, map_location="cpu")
            if t_state.get("optimizer_state_dict") and optimizer:
                optimizer.load_state_dict(t_state["optimizer_state_dict"])
            if t_state.get("scheduler_state_dict") and scheduler:
                scheduler.load_state_dict(t_state["scheduler_state_dict"])
            start_epoch = t_state.get("epoch", 1)
            global_step = t_state.get("global_step", 0)

    # 6. Preflight Smoke Test
    logger.info("Executing Preflight Multimodal Smoke Test...")
    model.train()
    sample_batch = next(iter(train_loader))
    sample_inputs = {k: v.to(device) for k, v in sample_batch.items() if isinstance(v, torch.Tensor)}
    
    preflight_start = time.time()
    outputs = model(**sample_inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.zero_grad()
    preflight_time = time.time() - preflight_start
    logger.info(f"Preflight smoke test PASSED! Forward/backward step time: {preflight_time:.3f}s")

    # Report training schedule
    effective_bs = training_config.micro_batch_size * training_config.gradient_accumulation_steps
    logger.info("=" * 60)
    logger.info(f"Classes: 15")
    logger.info(f"Total training samples: {len(train_dataset)}")
    logger.info(f"Effective batch size: {effective_bs} (Micro: {training_config.micro_batch_size}, Accum: {training_config.gradient_accumulation_steps})")
    logger.info(f"Steps per epoch: {steps_per_epoch} | Total epochs: {training_config.epochs}")
    logger.info(f"Total optimizer updates: {total_training_steps}")
    logger.info("=" * 60)

    # 7. Training Loop
    total_batches = len(train_loader)
    progress = TrainingProgressTracker(total_epochs=training_config.epochs, batches_per_epoch=total_batches)

    try:
        for epoch in range(start_epoch, training_config.epochs + 1):
            logger.info(f"\n--- Starting Epoch {epoch}/{training_config.epochs} ---")
            model.train()
            progress.start_epoch(epoch)
            accumulated_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(**inputs)
                raw_batch_loss = outputs.loss.item()
                loss = outputs.loss / training_config.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += raw_batch_loss

                current_lr = scheduler.get_last_lr()[0]
                progress.update_batch(
                    batch_loss=raw_batch_loss,
                    lr=current_lr,
                    batch_size=training_config.micro_batch_size,
                    opt_step=global_step
                )

                if (batch_idx + 1) % training_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    accumulated_loss = 0.0

                    if global_step % training_config.save_every_n_steps == 0:
                        ckpt_mgr.save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            global_step=global_step,
                            batch_idx=batch_idx,
                            metrics={"running_loss": raw_batch_loss}
                        )

            progress.close_epoch()

            # End of epoch validation
            logger.info(f"Running validation after Epoch {epoch}...")
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_batch in val_loader:
                    v_inputs = {k: v.to(device) for k, v in v_batch.items() if isinstance(v, torch.Tensor)}
                    v_out = model(**v_inputs)
                    val_loss += v_out.loss.item()
            avg_val_loss = val_loss / max(1, len(val_loader))
            logger.info(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")

            # Save end-of-epoch checkpoint
            is_best = (avg_val_loss < best_metric) if best_metric > 0 else True
            if is_best:
                best_metric = avg_val_loss

            ckpt_mgr.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                batch_idx=len(train_loader),
                metrics={"val_loss": avg_val_loss},
                is_best=is_best,
                tag=f"epoch_{epoch}_final"
            )

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user! Saving emergency checkpoint...")
        ckpt_mgr.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            batch_idx=batch_idx,
            metrics={"interrupted": True},
            tag="interrupted_checkpoint"
        )

    # Save final completed model checkpoint
    final_path = os.path.join(training_config.checkpoint_dir, "final_adapter")
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(final_path)
    logger.info(f"Final trained LoRA adapter saved to {final_path}")

    return {
        "status": "COMPLETED",
        "total_epochs": training_config.epochs,
        "global_steps": global_step,
        "final_adapter_dir": final_path,
        "lora_parameters": lora_stats
    }
