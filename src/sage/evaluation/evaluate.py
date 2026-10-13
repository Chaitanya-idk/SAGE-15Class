import os
from typing import Any, Dict, List
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sage.config import ClassesConfig, EvaluationConfig
from sage.data.labels import parse_model_prediction
from sage.data.streaming import SAGEImageDataset
from sage.evaluation.confusion import compute_confusion_matrix, find_top_confusions, plot_confusion_matrix
from sage.evaluation.metrics import calculate_classification_metrics
from sage.evaluation.report import generate_final_report
from sage.utils.hardware import get_hardware_info
from sage.utils.logging import get_logger

logger = get_logger("sage.evaluate")


def run_evaluation_pipeline(
    model: Any,
    processor: Any,
    manifest_df: pd.DataFrame,
    classes_config: ClassesConfig,
    eval_config: EvaluationConfig,
    split: str = "test",
    image_dir: str = None
) -> Dict[str, Any]:
    """Runs evaluation over the specified split, computes multi-tier metrics, and exports reports."""
    os.makedirs(eval_config.predictions_dir, exist_ok=True)
    os.makedirs(eval_config.reports_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    eval_dataset = SAGEImageDataset(
        manifest_df=manifest_df,
        split=split,
        image_dir=image_dir,
    )

    all_class_names = [c.class_name for c in classes_config.classes]

    records = []
    logger.info(f"Evaluating {len(eval_dataset)} samples on split '{split}'...")

    for idx in tqdm(range(len(eval_dataset)), desc=f"Evaluating {split}"):
        item = eval_dataset[idx]
        image = item["image"]
        ground_truth_crop = item["crop"]
        ground_truth_disease = item["disease"]
        ground_truth_class = item["class_name"]

        prompt = (
            f"<|im_start|>system\nYou are an expert precision agriculture AI agent.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Identify the crop and plant disease visible in the image. Return the output strictly in the format:\nCrop: <Crop>\nDisease: <Disease><|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=eval_config.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
            )

        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_text = processor.decode(gen_ids, skip_special_tokens=True).strip()

        parsed_crop, parsed_disease = parse_model_prediction(raw_text)
        predicted_class = f"{parsed_crop}::{parsed_disease}" if parsed_crop and parsed_disease else None

        records.append({
            "sample_id": item["sample_id"],
            "ground_truth_crop": ground_truth_crop,
            "ground_truth_disease": ground_truth_disease,
            "ground_truth_class": ground_truth_class,
            "raw_output": raw_text,
            "predicted_crop": parsed_crop,
            "predicted_disease": parsed_disease,
            "predicted_class": predicted_class,
            "exact_match": (ground_truth_class == predicted_class)
        })

    preds_df = pd.DataFrame(records)

    # Save test_predictions.parquet
    preds_df.to_parquet(eval_config.test_predictions_parquet, index=False)
    logger.info(f"Saved predictions to {eval_config.test_predictions_parquet}")

    # Metrics
    metrics = calculate_classification_metrics(
        ground_truth_crops=preds_df["ground_truth_crop"].tolist(),
        ground_truth_diseases=preds_df["ground_truth_disease"].tolist(),
        ground_truth_classes=preds_df["ground_truth_class"].tolist(),
        predicted_crops=preds_df["predicted_crop"].tolist(),
        predicted_diseases=preds_df["predicted_disease"].tolist(),
        predicted_classes=preds_df["predicted_class"].tolist(),
        all_class_names=all_class_names
    )

    # Confusion matrix
    clean_preds = [p if p is not None else "__PARSE_FAILURE__" for p in preds_df["predicted_class"]]
    cm = compute_confusion_matrix(preds_df["ground_truth_class"].tolist(), clean_preds, labels=all_class_names)
    plot_confusion_matrix(cm, labels=all_class_names, output_path=eval_config.confusion_matrix_png)
    top_confusions = find_top_confusions(cm, labels=all_class_names, top_k=10)

    # Failures
    failures_df = preds_df[~preds_df["exact_match"]].copy()

    # Generate Reports
    hw_info = get_hardware_info()
    generate_final_report(
        eval_metrics=metrics,
        top_confusions=top_confusions,
        eval_config=eval_config,
        failures_df=failures_df,
        hardware_info=hw_info
    )

    logger.info(f"Evaluation complete! Exact Match Accuracy: {metrics.get('exact_match_accuracy', 0.0):.4f}")
    return metrics
