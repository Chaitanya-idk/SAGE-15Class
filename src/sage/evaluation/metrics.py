from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score


def calculate_classification_metrics(
    ground_truth_crops: List[str],
    ground_truth_diseases: List[str],
    ground_truth_classes: List[str],
    predicted_crops: List[Optional[str]],
    predicted_diseases: List[Optional[str]],
    predicted_classes: List[Optional[str]],
    all_class_names: List[str]
) -> Dict[str, Any]:
    """Computes comprehensive multi-tiered classification metrics."""
    total_samples = len(ground_truth_classes)
    if total_samples == 0:
        return {}

    # 1. Parsing failures
    parsing_failures = sum(1 for c, d in zip(predicted_crops, predicted_diseases) if c is None or d is None)

    # 2. Crop-only accuracy
    crop_correct = sum(
        1 for gt, pred in zip(ground_truth_crops, predicted_crops)
        if pred is not None and str(gt).strip().lower() == str(pred).strip().lower()
    )
    crop_accuracy = crop_correct / total_samples

    # 3. Disease-only accuracy
    disease_correct = sum(
        1 for gt, pred in zip(ground_truth_diseases, predicted_diseases)
        if pred is not None and str(gt).strip().lower() == str(pred).strip().lower()
    )
    disease_accuracy = disease_correct / total_samples

    # 4. Exact match accuracy (Crop AND Disease correct)
    exact_correct = sum(
        1 for gt, pred in zip(ground_truth_classes, predicted_classes)
        if pred is not None and str(gt).strip().lower() == str(pred).strip().lower()
    )
    exact_accuracy = exact_correct / total_samples

    # For sklearn reports, replace None predictions with "__PARSE_FAILURE__"
    clean_preds = [p if p is not None else "__PARSE_FAILURE__" for p in predicted_classes]
    unique_labels = sorted(list(set(ground_truth_classes) | set(clean_preds)))

    # Classification report dict
    report_dict = classification_report(
        ground_truth_classes,
        clean_preds,
        labels=all_class_names,
        output_dict=True,
        zero_division=0
    )

    macro_f1 = f1_score(ground_truth_classes, clean_preds, labels=all_class_names, average="macro", zero_division=0)
    weighted_f1 = f1_score(ground_truth_classes, clean_preds, labels=all_class_names, average="weighted", zero_division=0)
    macro_precision = precision_score(ground_truth_classes, clean_preds, labels=all_class_names, average="macro", zero_division=0)
    macro_recall = recall_score(ground_truth_classes, clean_preds, labels=all_class_names, average="macro", zero_division=0)

    # Per-class metrics table
    per_class_metrics = []
    for c_name in all_class_names:
        c_stats = report_dict.get(c_name, {})
        per_class_metrics.append({
            "class_name": c_name,
            "precision": round(c_stats.get("precision", 0.0), 4),
            "recall": round(c_stats.get("recall", 0.0), 4),
            "f1_score": round(c_stats.get("f1-score", 0.0), 4),
            "support": int(c_stats.get("support", 0))
        })

    return {
        "total_samples": total_samples,
        "parsing_failures": parsing_failures,
        "parsing_failure_rate": round(parsing_failures / total_samples, 4),
        "overall_accuracy": round(exact_accuracy, 4),
        "exact_match_accuracy": round(exact_accuracy, 4),
        "crop_accuracy": round(crop_accuracy, 4),
        "disease_accuracy": round(disease_accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_metrics": per_class_metrics,
        "classification_report_raw": report_dict,
    }
