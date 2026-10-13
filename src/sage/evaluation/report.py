from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional
import pandas as pd
from sage.config import EvaluationConfig


def generate_final_report(
    eval_metrics: Dict[str, Any],
    top_confusions: List[Dict[str, Any]],
    eval_config: EvaluationConfig,
    failures_df: Optional[pd.DataFrame] = None,
    hardware_info: Optional[Dict[str, Any]] = None,
    training_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Generates all final report artifacts in Markdown, JSON, and CSV formats."""
    os.makedirs(eval_config.reports_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Save CSVs
    # Per-class metrics
    if "per_class_metrics" in eval_metrics:
        pd.DataFrame(eval_metrics["per_class_metrics"]).to_csv(eval_config.class_metrics_csv, index=False)

    # Classification report
    if "classification_report_raw" in eval_metrics:
        pd.DataFrame(eval_metrics["classification_report_raw"]).T.to_csv(eval_config.classification_report_csv)

    # Top confusions
    pd.DataFrame(top_confusions).to_csv(eval_config.top_confusions_csv, index=False)

    # Failures
    if failures_df is not None and not failures_df.empty:
        failures_df.to_csv(eval_config.failures_csv, index=False)
    else:
        pd.DataFrame(columns=["sample_id", "true_class", "raw_output", "parsed_crop", "parsed_disease"]).to_csv(eval_config.failures_csv, index=False)

    # 2. Save JSON
    final_json = {
        "timestamp_utc": timestamp,
        "evaluation_metrics": eval_metrics,
        "top_confusions": top_confusions,
        "hardware_info": hardware_info or {},
        "training_summary": training_summary or {},
    }
    with open(eval_config.final_report_json, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)

    # 3. Save Markdown Final Report
    md = f"""# SAGE 15-Class Crop Disease Diagnosis Final Report

**Evaluation Timestamp (UTC):** {timestamp}  
**Model Architecture:** `Qwen/Qwen2.5-VL-3B-Instruct` (QLoRA 4-bit NF4)  

---

## 1. Executive Performance Summary

| Metric | Score |
|---|---|
| **Exact Match Accuracy (Crop & Disease)** | **{eval_metrics.get('exact_match_accuracy', 0.0):.4f}** |
| **Crop Identification Accuracy** | **{eval_metrics.get('crop_accuracy', 0.0):.4f}** |
| **Disease Diagnosis Accuracy** | **{eval_metrics.get('disease_accuracy', 0.0):.4f}** |
| **Macro-Averaged F1 Score** | **{eval_metrics.get('macro_f1', 0.0):.4f}** |
| **Weighted F1 Score** | **{eval_metrics.get('weighted_f1', 0.0):.4f}** |
| **Macro Precision** | **{eval_metrics.get('macro_precision', 0.0):.4f}** |
| **Macro Recall** | **{eval_metrics.get('macro_recall', 0.0):.4f}** |
| **Parsing Failure Rate** | **{eval_metrics.get('parsing_failure_rate', 0.0):.2%}** ({eval_metrics.get('parsing_failures', 0)} / {eval_metrics.get('total_samples', 0)}) |

---

## 2. Per-Class Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
"""
    for row in eval_metrics.get("per_class_metrics", []):
        md += f"| `{row['class_name']}` | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} | {row['support']} |\n"

    md += f"""
---

## 3. Top Confused Class Pairs

| True Class | Predicted Class | Count |
|---|---|---|
"""
    for pair in top_confusions[:8]:
        md += f"| `{pair['true_class']}` | `{pair['predicted_class']}` | {pair['count']} |\n"

    md += f"""
---

## 4. Hardware & System Context
- **GPU Devices:** {hardware_info.get('devices', [{'name': 'None'}])[0].get('name', 'N/A') if hardware_info else 'N/A'}
- **BF16 Hardware Support:** {hardware_info.get('bf16_supported', False) if hardware_info else 'N/A'}
- **Environment:** {'Kaggle GPU Container' if hardware_info and hardware_info.get('is_kaggle') else 'Cloud/Local'}

---

## 5. Artifacts and Output Files
- Metrics Report: `artifacts/reports/final_report.json`
- Class Metrics CSV: `artifacts/reports/class_metrics.csv`
- Confusion Matrix Heatmap: `artifacts/reports/confusion_matrix.png`
- Detailed Predictions: `artifacts/predictions/test_predictions.parquet`
"""
    with open(eval_config.final_report_md, "w", encoding="utf-8") as f:
        f.write(md)
