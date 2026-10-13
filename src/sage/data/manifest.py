from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, Tuple
import pandas as pd
from sage.config import ClassesConfig, DatasetConfig
from sage.data.dedup import identify_duplicate_groups
from sage.data.sampling import ClassCounter, perform_balanced_sampling
from sage.utils.hashing import compute_string_hash


def build_sample_manifest(
    metadata_df: pd.DataFrame,
    classes_config: ClassesConfig,
    dataset_config: DatasetConfig,
    dataset_revision: str = "main",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Builds the complete reproducible sample manifest and all required metadata artifacts."""
    os.makedirs(dataset_config.output_dir, exist_ok=True)

    # 1. Deduplication / grouping
    grouped_df = identify_duplicate_groups(metadata_df)

    # 2. Balanced group-aware sampling
    sampled_df, counters = perform_balanced_sampling(grouped_df, classes_config, dataset_config)

    # 3. Add stable sample IDs and metadata columns
    sample_ids = []
    for idx, row in sampled_df.iterrows():
        key = f"{row['source']}::{row['filename']}::{row['crop']}::{row['disease']}::{idx}"
        sample_ids.append(f"sage_{compute_string_hash(key, algorithm='sha256')[:16]}")

    sampled_df["sample_id"] = sample_ids
    sampled_df["dataset_revision"] = dataset_revision
    sampled_df["selection_seed"] = dataset_config.seed

    manifest_columns = [
        "sample_id",
        "class_id",
        "class_name",
        "canonical_crop",
        "canonical_disease",
        "raw_label",
        "filename",
        "source",
        "group_id",
        "split",
        "selection_seed",
        "dataset_revision",
    ]
    manifest_df = sampled_df[manifest_columns].copy()

    # Save sample_manifest.parquet
    manifest_df.to_parquet(dataset_config.manifest_path, index=False)

    # 4. Generate class_registry.json & class_selection.csv
    class_registry = []
    for cid, c in sorted(counters.items()):
        class_registry.append({
            "class_id": c.class_id,
            "crop": c.crop,
            "disease": c.disease,
            "class_name": f"{c.crop}::{c.disease}",
            "available_records": c.discovered,
            "valid_unique_groups": c.valid_unique,
            "duplicate_records": c.duplicate_groups,
            "selected_total": c.selected_total,
            "train_count": c.train_count,
            "val_count": c.val_count,
            "test_count": c.test_count,
            "selection_status": "APPROVED",
            "target_train_met": c.train_count >= dataset_config.target_train_images_per_class
        })

    with open(dataset_config.class_registry_path, "w", encoding="utf-8") as f:
        json.dump(class_registry, f, indent=2)

    pd.DataFrame(class_registry).to_csv(dataset_config.class_selection_path, index=False)

    # 5. Generate split_summary.json
    split_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_selected_samples": len(manifest_df),
        "split_counts": manifest_df["split"].value_counts().to_dict(),
        "classes_count": len(class_registry),
        "per_class_split_distribution": manifest_df.groupby(["class_name", "split"]).size().unstack(fill_value=0).to_dict(orient="index")
    }
    with open(dataset_config.split_summary_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    # 6. Generate data_quality_report.md
    report_md = f"""# SAGE Dataset Audit and Quality Report

**Generated UTC:** {split_summary['timestamp_utc']}  
**Dataset Source:** `{dataset_config.hf_repo_id}`  
**Dataset Revision:** `{dataset_revision}`  
**Selection Seed:** `{dataset_config.seed}`  

## 1. Class Registry & Distribution

| Class ID | Class Name | Discovered | Train Count | Val Count | Test Count | Total Selected | Target Met |
|---|---|---|---|---|---|---|---|
"""
    for item in class_registry:
        report_md += f"| {item['class_id']} | `{item['class_name']}` | {item['available_records']:,} | {item['train_count']:,} | {item['val_count']:,} | {item['test_count']:,} | {item['selected_total']:,} | {'Yes' if item['target_train_met'] else 'No'} |\n"

    report_md += f"""
## 2. Split Summary
- **Total Selected Records:** {len(manifest_df):,}
- **Train Split (70%):** {split_summary['split_counts'].get('train', 0):,}
- **Validation Split (15%):** {split_summary['split_counts'].get('val', 0):,}
- **Test Split (15%):** {split_summary['split_counts'].get('test', 0):,}

## 3. Data Integrity & Leakage Prevention
- **Group Leakage Check:** Group IDs are assigned per source + base filename.
- Group-level stratification guarantees that exact duplicates and near-duplicates do not cross splits.
- Zero image binary payloads are stored in this metadata manifest to ensure fast, scalable loading.
"""
    with open(dataset_config.data_quality_report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    return manifest_df, split_summary


def validate_sample_manifest(manifest_path: str) -> Dict[str, Any]:
    """Validates the sample manifest for integrity, completeness, and split leakage."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_parquet(manifest_path)
    issues = []

    # Check required columns
    required_cols = ["sample_id", "class_id", "class_name", "split", "group_id"]
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    # Check class coverage
    num_classes = df["class_id"].nunique()
    if num_classes != 15:
        issues.append(f"Expected 15 classes, found {num_classes}")

    # Check split leakage across groups
    group_split_counts = df.groupby("group_id")["split"].nunique()
    leaked_groups = group_split_counts[group_split_counts > 1]
    if len(leaked_groups) > 0:
        issues.append(f"SPLIT LEAKAGE DETECTED: {len(leaked_groups)} groups appear in multiple splits!")

    # Check nulls
    null_counts = df[required_cols].isna().sum().to_dict()
    for col, n_nulls in null_counts.items():
        if n_nulls > 0:
            issues.append(f"Column '{col}' has {n_nulls} null values")

    return {
        "valid": len(issues) == 0,
        "total_samples": len(df),
        "num_classes": num_classes,
        "splits": df["split"].value_counts().to_dict(),
        "leaked_groups_count": len(leaked_groups),
        "issues": issues
    }
