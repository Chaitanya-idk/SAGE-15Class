from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, Tuple
import pandas as pd
from sage.config import ClassesConfig, DatasetConfig
from sage.data.labels import match_class
from sage.data.schema import SAGE_COLUMN_TYPES, validate_dataframe_schema


def audit_dataset_inventory(
    metadata_df: pd.DataFrame,
    classes_config: ClassesConfig,
    dataset_config: DatasetConfig,
    num_shards: int = 48,
    dataset_revision: str = "main"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Performs full schema audit and inventory scan, saving dataset_schema.json and dataset_inventory.json."""
    os.makedirs(dataset_config.output_dir, exist_ok=True)
    schema_path = os.path.join(dataset_config.output_dir, "dataset_schema.json")
    inventory_path = os.path.join(dataset_config.output_dir, "dataset_inventory.json")

    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Dataset schema artifact
    schema_validation = validate_dataframe_schema(metadata_df)
    schema_artifact = {
        "dataset_name": dataset_config.hf_repo_id,
        "dataset_revision": dataset_revision,
        "discovery_timestamp_utc": timestamp,
        "column_types": SAGE_COLUMN_TYPES,
        "present_columns": schema_validation["present_columns"],
        "total_records_discovered": len(metadata_df),
        "schema_valid": schema_validation["valid"],
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_artifact, f, indent=2)

    # 2. Inventory counts
    valid_crops = metadata_df["crop"].notna() & (metadata_df["crop"].str.strip() != "")
    valid_diseases = metadata_df["disease"].notna() & (metadata_df["disease"].str.strip() != "")
    valid_both = valid_crops & valid_diseases

    # Candidate class counts
    class_counts = {}
    for item in classes_config.classes:
        class_counts[item.class_name] = 0

    for _, row in metadata_df.iterrows():
        c_item = match_class(row.get("crop"), row.get("disease"), row.get("raw_label"), classes_config)
        if c_item is not None:
            class_counts[c_item.class_name] = class_counts.get(c_item.class_name, 0) + 1

    missing_fields_counts = metadata_df.isna().sum().to_dict()

    inventory_artifact = {
        "discovery_timestamp_utc": timestamp,
        "dataset_name": dataset_config.hf_repo_id,
        "dataset_revision": dataset_revision,
        "total_shards_reported": num_shards,
        "total_records_enumerated": len(metadata_df),
        "records_with_valid_crop": int(valid_crops.sum()),
        "records_with_valid_disease": int(valid_diseases.sum()),
        "records_with_valid_crop_and_disease": int(valid_both.sum()),
        "missing_fields_breakdown": {k: int(v) for k, v in missing_fields_counts.items()},
        "candidate_class_record_counts": class_counts,
        "unique_crops_count": int(metadata_df["crop"].nunique()),
        "unique_diseases_count": int(metadata_df["disease"].nunique()),
        "inventory_status": "COMPLETE_EXACT_SCAN"
    }
    with open(inventory_path, "w", encoding="utf-8") as f:
        json.dump(inventory_artifact, f, indent=2)

    return schema_artifact, inventory_artifact
