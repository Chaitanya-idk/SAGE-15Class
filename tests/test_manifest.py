import os
import pandas as pd
try:
    import pytest
except ImportError:
    pytest = None
from sage.config import load_classes_config, load_dataset_config
from sage.data.manifest import build_sample_manifest, validate_sample_manifest


def test_build_and_validate_manifest_synthetic(tmp_path):
    classes_cfg = load_classes_config("configs/classes.yaml")
    dataset_cfg = load_dataset_config("configs/dataset.yaml")

    # Override paths to tmp_path
    dataset_cfg.output_dir = str(tmp_path / "metadata")
    dataset_cfg.manifest_path = str(tmp_path / "metadata" / "sample_manifest.parquet")
    dataset_cfg.class_registry_path = str(tmp_path / "metadata" / "class_registry.json")
    dataset_cfg.class_selection_path = str(tmp_path / "metadata" / "class_selection.csv")
    dataset_cfg.split_summary_path = str(tmp_path / "metadata" / "split_summary.json")
    dataset_cfg.data_quality_report_path = str(tmp_path / "metadata" / "data_quality_report.md")
    dataset_cfg.target_train_images_per_class = 10

    # Create synthetic dataframe with all 15 classes
    rows = []
    for c in classes_cfg.classes:
        for i in range(25):
            rows.append({
                "crop": c.crop,
                "disease": c.disease,
                "source": "synthetic_source",
                "filename": f"{c.crop}_{c.disease}_{i}.jpg",
                "raw_label": f"{c.crop}/{c.disease.replace(' ', '_')}"
            })
    df_synthetic = pd.DataFrame(rows)

    manifest_df, split_summary = build_sample_manifest(
        metadata_df=df_synthetic,
        classes_config=classes_cfg,
        dataset_config=dataset_cfg
    )

    assert os.path.exists(dataset_cfg.manifest_path)
    assert os.path.exists(dataset_cfg.class_registry_path)
    assert os.path.exists(dataset_cfg.split_summary_path)
    assert os.path.exists(dataset_cfg.data_quality_report_path)

    # Validate
    validation_res = validate_sample_manifest(dataset_cfg.manifest_path)
    assert validation_res["valid"] is True
    assert validation_res["num_classes"] == 15
    assert validation_res["leaked_groups_count"] == 0
