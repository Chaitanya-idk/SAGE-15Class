import pandas as pd
try:
    import pytest
except ImportError:
    pytest = None
from sage.config import load_classes_config, load_dataset_config
from sage.data.dedup import identify_duplicate_groups
from sage.data.sampling import perform_balanced_sampling


def test_zero_split_leakage():
    classes_cfg = load_classes_config("configs/classes.yaml")
    dataset_cfg = load_dataset_config("configs/dataset.yaml")
    dataset_cfg.target_train_images_per_class = 20

    # Create synthetic records with explicit duplicate groups (same base filename)
    rows = []
    for c in classes_cfg.classes:
        for i in range(10):
            # Same base image with multiple file variations
            rows.append({
                "crop": c.crop,
                "disease": c.disease,
                "source": "lab_a",
                "filename": f"sample_{i}.jpg",
                "raw_label": f"{c.crop}/{c.disease}"
            })
            rows.append({
                "crop": c.crop,
                "disease": c.disease,
                "source": "lab_a",
                "filename": f"sample_{i}_copy.jpg",
                "raw_label": f"{c.crop}/{c.disease}"
            })
            rows.append({
                "crop": c.crop,
                "disease": c.disease,
                "source": "lab_a",
                "filename": f"sample_{i} (1).jpg",
                "raw_label": f"{c.crop}/{c.disease}"
            })

    df = pd.DataFrame(rows)
    grouped_df = identify_duplicate_groups(df)
    sampled_df, counters = perform_balanced_sampling(grouped_df, classes_cfg, dataset_cfg)

    # Verify no group appears in more than 1 split
    splits_per_group = sampled_df.groupby("group_id")["split"].nunique()
    leaks = splits_per_group[splits_per_group > 1]
    assert len(leaks) == 0, f"Found {len(leaks)} groups that leaked across multiple splits!"
