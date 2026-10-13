import os
import pandas as pd
try:
    import pytest
except ImportError:
    pytest = None
from sage.config import load_classes_config
from sage.data.labels import match_class


def test_classes_config_integrity():
    cfg = load_classes_config("configs/classes.yaml")
    assert cfg.target_classes_count == 15
    assert len(cfg.classes) == 15

    # Check distinct deterministic class IDs 0 to 14
    class_ids = [c.class_id for c in cfg.classes]
    assert sorted(class_ids) == list(range(15))

    # Check no duplicate class names
    class_names = [c.class_name for c in cfg.classes]
    assert len(set(class_names)) == 15


@pytest.mark.skipif(not os.path.exists("sage_metadata.parquet"), reason="sage_metadata.parquet not downloaded locally")
def test_all_15_classes_exist_in_metadata():
    cfg = load_classes_config("configs/classes.yaml")
    df = pd.read_parquet("sage_metadata.parquet")

    found_counts = {c.class_id: 0 for c in cfg.classes}
    for _, row in df.iterrows():
        c = match_class(row.get("crop"), row.get("disease"), row.get("raw_label"), cfg)
        if c is not None:
            found_counts[c.class_id] += 1

    for c in cfg.classes:
        count = found_counts[c.class_id]
        assert count >= c.min_required_samples, (
            f"Class {c.class_name} (ID {c.class_id}) has only {count} records, below required {c.min_required_samples}"
        )
