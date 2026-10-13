from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sage.config import ClassesConfig, DatasetConfig
from sage.data.labels import match_class


@dataclass
class ClassCounter:
    class_id: int
    crop: str
    disease: str
    discovered: int = 0
    valid_labels: int = 0
    valid_unique: int = 0
    duplicate_groups: int = 0
    selected_total: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0


def perform_balanced_sampling(
    df: pd.DataFrame,
    classes_config: ClassesConfig,
    dataset_config: DatasetConfig,
) -> Tuple[pd.DataFrame, Dict[int, ClassCounter]]:
    """Performs deterministic balanced sampling targeting 800 train images per class.
    
    Uses group-aware splitting to guarantee no duplicate group crosses train/val/test splits.
    """
    rng = np.random.RandomState(dataset_config.seed)
    counters: Dict[int, ClassCounter] = {}

    for item in classes_config.classes:
        counters[item.class_id] = ClassCounter(
            class_id=item.class_id,
            crop=item.crop,
            disease=item.disease
        )

    # Pre-filter crops to approved crops for massive speedup
    valid_crops = set()
    for item in classes_config.classes:
        valid_crops.add(item.crop.lower())
        for a in item.aliases.get("crops", []):
            valid_crops.add(a.lower())

    df = df.copy()
    crop_series = df["crop"].fillna("").astype(str).str.lower()
    df = df[crop_series.isin(valid_crops)].copy()

    # Fast row matching using itertuples
    matched_class_ids = []
    crops = df["crop"].tolist()
    diseases = df["disease"].tolist()
    raw_labels = df["raw_label"].tolist()

    for c, d, r in zip(crops, diseases, raw_labels):
        c_item = match_class(
            crop=c,
            disease=d,
            raw_label=r,
            classes_config=classes_config
        )
        if c_item is not None:
            matched_class_ids.append(c_item.class_id)
            counters[c_item.class_id].discovered += 1
            counters[c_item.class_id].valid_labels += 1
        else:
            matched_class_ids.append(-1)

    df = df.copy()
    df["class_id"] = matched_class_ids
    filtered_df = df[df["class_id"] >= 0].copy()

    selected_rows = []

    # Process each class separately to ensure balanced representation
    for item in classes_config.classes:
        cid = item.class_id
        class_df = filtered_df[filtered_df["class_id"] == cid].copy()
        c = counters[cid]

        # Group by group_id
        unique_groups = np.array(class_df["group_id"].unique())
        c.duplicate_groups = len(class_df) - len(unique_groups)
        c.valid_unique = len(unique_groups)

        # Shuffle groups deterministically
        rng.shuffle(unique_groups)

        # Calculate target counts
        target_train = dataset_config.target_train_images_per_class
        train_ratio = dataset_config.train_split
        val_ratio = dataset_config.val_split
        test_ratio = dataset_config.test_split

        # Total needed: target_train / train_ratio
        target_total = int(math.ceil(target_train / train_ratio))

        # Select groups until we reach target_total or run out of groups
        selected_group_list = []
        accumulated_count = 0
        for gid in unique_groups:
            group_rows = class_df[class_df["group_id"] == gid]
            selected_group_list.append(gid)
            accumulated_count += len(group_rows)
            if accumulated_count >= target_total:
                break

        # Now allocate selected groups to train, val, test
        # Determine number of groups for train, val, test
        num_sel_groups = len(selected_group_list)
        n_train_groups = int(math.floor(num_sel_groups * train_ratio))
        n_val_groups = int(math.floor(num_sel_groups * val_ratio))
        n_test_groups = num_sel_groups - n_train_groups - n_val_groups

        # Ensure at least 1 group in val and test if possible
        if n_val_groups == 0 and num_sel_groups >= 3:
            n_val_groups = 1
            n_train_groups -= 1
        if n_test_groups == 0 and num_sel_groups >= 3:
            n_test_groups = 1
            n_train_groups -= 1

        train_groups = set(selected_group_list[:n_train_groups])
        val_groups = set(selected_group_list[n_train_groups:n_train_groups + n_val_groups])
        test_groups = set(selected_group_list[n_train_groups + n_val_groups:])

        for gid in selected_group_list:
            g_df = class_df[class_df["group_id"] == gid].copy()
            if gid in train_groups:
                g_df["split"] = "train"
                c.train_count += len(g_df)
            elif gid in val_groups:
                g_df["split"] = "val"
                c.val_count += len(g_df)
            else:
                g_df["split"] = "test"
                c.test_count += len(g_df)

            selected_rows.append(g_df)

        c.selected_total = c.train_count + c.val_count + c.test_count

    if not selected_rows:
        raise ValueError("No records matched the approved class configuration.")

    sampled_df = pd.concat(selected_rows, ignore_index=True)
    sampled_df["canonical_crop"] = sampled_df["class_id"].map(lambda cid: counters[cid].crop)
    sampled_df["canonical_disease"] = sampled_df["class_id"].map(lambda cid: counters[cid].disease)
    sampled_df["class_name"] = sampled_df["canonical_crop"] + "::" + sampled_df["canonical_disease"]

    return sampled_df, counters
