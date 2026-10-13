import hashlib
import re
from typing import List
import pandas as pd


def extract_base_filename(filename: str) -> str:
    """Extracts base filename removing extension, resolution suffixes, or duplicate marks."""
    if not filename or pd.isna(filename):
        return "unknown"
    base = re.sub(r"\.[a-zA-Z0-9]+$", "", str(filename).strip())
    base = re.sub(r"[\(_\s]+(copy|\d+)[\)\s]*$", "", base, flags=re.IGNORECASE)
    return base.strip()


def identify_duplicate_groups(df: pd.DataFrame, group_columns: List[str] = None) -> pd.DataFrame:
    """Fast vectorized group_id assignment to prevent split leakage across duplicate specimens."""
    df = df.copy()

    source_col = df["source"].fillna("unknown").astype(str)
    fn_col = df["filename"].fillna("").astype(str).str.replace(r"\.[a-zA-Z0-9]+$", "", regex=True)
    crop_col = df["crop"].fillna("").astype(str)
    disease_col = df["disease"].fillna("").astype(str)

    combined_keys = (source_col + "::" + fn_col + "::" + crop_col + "::" + disease_col).values
    group_ids = [hashlib.md5(k.encode("utf-8")).hexdigest()[:12] for k in combined_keys]

    df["group_id"] = group_ids
    return df
