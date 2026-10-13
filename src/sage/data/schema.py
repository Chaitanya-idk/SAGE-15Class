from typing import Any, Dict, List, Optional
import pandas as pd


SAGE_RAW_COLUMNS = [
    "crop",
    "disease",
    "plant_organ",
    "visual_symptoms",
    "pathogen",
    "disease_type",
    "symptom_source",
    "symptom_quote",
    "source",
    "filename",
    "raw_label",
    "image" # optional in metadata-only file
]

# Minimal columns required for metadata discovery, sampling, and deduplication
SAGE_PROJECTED_COLUMNS = [
    "crop",
    "disease",
    "source",
    "filename",
    "raw_label"
]

SAGE_COLUMN_TYPES = {
    "crop": "string",
    "disease": "string",
    "plant_organ": "string",
    "visual_symptoms": "string",
    "pathogen": "string",
    "disease_type": "string",
    "symptom_source": "string",
    "symptom_quote": "string",
    "source": "string",
    "filename": "string",
    "raw_label": "string",
}


def validate_dataframe_schema(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validates dataframe columns against the SAGE schema and reports anomalies."""
    if required_columns is None:
        required_columns = SAGE_PROJECTED_COLUMNS

    present_columns = df.columns.tolist()
    missing_columns = [col for col in required_columns if col not in present_columns]
    unexpected_columns = [col for col in present_columns if col not in SAGE_RAW_COLUMNS]

    return {
        "valid": len(missing_columns) == 0,
        "present_columns": present_columns,
        "missing_required_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "total_rows": len(df)
    }


def project_metadata_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Projects only needed columns safely, inserting NaN for missing optional columns."""
    if columns is None:
        columns = SAGE_PROJECTED_COLUMNS

    projected_data = {}
    for col in columns:
        if col in df.columns:
            projected_data[col] = df[col]
        else:
            projected_data[col] = pd.Series([None] * len(df), index=df.index, dtype="object")

    return pd.DataFrame(projected_data)
