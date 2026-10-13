"""SAGE data pipeline module."""
from sage.data.schema import SAGE_RAW_COLUMNS, SAGE_PROJECTED_COLUMNS, validate_dataframe_schema
from sage.data.labels import normalize_label_string, match_class
from sage.data.inventory import audit_dataset_inventory
from sage.data.dedup import identify_duplicate_groups
from sage.data.sampling import perform_balanced_sampling
from sage.data.manifest import build_sample_manifest, validate_sample_manifest
from sage.data.image_validation import process_image_safely
from sage.data.streaming import SAGEImageDataset

__all__ = [
    "SAGE_RAW_COLUMNS",
    "SAGE_PROJECTED_COLUMNS",
    "validate_dataframe_schema",
    "normalize_label_string",
    "match_class",
    "audit_dataset_inventory",
    "identify_duplicate_groups",
    "perform_balanced_sampling",
    "build_sample_manifest",
    "validate_sample_manifest",
    "process_image_safely",
    "SAGEImageDataset",
]
