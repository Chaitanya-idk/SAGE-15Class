import re
from typing import Optional, Tuple
from sage.config import ClassItem, ClassesConfig


def normalize_label_string(s: Optional[str]) -> str:
    """Normalizes label text by stripping whitespace, standardizing separators, and casing."""
    if s is None:
        return ""
    # Replace underscores and hyphens with spaces
    cleaned = re.sub(r"[_\-]+", " ", str(s))
    # Replace multiple spaces with a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title()


def match_class(crop: Optional[str], disease: Optional[str], raw_label: Optional[str], classes_config: ClassesConfig) -> Optional[ClassItem]:
    """Matches a record's crop, disease, and raw_label against the approved class registry."""
    norm_crop = normalize_label_string(crop)
    norm_disease = normalize_label_string(disease)
    norm_raw = str(raw_label).strip() if raw_label else ""

    for item in classes_config.classes:
        # Check raw labels exact match
        if norm_raw and norm_raw in item.raw_labels:
            return item

        # Check crop match (canonical or alias)
        crop_match = (norm_crop.lower() == item.crop.lower()) or any(
            norm_crop.lower() == a.lower() for a in item.aliases.get("crops", [])
        )

        if not crop_match:
            continue

        # Check disease match (canonical or alias)
        disease_match = (norm_disease.lower() == item.disease.lower()) or any(
            norm_disease.lower() == a.lower() for a in item.aliases.get("diseases", [])
        )

        if disease_match:
            return item

    return None


def parse_model_prediction(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parses model response into (predicted_crop, predicted_disease).
    
    Expected format:
    Crop: <Crop>
    Disease: <Disease>
    """
    if not text:
        return None, None

    crop = None
    disease = None

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("crop:"):
            crop = normalize_label_string(line.split(":", 1)[1])
        elif line.lower().startswith("disease:"):
            disease = normalize_label_string(line.split(":", 1)[1])

    # Fallback heuristic if format wasn't strictly newline separated
    if crop is None or disease is None:
        crop_match = re.search(r"crop:\s*([^\n,]+)", text, re.IGNORECASE)
        disease_match = re.search(r"disease:\s*([^\n,]+)", text, re.IGNORECASE)
        if crop_match and crop is None:
            crop = normalize_label_string(crop_match.group(1))
        if disease_match and disease is None:
            disease = normalize_label_string(disease_match.group(1))

    return crop, disease
