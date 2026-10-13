try:
    import pytest
except ImportError:
    pytest = None
from sage.config import load_classes_config
from sage.data.labels import normalize_label_string, match_class, parse_model_prediction


def test_normalize_label_string():
    assert normalize_label_string("tomato_leaf_curl_virus") == "Tomato Leaf Curl Virus"
    assert normalize_label_string("corn-common-rust") == "Corn Common Rust"
    assert normalize_label_string("  rice   blast  ") == "Rice Blast"
    assert normalize_label_string(None) == ""


def test_match_class():
    cfg = load_classes_config("configs/classes.yaml")
    
    # Exact match
    c1 = match_class("Tomato", "Late Blight", "Tomato/Late_Blight", cfg)
    assert c1 is not None
    assert c1.class_id == 2
    assert c1.crop == "Tomato"
    assert c1.disease == "Late Blight"

    # Alias crop match (Maize -> Corn)
    c2 = match_class("Maize", "Common Rust", "Maize/Common_Rust", cfg)
    assert c2 is not None
    assert c2.class_id == 6
    assert c2.crop == "Corn"

    # Alias disease match (Bacterial Blight -> Bacterial Leaf Blight)
    c3 = match_class("Rice", "Bacterial Blight", "Rice/Bacterial_Blight", cfg)
    assert c3 is not None
    assert c3.class_id == 5

    # Non-matching crop/disease
    c_none = match_class("Apple", "Scab", "Apple/Scab", cfg)
    assert c_none is None


def test_parse_model_prediction():
    text_standard = "Crop: Tomato\nDisease: Late Blight"
    crop, disease = parse_model_prediction(text_standard)
    assert crop == "Tomato"
    assert disease == "Late Blight"

    text_messy = "Based on visual symptoms:\nCrop: Rice\nDisease: Blast (severe leaf lesions)"
    crop, disease = parse_model_prediction(text_messy)
    assert crop == "Rice"
    assert "Blast" in disease

    text_invalid = "I am unsure what plant this is."
    crop, disease = parse_model_prediction(text_invalid)
    assert crop is None
    assert disease is None
