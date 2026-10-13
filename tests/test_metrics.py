try:
    import pytest
except ImportError:
    pytest = None
from sage.evaluation.metrics import calculate_classification_metrics


def test_calculate_classification_metrics():
    gt_crops = ["Tomato", "Rice", "Corn", "Potato"]
    gt_diseases = ["Late Blight", "Blast", "Common Rust", "Late Blight"]
    gt_classes = [
        "Tomato::Late Blight",
        "Rice::Blast",
        "Corn::Common Rust",
        "Potato::Late Blight"
    ]

    # 3 correct, 1 wrong disease
    pred_crops = ["Tomato", "Rice", "Corn", "Potato"]
    pred_diseases = ["Late Blight", "Blast", "Common Rust", "Early Blight"]
    pred_classes = [
        "Tomato::Late Blight",
        "Rice::Blast",
        "Corn::Common Rust",
        "Potato::Early Blight"
    ]

    all_classes = [
        "Tomato::Late Blight",
        "Rice::Blast",
        "Corn::Common Rust",
        "Potato::Late Blight",
        "Potato::Early Blight"
    ]

    metrics = calculate_classification_metrics(
        ground_truth_crops=gt_crops,
        ground_truth_diseases=gt_diseases,
        ground_truth_classes=gt_classes,
        predicted_crops=pred_crops,
        predicted_diseases=pred_diseases,
        predicted_classes=pred_classes,
        all_class_names=all_classes
    )

    assert metrics["crop_accuracy"] == 1.0 # all 4 crops correct
    assert metrics["disease_accuracy"] == 0.75 # 3 out of 4 diseases correct
    assert metrics["exact_match_accuracy"] == 0.75 # 3 out of 4 exact matches
    assert metrics["parsing_failures"] == 0
    assert metrics["macro_f1"] > 0.0
