"""Evaluation and reporting pipeline for SAGE classification."""
from sage.evaluation.metrics import calculate_classification_metrics
from sage.evaluation.confusion import compute_confusion_matrix, find_top_confusions, plot_confusion_matrix
from sage.evaluation.report import generate_final_report


def run_evaluation_pipeline(*args, **kwargs):
    from sage.evaluation.evaluate import run_evaluation_pipeline as _fn
    return _fn(*args, **kwargs)


__all__ = [
    "calculate_classification_metrics",
    "compute_confusion_matrix",
    "find_top_confusions",
    "plot_confusion_matrix",
    "generate_final_report",
    "run_evaluation_pipeline",
]
