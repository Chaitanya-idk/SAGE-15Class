from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import yaml


@dataclass
class ClassItem:
    class_id: int
    crop: str
    disease: str
    class_name: str
    aliases: Dict[str, List[str]] = field(default_factory=dict)
    raw_labels: List[str] = field(default_factory=list)
    min_required_samples: int = 1000


@dataclass
class ClassesConfig:
    version: str
    description: str
    target_classes_count: int
    classes: List[ClassItem]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassesConfig":
        classes = [ClassItem(**c) for c in data.get("classes", [])]
        return cls(
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            target_classes_count=data.get("target_classes_count", len(classes)),
            classes=classes
        )


@dataclass
class DatasetConfig:
    source_type: str = "huggingface"
    hf_repo_id: str = "tirtho149/SAGE"
    metadata_file: str = "sage_metadata.parquet"
    canonical_mapping_file: str = "canonical_mapping.json"
    kaggle_input_dir: str = "/kaggle/input/sage"
    cache_dir: str = "artifacts/cache"
    max_cache_size_gb: float = 10.0

    # Sampling
    seed: int = 42
    target_train_images_per_class: int = 800
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    group_by_columns: List[str] = field(default_factory=lambda: ["source", "filename"])

    # Preprocessing
    target_image_size: List[int] = field(default_factory=lambda: [488, 488])
    preserve_aspect_ratio: bool = True
    interpolation: str = "bicubic"
    apply_exif_orientation: bool = True
    to_rgb: bool = True

    # Manifest paths
    output_dir: str = "artifacts/metadata"
    manifest_path: str = "artifacts/metadata/sample_manifest.parquet"
    class_registry_path: str = "artifacts/metadata/class_registry.json"
    class_selection_path: str = "artifacts/metadata/class_selection.csv"
    split_summary_path: str = "artifacts/metadata/split_summary.json"
    data_quality_report_path: str = "artifacts/metadata/data_quality_report.md"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        d_sec = data.get("dataset", {})
        s_sec = data.get("sampling", {})
        p_sec = data.get("preprocessing", {})
        m_sec = data.get("manifest", {})
        splits = s_sec.get("split_ratios", {})

        return cls(
            source_type=d_sec.get("source_type", "huggingface"),
            hf_repo_id=d_sec.get("hf_repo_id", "tirtho149/SAGE"),
            metadata_file=d_sec.get("metadata_file", "sage_metadata.parquet"),
            canonical_mapping_file=d_sec.get("canonical_mapping_file", "canonical_mapping.json"),
            kaggle_input_dir=d_sec.get("kaggle_input_dir", "/kaggle/input/sage"),
            cache_dir=d_sec.get("cache_dir", "artifacts/cache"),
            max_cache_size_gb=d_sec.get("max_cache_size_gb", 10.0),
            seed=s_sec.get("seed", 42),
            target_train_images_per_class=s_sec.get("target_train_images_per_class", 800),
            train_split=splits.get("train", 0.70),
            val_split=splits.get("val", 0.15),
            test_split=splits.get("test", 0.15),
            group_by_columns=s_sec.get("group_by_columns", ["source", "filename"]),
            target_image_size=p_sec.get("target_image_size", [488, 488]),
            preserve_aspect_ratio=p_sec.get("preserve_aspect_ratio", True),
            interpolation=p_sec.get("interpolation", "bicubic"),
            apply_exif_orientation=p_sec.get("apply_exif_orientation", True),
            to_rgb=p_sec.get("to_rgb", True),
            output_dir=m_sec.get("output_dir", "artifacts/metadata"),
            manifest_path=m_sec.get("manifest_path", "artifacts/metadata/sample_manifest.parquet"),
            class_registry_path=m_sec.get("class_registry_path", "artifacts/metadata/class_registry.json"),
            class_selection_path=m_sec.get("class_selection_path", "artifacts/metadata/class_selection.csv"),
            split_summary_path=m_sec.get("split_summary_path", "artifacts/metadata/split_summary.json"),
            data_quality_report_path=m_sec.get("data_quality_report_path", "artifacts/metadata/data_quality_report.md")
        )


@dataclass
class TrainingConfig:
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    torch_dtype: str = "auto"
    use_gradient_checkpointing: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # Loop
    epochs: int = 2
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    seed: int = 42
    save_every_n_steps: int = 100
    logging_steps: int = 10

    # Prompt
    instruction: str = "Identify the crop and plant disease visible in the image. Return the output strictly in the format:\nCrop: <Crop>\nDisease: <Disease>"
    max_new_tokens: int = 64

    # Checkpoint
    checkpoint_dir: str = "artifacts/checkpoints"
    save_best: bool = True
    progress_file: str = "artifacts/checkpoints/training_progress.json"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        m_sec = data.get("model", {})
        q_sec = data.get("quantization", {})
        l_sec = data.get("lora", {})
        t_sec = data.get("training", {})
        p_sec = data.get("prompt", {})
        c_sec = data.get("checkpointing", {})

        return cls(
            model_id=m_sec.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct"),
            torch_dtype=m_sec.get("torch_dtype", "auto"),
            use_gradient_checkpointing=m_sec.get("use_gradient_checkpointing", True),
            load_in_4bit=q_sec.get("load_in_4bit", True),
            bnb_4bit_quant_type=q_sec.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=q_sec.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=q_sec.get("bnb_4bit_compute_dtype", "bfloat16"),
            lora_r=l_sec.get("r", 16),
            lora_alpha=l_sec.get("lora_alpha", 32),
            lora_dropout=l_sec.get("lora_dropout", 0.05),
            target_modules=l_sec.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            epochs=t_sec.get("epochs", 2),
            micro_batch_size=t_sec.get("micro_batch_size", 2),
            gradient_accumulation_steps=t_sec.get("gradient_accumulation_steps", 8),
            learning_rate=float(t_sec.get("learning_rate", 2.0e-4)),
            lr_scheduler_type=t_sec.get("lr_scheduler_type", "cosine"),
            warmup_ratio=t_sec.get("warmup_ratio", 0.05),
            weight_decay=t_sec.get("weight_decay", 0.01),
            seed=t_sec.get("seed", 42),
            save_every_n_steps=t_sec.get("save_every_n_steps", 100),
            logging_steps=t_sec.get("logging_steps", 10),
            instruction=p_sec.get("instruction", "Identify the crop and plant disease visible in the image. Return the output strictly in the format:\nCrop: <Crop>\nDisease: <Disease>"),
            max_new_tokens=p_sec.get("max_new_tokens", 64),
            checkpoint_dir=c_sec.get("checkpoint_dir", "artifacts/checkpoints"),
            save_best=c_sec.get("save_best", True),
            progress_file=c_sec.get("progress_file", "artifacts/checkpoints/training_progress.json")
        )


@dataclass
class EvaluationConfig:
    batch_size: int = 4
    splits: List[str] = field(default_factory=lambda: ["val", "test"])
    temperature: float = 0.0
    max_new_tokens: int = 64
    reports_dir: str = "artifacts/reports"
    predictions_dir: str = "artifacts/predictions"
    final_report_md: str = "artifacts/reports/final_report.md"
    final_report_json: str = "artifacts/reports/final_report.json"
    classification_report_csv: str = "artifacts/reports/classification_report.csv"
    class_metrics_csv: str = "artifacts/reports/class_metrics.csv"
    crop_metrics_csv: str = "artifacts/reports/crop_metrics.csv"
    top_confusions_csv: str = "artifacts/reports/top_confusions.csv"
    failures_csv: str = "artifacts/reports/failures.csv"
    confusion_matrix_png: str = "artifacts/reports/confusion_matrix.png"
    test_predictions_parquet: str = "artifacts/predictions/test_predictions.parquet"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        e_sec = data.get("evaluation", {})
        o_sec = data.get("outputs", {})
        return cls(
            batch_size=e_sec.get("batch_size", 4),
            splits=e_sec.get("splits", ["val", "test"]),
            temperature=e_sec.get("temperature", 0.0),
            max_new_tokens=e_sec.get("max_new_tokens", 64),
            reports_dir=o_sec.get("reports_dir", "artifacts/reports"),
            predictions_dir=o_sec.get("predictions_dir", "artifacts/predictions"),
            final_report_md=o_sec.get("final_report_md", "artifacts/reports/final_report.md"),
            final_report_json=o_sec.get("final_report_json", "artifacts/reports/final_report.json"),
            classification_report_csv=o_sec.get("classification_report_csv", "artifacts/reports/classification_report.csv"),
            class_metrics_csv=o_sec.get("class_metrics_csv", "artifacts/reports/class_metrics.csv"),
            crop_metrics_csv=o_sec.get("crop_metrics_csv", "artifacts/reports/crop_metrics.csv"),
            top_confusions_csv=o_sec.get("top_confusions_csv", "artifacts/reports/top_confusions.csv"),
            failures_csv=o_sec.get("failures_csv", "artifacts/reports/failures.csv"),
            confusion_matrix_png=o_sec.get("confusion_matrix_png", "artifacts/reports/confusion_matrix.png"),
            test_predictions_parquet=o_sec.get("test_predictions_parquet", "artifacts/predictions/test_predictions.parquet")
        )


def load_yaml(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_classes_config(file_path: str = "configs/classes.yaml") -> ClassesConfig:
    return ClassesConfig.from_dict(load_yaml(file_path))


def load_dataset_config(file_path: str = "configs/dataset.yaml") -> DatasetConfig:
    return DatasetConfig.from_dict(load_yaml(file_path))


def load_training_config(file_path: str = "configs/training.yaml") -> TrainingConfig:
    return TrainingConfig.from_dict(load_yaml(file_path))


def load_evaluation_config(file_path: str = "configs/evaluation.yaml") -> EvaluationConfig:
    return EvaluationConfig.from_dict(load_yaml(file_path))
