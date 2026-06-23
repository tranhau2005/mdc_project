from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class PathConfig:
    project_root: Path
    data_dir: Path
    pdf_dir: Path
    xml_dir: Path
    train_labels_path: Optional[Path]
    database_dir: Path
    output_dir: Path
    log_dir: Path


@dataclass
class RetrievalConfig:
    text_chunk_size: int = 500
    doi_min_len: int = 5
    doi_max_len: int = 100
    use_xml_fallback_for_doi: bool = True
    use_xml_for_accession: bool = True
    max_xml_acc_additions: int = 10
    pdf_text_mode: str = "plain"


@dataclass
class ModelConfig:
    model_paths: list[str] = field(default_factory=list)
    tokenizer_path: str = ""
    max_length: int = 2048
    batch_size: int = 8
    device: str = "cuda"
    use_fp16: bool = True


@dataclass
class ThresholdConfig:
    doi_primary_threshold: float = 0.5
    acc_secondary_quantile: float = 0.1


@dataclass
class RuntimeConfig:
    seed: int = 42
    num_workers: int = 4
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    linux_mode: bool = True
    debug: bool = False




@dataclass
class TrainConfig:
    input_csv: Optional[str] = None
    base_model_path: str = ""
    tokenizer_path: str = ""
    output_dir: str = "./artifacts/models"
    run_name_prefix: str = "deberta-new"
    num_folds: int = 6
    fold_index: Optional[int] = None
    epochs: float = 3.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clipping: float = 1.0
    gradient_accumulation_steps: int = 1
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    logging_steps: int = 25
    ema_decay: float = 0.9995
    ema_update_every: int = 1
    ema_update_after_step: int = 50
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    dataloader_num_workers: int = 0
    report_to: list[str] = field(default_factory=list)
    drop_missing: bool = True
    use_class_weights: bool = True
    apply_title_heuristic: bool = True

@dataclass
class AppConfig:
    paths: PathConfig
    retrieval: RetrievalConfig
    model: ModelConfig
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _resolve_path(project_root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def load_config(config_path: str | Path) -> AppConfig:
    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text())

    root_from_yaml = raw["paths"].get("project_root", ".")
    project_root = _resolve_path(config_path.parent, root_from_yaml)
    assert project_root is not None

    paths = PathConfig(
        project_root=project_root,
        data_dir=_resolve_path(project_root, raw["paths"]["data_dir"]),
        pdf_dir=_resolve_path(project_root, raw["paths"]["pdf_dir"]),
        xml_dir=_resolve_path(project_root, raw["paths"]["xml_dir"]),
        train_labels_path=_resolve_path(project_root, raw["paths"].get("train_labels_path")),
        database_dir=_resolve_path(project_root, raw["paths"]["database_dir"]),
        output_dir=_resolve_path(project_root, raw["paths"]["output_dir"]),
        log_dir=_resolve_path(project_root, raw["paths"]["log_dir"]),
    )

    retrieval = RetrievalConfig(**raw.get("retrieval", {}))
    model = ModelConfig(**raw.get("model", {}))
    thresholds = ThresholdConfig(**raw.get("thresholds", {}))
    runtime = RuntimeConfig(**raw.get("runtime", {}))
    train = TrainConfig(**raw.get("train", {}))

    return AppConfig(paths=paths, retrieval=retrieval, model=model, thresholds=thresholds, runtime=runtime, train=train)
