from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


class DebertaClassifier:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_length: int = 2048,
        batch_size: int = 8,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.use_fp16 = use_fp16
        self.model = None
        self.tokenizer = None

    def load(self) -> "DebertaClassifier":
        cfg = AutoConfig.from_pretrained(
            f"{self.model_path}/ema_model",
            num_labels=2,
            problem_type="single_label_classification",
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            f"{self.model_path}/ema_model",
            config=cfg,
        ).to(self.device)
        self.model.eval()
        if self.use_fp16 and self.device.startswith("cuda"):
            self.model = self.model.half()
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return self

    @torch.no_grad()
    def predict_proba(self, prompts: Sequence[str]) -> np.ndarray:
        outputs = []
        for start in range(0, len(prompts), self.batch_size):
            batch = list(prompts[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
                add_special_tokens=True,
            ).to(self.device)
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            outputs.append(probs)
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 2))
